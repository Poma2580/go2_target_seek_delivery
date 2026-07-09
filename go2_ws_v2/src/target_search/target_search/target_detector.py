import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped, PoseStamped
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import math
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_point

COCO_CLASSES = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic_light',
    10: 'fire_hydrant',
    11: 'stop_sign',
    12: 'parking_meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports_ball',
    33: 'kite',
    34: 'baseball_bat',
    35: 'baseball_glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis_racket',
    39: 'bottle',
    40: 'wine_glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot_dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted_plant',
    59: 'bed',
    60: 'dining_table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell_phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy_bear',
    78: 'hair_drier',
    79: 'toothbrush'
}

CLASS_NAME_TO_ID = {v: k for k, v in COCO_CLASSES.items()}


class TargetDetectorNode(Node):
    def __init__(self):
        super().__init__('target_detector')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('rgb_topic', '/go2_1/camera/image_raw'),
                ('depth_topic', '/go2_1/camera/depth/image_raw'),
                ('odom_topic', '/go2_1/odom'),
                ('camera_link_frame', 'camera_link'),
                ('base_link_frame', 'base_link'),
                ('odom_frame', 'odom'),
                ('map_frame', 'map'),
                ('target_class', 'airplane'),
                ('target_class_id', 4),
                ('confidence_threshold', 0.3),
                ('image_width', 640),
                ('image_height', 480),
                ('fov_rad', 1.047),
                ('depth_min', 0.05),
                ('depth_max', 20.0),
                ('yolo_model_path', ''),
                ('yolo_model_type', 'yolov8'),
                ('publish_visualization', True),
                ('use_compressed', False),
                ('detection_rate', 10.0),
                ('camera_offset_x', 0.28),
                ('camera_offset_y', 0.0),
                ('camera_offset_z', 0.12),
                ('rgb_offset_x', 0.02),
                ('rgb_offset_y', -0.02),
                ('rgb_offset_z', 0.0),
                ('depth_offset_x', 0.02),
                ('depth_offset_y', 0.02),
                ('depth_offset_z', 0.0),
            ]
        )

        self.rgb_topic = self.get_parameter('rgb_topic').value
        self.depth_topic = self.get_parameter('depth_topic').value
        self.odom_topic = self.get_parameter('odom_topic').value
        self.camera_link_frame = self.get_parameter('camera_link_frame').value
        self.base_link_frame = self.get_parameter('base_link_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.map_frame = self.get_parameter('map_frame').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.image_width = self.get_parameter('image_width').value
        self.image_height = self.get_parameter('image_height').value
        self.fov_rad = self.get_parameter('fov_rad').value
        self.depth_min = self.get_parameter('depth_min').value
        self.depth_max = self.get_parameter('depth_max').value
        self.yolo_model_path = self.get_parameter('yolo_model_path').value
        self.yolo_model_type = self.get_parameter('yolo_model_type').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        self.use_compressed = self.get_parameter('use_compressed').value
        self.detection_rate = self.get_parameter('detection_rate').value

        self.camera_offset_x = self.get_parameter('camera_offset_x').value
        self.camera_offset_y = self.get_parameter('camera_offset_y').value
        self.camera_offset_z = self.get_parameter('camera_offset_z').value
        self.rgb_offset_x = self.get_parameter('rgb_offset_x').value
        self.rgb_offset_y = self.get_parameter('rgb_offset_y').value
        self.rgb_offset_z = self.get_parameter('rgb_offset_z').value
        self.depth_offset_x = self.get_parameter('depth_offset_x').value
        self.depth_offset_y = self.get_parameter('depth_offset_y').value
        self.depth_offset_z = self.get_parameter('depth_offset_z').value

        target_class_name = self.get_parameter('target_class').value
        target_class_id = self.get_parameter('target_class_id').value

        if target_class_name in CLASS_NAME_TO_ID:
            self.target_class_id = CLASS_NAME_TO_ID[target_class_name]
            self.target_class = target_class_name
            self.get_logger().info(f'Target class set by name: {self.target_class} (ID: {self.target_class_id})')
        else:
            if target_class_id in COCO_CLASSES:
                self.target_class_id = target_class_id
                self.target_class = COCO_CLASSES[target_class_id]
                self.get_logger().info(f'Target class set by ID: {self.target_class} (ID: {self.target_class_id})')
            else:
                self.target_class_id = 0
                self.target_class = 'person'
                self.get_logger().warn(f'Invalid target_class "{target_class_name}" and target_class_id "{target_class_id}". Using default: person (ID: 0)')

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.latest_rgb_image = None
        self.latest_depth_image = None
        self.latest_odom = None
        self.first_detection_done = False

        self.fx = self.image_width / (2.0 * math.tan(self.fov_rad / 2.0))
        self.fy = self.fx
        self.cx = self.image_width / 2.0
        self.cy = self.image_height / 2.0

        self.frame_candidates = [
            self.map_frame,
            self.odom_frame,
            'go2_1/odom',
            'go2_1/base_link',
        ]

        if self.use_compressed:
            self.sub_rgb = self.create_subscription(
                CompressedImage,
                self.rgb_topic,
                self.rgb_callback_compressed,
                10
            )
            self.sub_depth = self.create_subscription(
                CompressedImage,
                self.depth_topic,
                self.depth_callback_compressed,
                10
            )
        else:
            self.sub_rgb = self.create_subscription(
                Image,
                self.rgb_topic,
                self.rgb_callback_uncompressed,
                10
            )
            self.sub_depth = self.create_subscription(
                Image,
                self.depth_topic,
                self.depth_callback_uncompressed,
                10
            )

        self.sub_odom = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10
        )

        self.pub_target_point = self.create_publisher(
            PoseStamped,
            '/target_search/target_pose',
            10
        )
        self.pub_visualization = self.create_publisher(
            Image,
            '/target_detector/detected_image',
            10
        )

        timer_period = 1.0 / self.detection_rate
        self.timer = self.create_timer(timer_period, self.process_frame)

        self.yolo_model = None
        self.init_yolo()

    def init_yolo(self):
        try:
            if self.yolo_model_type.lower() == 'yolov8':
                from ultralytics import YOLO
                if self.yolo_model_path:
                    self.yolo_model = YOLO(self.yolo_model_path)
                else:
                    self.yolo_model = YOLO('yolov8n.pt')
            else:
                self.get_logger().error(f'Unsupported YOLO model type: {self.yolo_model_type}')
        except ImportError:
            self.get_logger().error('Failed to import YOLO library. Install ultralytics first.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize YOLO model: {str(e)}')

    def rgb_callback_uncompressed(self, msg):
        try:
            self.latest_rgb_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'RGB conversion error: {e}')

    def depth_callback_uncompressed(self, msg):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
            self._process_depth(cv_depth)
        except CvBridgeError as e:
            self.get_logger().error(f'Depth conversion error: {e}')
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def rgb_callback_compressed(self, msg):
        try:
            self.latest_rgb_image = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'RGB compressed conversion error: {e}')

    def depth_callback_compressed(self, msg):
        try:
            cv_depth = self.bridge.compressed_imgmsg_to_cv2(msg, 'passthrough')
            self._process_depth(cv_depth)
        except CvBridgeError as e:
            self.get_logger().error(f'Depth compressed conversion error: {e}')
        except Exception as e:
            self.get_logger().error(f'Depth processing error: {e}')

    def _process_depth(self, cv_depth):
        if cv_depth.dtype == np.uint16:
            self.latest_depth_image = cv_depth.astype(np.float32) / 1000.0
        elif cv_depth.dtype == np.float32:
            self.latest_depth_image = cv_depth
        else:
            self.latest_depth_image = cv_depth.astype(np.float32)

    def odom_callback(self, msg):
        self.latest_odom = msg

    def pixel_to_camera(self, u, v, depth):
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        z = depth
        return float(x), float(y), float(z)

    def get_valid_transform(self, source_frame):
        for target_frame in self.frame_candidates:
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame,
                    source_frame,
                    rclpy.time.Time()
                )
                return transform, target_frame
            except Exception:
                pass
        return None, None

    def process_frame(self):
        if self.latest_rgb_image is None or self.latest_depth_image is None:
            return

        if self.yolo_model is None:
            return

        display_image = self.latest_rgb_image.copy()

        try:
            results = self.yolo_model(self.latest_rgb_image, verbose=False)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])

                    if class_id == self.target_class_id and confidence >= self.confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        center_u = (x1 + x2) / 2.0
                        center_v = (y1 + y2) / 2.0

                        if self.publish_visualization:
                            cv2.rectangle(display_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            label = f'{self.target_class} {confidence:.2f}'
                            cv2.putText(display_image, label, (int(x1), int(y1)-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        u_int = int(round(center_u))
                        v_int = int(round(center_v))
                        if not (0 <= u_int < self.latest_depth_image.shape[1] and
                                0 <= v_int < self.latest_depth_image.shape[0]):
                            self.get_logger().error(f'Pixel coordinates out of bounds: ({u_int}, {v_int})')
                            continue

                        depth_val = float(self.latest_depth_image[v_int, u_int])
                        if depth_val < self.depth_min or depth_val > self.depth_max or np.isnan(depth_val) or np.isinf(depth_val):
                            self.get_logger().error(f'Invalid depth value: {depth_val} (min={self.depth_min}, max={self.depth_max})')
                            continue

                        self.get_logger().error(f'Depth value at ({u_int}, {v_int}): {depth_val}')

                        cam_x, cam_y, cam_z = self.pixel_to_camera(center_u, center_v, depth_val)

                        cam_x += self.depth_offset_x
                        cam_y += self.depth_offset_y
                        cam_z += self.depth_offset_z

                        target_point_camera = PointStamped()
                        target_point_camera.header.stamp = self.get_clock().now().to_msg()
                        target_point_camera.header.frame_id = self.camera_link_frame
                        target_point_camera.point.x = cam_x
                        target_point_camera.point.y = cam_y
                        target_point_camera.point.z = cam_z

                        self.get_logger().error(f'Target in camera frame: x={cam_x:.3f}, y={cam_y:.3f}, z={cam_z:.3f}')

                        transform, target_frame_used = self.get_valid_transform(self.camera_link_frame)

                        if transform is None:
                            self.get_logger().error('TF transform not available, trying odom-based calculation')
                            if self.latest_odom is not None:
                                global_pose = self.calculate_global_coords_odom(cam_x, cam_y, cam_z)
                                if global_pose is not None:
                                    target_pose = PoseStamped()
                                    target_pose.header.stamp = self.get_clock().now().to_msg()
                                    target_pose.header.frame_id = self.odom_frame
                                    target_pose.pose.position.x = float(global_pose.x)
                                    target_pose.pose.position.y = float(global_pose.y)
                                    target_pose.pose.position.z = float(global_pose.z)
                                    target_pose.pose.orientation.w = 1.0
                                    self.pub_target_point.publish(target_pose)
                                    self.get_logger().error(f'Published target pose via odom: ({global_pose.x:.3f}, {global_pose.y:.3f}, {global_pose.z:.3f})')

                                    if not self.first_detection_done:
                                        self.on_first_detection(target_pose)
                                        self.first_detection_done = True
                            else:
                                self.get_logger().error('No odom data available for fallback calculation')
                            continue

                        self.get_logger().error(f'TF transform found, target_frame: {target_frame_used}')

                        target_point_global = do_transform_point(target_point_camera, transform)

                        target_pose = PoseStamped()
                        target_pose.header = target_point_global.header
                        target_pose.pose.position.x = float(target_point_global.point.x)
                        target_pose.pose.position.y = float(target_point_global.point.y)
                        target_pose.pose.position.z = float(target_point_global.point.z)
                        target_pose.pose.orientation.w = 1.0
                        self.pub_target_point.publish(target_pose)
                        self.get_logger().error(f'Published target pose via TF: ({target_point_global.point.x:.3f}, {target_point_global.point.y:.3f}, {target_point_global.point.z:.3f})')

                        if not self.first_detection_done:
                            self.on_first_detection(target_pose)
                            self.first_detection_done = True

            if self.publish_visualization and self.pub_visualization.get_subscription_count() > 0:
                vis_msg = self.bridge.cv2_to_imgmsg(display_image, 'bgr8')
                vis_msg.header.stamp = self.get_clock().now().to_msg()
                vis_msg.header.frame_id = self.camera_link_frame
                self.pub_visualization.publish(vis_msg)

        except Exception as e:
            self.get_logger().error(f'Detection error: {e}')

    def calculate_global_coords_odom(self, cam_x, cam_y, cam_z):
        if self.latest_odom is None:
            return None

        odom_pose = self.latest_odom.pose.pose
        robot_x = odom_pose.position.x
        robot_y = odom_pose.position.y
        robot_z = odom_pose.position.z

        qx = odom_pose.orientation.x
        qy = odom_pose.orientation.y
        qz = odom_pose.orientation.z
        qw = odom_pose.orientation.w

        cam_rel_x = self.camera_offset_x
        cam_rel_y = self.camera_offset_y
        cam_rel_z = self.camera_offset_z

        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        target_rel_x = cam_rel_x + cam_z * math.cos(yaw) - cam_x * math.sin(yaw)
        target_rel_y = cam_rel_y + cam_z * math.sin(yaw) + cam_x * math.cos(yaw)
        target_rel_z = cam_rel_z + cam_y

        global_x = robot_x + target_rel_x
        global_y = robot_y + target_rel_y
        global_z = robot_z + target_rel_z

        point = PointStamped()
        point.point.x = global_x
        point.point.y = global_y
        point.point.z = global_z
        return point.point

    def on_first_detection(self, target_pose):
        """
        Called when the target is first detected.
        Override this function in derived classes for custom behavior.
        """
        pass


def main(args=None):
    rclpy.init(args=args)
    node = TargetDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
