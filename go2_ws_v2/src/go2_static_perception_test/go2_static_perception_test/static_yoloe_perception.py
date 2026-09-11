#!/usr/bin/env python3
"""Single-prompt static RGB-D perception, independent of the walking-target chain."""

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

import message_filters
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, PoseStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
import tf2_geometry_msgs
import tf2_ros


DEFAULTS = {
    'robot_namespace': 'go2_1', 'model_path': 'yoloe-26s-seg.pt',
    'model_cache_dir': '~/.cache/go2_static_yolo_smoke',
    'target_prompt': 'person', 'rgb_topic': '', 'depth_topic': '',
    'camera_info_topic': '', 'pose_topic': '', 'result_status_topic': '',
    'debug_image_topic': '', 'target_frame': '',
    'conf': 0.15, 'imgsz': 960, 'device': 'cpu',
    'depth_window': 5, 'depth_percentile': 30.0, 'min_depth_samples': 20,
    'min_depth': 0.3, 'max_depth': 25.0,
    'roi_x_min': 0.25, 'roi_x_max': 0.75, 'roi_y_min': 0.20, 'roi_y_max': 0.70,
    'sync_slop': 0.05, 'inference_rate': 8.0, 'max_image_age': 0.30,
    'tf_timeout': 0.1, 'publish_debug': True,
}


def validate_prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('target_prompt must be a non-empty string')
    return value.strip()


def result_status_json(stamp, sample_id, prompt, recognition_success=False,
                       confidence=None, bbox=None, localization_success=False):
    """Serialize the complete v1 status; never emit non-standard JSON numbers."""
    prompt = validate_prompt(prompt)
    if localization_success and not recognition_success:
        raise ValueError('localization requires recognition')
    if not recognition_success:
        confidence, bbox = None, None
    else:
        if confidence is None or bbox is None or len(bbox) != 4:
            raise ValueError('recognition requires confidence and a four-value bbox')
        confidence = float(confidence)
        bbox = [float(value) for value in bbox]
        if not all(math.isfinite(v) for v in [confidence, *bbox]):
            raise ValueError('confidence and bbox must be finite')
    return json.dumps({
        'schema_version': 1,
        'stamp': {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)},
        'sample_id': int(sample_id), 'prompt': prompt,
        'recognition_success': bool(recognition_success),
        'confidence': confidence, 'bbox': bbox,
        'localization_success': bool(localization_success),
    }, allow_nan=False, ensure_ascii=False, separators=(',', ':'))


@contextmanager
def working_directory(path):
    """Temporarily change cwd and restore it even when model loading fails."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def load_model(model_path, prompt, model_cache_dir='~/.cache/go2_static_yolo_smoke'):
    """Load YOLOE and its text encoder through one cwd-independent cache."""
    from ultralytics import YOLOE

    cache_dir = Path(model_cache_dir).expanduser().resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(f'cannot create model cache directory {cache_dir}: {error}') from error

    source = Path(model_path).expanduser()
    cached = cache_dir / source.name
    if source.is_file():
        model_path = str(source.resolve())
    elif source.parent == Path('.') and cached.is_file():
        model_path = str(cached.resolve())
    elif source.is_absolute() or source.parent != Path('.'):
        raise ValueError(f'model file does not exist: {source}')

    # Ultralytics resolves both a bare model name and mobileclip2_b.ts relative
    # to cwd. Keeping that cwd change inside startup makes launch location
    # irrelevant and places both automatic downloads in the same cache.
    with working_directory(cache_dir):
        model = YOLOE(model_path)
        embedding = model.get_text_pe([prompt])
        model.set_classes([prompt], embedding)
    return model


class StaticYoloePerception(Node):
    def __init__(self):
        super().__init__('static_yoloe_perception')
        # These are startup parameters: rejecting changes prevents reported
        # parameters from silently disagreeing with the model/subscriptions.
        for name, default in DEFAULTS.items():
            self.declare_parameter(name, default, ParameterDescriptor(read_only=True))
        values = {key: self.get_parameter(key).value for key in DEFAULTS}
        for name, value in values.items():
            setattr(self, name, value)
        self.target_prompt = validate_prompt(self.target_prompt)
        self.robot_namespace = self.robot_namespace.strip('/')
        if self.robot_namespace != 'go2_1':
            raise ValueError('this phase supports robot_namespace=go2_1 only')
        self._validate_parameters()
        ns = self.robot_namespace
        self.target_frame = self.target_frame or f'{ns}/odom'
        self.rgb_topic = self.rgb_topic or f'/{ns}/camera/image_raw'
        self.depth_topic = self.depth_topic or f'/{ns}/camera/depth/image_raw'
        self.camera_info_topic = self.camera_info_topic or f'/{ns}/camera/depth/camera_info'
        prefix = f'/{ns}/static_perception'
        self.pose_topic = self.pose_topic or prefix + '/target_pose_estimated'
        self.result_status_topic = self.result_status_topic or prefix + '/result_status'
        self.debug_image_topic = self.debug_image_topic or prefix + '/debug_image'

        self.get_logger().info(f'Loading YOLOE {self.model_path}; prompt={self.target_prompt!r}')
        self.model = load_model(
            self.model_path, self.target_prompt, self.model_cache_dir)
        self._yolo_kwargs = dict(conf=self.conf, imgsz=self.imgsz,
                                 device=self.device, verbose=False)
        self.bridge = CvBridge()
        self.camera_info = None
        self._latest_rgbd = None
        self._latest_pair_id = 0
        self._processed_pair_id = -1
        self._last_sensor_stamp = None
        self._last_clock = None
        self._next_sample_id = 0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)
        self.status_pub = self.create_publisher(String, self.result_status_topic, 10)
        self.dbg_pub = (self.create_publisher(Image, self.debug_image_topic,
                                            qos_profile_sensor_data)
                        if self.publish_debug else None)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic,
                                                  self._info_cb, qos_profile_sensor_data)
        self.rgb_sub = message_filters.Subscriber(
            self, Image, self.rgb_topic, qos_profile=qos_profile_sensor_data)
        self.depth_sub = message_filters.Subscriber(
            self, Image, self.depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub], queue_size=10, slop=self.sync_slop)
        self.sync.registerCallback(self._rgbd_cb)
        self.timer = self.create_timer(1.0 / self.inference_rate, self._process_latest_rgbd)
        self.get_logger().info(
            f'RGB={self.rgb_topic}; depth={self.depth_topic}; info={self.camera_info_topic}; '
            f'pose={self.pose_topic} ({self.target_frame}); status={self.result_status_topic}')

    def _validate_parameters(self):
        numeric = ('conf', 'imgsz', 'depth_window', 'depth_percentile',
                   'min_depth_samples', 'min_depth', 'max_depth', 'sync_slop',
                   'inference_rate', 'max_image_age', 'tf_timeout',
                   'roi_x_min', 'roi_x_max', 'roi_y_min', 'roi_y_max')
        if not all(math.isfinite(float(getattr(self, k))) for k in numeric):
            raise ValueError('numeric parameters must be finite')
        if not 0 <= self.conf <= 1 or self.imgsz <= 0:
            raise ValueError('conf must be in [0,1] and imgsz positive')
        if self.depth_window < 0 or self.min_depth_samples < 1:
            raise ValueError('invalid depth_window/min_depth_samples')
        if not 0 <= self.depth_percentile <= 100 or not 0 <= self.min_depth < self.max_depth:
            raise ValueError('invalid depth limits/percentile')
        if self.inference_rate <= 0 or self.sync_slop <= 0:
            raise ValueError('inference_rate and sync_slop must be positive')
        if self.max_image_age <= 0 or self.tf_timeout < 0:
            raise ValueError('invalid max_image_age/tf_timeout')
        for axis in ('x', 'y'):
            if not 0 <= getattr(self, f'roi_{axis}_min') < getattr(self, f'roi_{axis}_max') <= 1:
                raise ValueError('ROI fractions must satisfy 0 <= min < max <= 1')
        if (not self.model_path.strip() or not self.model_cache_dir.strip() or
                not self.device.strip()):
            raise ValueError('model_path/model_cache_dir/device cannot be empty')

    def _warn(self, message):
        self.get_logger().warning(message, throttle_duration_sec=2.0)

    def _info_cb(self, message):
        self.camera_info = message

    def _rgbd_cb(self, rgb, depth):
        self._latest_pair_id += 1
        self._latest_rgbd = self._latest_pair_id, rgb, depth

    def _process_latest_rgbd(self):
        if self._latest_rgbd is None:
            return
        pair_id, rgb, depth = self._latest_rgbd
        if pair_id == self._processed_pair_id:
            return
        self._processed_pair_id = pair_id
        now = self.get_clock().now().nanoseconds
        if self._last_clock is not None and now < self._last_clock:
            self._last_sensor_stamp = None  # Gazebo reset; sample IDs stay monotonic.
        self._last_clock = now
        stamps = (Time.from_msg(rgb.header.stamp).nanoseconds,
                  Time.from_msg(depth.header.stamp).nanoseconds)
        if self._last_sensor_stamp is not None and stamps[1] <= self._last_sensor_stamp:
            return
        self._last_sensor_stamp = stamps[1]
        if any((now - stamp) * 1e-9 > self.max_image_age for stamp in stamps):
            self._warn('Discarding stale RGB-D pair before inference')
            return
        try:
            image = self.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
        except Exception as error:
            self._warn(f'RGB conversion failed before inference: {error}')
            return
        self._infer_sample(image, rgb, depth)

    def _infer_sample(self, image, rgb, depth):
        sample_id = self._next_sample_id
        self._next_sample_id += 1
        recognized, localized = False, False
        confidence, bbox, z, point = None, None, None, None
        reason = 'no detection'
        try:
            best = self._pick_target(self.model(image, **self._yolo_kwargs))
            if best is not None:
                *bbox, confidence = best
                recognized = True
                try:
                    point, z = self._localize(image, rgb, depth, bbox)
                    self._publish_pose(point, depth.header.stamp)
                    localized = True
                    reason = ''
                except Exception as error:
                    reason = str(error)
                    self._warn(f'Localization failed: {error}')
        except Exception as error:
            reason = f'inference failed: {error}'
            self._warn(reason)
        # Status comes before debug; visualization can never erase a result.
        self.status_pub.publish(String(data=result_status_json(
            depth.header.stamp, sample_id, self.target_prompt,
            recognized, confidence, bbox, localized)))
        if self.dbg_pub is not None:
            try:
                self._publish_debug(image, rgb.header, bbox, confidence, z,
                                    point if localized else None, reason)
            except Exception as error:
                self._warn(f'Debug image failed: {error}')

    @staticmethod
    def _pick_target(results):
        best, best_area = None, 0.0
        for result in results or []:
            boxes = getattr(result, 'boxes', None)
            if boxes is None:
                continue
            for box in boxes:
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf[0])
                if len(xyxy) != 4 or not all(math.isfinite(v) for v in [*xyxy, score]):
                    continue
                x1, y1, x2, y2 = xyxy
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                if 0 <= score <= 1 and area > best_area:
                    best, best_area = (x1, y1, x2, y2, score), area
        return best

    def _localize(self, rgb_array, rgb, depth_msg, bbox):
        info = self.camera_info
        if info is None:
            raise ValueError('camera_info unavailable')
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        if not all(math.isfinite(v) for v in (fx, fy, cx, cy)) or fx <= 0 or fy <= 0:
            raise ValueError('invalid camera intrinsics')
        frame = depth_msg.header.frame_id
        if not frame or 'optical' not in frame or info.header.frame_id != frame:
            raise ValueError('depth/CameraInfo optical frames do not match')
        # Gazebo RGB and depth may name distinct but colocated optical frames.
        if not rgb.header.frame_id or 'optical' not in rgb.header.frame_id:
            raise ValueError('RGB optical frame missing')
        if rgb.header.frame_id != frame:
            alignment = self.tf_buffer.lookup_transform(
                frame, rgb.header.frame_id, Time.from_msg(depth_msg.header.stamp),
                timeout=Duration(seconds=self.tf_timeout))
            t, q = alignment.transform.translation, alignment.transform.rotation
            if (math.sqrt(t.x*t.x + t.y*t.y + t.z*t.z) > 1e-5 or
                    max(abs(q.x), abs(q.y), abs(q.z)) > 1e-5 or
                    abs(abs(q.w) - 1.0) > 1e-5):
                raise ValueError('RGB/depth are not registered (non-identity optical TF)')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim != 2 or depth.shape != rgb_array.shape[:2]:
            raise ValueError('RGB/depth dimensions do not match')
        if (info.height, info.width) != depth.shape:
            raise ValueError('CameraInfo/depth dimensions do not match')
        if depth_msg.encoding in ('16UC1', 'mono16'):
            depth = depth / 1000.0
        elif depth_msg.encoding != '32FC1':
            raise ValueError(f'unsupported depth encoding: {depth_msg.encoding}')
        x1, y1, x2, y2 = bbox
        u, v = int(round((x1+x2)/2)), int(round((y1+y2)/2))
        if not (0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]):
            raise ValueError('bbox center outside image')
        z = self._sample_depth_from_bbox(depth, bbox, u, v)
        if z is None:
            raise ValueError('no valid depth')
        point = PointStamped()
        point.header.frame_id, point.header.stamp = frame, depth_msg.header.stamp
        point.point.x = float((u-cx)*z/fx)
        point.point.y = float((v-cy)*z/fy)
        point.point.z = float(z)
        try:
            transformed = self.tf_buffer.transform(
                point, self.target_frame, timeout=Duration(seconds=self.tf_timeout))
        except Exception:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame, frame, Time(), timeout=Duration(seconds=self.tf_timeout))
            transformed = tf2_geometry_msgs.do_transform_point(point, transform)
        p = transformed.point
        if not all(math.isfinite(v) for v in (p.x, p.y, p.z)):
            raise ValueError('non-finite transformed point')
        return p, z

    def _sample_depth(self, depth, u, v):
        h, w = depth.shape
        if not (0 <= u < w and 0 <= v < h):
            return None
        win = self.depth_window
        patch = depth[max(0, v-win):min(h, v+win+1),
                      max(0, u-win):min(w, u+win+1)].reshape(-1)
        valid = patch[np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)]
        return float(np.median(valid)) if valid.size else None

    def _sample_depth_from_bbox(self, depth, bbox, u, v):
        # Deliberately copied from the established node, without importing it.
        h, w = depth.shape
        x1, y1, x2, y2 = bbox
        bx0, bx1 = (max(0, min(w, int(round(x)))) for x in (x1, x2))
        by0, by1 = (max(0, min(h, int(round(y)))) for y in (y1, y2))
        bw, bh = bx1-bx0, by1-by0
        if bw > 0 and bh > 0:
            rx0 = bx0 + int(round(bw*self.roi_x_min))
            rx1 = bx0 + int(round(bw*self.roi_x_max))
            ry0 = by0 + int(round(bh*self.roi_y_min))
            ry1 = by0 + int(round(bh*self.roi_y_max))
            patch = depth[ry0:ry1, rx0:rx1].reshape(-1)
            valid = patch[np.isfinite(patch) & (patch > self.min_depth) & (patch < self.max_depth)]
            if valid.size >= self.min_depth_samples:
                return float(np.percentile(valid, self.depth_percentile))
        return self._sample_depth(depth, u, v)

    def _publish_pose(self, point, stamp):
        message = PoseStamped()
        message.header.stamp, message.header.frame_id = stamp, self.target_frame
        message.pose.position = point
        message.pose.orientation.w = 1.0
        self.pose_pub.publish(message)

    def _publish_debug(self, image, header, bbox, confidence, z, point, reason):
        import cv2

        image = image.copy()
        if bbox is not None:
            x1, y1, x2, y2 = (int(v) for v in bbox)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lines = [f'{self.target_prompt} ' + (f'{confidence:.2f}' if confidence is not None else 'N/A'),
                 f'Z={z:.2f}m' if z is not None else 'Z=N/A',
                 f'({point.x:.2f}, {point.y:.2f})' if point is not None else '(x=N/A, y=N/A)']
        if reason:
            lines.append(reason)
        for index, label in enumerate(lines):
            cv2.putText(image, label, (10, 25 + index*25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)
        message = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        message.header = header
        self.dbg_pub.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = StaticYoloePerception()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
