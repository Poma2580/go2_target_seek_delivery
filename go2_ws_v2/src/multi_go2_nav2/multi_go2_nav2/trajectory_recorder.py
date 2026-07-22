"""Publish all three ground-truth trajectories in the shared map frame."""

import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Empty
from tf2_ros import Buffer, TransformException, TransformListener
import tf2_geometry_msgs  # noqa: F401: registers PoseStamped TF conversions

from multi_go2_nav2.scene_config import ROBOT_NAMES, load_scene_config


class TrajectoryRecorder(Node):
    """Accumulate odometry samples, transform them, and publish map-frame paths."""

    def __init__(self):
        super().__init__('trajectory_recorder')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('sample_distance', 0.05)
        self.declare_parameter('max_points', 10000)
        self.declare_parameter('publish_rate', 2.0)
        config_file = self.get_parameter('scene_config').value
        if not config_file:
            raise ValueError('scene_config parameter is required')
        self.config = load_scene_config(config_file)
        if self.config.map is None:
            raise ValueError('trajectory visualization requires a map frame')
        self.sample_distance = float(self.get_parameter('sample_distance').value)
        self.max_points = int(self.get_parameter('max_points').value)
        publish_rate = float(self.get_parameter('publish_rate').value)
        if self.sample_distance <= 0.0 or self.max_points < 2 or publish_rate <= 0.0:
            raise ValueError('trajectory recorder parameters are invalid')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.paths = {}
        self.path_publishers = {}
        for name in ROBOT_NAMES:
            path = Path()
            path.header.frame_id = self.config.map.frame_id
            self.paths[name] = path
            self.path_publishers[name] = self.create_publisher(
                Path, f'/{name}/actual_path', latched_qos)
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda message, robot=name: self._odom_callback(robot, message),
                20,
            )
        self.create_service(Empty, '~/clear', self._clear_callback)
        self.timer = self.create_timer(1.0 / publish_rate, self._publish)

    def _odom_callback(self, robot_name, message):
        source = PoseStamped()
        source.header = message.header
        source.pose = message.pose.pose
        try:
            transformed = self.tf_buffer.transform(
                source,
                self.config.map.frame_id,
                timeout=Duration(seconds=0.05),
            )
        except TransformException as error:
            self.get_logger().warn(
                f'Cannot transform {robot_name} odometry into map: {error}',
                throttle_duration_sec=5.0,
            )
            return
        path = self.paths[robot_name]
        if path.poses:
            previous = path.poses[-1].pose.position
            current = transformed.pose.position
            if math.hypot(current.x - previous.x, current.y - previous.y) \
                    < self.sample_distance:
                return
        path.poses.append(transformed)
        if len(path.poses) > self.max_points:
            del path.poses[:len(path.poses) - self.max_points]

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        for name in ROBOT_NAMES:
            self.paths[name].header.stamp = stamp
            self.path_publishers[name].publish(self.paths[name])

    def _clear_callback(self, _request, response):
        for path in self.paths.values():
            path.poses.clear()
        self.get_logger().info('Cleared all three actual trajectory histories.')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()
