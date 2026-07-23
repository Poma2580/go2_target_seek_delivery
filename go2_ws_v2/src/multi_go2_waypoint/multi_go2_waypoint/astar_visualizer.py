#!/usr/bin/env python3
"""旧版 A* 的地图、实际轨迹和 world 坐标系可视化节点。"""

import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster

from multi_go2_waypoint.grid_astar import GridMap
from multi_go2_waypoint.waypoint_encircle import (
    load_scene_config,
    load_visualization_config,
)


def latched_qos():
    """用于地图和轨迹，确保后启动的 RViz 也可收到最近一次数据。"""
    return QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def occupancy_grid_from_map(grid_map, frame_id, stamp):
    """将旧版 A* 内部栅格转为 RViz 可显示的 OccupancyGrid。"""
    message = OccupancyGrid()
    message.header.frame_id = frame_id
    message.header.stamp = stamp
    message.info.map_load_time = stamp
    message.info.resolution = grid_map.resolution
    message.info.width = grid_map.width
    message.info.height = grid_map.height
    message.info.origin.position.x = grid_map.origin_x
    message.info.origin.position.y = grid_map.origin_y
    message.info.origin.orientation.w = 1.0
    message.data = [100 if blocked else 0 for blocked in grid_map.blocked]
    return message


class TrajectoryBuffer:
    """按位移阈值累计一条固定上限的 world 坐标轨迹。"""

    def __init__(self, frame_id, sample_distance, max_points):
        self.frame_id = frame_id
        self.sample_distance = sample_distance
        self.max_points = max_points
        self.path = NavPath()
        self.path.header.frame_id = frame_id

    def append_odometry(self, message):
        position = message.pose.pose.position
        if self.path.poses:
            previous = self.path.poses[-1].pose.position
            if math.hypot(position.x - previous.x, position.y - previous.y) \
                    < self.sample_distance:
                return False

        pose = PoseStamped()
        pose.header.stamp = message.header.stamp
        pose.header.frame_id = self.frame_id
        pose.pose = message.pose.pose
        self.path.poses.append(pose)
        if len(self.path.poses) > self.max_points:
            del self.path.poses[:len(self.path.poses) - self.max_points]
        return True

    def message(self, stamp):
        self.path.header.stamp = stamp
        return self.path


def world_to_odom_transforms(frame_id, robot_names, stamp):
    """为使用全局数值 odom 的三只狗建立单位 world -> odom 静态 TF。"""
    transforms = []
    for robot_name in robot_names:
        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = frame_id
        transform.child_frame_id = f'{robot_name}/odom'
        transform.transform.rotation.w = 1.0
        transforms.append(transform)
    return transforms


class AStarVisualizer(Node):
    """发布旧版 A* 的地图、实际轨迹与 RobotModel 所需根 TF。"""

    def __init__(self):
        super().__init__('astar_visualizer')
        self.declare_parameter('scene', 'airport')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('visualization_config', '')
        self.declare_parameter('visualization_frame', '')
        self.declare_parameter('num_dogs', 3)
        optional_value = ParameterDescriptor(dynamic_typing=True)
        for name in (
                'trajectory_sample_distance', 'max_trajectory_points',
                'trajectory_publish_rate', 'publish_world_to_odom_tf'):
            self.declare_parameter(name, descriptor=optional_value)

        self.scene_name = str(self.get_parameter('scene').value).lower()
        self.robot_names = [
            f'go2_{index + 1}'
            for index in range(int(self.get_parameter('num_dogs').value))
        ]
        if not self.robot_names:
            raise ValueError('num_dogs 必须大于零。')

        scene = load_scene_config(self._config_path(
            'scene_config', 'scenes', f'{self.scene_name}.yaml'))
        if scene.name != self.scene_name:
            raise ValueError(
                f'场景配置 scene={scene.name!r} 与 '
                f'ROS 参数 scene={self.scene_name!r} 不一致。')
        missing = [
            name for name in self.robot_names if name not in scene.robots]
        if missing:
            raise ValueError(
                f'场景 {self.scene_name!r} 未配置 {", ".join(missing)}。')

        visualization = load_visualization_config(self._config_path(
            'visualization_config', 'visualization', 'astar_rviz.yaml'))
        frame_override = str(
            self.get_parameter('visualization_frame').value).strip()
        self.frame_id = frame_override or visualization.frame_id
        self.sample_distance = self._override_number(
            'trajectory_sample_distance',
            visualization.trajectory_sample_distance,
            positive=True)
        self.max_points = self._override_integer(
            'max_trajectory_points', visualization.max_trajectory_points,
            minimum=2)
        self.publish_rate = self._override_number(
            'trajectory_publish_rate', visualization.trajectory_publish_rate,
            positive=True)
        self.publish_world_to_odom_tf = self._override_boolean(
            'publish_world_to_odom_tf', visualization.publish_world_to_odom_tf)

        self.map_publisher = self.create_publisher(
            OccupancyGrid, '/astar_map', latched_qos())
        self.path_publishers = {
            name: self.create_publisher(
                NavPath, f'/{name}/astar_actual_path', latched_qos())
            for name in self.robot_names
        }
        self.trajectories = {
            name: TrajectoryBuffer(
                self.frame_id, self.sample_distance, self.max_points)
            for name in self.robot_names
        }
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda message, robot=name: self._odom_callback(
                    robot, message),
                20,
            )

        if self.publish_world_to_odom_tf:
            broadcaster = StaticTransformBroadcaster(self)
            broadcaster.sendTransform(world_to_odom_transforms(
                self.frame_id, self.robot_names,
                self.get_clock().now().to_msg()))
            self.tf_broadcaster = broadcaster
        else:
            self.tf_broadcaster = None

        self._publish_scene_map(scene)
        self.timer = self.create_timer(
            1.0 / self.publish_rate, self._publish_paths)
        self.get_logger().info(
            f'A* RViz 可视化已启动：scene={self.scene_name}, '
            f'frame={self.frame_id}, '
            f'sample_distance={self.sample_distance:.3f}m。')

    def _config_path(self, parameter_name, *default_parts):
        configured = str(self.get_parameter(parameter_name).value).strip()
        if configured:
            return Path(configured).expanduser()
        package_share = Path(get_package_share_directory('multi_go2_waypoint'))
        return package_share.joinpath('config', *default_parts)

    def _override_number(self, name, default, *, positive=False):
        value = self.get_parameter(name).value
        if value is None:
            return default
        if isinstance(value, bool):
            raise ValueError(f'参数 {name} 必须是数值。')
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'参数 {name} 必须是数值。') from error
        if not math.isfinite(value) or (positive and value <= 0.0):
            raise ValueError(f'参数 {name} 必须是大于零的有限数值。')
        return value

    def _override_integer(self, name, default, *, minimum):
        value = self.get_parameter(name).value
        if value is None:
            return default
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'参数 {name} 必须是整数。')
        if value < minimum:
            raise ValueError(f'参数 {name} 不能小于 {minimum}。')
        return value

    def _override_boolean(self, name, default):
        value = self.get_parameter(name).value
        if value is None:
            return default
        if not isinstance(value, bool):
            raise ValueError(f'参数 {name} 必须是布尔值。')
        return value

    def _publish_scene_map(self, scene):
        if scene.map_package is None or scene.map_yaml is None:
            self.get_logger().warn(
                f'场景 {self.scene_name!r} 没有离线地图，不发布 /astar_map。')
            return
        map_yaml = (
            Path(get_package_share_directory(scene.map_package)) /
            scene.map_yaml)
        grid_map = GridMap.from_yaml(map_yaml)
        self.map_publisher.publish(occupancy_grid_from_map(
            grid_map, self.frame_id, self.get_clock().now().to_msg()))
        self.get_logger().info(
            f'已发布 /astar_map：{map_yaml}，'
            f'{grid_map.width}x{grid_map.height}。')

    def _odom_callback(self, robot_name, message):
        self.trajectories[robot_name].append_odometry(message)

    def _publish_paths(self):
        stamp = self.get_clock().now().to_msg()
        for name in self.robot_names:
            self.path_publishers[name].publish(
                self.trajectories[name].message(stamp))


def main(args=None):
    rclpy.init(args=args)
    node = AStarVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
