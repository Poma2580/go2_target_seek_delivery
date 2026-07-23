#!/usr/bin/env python3
"""多 Go2 联合 waypoint 围捕控制节点。

控制多只 Go2 各自沿场景配置的中途 waypoint 行进，最终在目标周围均匀围捕，
并对准目标中心后停车。

前提：waypoint 为世界系绝对坐标，要求 /go2_N/odom 是世界系里程计
（用 spawn launch 的 use_ground_truth_odom:=true 启动）。
"""

import itertools
import math
from dataclasses import dataclass
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry, Path as NavPath
from visualization_msgs.msg import Marker, MarkerArray

from multi_go2_waypoint.grid_astar import (
    GridMap,
    PlanningError,
    astar,
    grid_path_to_waypoints,
    path_length_world,
    simplify_grid_path,
)


class WaypointConfigError(ValueError):
    """waypoint 配置缺失、格式错误或包含非法数值。"""


@dataclass(frozen=True)
class SceneConfig:
    """静态围捕场景的目标、地图、控制器和机器人配置。"""

    name: str
    map_package: str | None
    map_yaml: str | None
    target_x: float
    target_y: float
    target_yaw: float
    encircle_radius: float
    controller: dict
    robots: dict


@dataclass(frozen=True)
class VisualizationConfig:
    """A* RViz 显示与实际轨迹记录参数。"""

    frame_id: str
    trajectory_sample_distance: float
    max_trajectory_points: int
    trajectory_publish_rate: float
    publish_world_to_odom_tf: bool


def _read_mapping(path, description):
    path = Path(path).expanduser().resolve()
    try:
        content = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, yaml.YAMLError) as error:
        raise WaypointConfigError(
            f'无法读取{description} {path}: {error}') from error
    if not isinstance(content, dict):
        raise WaypointConfigError(f'{description}根节点必须是字典：{path}')
    if content.get('schema_version') != 1:
        raise WaypointConfigError(
            f'{description}的 schema_version 必须为 1：{path}')
    return content


def _mapping(value, name):
    if not isinstance(value, dict):
        raise WaypointConfigError(f'{name}必须是字典。')
    return value


def _string(value, name):
    if not isinstance(value, str) or not value:
        raise WaypointConfigError(f'{name}必须是非空字符串。')
    return value


def _number(value, name, *, minimum=None, strictly_positive=False):
    if isinstance(value, bool):
        raise WaypointConfigError(f'{name}必须是数值。')
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise WaypointConfigError(f'{name}必须是数值。') from error
    if not math.isfinite(result):
        raise WaypointConfigError(f'{name}必须是有限数值。')
    if strictly_positive and result <= 0.0:
        raise WaypointConfigError(f'{name}必须大于零。')
    if minimum is not None and result < minimum:
        raise WaypointConfigError(f'{name}不能小于 {minimum}。')
    return result


def load_scene_config(path):
    """读取 waypoint 围捕场景配置。"""
    config = _read_mapping(path, '场景配置')
    name = _string(config.get('scene'), 'scene')
    _mapping(config.get('world'), 'world')

    target = _mapping(config.get('target'), 'target')
    encircle = _mapping(config.get('encircle'), 'encircle')
    map_config = config.get('map')
    map_package = None
    map_yaml = None
    if map_config is not None:
        map_config = _mapping(map_config, 'map')
        map_package = _string(map_config.get('package'), 'map.package')
        map_yaml = _string(map_config.get('yaml'), 'map.yaml')

    robots_config = _mapping(config.get('robots'), 'robots')
    robots = {}
    for robot_name, robot_config in robots_config.items():
        robot_name = _string(robot_name, 'robots 的名称')
        robot_config = _mapping(robot_config, f'robots.{robot_name}')
        spawn = _mapping(
            robot_config.get('spawn'), f'robots.{robot_name}.spawn')
        robots[robot_name] = (
            _number(spawn.get('x'), f'robots.{robot_name}.spawn.x'),
            _number(spawn.get('y'), f'robots.{robot_name}.spawn.y'),
        )

    controller_config = config.get('controller')
    controller = {}
    if controller_config is not None:
        controller_config = _mapping(controller_config, 'controller')
        validators = {
            'reach_threshold': {'strictly_positive': True},
            'yaw_threshold': {'strictly_positive': True},
            'turn_in_place_thresh': {'minimum': 0.0},
            'max_linear': {'minimum': 0.0},
            'max_angular': {'minimum': 0.0},
            'k_linear': {'minimum': 0.0},
            'k_angular': {'minimum': 0.0},
            'control_period': {'strictly_positive': True},
        }
        unknown = set(controller_config) - set(validators)
        if unknown:
            raise WaypointConfigError(
                f'controller 包含未知参数：{", ".join(sorted(unknown))}。')
        for parameter_name, validation in validators.items():
            if parameter_name in controller_config:
                controller[parameter_name] = _number(
                    controller_config[parameter_name],
                    f'controller.{parameter_name}',
                    **validation)

    return SceneConfig(
        name=name,
        map_package=map_package,
        map_yaml=map_yaml,
        target_x=_number(target.get('x'), 'target.x'),
        target_y=_number(target.get('y'), 'target.y'),
        target_yaw=_number(target.get('yaw'), 'target.yaw'),
        encircle_radius=_number(encircle.get('radius'), 'encircle.radius',
                                strictly_positive=True),
        controller=controller,
        robots=robots,
    )


def load_astar_config(path):
    """读取 A* 的障碍膨胀、航点采样和目标扩张参数。"""
    config = _read_mapping(path, 'A* 配置')
    astar = _mapping(config.get('astar'), 'astar')
    return {
        'inflation_radius': _number(
            astar.get('inflation_radius'), 'astar.inflation_radius',
            minimum=0.0),
        'max_waypoint_spacing': _number(
            astar.get('max_waypoint_spacing'), 'astar.max_waypoint_spacing',
            strictly_positive=True),
        'max_goal_radius_expansion': _number(
            astar.get('max_goal_radius_expansion'),
            'astar.max_goal_radius_expansion', minimum=0.0),
    }


def load_manual_config(path):
    """读取三个场景的人工中途 waypoint。"""
    config = _read_mapping(path, '人工航点配置')
    manual = _mapping(config.get('manual'), 'manual')
    result = {}
    for scene_name, scene_paths in manual.items():
        scene_name = _string(scene_name, 'manual 场景名称')
        scene_paths = _mapping(scene_paths, f'manual.{scene_name}')
        result[scene_name] = {}
        for robot_name, waypoints in scene_paths.items():
            robot_name = _string(
                robot_name, f'manual.{scene_name} 的机器人名称')
            if not isinstance(waypoints, list):
                raise WaypointConfigError(
                    f'manual.{scene_name}.{robot_name}必须是 waypoint 列表。')
            parsed = []
            for index, waypoint in enumerate(waypoints):
                if (not isinstance(waypoint, (list, tuple))
                        or len(waypoint) != 2):
                    raise WaypointConfigError(
                        f'manual.{scene_name}.{robot_name}[{index}]'
                        '必须为 [x, y]。')
                parsed.append((
                    _number(
                        waypoint[0],
                        f'manual.{scene_name}.{robot_name}[{index}][0]'),
                    _number(
                        waypoint[1],
                        f'manual.{scene_name}.{robot_name}[{index}][1]'),
                ))
            result[scene_name][robot_name] = parsed
    return result


def load_controller_config(path):
    """读取全场景共用的底层 P 控制器参数。"""
    config = _read_mapping(path, 'P 控制器配置')
    controller = _mapping(config.get('controller'), 'controller')
    return {
        'reach_threshold': _number(
            controller.get('reach_threshold'), 'controller.reach_threshold',
            strictly_positive=True),
        'yaw_threshold': _number(
            controller.get('yaw_threshold'), 'controller.yaw_threshold',
            strictly_positive=True),
        'turn_in_place_thresh': _number(
            controller.get('turn_in_place_thresh'),
            'controller.turn_in_place_thresh', minimum=0.0),
        'max_linear': _number(
            controller.get('max_linear'), 'controller.max_linear',
            minimum=0.0),
        'max_angular': _number(
            controller.get('max_angular'), 'controller.max_angular',
            minimum=0.0),
        'k_linear': _number(
            controller.get('k_linear'), 'controller.k_linear', minimum=0.0),
        'k_angular': _number(
            controller.get('k_angular'), 'controller.k_angular', minimum=0.0),
        'control_period': _number(
            controller.get('control_period'), 'controller.control_period',
            strictly_positive=True),
    }


def load_visualization_config(path):
    """读取 A* RViz 显示与实际轨迹记录参数。"""
    config = _read_mapping(path, 'RViz 配置')
    visualization = _mapping(config.get('visualization'), 'visualization')
    max_points = visualization.get('max_trajectory_points')
    if isinstance(max_points, bool) or not isinstance(max_points, int):
        raise WaypointConfigError(
            'visualization.max_trajectory_points 必须是整数。')
    if max_points < 2:
        raise WaypointConfigError(
            'visualization.max_trajectory_points 必须至少为 2。')
    publish_tf = visualization.get('publish_world_to_odom_tf')
    if not isinstance(publish_tf, bool):
        raise WaypointConfigError(
            'visualization.publish_world_to_odom_tf 必须是布尔值。')
    return VisualizationConfig(
        frame_id=_string(
            visualization.get('frame_id'), 'visualization.frame_id'),
        trajectory_sample_distance=_number(
            visualization.get('trajectory_sample_distance'),
            'visualization.trajectory_sample_distance',
            strictly_positive=True),
        max_trajectory_points=max_points,
        trajectory_publish_rate=_number(
            visualization.get('trajectory_publish_rate'),
            'visualization.trajectory_publish_rate', strictly_positive=True),
        publish_world_to_odom_tf=publish_tf,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------
def quaternion_to_yaw(q):
    """从四元数（geometry_msgs/Quaternion）提取 yaw（绕 z 轴）。"""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]。"""
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def yaw_to_quaternion(yaw):
    """返回仅绕 z 轴旋转的四元数分量 (z, w)。"""
    half_yaw = yaw * 0.5
    return math.sin(half_yaw), math.cos(half_yaw)


def latched_qos():
    """规划路径和标记在 RViz 晚启动时仍可见。"""
    return QoSProfile(
        depth=1,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        reliability=ReliabilityPolicy.RELIABLE,
    )


def execution_path_from_waypoints(start, waypoints, frame_id, stamp):
    """将实际执行的 A* 航点转换为 RViz 可显示的 Path。"""
    start_x, start_y, start_yaw = start
    path = NavPath()
    path.header.frame_id = frame_id
    path.header.stamp = stamp

    start_pose = PoseStamped()
    start_pose.header.frame_id = frame_id
    start_pose.header.stamp = stamp
    start_pose.pose.position.x = start_x
    start_pose.pose.position.y = start_y
    start_pose.pose.orientation.z, start_pose.pose.orientation.w = \
        yaw_to_quaternion(start_yaw)
    path.poses.append(start_pose)

    for x, y, yaw in waypoints:
        pose = PoseStamped()
        pose.header.frame_id = frame_id
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        if yaw is None:
            pose.pose.orientation.w = 1.0
        else:
            pose.pose.orientation.z, pose.pose.orientation.w = \
                yaw_to_quaternion(yaw)
        path.poses.append(pose)
    return path


def solve_encircle_points(target_x, target_y, radius, num_dogs, target_yaw=0.0,
                          start_angle=None):
    """均匀求解目标周围的围捕终点，每个终点朝向目标中心。"""
    if num_dogs < 1:
        raise ValueError('num_dogs 必须大于 0。')
    if radius <= 0.0:
        raise ValueError('encircle_radius 必须大于 0。')

    if start_angle is None:
        start_angle = normalize_angle(target_yaw + math.pi)

    points = []
    for index in range(num_dogs):
        angle = normalize_angle(start_angle + 2.0 * math.pi * index / num_dogs)
        point_x = target_x + radius * math.cos(angle)
        point_y = target_y + radius * math.sin(angle)
        point_yaw = normalize_angle(angle + math.pi)
        points.append((point_x, point_y, point_yaw))
    return points


def assign_encircle_points(dog_refs, points):
    """按总行进距离最小，将围捕终点一一分配给各狗。"""
    if len(dog_refs) != len(points):
        raise ValueError('dog_refs 与 points 的数量必须一致。')

    names = [reference[0] for reference in dog_refs]
    best_permutation = None
    best_cost = float('inf')
    for permutation in itertools.permutations(points):
        cost = sum(
            math.hypot(permutation[index][0] - dog_refs[index][1],
                       permutation[index][1] - dog_refs[index][2])
            for index in range(len(names))
        )
        if cost < best_cost:
            best_cost = cost
            best_permutation = permutation

    return dict(zip(names, best_permutation))


# ---------------------------------------------------------------------------
# 单狗控制器（状态容器）
# ---------------------------------------------------------------------------
class DogController:
    def __init__(self, node, name, waypoints):
        self.node = node
        self.name = name
        self.waypoints = waypoints
        self.current_waypoint_index = 0
        self.state = 'tracking'  # 'tracking' / 'final_yaw_align' / 'done'

        # 最新里程计位姿
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.odom_received = False

        self.cmd_pub = node.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self.odom_sub = node.create_subscription(
            Odometry, f'/{name}/odom', self.odom_callback, 10)

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.odom_received = True

    def publish_zero(self):
        self.cmd_pub.publish(Twist())


# ---------------------------------------------------------------------------
# 主节点
# ---------------------------------------------------------------------------
class WaypointEncircle(Node):
    def __init__(self):
        super().__init__('multi_go2_waypoint_encircle')

        self.declare_parameter('scene', 'city')
        self.declare_parameter('num_dogs', 3)
        self.declare_parameter('planner_mode', 'manual')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('planner_config', '')
        self.declare_parameter('controller_config', '')
        self.declare_parameter('visualization_config', '')
        self.declare_parameter('visualization_frame', '')
        self.declare_parameter('map_yaml', '')
        # 默认 None 表示采用 YAML 文件值，命令行 -p 则覆盖 YAML。
        optional_value = ParameterDescriptor(dynamic_typing=True)
        for name in (
                'target_x', 'target_y', 'target_yaw', 'encircle_radius',
                'inflation_radius', 'max_waypoint_spacing',
                'max_goal_radius_expansion', 'reach_threshold',
                'yaw_threshold',
                'turn_in_place_thresh', 'max_linear', 'max_angular',
                'k_linear', 'k_angular', 'control_period'):
            self.declare_parameter(name, descriptor=optional_value)

        scene_name = str(self.get_parameter('scene').value).lower()
        planner_mode = str(self.get_parameter('planner_mode').value).lower()
        if planner_mode not in ('manual', 'astar'):
            raise ValueError('planner_mode 只能为 manual 或 astar。')

        scene_config_path = self._config_path(
            'scene_config', 'scenes', f'{scene_name}.yaml')
        scene = load_scene_config(scene_config_path)
        if scene.name != scene_name:
            raise ValueError(
                f'场景配置 {scene_config_path} 的 scene={scene.name!r}，'
                f'与 ROS 参数 scene={scene_name!r} 不一致。')

        num_dogs = self.get_parameter('num_dogs').value
        if num_dogs < 1:
            raise ValueError('num_dogs 必须大于 0。')

        target_x = self._parameter_or_default('target_x', scene.target_x)
        target_y = self._parameter_or_default('target_y', scene.target_y)
        target_yaw = self._parameter_or_default('target_yaw', scene.target_yaw)
        radius = self._parameter_or_default(
            'encircle_radius', scene.encircle_radius)
        if radius <= 0.0:
            raise ValueError('encircle_radius 必须大于零。')

        controller_config_path = self._config_path(
            'controller_config', 'controller', 'p_controller.yaml')
        controller_defaults = load_controller_config(controller_config_path)
        controller_defaults.update(scene.controller)
        self.reach_threshold = self._parameter_or_default(
            'reach_threshold', controller_defaults['reach_threshold'])
        self.yaw_threshold = self._parameter_or_default(
            'yaw_threshold', controller_defaults['yaw_threshold'])
        self.turn_in_place_thresh = self._parameter_or_default(
            'turn_in_place_thresh',
            controller_defaults['turn_in_place_thresh'])
        self.max_linear = self._parameter_or_default(
            'max_linear', controller_defaults['max_linear'])
        self.max_angular = self._parameter_or_default(
            'max_angular', controller_defaults['max_angular'])
        self.k_linear = self._parameter_or_default(
            'k_linear', controller_defaults['k_linear'])
        self.k_angular = self._parameter_or_default(
            'k_angular', controller_defaults['k_angular'])
        self.control_period = self._parameter_or_default(
            'control_period', controller_defaults['control_period'])
        self._validate_controller_parameters()

        visualization_defaults = load_visualization_config(self._config_path(
            'visualization_config', 'visualization', 'astar_rviz.yaml'))
        visualization_frame = str(
            self.get_parameter('visualization_frame').value).strip()
        self.visualization_frame = (
            visualization_frame or visualization_defaults.frame_id)

        self.scene_name = scene_name
        self.target_x = target_x
        self.target_y = target_y
        self.target_yaw = target_yaw
        self.requested_radius = radius
        self.planner_mode = planner_mode
        self.planning_complete = planner_mode == 'manual'
        self.planning_failed = False
        self.grid_map = None

        dog_names = [f'go2_{index + 1}' for index in range(num_dogs)]
        self.plan_publishers = {
            name: self.create_publisher(
                NavPath, f'/{name}/astar_plan', latched_qos())
            for name in dog_names
        }
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/astar_encircle_markers', latched_qos())
        missing_dog_configs = [
            name for name in dog_names
            if name not in scene.robots
        ]
        if missing_dog_configs:
            raise ValueError(
                f'场景 {scene_name!r} 未配置 '
                f'{", ".join(missing_dog_configs)} 的出生点。')

        if planner_mode == 'manual':
            manual_config_path = self._config_path(
                'planner_config', 'planner', 'manual.yaml')
            manual_paths = load_manual_config(manual_config_path)
            scene_paths = manual_paths.get(scene_name)
            if scene_paths is None:
                raise ValueError(
                    f'人工航点配置 {manual_config_path} 未配置场景 {scene_name!r}。')
            missing_manual_paths = [
                name for name in dog_names if name not in scene_paths]
            if missing_manual_paths:
                raise ValueError(
                    f'人工航点配置 {manual_config_path} 未配置 '
                    f'{", ".join(missing_manual_paths)} 的 waypoint。')
            points = solve_encircle_points(
                target_x, target_y, radius, num_dogs, target_yaw)
            dog_refs = []
            for name in dog_names:
                approach = scene_paths[name]
                reference_x, reference_y = (
                    approach[-1] if approach else scene.robots[name])
                dog_refs.append((name, reference_x, reference_y))

            assigned_points = assign_encircle_points(dog_refs, points)
            self.dogs = []
            for name in dog_names:
                waypoints = [(*point, None) for point in scene_paths[name]]
                waypoints.append(assigned_points[name])
                self.dogs.append(DogController(self, name, waypoints))
        else:
            astar_config_path = self._config_path(
                'planner_config', 'planner', 'astar.yaml')
            astar_defaults = load_astar_config(astar_config_path)
            self.max_waypoint_spacing = self._parameter_or_default(
                'max_waypoint_spacing', astar_defaults['max_waypoint_spacing'])
            self.max_goal_radius_expansion = self._parameter_or_default(
                'max_goal_radius_expansion',
                astar_defaults['max_goal_radius_expansion'])
            self.inflation_radius = self._parameter_or_default(
                'inflation_radius', astar_defaults['inflation_radius'])
            if self.inflation_radius < 0.0:
                raise ValueError('inflation_radius 不能小于零。')
            if self.max_waypoint_spacing <= 0.0:
                raise ValueError('max_waypoint_spacing 必须大于零。')
            if self.max_goal_radius_expansion < 0.0:
                raise ValueError('max_goal_radius_expansion 不能小于零。')

            map_yaml_value = str(self.get_parameter('map_yaml').value)
            if map_yaml_value:
                map_yaml = Path(map_yaml_value).expanduser()
            else:
                if scene.map_package is None or scene.map_yaml is None:
                    raise ValueError(
                        f'场景 {scene_name!r} 没有离线地图；使用 A* 时必须显式传入 map_yaml。')
                package_share = Path(
                    get_package_share_directory(scene.map_package))
                map_yaml = package_share / scene.map_yaml
            raw_map = GridMap.from_yaml(map_yaml)
            self.grid_map = raw_map.inflated(self.inflation_radius)
            self.dogs = [
                DogController(self, name, [])
                for name in dog_names
            ]
            self.get_logger().info(
                f'A* 地图已加载：{map_yaml}，{raw_map.width}x{raw_map.height}，'
                f'resolution={raw_map.resolution:.3f} m，'
                f'inflation_radius={self.inflation_radius:.3f} m。')

        self.timer = self.create_timer(self.control_period, self.control_loop)
        self._all_done_logged = False
        self.get_logger().info(
            f'多狗围捕已配置：scene={scene_name}, planner={planner_mode}, '
            f'num_dogs={num_dogs}, '
            f'target=({target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}), '
            f'radius={radius:.3f}。')
        self.get_logger().info('multi_go2_waypoint_encircle 已启动，等待里程计...')

    def _config_path(self, parameter_name, *default_parts):
        """优先使用命令行指定的配置路径，否则从本包安装目录取默认文件。"""
        configured = str(self.get_parameter(parameter_name).value).strip()
        if configured:
            return Path(configured).expanduser()
        package_share = Path(get_package_share_directory('multi_go2_waypoint'))
        return package_share.joinpath('config', *default_parts)

    def _parameter_or_default(self, name, default):
        """未传入的动态 ROS 参数使用已加载 YAML 的值。"""
        value = self.get_parameter(name).value
        if value is None:
            return default
        try:
            value = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f'参数 {name} 必须是数值。') from error
        if not math.isfinite(value):
            raise ValueError(f'参数 {name} 必须是有限数值。')
        return value

    def _validate_controller_parameters(self):
        """保证 P 控制器覆盖 YAML 后仍处于可执行范围。"""
        if self.reach_threshold <= 0.0:
            raise ValueError('reach_threshold 必须大于零。')
        if self.yaw_threshold <= 0.0:
            raise ValueError('yaw_threshold 必须大于零。')
        if self.turn_in_place_thresh < 0.0:
            raise ValueError('turn_in_place_thresh 不能小于零。')
        if self.max_linear < 0.0 or self.max_angular < 0.0:
            raise ValueError('max_linear 和 max_angular 不能小于零。')
        if self.k_linear < 0.0 or self.k_angular < 0.0:
            raise ValueError('k_linear 和 k_angular 不能小于零。')
        if self.control_period <= 0.0:
            raise ValueError('control_period 必须大于零。')

    def control_loop(self):
        if self.planning_failed:
            for dog in self.dogs:
                dog.publish_zero()
            return

        if not self.planning_complete:
            if not all(dog.odom_received for dog in self.dogs):
                return
            try:
                self._prepare_astar_waypoints()
            except (PlanningError, ValueError) as error:
                self.planning_failed = True
                for dog in self.dogs:
                    dog.publish_zero()
                self.get_logger().error(f'A* 规划失败，所有机器狗保持停止：{error}')
            return

        for dog in self.dogs:
            self.control_one(dog)

        if (all(d.state == 'done' for d in self.dogs)
                and not self._all_done_logged):
            self.get_logger().info('所有机器狗已完成围捕，持续发布零速度。')
            self._all_done_logged = True

    def _find_feasible_encircle_points(self):
        """在膨胀地图上同步扩大围捕半径，保持等边三角形结构。"""
        resolution = self.grid_map.resolution
        max_steps = int(math.floor(
            self.max_goal_radius_expansion / resolution + 1e-9))
        for step in range(max_steps + 1):
            radius = self.requested_radius + step * resolution
            points = solve_encircle_points(
                self.target_x, self.target_y, radius, len(self.dogs),
                self.target_yaw)
            if all(self.grid_map.is_free_world(x, y) for x, y, _ in points):
                if step:
                    self.get_logger().warning(
                        f'请求的围捕半径 {self.requested_radius:.3f} m 在膨胀地图上'
                        f'不可用，已自动扩大为 {radius:.3f} m。')
                return points, radius
        raise PlanningError(
            f'从围捕半径 {self.requested_radius:.3f} m 扩大 '
            f'{self.max_goal_radius_expansion:.3f} m 后仍找不到全部自由的围捕点。')

    def _prepare_astar_waypoints(self):
        """从首次真实 odom 生成三条静态 A* 路径，并交给原航点控制器。"""
        points, actual_radius = self._find_feasible_encircle_points()
        dog_refs = [(dog.name, dog.x, dog.y) for dog in self.dogs]
        assigned_points = assign_encircle_points(dog_refs, points)
        planned = {}
        for dog in self.dogs:
            goal_x, goal_y, goal_yaw = assigned_points[dog.name]
            start_cell = self.grid_map.world_to_grid(dog.x, dog.y)
            goal_cell = self.grid_map.world_to_grid(goal_x, goal_y)
            result = astar(self.grid_map, start_cell, goal_cell)
            simplified = simplify_grid_path(self.grid_map, result.cells)
            waypoints = grid_path_to_waypoints(
                self.grid_map, simplified, (dog.x, dog.y),
                (goal_x, goal_y), goal_yaw, self.max_waypoint_spacing)
            planned[dog.name] = waypoints
            length = path_length_world(waypoints, (dog.x, dog.y))
            self.get_logger().info(
                f'{dog.name} A* 完成：展开 {result.expanded} 个栅格，'
                f'原始路径 {len(result.cells)} 点，简化折线 {len(simplified)} 点，'
                f'控制航点 {len(waypoints)} 个，长度 {length:.2f} m。')

        # 三条路径全部成功后再统一启用，避免部分机器狗提前运动。
        for dog in self.dogs:
            dog.waypoints = planned[dog.name]
            dog.current_waypoint_index = 0
            dog.state = 'tracking'
        self.planning_complete = True
        self._publish_astar_visualization(assigned_points)
        self.get_logger().info(
            f'全部 A* 路径准备完成，实际围捕半径 {actual_radius:.3f} m，开始执行。')

    def _publish_astar_visualization(self, assigned_points):
        """发布三条锁存 A* 执行路径以及围捕目标、终点标记。"""
        stamp = self.get_clock().now().to_msg()
        for dog in self.dogs:
            path = execution_path_from_waypoints(
                (dog.x, dog.y, dog.yaw), dog.waypoints,
                self.visualization_frame, stamp)
            self.plan_publishers[dog.name].publish(path)

        markers = MarkerArray()
        target = Marker()
        target.header.frame_id = self.visualization_frame
        target.header.stamp = stamp
        target.ns = 'astar_target'
        target.id = 0
        target.type = Marker.CYLINDER
        target.action = Marker.ADD
        target.pose.position.x = self.target_x
        target.pose.position.y = self.target_y
        target.pose.position.z = 0.10
        target.pose.orientation.w = 1.0
        target.scale.x = 0.75
        target.scale.y = 0.75
        target.scale.z = 0.20
        target.color.r = 1.0
        target.color.g = 0.55
        target.color.a = 1.0
        markers.markers.append(target)

        colors = {
            'go2_1': (0.90, 0.15, 0.15),
            'go2_2': (0.10, 0.75, 0.20),
            'go2_3': (0.15, 0.35, 0.95),
        }
        for index, dog in enumerate(self.dogs, start=1):
            x, y, yaw = assigned_points[dog.name]
            red, green, blue = colors.get(dog.name, (0.85, 0.85, 0.85))
            arrow = Marker()
            arrow.header.frame_id = self.visualization_frame
            arrow.header.stamp = stamp
            arrow.ns = 'astar_goals'
            arrow.id = index
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose.position.x = x
            arrow.pose.position.y = y
            arrow.pose.position.z = 0.12
            arrow.pose.orientation.z, arrow.pose.orientation.w = \
                yaw_to_quaternion(yaw)
            arrow.scale.x = 0.80
            arrow.scale.y = 0.20
            arrow.scale.z = 0.20
            arrow.color.r = red
            arrow.color.g = green
            arrow.color.b = blue
            arrow.color.a = 1.0
            markers.markers.append(arrow)

            label = Marker()
            label.header.frame_id = self.visualization_frame
            label.header.stamp = stamp
            label.ns = 'astar_goal_labels'
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.65
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r = red
            label.color.g = green
            label.color.b = blue
            label.color.a = 1.0
            label.text = dog.name
            markers.markers.append(label)

        self.marker_publisher.publish(markers)

    def control_one(self, dog):
        # 1. 已完成：持续发零速
        if dog.state == 'done':
            dog.publish_zero()
            return

        # 2. 未收到里程计：不发指令
        if not dog.odom_received:
            return

        wx, wy, wyaw = dog.waypoints[dog.current_waypoint_index]
        dx = wx - dog.x
        dy = wy - dog.y
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)

        cmd = Twist()

        # 6. 最终朝向对齐状态
        if dog.state == 'final_yaw_align':
            yaw_err = normalize_angle(wyaw - dog.yaw)
            if abs(yaw_err) < self.yaw_threshold:
                dog.state = 'done'
                dog.publish_zero()
                self.get_logger().info(f'{dog.name} 已对准目标朝向，任务完成。')
                return
            cmd.linear.x = 0.0
            cmd.angular.z = clamp(self.k_angular * yaw_err,
                                  -self.max_angular, self.max_angular)
            dog.cmd_pub.publish(cmd)
            return

        # 4. 位置追踪
        if dist > self.reach_threshold:
            yaw_err = normalize_angle(desired_yaw - dog.yaw)
            if abs(yaw_err) > self.turn_in_place_thresh:
                cmd.linear.x = 0.0  # 误差大，原地转向
            else:
                cmd.linear.x = clamp(
                    self.k_linear * dist, 0.0, self.max_linear)
            cmd.angular.z = clamp(self.k_angular * yaw_err,
                                  -self.max_angular, self.max_angular)
            dog.cmd_pub.publish(cmd)
            return

        # 5. 已到达当前 waypoint
        if wyaw is None:
            # 中途路点：切到下一个
            if dog.current_waypoint_index < len(dog.waypoints) - 1:
                dog.current_waypoint_index += 1
                self.get_logger().info(
                    f'{dog.name} 到达路点 {dog.current_waypoint_index - 1}，'
                    f'前往路点 {dog.current_waypoint_index}。')
            else:
                # 兜底：最后一个点却没有 yaw，直接完成
                dog.state = 'done'
            dog.publish_zero()
        else:
            # 终点路点：进入最终朝向对齐
            dog.state = 'final_yaw_align'
            self.get_logger().info(f'{dog.name} 到达终点位置，开始对准朝向。')
            dog.publish_zero()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointEncircle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 退出前发一次零速度，防止狗继续运动
        if rclpy.ok():
            for dog in node.dogs:
                dog.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
