#!/usr/bin/env python3
"""多 Go2 联合 waypoint 围捕控制节点。

控制多只 Go2 各自沿场景配置的中途 waypoint 行进，最终在目标周围均匀围捕，
并对准目标中心后停车。

前提：waypoint 为世界系绝对坐标，要求 /go2_N/odom 是世界系里程计
（用 spawn launch 的 use_ground_truth_odom:=true 启动）。
"""

import itertools
import math
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_share_directory
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

from multi_go2_waypoint.grid_astar import (
    GridMap,
    PlanningError,
    astar,
    grid_path_to_waypoints,
    path_length_world,
    simplify_grid_path,
)


# ---------------------------------------------------------------------------
# 场景配置
#
# target: (x, y, yaw)，yaw 用于确定围捕点的起始方位。
# approach: 中途引导点；终点由运行时根据 target、radius 和 num_dogs 自动计算。
# ---------------------------------------------------------------------------
SCENES = {
    'city': {
        'target': (42.0, -20.0, -0.5),
        'radius': 2.0,
        'spawn': {
            'go2_1': (0.0, -4.0),
            'go2_2': (10.0, -17.0),
            'go2_3': (60.0, 10.0),
        },
        'approach': {
            'go2_1': [(21.0, 0.0), (38.0, -8.0)],
            'go2_2': [(30.0, -30.0)],
            'go2_3': [(45.0, 0.0)],
        },
    },
    'forest': {
        # 占位目标；在 forest 世界中部署实际目标后，按其位置修改或用 ROS 参数覆盖。
        'target': (0.0, 0.0, 0.0),
        'radius': 2.0,
        'spawn': {
            'go2_1': (0.0, -4.0),
            'go2_2': (2.0, -4.0),
            'go2_3': (0.0, -6.0),
        },
        'approach': {
            'go2_1': [],
            'go2_2': [],
            'go2_3': [],
        },
    },
    'airport': {
        # KD_MODEL/world/airport 中 airpor_cart_target（pickup）的位姿。
        'target': (80.0, -25.0, -1.0),
        'radius': 3.0,
        'spawn': {
            'go2_1': (0.0, -4.0),
            'go2_2': (2.0, -4.0),
            'go2_3': (0.0, -6.0),
        },
        'approach': {'go2_1': [], 'go2_2': [], 'go2_3': []},
    },
}


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
    # 控制参数
    REACH_THRESHOLD = 0.25      # 到点距离阈值 (m)
    YAW_THRESHOLD = 0.1         # 最终朝向对齐阈值 (rad)
    TURN_IN_PLACE_THRESH = 0.7  # yaw 误差大于此值则原地转向 (rad)
    MAX_LINEAR = 0.65           # 最大线速度 (m/s)
    MAX_ANGULAR = 0.9           # 最大角速度 (rad/s)
    K_LINEAR = 0.6              # 线速度比例增益
    K_ANGULAR = 1.5            # 角速度比例增益
    CONTROL_PERIOD = 0.05       # 控制周期 (s) -> 20 Hz

    def __init__(self):
        super().__init__('multi_go2_waypoint_encircle')

        self.declare_parameter('scene', 'city')
        self.declare_parameter('num_dogs', 3)
        optional_float = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter('target_x', descriptor=optional_float)
        self.declare_parameter('target_y', descriptor=optional_float)
        self.declare_parameter('target_yaw', descriptor=optional_float)
        self.declare_parameter('encircle_radius', descriptor=optional_float)
        self.declare_parameter('planner_mode', 'manual')
        self.declare_parameter('map_yaml', '')
        self.declare_parameter('inflation_radius', 0.55)
        self.declare_parameter('max_waypoint_spacing', 1.5)
        self.declare_parameter('max_goal_radius_expansion', 3.0)

        scene_name = self.get_parameter('scene').value
        if scene_name not in SCENES:
            available_scenes = ', '.join(SCENES)
            raise ValueError(
                f'未知场景 {scene_name!r}，可选值为：{available_scenes}。')

        scene = SCENES[scene_name]
        num_dogs = self.get_parameter('num_dogs').value
        if num_dogs < 1:
            raise ValueError('num_dogs 必须大于 0。')

        default_target_x, default_target_y, default_target_yaw = scene['target']
        target_x = self._parameter_or_default('target_x', default_target_x)
        target_y = self._parameter_or_default('target_y', default_target_y)
        target_yaw = self._parameter_or_default('target_yaw', default_target_yaw)
        radius = self._parameter_or_default('encircle_radius', scene['radius'])
        planner_mode = str(self.get_parameter('planner_mode').value).lower()
        if planner_mode not in ('manual', 'astar'):
            raise ValueError('planner_mode 只能为 manual 或 astar。')

        self.scene_name = scene_name
        self.target_x = target_x
        self.target_y = target_y
        self.target_yaw = target_yaw
        self.requested_radius = radius
        self.planner_mode = planner_mode
        self.planning_complete = planner_mode == 'manual'
        self.planning_failed = False
        self.grid_map = None
        self.max_waypoint_spacing = float(
            self.get_parameter('max_waypoint_spacing').value)
        self.max_goal_radius_expansion = float(
            self.get_parameter('max_goal_radius_expansion').value)
        if self.max_waypoint_spacing <= 0.0:
            raise ValueError('max_waypoint_spacing 必须大于零。')
        if self.max_goal_radius_expansion < 0.0:
            raise ValueError('max_goal_radius_expansion 不能小于零。')

        dog_names = [f'go2_{index + 1}' for index in range(num_dogs)]
        missing_dog_configs = [
            name for name in dog_names
            if name not in scene['spawn'] or name not in scene['approach']
        ]
        if missing_dog_configs:
            raise ValueError(
                f'场景 {scene_name!r} 未配置 {", ".join(missing_dog_configs)} 的 '
                '出生点和中途路点；请先补充 SCENES 配置。')

        if planner_mode == 'manual':
            points = solve_encircle_points(
                target_x, target_y, radius, num_dogs, target_yaw)
            dog_refs = []
            for name in dog_names:
                approach = scene['approach'][name]
                reference_x, reference_y = (
                    approach[-1] if approach else scene['spawn'][name])
                dog_refs.append((name, reference_x, reference_y))

            assigned_points = assign_encircle_points(dog_refs, points)
            self.dogs = []
            for name in dog_names:
                waypoints = [(*point, None) for point in scene['approach'][name]]
                waypoints.append(assigned_points[name])
                self.dogs.append(DogController(self, name, waypoints))
        else:
            map_yaml_value = str(self.get_parameter('map_yaml').value)
            if map_yaml_value:
                map_yaml = Path(map_yaml_value).expanduser()
            else:
                if scene_name != 'airport':
                    raise ValueError(
                        '非 airport 场景使用 A* 时必须显式传入 map_yaml。')
                package_share = Path(
                    get_package_share_directory('multi_go2_waypoint'))
                map_yaml = package_share / 'maps' / 'airport.yaml'
            inflation_radius = float(
                self.get_parameter('inflation_radius').value)
            if inflation_radius < 0.0:
                raise ValueError('inflation_radius 不能小于零。')
            raw_map = GridMap.from_yaml(map_yaml)
            self.grid_map = raw_map.inflated(inflation_radius)
            self.dogs = [
                DogController(self, name, [])
                for name in dog_names
            ]
            self.get_logger().info(
                f'A* 地图已加载：{map_yaml}，{raw_map.width}x{raw_map.height}，'
                f'resolution={raw_map.resolution:.3f} m，'
                f'inflation_radius={inflation_radius:.3f} m。')

        self.timer = self.create_timer(self.CONTROL_PERIOD, self.control_loop)
        self._all_done_logged = False
        self.get_logger().info(
            f'多狗围捕已配置：scene={scene_name}, planner={planner_mode}, '
            f'num_dogs={num_dogs}, '
            f'target=({target_x:.3f}, {target_y:.3f}, {target_yaw:.3f}), '
            f'radius={radius:.3f}。')
        self.get_logger().info('multi_go2_waypoint_encircle 已启动，等待里程计...')

    def _parameter_or_default(self, name, default):
        """未传入的可选 ROS 参数使用场景默认值。"""
        value = self.get_parameter(name).value
        return default if value is None else float(value)

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

        if all(d.state == 'done' for d in self.dogs) and not self._all_done_logged:
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
        self.get_logger().info(
            f'全部 A* 路径准备完成，实际围捕半径 {actual_radius:.3f} m，开始执行。')

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
            if abs(yaw_err) < self.YAW_THRESHOLD:
                dog.state = 'done'
                dog.publish_zero()
                self.get_logger().info(f'{dog.name} 已对准目标朝向，任务完成。')
                return
            cmd.linear.x = 0.0
            cmd.angular.z = clamp(self.K_ANGULAR * yaw_err,
                                  -self.MAX_ANGULAR, self.MAX_ANGULAR)
            dog.cmd_pub.publish(cmd)
            return

        # 4. 位置追踪
        if dist > self.REACH_THRESHOLD:
            yaw_err = normalize_angle(desired_yaw - dog.yaw)
            if abs(yaw_err) > self.TURN_IN_PLACE_THRESH:
                cmd.linear.x = 0.0  # 误差大，原地转向
            else:
                cmd.linear.x = clamp(self.K_LINEAR * dist, 0.0, self.MAX_LINEAR)
            cmd.angular.z = clamp(self.K_ANGULAR * yaw_err,
                                  -self.MAX_ANGULAR, self.MAX_ANGULAR)
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
