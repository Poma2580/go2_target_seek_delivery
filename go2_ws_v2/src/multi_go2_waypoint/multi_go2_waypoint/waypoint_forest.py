#!/usr/bin/env python3
"""森林场景多 Go2 waypoint 围捕控制节点。

控制 go2_1 / go2_2 / go2_3 三只机器狗，各自沿预设 waypoint 序列行进，
最终在森林场景 cessna_c172 目标周围按半径 r=3.0 的三角形完成围捕。

前提：waypoint 为世界系绝对坐标，要求 /go2_N/odom 是世界系里程计。
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# 森林目标与几何参数
# target: KD_MODEL/world/forestV3.world 中的 cessna_c172
# ---------------------------------------------------------------------------
TARGET_X = 8.4
TARGET_Y = 38.3395
TARGET_YAW = 1.59854
ENCIRCLE_RADIUS = 3.0

# 每只狗的 waypoint 序列，每项为 (x, y, yaw)，yaw 为 None 表示中途路点。
# 初始位置建议：
#   go2_1: (20, 18), yaw 2.19
#   go2_2: (-8, 42), yaw 0.00
#   go2_3: (36, 40), yaw -2.92
WAYPOINTS = {
    'go2_1': [
        (15.0, 25.0, None),
        (10.0, 31.0, None),
        (8.4, 35.3395, 1.571),
    ],
    'go2_2': [
        (-3.0, 42.0, None),
        (3.0, 40.0, None),
        (5.4, 38.3395, 0.0),
    ],
    'go2_3': [
        (27.0, 38.0, None),
        (12.0, 42.0, None),
        (10.998, 39.8395, -2.618),
    ],
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
class WaypointForest(Node):
    # 森林中障碍物更密，速度比城市版保守一些。
    REACH_THRESHOLD = 0.35      # 到点距离阈值 (m)
    YAW_THRESHOLD = 0.1         # 最终朝向对齐阈值 (rad)
    TURN_IN_PLACE_THRESH = 0.7  # yaw 误差大于此值则原地转向 (rad)
    MAX_LINEAR = 0.4            # 最大线速度 (m/s)
    MAX_ANGULAR = 0.8           # 最大角速度 (rad/s)
    K_LINEAR = 0.6              # 线速度比例增益
    K_ANGULAR = 1.5             # 角速度比例增益
    CONTROL_PERIOD = 0.05       # 控制周期 (s) -> 20 Hz

    def __init__(self):
        super().__init__('multi_go2_waypoint_forest')

        self.dogs = [
            DogController(self, name, list(wps))
            for name, wps in WAYPOINTS.items()
        ]

        self.timer = self.create_timer(self.CONTROL_PERIOD, self.control_loop)
        self._all_done_logged = False
        self.get_logger().info('multi_go2_waypoint_forest 已启动，等待里程计...')

    def control_loop(self):
        for dog in self.dogs:
            self.control_one(dog)

        if all(d.state == 'done' for d in self.dogs) and not self._all_done_logged:
            self.get_logger().info('所有机器狗已完成森林目标围捕，持续发布零速度。')
            self._all_done_logged = True

    def control_one(self, dog):
        if dog.state == 'done':
            dog.publish_zero()
            return

        if not dog.odom_received:
            return

        wx, wy, wyaw = dog.waypoints[dog.current_waypoint_index]
        dx = wx - dog.x
        dy = wy - dog.y
        dist = math.hypot(dx, dy)
        desired_yaw = math.atan2(dy, dx)

        cmd = Twist()

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

        if dist > self.REACH_THRESHOLD:
            yaw_err = normalize_angle(desired_yaw - dog.yaw)
            if abs(yaw_err) > self.TURN_IN_PLACE_THRESH:
                cmd.linear.x = 0.0
            else:
                cmd.linear.x = clamp(self.K_LINEAR * dist, 0.0, self.MAX_LINEAR)
            cmd.angular.z = clamp(self.K_ANGULAR * yaw_err,
                                  -self.MAX_ANGULAR, self.MAX_ANGULAR)
            dog.cmd_pub.publish(cmd)
            return

        if wyaw is None:
            if dog.current_waypoint_index < len(dog.waypoints) - 1:
                dog.current_waypoint_index += 1
                self.get_logger().info(
                    f'{dog.name} 到达路点 {dog.current_waypoint_index - 1}，'
                    f'前往路点 {dog.current_waypoint_index}。')
            else:
                dog.state = 'done'
            dog.publish_zero()
        else:
            dog.state = 'final_yaw_align'
            self.get_logger().info(f'{dog.name} 到达终点位置，开始对准朝向。')
            dog.publish_zero()


def main(args=None):
    rclpy.init(args=args)
    node = WaypointForest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for dog in node.dogs:
            dog.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
