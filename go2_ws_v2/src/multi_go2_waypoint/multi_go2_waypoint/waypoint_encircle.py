#!/usr/bin/env python3
"""多 Go2 联合 waypoint 围捕控制节点。

控制 go2_1 / go2_2 / go2_3 三只机器狗，各自沿预设 waypoint 序列行进，
最终在目标 SUV 周围按等边三角形（半径 r=2.0）完成围捕，并对准指定朝向后停车。

前提：waypoint 为世界系绝对坐标，要求 /go2_N/odom 是世界系里程计
（用 spawn launch 的 use_ground_truth_odom:=true 启动）。
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# 目标与几何参数（仅用于记录/说明，实际控制使用下面的 waypoint 序列）
# ---------------------------------------------------------------------------
TARGET_X = 42.0
TARGET_Y = -20.0
TARGET_YAW = -0.5
ENCIRCLE_RADIUS = 2.0

# 每只狗的 waypoint 序列，每项为 (x, y, yaw)，yaw 为 None 表示中途路点（到点即切下一个）。
WAYPOINTS = {
    'go2_1': [
        (21.0, 0.0, None),
        (38.0, -8.0, None),
        (39.245, -18.041, -0.500),
    ],
    'go2_2': [
        (30.0, -30.0, None),
        (41.953, -21.999, 1.594),
    ],
    'go2_3': [
        (45.0, 0.0, None),
        (43.707, -18.957, -2.594),
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

        self.dogs = [
            DogController(self, name, list(wps))
            for name, wps in WAYPOINTS.items()
        ]

        self.timer = self.create_timer(self.CONTROL_PERIOD, self.control_loop)
        self._all_done_logged = False
        self.get_logger().info('multi_go2_waypoint_encircle 已启动，等待里程计...')

    def control_loop(self):
        for dog in self.dogs:
            self.control_one(dog)

        if all(d.state == 'done' for d in self.dogs) and not self._all_done_logged:
            self.get_logger().info('所有机器狗已完成围捕，持续发布零速度。')
            self._all_done_logged = True

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
        for dog in node.dogs:
            dog.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
