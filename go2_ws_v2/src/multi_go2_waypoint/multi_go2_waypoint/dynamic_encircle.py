#!/usr/bin/env python3
"""三 Go2 动态行人追踪与围捕节点（三段式：approach -> catch_up -> formation）。

订阅行人里程计 /walking_target/odom 与三只狗 /go2_N/odom，让三只狗：
  1. approach   ：走用户预设的安全路点，绕开建筑、抵达行人回路（环）附近。
  2. catch_up   ：把狗与行人都投影到环上，狗沿环（就近方向、贴角、不穿心）追上行人。
  3. formation  ：追到行人身边后，做绕行人 1.5m 的旋转三角围捕。

要点：
- 编队朝向用行人“运动航向”atan2(vy,vx)，不用 actor 那个在拐角会猛甩的 yaw。
- catch_up 严格沿矩形环走，结构上不可能穿过中间房子；方向取最短弧、不绕大圈。
- 狗朝向始终对齐自身前进方向；永不进入 done；行人/自身里程计超时即持续发零速度。
"""

import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


# ---------------------------------------------------------------------------
# 辅助函数（与 waypoint_encircle 保持一致）
# ---------------------------------------------------------------------------
def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 行人回路（环）几何：闭合多边形的弧长参数化
# 角点须与 QY_MODEL/target_seek 里行人 actor 的（加宽后）矩形一致。
# ---------------------------------------------------------------------------
LOOP_CORNERS = [(41.0, 4.0), (41.0, 36.0), (-13.0, 36.0), (-13.0, 4.0)]


class Loop:
    def __init__(self, corners):
        self.corners = corners
        self.n = len(corners)
        self.edges = []        # (x0, y0, x1, y1, length)
        self.cum = [0.0]       # 每条边起点的累计弧长
        for i in range(self.n):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % self.n]
            d = math.hypot(x1 - x0, y1 - y0)
            self.edges.append((x0, y0, x1, y1, d))
            self.cum.append(self.cum[-1] + d)
        self.L = self.cum[-1]  # 周长

    def project(self, px, py):
        """把点投影到环上最近处，返回弧长 s。"""
        best_s, best_d2 = 0.0, float('inf')
        for i, (x0, y0, x1, y1, d) in enumerate(self.edges):
            if d < 1e-9:
                continue
            t = ((px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)) / (d * d)
            t = clamp(t, 0.0, 1.0)
            cx, cy = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            dd = (px - cx) ** 2 + (py - cy) ** 2
            if dd < best_d2:
                best_d2, best_s = dd, self.cum[i] + t * d
        return best_s

    def point_at(self, s):
        """弧长 -> 坐标（自动按周长取模）。"""
        s = s % self.L
        for i, (x0, y0, x1, y1, d) in enumerate(self.edges):
            if s <= self.cum[i + 1] or i == self.n - 1:
                t = (s - self.cum[i]) / d if d > 1e-9 else 0.0
                return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
        return self.corners[0]

    def signed_arc(self, s_from, s_to):
        """s_from -> s_to 沿环的最短带符号弧长，范围 (-L/2, L/2]。"""
        d = (s_to - s_from) % self.L
        if d > self.L / 2.0:
            d -= self.L
        return d

    def dist_to_next_corner(self, s, direction):
        """沿 direction(+1/-1) 从 s 到下一个角点的距离（用于防切角）。"""
        best = self.L
        for b in self.cum[:-1]:        # 角点弧长 [0, e0, e0+e1, ...]
            c = ((b - s) * direction) % self.L
            if c < 1e-6:
                c = self.L
            best = min(best, c)
        return best


# 三只狗在编队里的角度偏置（相对行人运动航向）
FORMATION_OFFSETS = {
    'go2_1': 0.0,
    'go2_2': 2.0 * math.pi / 3.0,    # +120°
    'go2_3': -2.0 * math.pi / 3.0,   # -120°
}

# 用户手工设计的接近安全路点：每只狗一串 (x, y)，绕开建筑到达环附近。
# 留空列表 [] 表示该狗跳过 approach、直接进入 catch_up（沿环追人）。
APPROACH_WAYPOINTS = {
    'go2_1': [(2.0, 4.0)],
    'go2_2': [(17.0, -27.0), (10.0, -17.0), (9.0, 5.0)],
    'go2_3': [(41.0, 10.0)],
}


class DogState:
    def __init__(self, node, name):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_stamp = None
        self.received = False

        # 三段状态机
        self.phase = 'approach' if APPROACH_WAYPOINTS.get(name) else 'catch_up'
        self.approach_index = 0

        # 上一拍下发的速度（用于限加速度）
        self.prev_lin = 0.0
        self.prev_ang = 0.0

        self.cmd_pub = node.create_publisher(Twist, f'/{name}/cmd_vel', 10)
        self.odom_sub = node.create_subscription(
            Odometry, f'/{name}/odom', self._odom_cb, 10)

    def _odom_cb(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.last_stamp = msg.header.stamp
        self.received = True

    def publish_zero(self):
        self.prev_lin = 0.0
        self.prev_ang = 0.0
        self.cmd_pub.publish(Twist())


class DynamicEncircle(Node):
    def __init__(self):
        super().__init__('multi_go2_dynamic_encircle')

        # 参数
        self.declare_parameter('formation_radius', 1.5)
        self.declare_parameter('control_rate', 20.0)
        self.declare_parameter('target_timeout', 0.5)
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('max_linear', 0.65)
        self.declare_parameter('max_angular', 0.9)
        self.declare_parameter('position_deadband', 0.25)
        self.declare_parameter('k_linear', 0.8)
        self.declare_parameter('k_angular', 1.5)
        self.declare_parameter('turn_in_place_thresh', 0.7)
        self.declare_parameter('heading_speed_thresh', 0.08)   # 低于此速度冻结编队航向
        self.declare_parameter('heading_slew_rate', 1.0)       # 编队航向变化率限幅 rad/s
        self.declare_parameter('accel_lin', 1.0)               # 线/角加速度限幅
        self.declare_parameter('accel_ang', 3.0)
        # 三段式相关
        self.declare_parameter('approach_reach', 0.35)         # approach 路点到达阈值
        self.declare_parameter('catch_lookahead', 1.5)         # catch_up 沿环前瞻距离(小→贴边)
        self.declare_parameter('catch_speed', 0.6)             # catch_up 恒定巡航速度
        self.declare_parameter('catch_radius', 2.0)            # 进 formation 的距离
        self.declare_parameter('revert_radius', 4.0)           # 退回 catch_up 的距离(滞回)

        self.r = self.get_parameter('formation_radius').value
        self.rate = float(self.get_parameter('control_rate').value)
        self.target_timeout = self.get_parameter('target_timeout').value
        self.odom_timeout = self.get_parameter('odom_timeout').value
        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.deadband = self.get_parameter('position_deadband').value
        self.k_linear = self.get_parameter('k_linear').value
        self.k_angular = self.get_parameter('k_angular').value
        self.turn_in_place_thresh = self.get_parameter('turn_in_place_thresh').value
        self.heading_speed_thresh = self.get_parameter('heading_speed_thresh').value
        self.heading_slew_rate = self.get_parameter('heading_slew_rate').value
        self.accel_lin = self.get_parameter('accel_lin').value
        self.accel_ang = self.get_parameter('accel_ang').value
        self.approach_reach = self.get_parameter('approach_reach').value
        self.catch_lookahead = self.get_parameter('catch_lookahead').value
        self.catch_speed = self.get_parameter('catch_speed').value
        self.catch_radius = self.get_parameter('catch_radius').value
        self.revert_radius = self.get_parameter('revert_radius').value

        self.dt = 1.0 / self.rate
        self.loop = Loop(LOOP_CORNERS)

        # 行人状态
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_stamp = None
        self.target_received = False
        self.formation_heading = None   # 平滑后的编队航向（运动方向）

        self.dogs = {name: DogState(self, name) for name in FORMATION_OFFSETS}

        self.target_sub = self.create_subscription(
            Odometry, '/walking_target/odom', self._target_cb, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self._lost_logged = False
        self.get_logger().info('multi_go2_dynamic_encircle 已启动，等待行人与里程计...')

    def _target_cb(self, msg):
        self.target_x = msg.pose.pose.position.x
        self.target_y = msg.pose.pose.position.y
        self.target_vx = msg.twist.twist.linear.x
        self.target_vy = msg.twist.twist.linear.y
        self.target_stamp = msg.header.stamp
        self.target_received = True

    def _age(self, stamp):
        if stamp is None:
            return float('inf')
        now = self.get_clock().now()
        return (now - rclpy.time.Time.from_msg(stamp)).nanoseconds * 1e-9

    def _update_formation_heading(self):
        """用行人运动方向更新编队航向：低速冻结，并做变化率限幅。"""
        speed = math.hypot(self.target_vx, self.target_vy)
        if speed >= self.heading_speed_thresh:
            desired = math.atan2(self.target_vy, self.target_vx)
            if self.formation_heading is None:
                self.formation_heading = desired
            else:
                err = normalize_angle(desired - self.formation_heading)
                step = clamp(err, -self.heading_slew_rate * self.dt,
                             self.heading_slew_rate * self.dt)
                self.formation_heading = normalize_angle(self.formation_heading + step)

    # ----- 底层：朝某点的速度指令（误差大原地转，否则比例前进）-----
    def _goto(self, dog, gx, gy):
        dx, dy = gx - dog.x, gy - dog.y
        dist = math.hypot(dx, dy)
        yaw_err = normalize_angle(math.atan2(dy, dx) - dog.yaw)
        if abs(yaw_err) > self.turn_in_place_thresh:
            lin = 0.0
        else:
            lin = clamp(self.k_linear * dist, 0.0, self.max_linear)
        ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
        return lin, ang, dist

    def _emit(self, dog, lin, ang):
        # 限加速度后下发
        lin = clamp(lin, dog.prev_lin - self.accel_lin * self.dt,
                    dog.prev_lin + self.accel_lin * self.dt)
        ang = clamp(ang, dog.prev_ang - self.accel_ang * self.dt,
                    dog.prev_ang + self.accel_ang * self.dt)
        dog.prev_lin, dog.prev_ang = lin, ang
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        dog.cmd_pub.publish(cmd)

    def control_loop(self):
        # 行人超时：全员零速
        if not self.target_received or self._age(self.target_stamp) > self.target_timeout:
            for dog in self.dogs.values():
                dog.publish_zero()
            if not self._lost_logged:
                self.get_logger().warn('行人里程计缺失/超时，三狗保持静止。')
                self._lost_logged = True
            return
        self._lost_logged = False

        self._update_formation_heading()

        for name, dog in self.dogs.items():
            self.control_one(name, dog)

    def control_one(self, name, dog):
        # 自身里程计超时：零速
        if not dog.received or self._age(dog.last_stamp) > self.odom_timeout:
            dog.publish_zero()
            return

        dist_to_ped = math.hypot(self.target_x - dog.x, self.target_y - dog.y)

        # ---- 阶段1：approach（走预设安全路点）----
        if dog.phase == 'approach':
            wps = APPROACH_WAYPOINTS.get(name, [])
            if dog.approach_index >= len(wps):
                dog.phase = 'catch_up'
                self.get_logger().info(f'{name}: approach 完成 -> catch_up')
            else:
                gx, gy = wps[dog.approach_index]
                lin, ang, d = self._goto(dog, gx, gy)
                if d < self.approach_reach:
                    dog.approach_index += 1
                self._emit(dog, lin, ang)
                return

        # ---- 阶段2：catch_up（沿环就近追人，贴角、不穿心）----
        if dog.phase == 'catch_up':
            if dist_to_ped < self.catch_radius:
                dog.phase = 'formation'
                self.get_logger().info(f'{name}: 追上行人 -> formation')
            else:
                s_dog = self.loop.project(dog.x, dog.y)
                s_ped = self.loop.project(self.target_x, self.target_y)
                delta = self.loop.signed_arc(s_dog, s_ped)
                direction = 1.0 if delta >= 0.0 else -1.0
                # 固定前瞻：沿环（point_at 会自动绕过角点）取前瞻点，不在角点收缩；
                # 不越过行人本身（abs(delta) 封顶）。
                step = min(self.catch_lookahead, abs(delta))
                cx, cy = self.loop.point_at(s_dog + direction * step)
                # 速度与转向解耦：恒定巡航速度沿矩形持续推进，仅大转角处短暂原地转
                yaw_err = normalize_angle(math.atan2(cy - dog.y, cx - dog.x) - dog.yaw)
                if abs(yaw_err) > self.turn_in_place_thresh:
                    lin = 0.0
                else:
                    lin = self.catch_speed
                ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
                self._emit(dog, lin, ang)
                return

        # ---- 阶段3：formation（绕行人 1.5m 旋转三角围捕）----
        # 被甩远则退回 catch_up（滞回，避免直线抄近路撞楼）
        if dist_to_ped > self.revert_radius:
            dog.phase = 'catch_up'
            self.get_logger().info(f'{name}: 被甩开 -> catch_up')
            dog.publish_zero()
            return

        if self.formation_heading is None:
            # 行人还没动过，没有可用航向；原地等
            dog.publish_zero()
            return

        angle = self.formation_heading + FORMATION_OFFSETS[name]
        goal_x = self.target_x + self.r * math.cos(angle)
        goal_y = self.target_y + self.r * math.sin(angle)
        dx, dy = goal_x - dog.x, goal_y - dog.y
        dist = math.hypot(dx, dy)

        if dist > self.deadband:
            yaw_err = normalize_angle(math.atan2(dy, dx) - dog.yaw)
            if abs(yaw_err) > self.turn_in_place_thresh:
                lin = 0.0
            else:
                lin = clamp(self.k_linear * dist, 0.0, self.max_linear)
                ff = self.target_vx * math.cos(dog.yaw) + self.target_vy * math.sin(dog.yaw)
                lin = clamp(lin + max(0.0, ff), 0.0, self.max_linear)
            ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
        else:
            # 已在编队点附近：朝行人运动方向对齐，匀速跟随
            yaw_err = normalize_angle(self.formation_heading - dog.yaw)
            ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
            ff = self.target_vx * math.cos(dog.yaw) + self.target_vy * math.sin(dog.yaw)
            lin = clamp(max(0.0, ff), 0.0, self.max_linear)

        self._emit(dog, lin, ang)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicEncircle()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        for dog in node.dogs.values():
            dog.publish_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
