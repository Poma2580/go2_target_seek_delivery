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


# 三只狗在编队里的角度偏置（相对行人运动航向）。
# go2_1 是唯一的视觉感知源：放在行人“正后方”(π)，朝运动方向（朝前）走时相机始终
# 对着行人背影，formation 阶段不会因背对行人而丢检测、连累全队停车。
# go2_2/go2_3 放在 ±60° 偏前，与 go2_1 仍构成等边三角形（间隔 120°）。
FORMATION_OFFSETS = {
    'go2_1': math.pi,                # 180°（行人正后方，保持感知）
    'go2_2': math.pi / 3.0,          # +60°
    'go2_3': -math.pi / 3.0,         # -60°
}

# 用户手工设计的接近安全路点：每只狗一串 (x, y)，绕开建筑到达环附近。
# 留空列表 [] 表示该狗跳过 approach、直接进入 catch_up（沿环追人）。
# go2_1 初始即能感知到行人，直接进入 catch_up，故留空。
APPROACH_WAYPOINTS = {
    'go2_1': [],
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
        self.declare_parameter('formation_radius', 3.0)  # 编队半径（绕行人距离）
        self.declare_parameter('control_rate', 20.0)
        # 目标里程计话题：默认真值，切到在线感知时传 /go2_1/target_estimated/odom。
        self.declare_parameter('target_odom_topic', '/walking_target/odom')
        self.declare_parameter('target_timeout', 5.0)         # 超过此龄视为“非新鲜”，进入 coast
        self.declare_parameter('target_hold', 8.0)            # coast 最长保活时间，超过视为真丢失
        self.declare_parameter('odom_timeout', 0.5)
        self.declare_parameter('max_linear', 0.65)
        self.declare_parameter('max_angular', 0.9)
        self.declare_parameter('max_coast_speed', 0.65)       # coast 外推时的速度模长上限
        self.declare_parameter('position_deadband', 0.25)
        self.declare_parameter('k_linear', 0.8)
        self.declare_parameter('k_angular', 0.9)
        self.declare_parameter('turn_in_place_thresh', 1.2)  # 航向误差大于此阈值原地转
        self.declare_parameter('heading_speed_thresh', 0.08)   # 低于此速度冻结编队航向
        self.declare_parameter('heading_slew_rate', 1.0)       # 编队航向变化率限幅 rad/s
        self.declare_parameter('accel_lin', 1.0)               # 线/角加速度限幅
        self.declare_parameter('accel_ang', 3.0)
        # 三段式相关
        self.declare_parameter('approach_reach', 0.35)         # approach 路点到达阈值
        self.declare_parameter('catch_lookahead', 1.5)         # catch_up 沿环前瞻距离(小→贴边)
        self.declare_parameter('catch_speed', 0.6)             # catch_up 恒定巡航速度
        self.declare_parameter('catch_radius', 3.5)            # 进 formation 的距离
        self.declare_parameter('revert_radius', 8.0)           # 退回 catch_up 的距离(滞回)

        self.r = self.get_parameter('formation_radius').value
        self.rate = float(self.get_parameter('control_rate').value)
        self.target_odom_topic = self.get_parameter('target_odom_topic').value
        self.target_timeout = self.get_parameter('target_timeout').value
        self.target_hold = self.get_parameter('target_hold').value
        self.odom_timeout = self.get_parameter('odom_timeout').value
        self.max_linear = self.get_parameter('max_linear').value
        self.max_angular = self.get_parameter('max_angular').value
        self.max_coast_speed = self.get_parameter('max_coast_speed').value
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

        # 行人状态：target_* 为“本拍实际使用”的目标（可能是新鲜值或 coast 外推值）；
        # last_good_* 为最后一次新鲜估计的快照，coast 时据此外推。
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.last_good_x = 0.0
        self.last_good_y = 0.0
        self.last_good_vx = 0.0
        self.last_good_vy = 0.0
        self.last_good_time = None       # rclpy.time.Time，最后一次新鲜估计的接收时刻
        self.target_received = False
        self.target_ok = False           # 本拍目标是否可用（新鲜或 coast 期内）
        self.formation_heading = None    # 平滑后的编队航向（运动方向）

        self.dogs = {name: DogState(self, name) for name in FORMATION_OFFSETS}

        self.target_sub = self.create_subscription(
            Odometry, self.target_odom_topic, self._target_cb, 10)

        self.timer = self.create_timer(self.dt, self.control_loop)
        self._lost_logged = False
        self.get_logger().info(
            f'multi_go2_dynamic_encircle 已启动，目标源={self.target_odom_topic}，'
            f'等待行人与里程计...')

    def _target_cb(self, msg):
        # 只缓存“最后一次新鲜估计”的快照；本拍实际使用值在 control_loop 里由
        # _resolve_target 决定（新鲜则直接用，短断则据此外推）。
        self.last_good_x = msg.pose.pose.position.x
        self.last_good_y = msg.pose.pose.position.y
        self.last_good_vx = msg.twist.twist.linear.x
        self.last_good_vy = msg.twist.twist.linear.y
        self.last_good_time = self.get_clock().now()
        self.target_received = True

    def _age(self, stamp):
        if stamp is None:
            return float('inf')
        now = self.get_clock().now()
        return (now - rclpy.time.Time.from_msg(stamp)).nanoseconds * 1e-9

    def _resolve_target(self):
        """决定本拍使用的目标状态，并写入 self.target_x/y/vx/vy 与 self.target_ok。

        三档：
          - 新鲜（龄 ≤ target_timeout）：直接用最后估计，返回 fresh=True。
          - 短断（target_timeout < 龄 ≤ target_hold）：用最后估计按最后速度外推，
            速度模长先钳到 max_coast_speed，避免噪声让外推乱飞；返回 fresh=False。
          - 真丢失（龄 > target_hold 或从未收到）：target_ok=False，返回 fresh=False。
        """
        if not self.target_received or self.last_good_time is None:
            self.target_ok = False
            return False

        age = (self.get_clock().now() - self.last_good_time).nanoseconds * 1e-9

        if age <= self.target_timeout:
            self.target_x = self.last_good_x
            self.target_y = self.last_good_y
            self.target_vx = self.last_good_vx
            self.target_vy = self.last_good_vy
            self.target_ok = True
            return True

        if age <= self.target_hold:
            # coast：按最后速度外推位置（速度模长钳制）。航向用钳制后的速度，
            # 这样 _update_formation_heading 在 coast 期保持一致。
            vx, vy = self.last_good_vx, self.last_good_vy
            speed = math.hypot(vx, vy)
            if speed > self.max_coast_speed and speed > 1e-6:
                scale = self.max_coast_speed / speed
                vx, vy = vx * scale, vy * scale
            self.target_x = self.last_good_x + vx * age
            self.target_y = self.last_good_y + vy * age
            self.target_vx = vx
            self.target_vy = vy
            self.target_ok = True
            return False

        self.target_ok = False
        return False

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
        # 解析本拍目标：新鲜 / coast 外推 / 真丢失。不再“无目标整段 return 全员零速”——
        # approach 段只走预设路点、不依赖目标，真丢失时仍可推进；catch_up/formation 段
        # 在 control_one 内部按 self.target_ok 决定是否停。
        fresh = self._resolve_target()

        if not self.target_ok:
            if not self._lost_logged:
                self.get_logger().warn(
                    f'行人估计缺失/超过保活({self.target_hold:.1f}s)，'
                    f'追捕段保持静止；approach 段继续。')
                self._lost_logged = True
        else:
            self._lost_logged = False

        # 编队航向只在估计新鲜时更新，coast 期冻结在最后值。
        if fresh:
            self._update_formation_heading()

        for name, dog in self.dogs.items():
            self.control_one(name, dog)

    def control_one(self, name, dog):
        # 自身里程计超时：零速
        if not dog.received or self._age(dog.last_stamp) > self.odom_timeout:
            dog.publish_zero()
            return

        # ---- 阶段1：approach（走预设安全路点，不依赖目标）----
        # 放在目标可用性判断之前：approach 段只走预设路点，目标暂缺也照常推进。
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

        # catch_up / formation 都需要目标位置；目标真丢失（超过保活）则原地停。
        if not self.target_ok:
            dog.publish_zero()
            return

        dist_to_ped = math.hypot(self.target_x - dog.x, self.target_y - dog.y)

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

        # ============================ 阶段3：formation ============================
        # 目标：让每只狗贴到“行人 + 固定角度偏置 + 半径 r”的编队点上，绕着行人成三角。
        # go2_1 的偏置是 π（行人正后方 r 米处），朝运动方向走时前向相机正好看行人背影。
        #
        # 原“被甩远退回 catch_up”的滞回已被注释掉（下面几行）——注意：这样一来 formation
        # 是“只进不出”的终态，一旦某拍编队点算歪把狗带偏，也不会再回 catch_up 纠正。
        # if dist_to_ped > self.revert_radius:
        #     dog.phase = 'catch_up'
        #     self.get_logger().info(f'{name}: 被甩开 -> catch_up')
        #     dog.publish_zero()
        #     return

        # if self.formation_heading is None:
        #     # 编队点方位依赖“行人运动航向”；行人若从未动过就没有航向，先原地等。
        #     dog.publish_zero()
        #     return

        # # --- 1) 计算编队目标点 ---
        # # angle = 行人运动航向 + 本狗偏置。go2_1 偏置 π ⇒ angle 指向“运动反方向”，
        # # 故 goal 落在行人正后方 r 米处。注意：formation_heading 会随行人拐弯而摆动
        # # （虽有 slew 限幅），goal 因此是一个“会绕着行人转”的移动点。
        # angle = self.formation_heading + FORMATION_OFFSETS[name]
        # goal_x = self.target_x + self.r * math.cos(angle)
        # goal_y = self.target_y + self.r * math.sin(angle)
        # dx, dy = goal_x - dog.x, goal_y - dog.y
        # dist = math.hypot(dx, dy)      # 狗到“编队点”的距离（不是到行人的距离）

        # if dist > self.deadband:
        #     # --- 2a) 离编队点还远：朝编队点走（点跟随）---
        #     # 朝向目标 = 指向编队点的方向 atan2(dy,dx)。
        #     # ⚠️ 隐患：当狗逼近编队点时 (dx,dy)→0，atan2 对微小位置抖动极敏感，方向会
        #     #    突然乱跳，配合下面的“大误差原地转”会表现为“近处突然大角度转向”。
        #     #    此外 go2_1 编队点在行人正后方，若狗此刻在行人侧/前方（catch_up 刚按最短弧
        #     #    追上），朝编队点方向可能背对行人 ⇒ 掉头绕后时相机扫离行人 ⇒ 丢视野。
        #     yaw_err = normalize_angle(math.atan2(dy, dx) - dog.yaw)
        #     if abs(yaw_err) > self.turn_in_place_thresh:
        #         lin = 0.0                      # 朝向误差过大：先原地转对准，不前进
        #     else:
        #         # 比例前进 + 行人速度前馈 ff（只取沿狗朝向的正向分量，帮助跟上移动目标）
        #         lin = clamp(self.k_linear * dist, 0.0, self.max_linear)
        #         ff = self.target_vx * math.cos(dog.yaw) + self.target_vy * math.sin(dog.yaw)
        #         lin = clamp(lin + max(0.0, ff), 0.0, self.max_linear)
        #     ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
        # else:
        #     # --- 2b) 已在编队点附近（dist ≤ deadband）：保持队形、匀速跟随 ---
        #     # 此时不再朝“编队点方向”对齐（会因 atan2 奇异乱转），改为对齐“行人运动航向”，
        #     # 让狗跟着行人同向走；go2_1 在正后方同向 ⇒ 相机稳定看向行人。
        #     # 前进只用前馈 ff（≈行人速度），角速度与线速度都各减半以求平稳。
        #     yaw_err = normalize_angle(self.formation_heading - dog.yaw)
        #     ang = clamp(self.k_angular * yaw_err, -self.max_angular * 0.5, self.max_angular * 0.5)
        #     ff = self.target_vx * math.cos(dog.yaw) + self.target_vy * math.sin(dog.yaw)
        #     lin = clamp(max(0.0, ff), 0.0, self.max_linear * 0.5)
        # 把控制解耦成两路，互不干扰：
        #   角速度 → 机身始终正对行人，前向相机死锁目标，永不丢视野；
        #   线速度 → 维持“狗到行人的距离”= r（编队半径），远则前进、近则后退。
        # 不再用“编队点 goal + atan2(goal-狗)”，从根上消除逼近编队点时的方向奇异，
        # 也不再依赖 formation_heading（行人拐角航向摆动不再干扰跟随）。

        # 1) 朝向：对准行人本身
        bearing = math.atan2(self.target_y - dog.y, self.target_x - dog.x)
        yaw_err = normalize_angle(bearing - dog.yaw)
        ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)

        # 2) 线速度：维持到行人的距离 = r
        rng = math.hypot(self.target_x - dog.x, self.target_y - dog.y)
        e = rng - self.r                      # >0 太远(前进)  <0 太近(后退)
        if abs(e) < self.deadband:            # 距离死区，稳态不抖
            e = 0.0
        ff = self.target_vx * math.cos(bearing) + self.target_vy * math.sin(bearing)
        gate = max(math.cos(yaw_err), 0.25)   # 没对准就收着走，但不硬切0(避免原地转卡死)
        lin = clamp((self.k_linear * e + ff) * gate,
                    -0.5 * self.max_linear, self.max_linear)

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
