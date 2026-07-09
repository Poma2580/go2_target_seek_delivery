#!/usr/bin/env python3
"""三 Go2 动态行人追踪与“拦截式围捕”节点。

订阅目标里程计（默认 /walking_target/odom，可切到 /go2_1/target_estimated/odom）
与三只狗 /go2_N/odom，按固定角色完成动态追踪与合围：
  1. go2_1：感知手，沿基准矩形回路 catch_up 追上目标，贴近后进入 formation，
     在目标后方定距跟随并始终正对目标，不参与最终冲刺。
  2. go2_2/go2_3：冲刺手，状态机为 approach -> to_stage -> staged -> charge -> done；
     先沿各自外扩车道绕到目标前方，双方都 staged 后同步直线冲刺，冲进 r_final 后冻结。

要点：
- catch_up/to_stage 严格沿矩形环或外扩车道走，结构上不穿过中间房子。
- staged 只表示已到位，冲刺手仍持续跟随移动 stage 点，等待同步屏障统一放行。
- 目标估计缺失超过保活时间时追捕段停车；Ctrl+C 退出前会给三只狗补发零速度。
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

# 矩形回路的形心，用于求“外侧法向”（点 - 形心 归一化）；行人绕此街区逆时针走，
# 内侧是建筑，狗只能在外侧街道活动，故斜插手沿外侧法向外移。
LOOP_CENTER = (
    sum(x for x, _ in LOOP_CORNERS) / len(LOOP_CORNERS),
    sum(y for _, y in LOOP_CORNERS) / len(LOOP_CORNERS),
)


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

# 拦截式围捕的角色分工：go2_1 后方跟随做感知（不冲刺）；go2_2/go2_3 为“冲刺手”，
# 沿回路超车绕到行人前方 staging 点、等两只都就位后一起冲刺。
#   INTERCEPTOR：直接拦在行人前方路径上（弧长在前 ahead_intercept 米）。
#   SIDE_FLANKER：偏侧、并沿外侧法向外移 side_offset，从外侧街道斜插。
INTERCEPTOR = 'go2_2'
SIDE_FLANKER = 'go2_3'
FLANKERS = (INTERCEPTOR, SIDE_FLANKER)

# 用户手工设计的接近安全路点：每只狗一串 (x, y)，绕开建筑到达环附近。
# 留空列表 [] 表示该狗跳过 approach、直接进入 catch_up（沿环追人）。
# go2_1 初始即能感知到行人，直接进入 catch_up，故留空。
APPROACH_WAYPOINTS = {
    'go2_1': [],
    'go2_2': [(9.0, 5.0)],
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

        # 状态机：
        #   go2_1(感知手)     catch_up(沿基准回路锚定追人) -> formation(贴近 catch_radius 才切)
        #   go2_2/go2_3(冲刺手) approach -> to_stage -> staged -> charge -> done
        if name in FLANKERS:
            self.phase = 'approach' if APPROACH_WAYPOINTS.get(name) else 'to_stage'
        else:
            # 与原版严格对齐：go2_1 空路点 -> 直接 catch_up（回路锚定，绝不穿进内侧建筑），
            # 只有贴近行人 catch_radius 才切入自由 B 方案 formation。
            self.phase = 'approach' if APPROACH_WAYPOINTS.get(name) else 'catch_up'
        self.approach_index = 0
        self.staged = False        # 冲刺手：已到达 staging 点、等待另一只（同步屏障用）

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
        self.declare_parameter('formation_radius', 2.0)  # 编队半径（绕行人距离）
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
        # 拦截式围捕（go2_2/go2_3 的 to_stage -> staged -> charge）
        self.declare_parameter('ahead_intercept', 15.0)         # 拦截手：绕到行人前方的弧长距离
        self.declare_parameter('ahead_flank', 12.0)             # 斜插手：偏侧的弧长距离(更偏侧)
        self.declare_parameter('side_offset', 3.0)             # 斜插手：外侧法向外移量(车道)
        self.declare_parameter('intercept_offset', 1.5)        # 拦截手：外侧车道外移量(与 go2_1/斜插手分道避撞)
        self.declare_parameter('r_final', 5.0)                 # 冲进此距离即任务完成
        self.declare_parameter('stage_arrive_tol', 1.0)        # 到达 staging 点判定容差
        self.declare_parameter('charge_speed', 0.65)           # 冲刺速度(=max_linear)

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
        self.ahead_intercept = self.get_parameter('ahead_intercept').value
        self.ahead_flank = self.get_parameter('ahead_flank').value
        self.side_offset = self.get_parameter('side_offset').value
        self.intercept_offset = self.get_parameter('intercept_offset').value
        self.r_final = self.get_parameter('r_final').value
        self.stage_arrive_tol = self.get_parameter('stage_arrive_tol').value
        self.charge_speed = self.get_parameter('charge_speed').value

        self.dt = 1.0 / self.rate
        self.loop = Loop(LOOP_CORNERS)
        # 每只冲刺手一条“同心外扩矩形”车道回路：把基准回路四角整体外扩各自 lane 米。
        # 直边垂直外移、拐角干净、弧长单调 —— 取代 v3 的“径向逐点外移”（拐角会转圈的根源）。
        self.lane_loops = {
            name: Loop(self._expanded_corners(self._lane_offset(name)))
            for name in FLANKERS
        }

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
        self._charge_logged = False      # 「同时冲刺」只打印一次
        self._capture_logged = False     # 「围捕完成」只打印一次
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

    # ----- 拦截式围捕：几何与冲刺手状态机 -----
    def _lane_offset(self, name):
        """各冲刺手的车道外扩量：go2_2 小、go2_3 大，两条车道横向错开避撞。"""
        return self.intercept_offset if name == INTERCEPTOR else self.side_offset

    def _expanded_corners(self, d):
        """把基准矩形回路四角整体外扩 d 米，得到同心大矩形（车道回路的角点）。

        copysign 保证每个角沿各自象限外移（外扩而非缩小），直边平移 d、拐角干净，
        弧长单调——避免 v3 “径向逐点外移” 在拐角处的非单调打转。
        """
        cx, cy = LOOP_CENTER
        return [(x + math.copysign(d, x - cx), y + math.copysign(d, y - cy))
                for (x, y) in LOOP_CORNERS]

    def _control_flanker(self, name, dog, dist_to_ped):
        # --- to_stage / staged：在自己的车道回路上跟到行人前方 ahead 处，并【持续保持】---
        # staged 不站死：继续跟随随行人前移的 staging 点，行人往前走狗同步走，绝不被追上/走过；
        # 只是 phase=staged 时不 charge，等两只都就位、屏障在 control_loop 里一起放行。
        if dog.phase in ('to_stage', 'staged'):
            lane_loop = self.lane_loops[name]
            ahead = self.ahead_intercept if name == INTERCEPTOR else self.ahead_flank
            s_ped = lane_loop.project(self.target_x, self.target_y)   # 行人投影到本车道
            s_goal = s_ped + ahead                                    # 行人前方(travel_dir=+1)
            s_dog = lane_loop.project(dog.x, dog.y)
            delta = lane_loop.signed_arc(s_dog, s_goal)               # 狗->staging 带符号弧长
            direction = 1.0 if delta >= 0.0 else -1.0
            step = min(self.catch_lookahead, abs(delta))
            cx, cy = lane_loop.point_at(s_dog + direction * step)     # 干净车道前瞻点，无逐点外移
            yaw_err = normalize_angle(math.atan2(cy - dog.y, cx - dog.x) - dog.yaw)
            gate = max(math.cos(yaw_err), 0.25)                       # 边走边转，禁止硬原地转
            # 比例限速：离 staging 弧长越近越慢，稳定“跟着行人保持在其前方”，不冲过头也不站死。
            lin = clamp(self.k_linear * abs(delta), 0.0, self.catch_speed) * gate
            ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
            # 首次到位即置 staged（sticky，供屏障判定）；之后继续保持跟随。
            if dog.phase == 'to_stage':
                sx, sy = lane_loop.point_at(s_goal)
                if math.hypot(sx - dog.x, sy - dog.y) < self.stage_arrive_tol:
                    dog.staged = True
                    dog.phase = 'staged'
                    self.get_logger().info(
                        f'{name}: 就位 staging -> staged（继续保持在行人前方，等待另一只）')
            self._emit(dog, lin, ang)
            return

        # --- charge：对准行人直线冲刺，冲进 r_final 即 done ---
        if dog.phase == 'charge':
            if dist_to_ped < self.r_final:
                dog.phase = 'done'
                self.get_logger().info(f'{name}: 冲进 {self.r_final:.1f}m -> done（冻结）')
                dog.publish_zero()
                return
            bearing = math.atan2(self.target_y - dog.y, self.target_x - dog.x)
            yaw_err = normalize_angle(bearing - dog.yaw)
            ang = clamp(self.k_angular * yaw_err, -self.max_angular, self.max_angular)
            gate = max(math.cos(yaw_err), 0.25)       # 未对准先慢转(似"停下转身")，对准后全速冲
            lin = clamp(self.charge_speed * gate, 0.0, self.max_linear)
            self._emit(dog, lin, ang)
            return

        # --- done：冻结 ---
        dog.publish_zero()

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

        # 同步屏障：两只冲刺手都 staged（各自到位）才一起切 charge，保证同时冲刺。
        if all(self.dogs[n].staged for n in FLANKERS):
            for n in FLANKERS:
                if self.dogs[n].phase == 'staged':
                    self.dogs[n].phase = 'charge'
            if not getattr(self, '_charge_logged', False):
                self.get_logger().info('两只冲刺手均就位 -> 同时冲刺 charge')
                self._charge_logged = True

        for name, dog in self.dogs.items():
            self.control_one(name, dog)

        # 围捕完成判定：两只冲刺手都冲进 r_final（done）即算成功，打印一次。
        if (not self._capture_logged
                and all(self.dogs[n].phase == 'done' for n in FLANKERS)):
            self.get_logger().info(
                f'★ 围捕完成：go2_2/go2_3 均冲进 {self.r_final:.1f}m，任务成功。')
            self._capture_logged = True

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
                # 只有冲刺手(go2_2/go2_3)会走 approach；结束后进入车道跟随 to_stage。
                dog.phase = 'to_stage'
                self.get_logger().info(f'{name}: approach 完成 -> to_stage')
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

        # ---- 冲刺手（go2_2/go2_3）：to_stage -> staged -> charge -> done ----
        # 与 go2_1 完全分流：冲刺手在自己的车道回路上跟到行人前方 staging、两只都就位后
        # 一起冲刺围捕。go2_1 走下面原版的 catch_up -> formation。
        if name in FLANKERS:
            self._control_flanker(name, dog, dist_to_ped)
            return

        # ---- 阶段2：catch_up（沿基准回路就近追人，贴角、不穿心）——仅 go2_1（与原版对齐）----
        # 把狗与行人都投影到矩形回路上，沿回路最短弧贴边追；回路在建筑外沿，结构上不可能
        # 穿进内侧建筑。贴近行人 catch_radius 才切入自由 B 方案 formation。
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

        # ============================ go2_1：formation（B 方案跟随）============================
        # 贴近行人 catch_radius 后进入本段：机身正对行人、维持距离 r，前向相机死锁目标做
        # 感知，作为围捕后方一角，不参与冲刺。

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
