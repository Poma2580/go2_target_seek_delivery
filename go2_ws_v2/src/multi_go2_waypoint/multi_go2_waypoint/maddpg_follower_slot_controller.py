#!/usr/bin/env python3
"""Gazebo controller for the follower_slot_tracking_v0 MADDPG policy.

go1 can be controlled by a simple dynamic_encircle-style target tracking law.
go2/go3 are controlled by the two trained follower actors from one MADDPG
checkpoint.
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def find_repo_root():
    env_root = os.environ.get("DELIVERY_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()

    here = Path(__file__).resolve()
    for path in (here, *here.parents):
        for candidate in (path, path.parent):
            if (candidate / "三角形MADDPG").exists() and (candidate / "go2_ws_v2").exists():
                return candidate.resolve()

    # Fallback for the current project layout. Users can still override this
    # with the DELIVERY_ROOT environment variable or the maddpg_root parameter.
    return Path("/home/wangantong/KD_all/go2_target_seek_delivery").resolve()


REPO_ROOT = find_repo_root()
DEFAULT_MADDPG_ROOT = REPO_ROOT / "三角形MADDPG"
DEFAULT_MODEL_PATH = (
    DEFAULT_MADDPG_ROOT
    / "runs"
    / "follower_slot_tracking_v0"
    / "MADDPG"
    / "stage5_b512_usteps20_g0.99_t0.005_alr0.0003_clr0.0005_n0.25_minn0.03_h128,128_20260729_134612"
    / "best_model.pt"
)

POS_SCALE = 8.0
VEL_SCALE = 0.60
DIST_PARAM_SCALE = 3.0
DEFAULT_SIDE_DIST = 1.20
DEFAULT_LEADER_FOLLOW_DIST = 1.80

# Same urban loop used by dynamic_encircle.py for go2_1 catch_up.
LOOP_CORNERS = [(41.0, 4.0), (41.0, 36.0), (-13.0, 36.0), (-13.0, 4.0)]


@dataclass
class EntityState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    received: bool = False
    receive_time = None


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def unit_from_angle(theta):
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def rot90(v):
    return np.array([-v[1], v[0]], dtype=np.float32)


def body_frame(yaw, vec):
    c, s = math.cos(yaw), math.sin(yaw)
    x = c * vec[0] + s * vec[1]
    y = -s * vec[0] + c * vec[1]
    return np.array([x, y], dtype=np.float32)


def world_vel_from_odom(state, msg, twist_in_body_frame):
    vx = float(msg.twist.twist.linear.x)
    vy = float(msg.twist.twist.linear.y)
    if not twist_in_body_frame:
        return vx, vy
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return c * vx - s * vy, s * vx + c * vy


class Loop:
    def __init__(self, corners):
        self.corners = corners
        self.n = len(corners)
        self.edges = []
        self.cum = [0.0]
        for i in range(self.n):
            x0, y0 = corners[i]
            x1, y1 = corners[(i + 1) % self.n]
            d = math.hypot(x1 - x0, y1 - y0)
            self.edges.append((x0, y0, x1, y1, d))
            self.cum.append(self.cum[-1] + d)
        self.L = self.cum[-1]

    def project(self, px, py):
        best_s, best_d2 = 0.0, float("inf")
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
        s = s % self.L
        for i, (x0, y0, x1, y1, d) in enumerate(self.edges):
            if s <= self.cum[i + 1] or i == self.n - 1:
                t = (s - self.cum[i]) / d if d > 1e-9 else 0.0
                return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        return self.corners[0]

    def signed_arc(self, s_from, s_to):
        d = (s_to - s_from) % self.L
        if d > self.L / 2.0:
            d -= self.L
        return d


class MaddpgFollowerSlotController(Node):
    def __init__(self):
        super().__init__("maddpg_follower_slot_controller")

        self.declare_parameter("maddpg_root", str(DEFAULT_MADDPG_ROOT))
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("target_odom_topic", "/walking_target/odom")
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("target_timeout", 2.0)
        self.declare_parameter("max_linear", 0.60)
        self.declare_parameter("max_angular", 1.00)
        self.declare_parameter("follower_max_linear", 0.60)
        self.declare_parameter("follower_max_angular", 0.80)
        self.declare_parameter("follower_allow_reverse", False)
        self.declare_parameter("follower_accel_lin", 0.40)
        self.declare_parameter("follower_accel_ang", 0.60)
        self.declare_parameter("follower_action_mode", "accel")
        self.declare_parameter("follower_turn_slowdown", True)
        self.declare_parameter("side_dist", DEFAULT_SIDE_DIST)
        self.declare_parameter("leader_follow_dist", DEFAULT_LEADER_FOLLOW_DIST)
        self.declare_parameter("control_leader", True)
        self.declare_parameter("leader_k_linear", 0.8)
        self.declare_parameter("leader_k_angular", 0.9)
        self.declare_parameter("leader_distance_deadband", 0.25)
        self.declare_parameter("leader_catch_radius", 3.5)
        self.declare_parameter("leader_catch_lookahead", 1.5)
        self.declare_parameter("leader_catch_speed", 0.6)
        self.declare_parameter("leader_turn_in_place_thresh", 1.2)
        self.declare_parameter("leader_accel_lin", 1.0)
        self.declare_parameter("leader_accel_ang", 3.0)
        self.declare_parameter("leader_initial_phase", "formation")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("publish_zero_when_not_ready", True)
        self.declare_parameter("odom_twist_in_body_frame", False)
        self.declare_parameter("log_rate", 2.0)
        self.declare_parameter("debug_policy", True)

        self.maddpg_root = Path(self.get_parameter("maddpg_root").value).expanduser().resolve()
        self.model_path = Path(self.get_parameter("model_path").value).expanduser().resolve()
        self.target_odom_topic = str(self.get_parameter("target_odom_topic").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.follower_max_linear = float(self.get_parameter("follower_max_linear").value)
        self.follower_max_angular = float(self.get_parameter("follower_max_angular").value)
        self.follower_allow_reverse = bool(self.get_parameter("follower_allow_reverse").value)
        self.follower_accel_lin = float(self.get_parameter("follower_accel_lin").value)
        self.follower_accel_ang = float(self.get_parameter("follower_accel_ang").value)
        self.follower_action_mode = str(self.get_parameter("follower_action_mode").value)
        if self.follower_action_mode not in ("accel", "velocity"):
            raise ValueError("follower_action_mode must be 'accel' or 'velocity'")
        self.follower_turn_slowdown = bool(self.get_parameter("follower_turn_slowdown").value)
        self.side_dist = float(self.get_parameter("side_dist").value)
        self.leader_follow_dist = float(self.get_parameter("leader_follow_dist").value)
        self.control_leader = bool(self.get_parameter("control_leader").value)
        self.leader_k_linear = float(self.get_parameter("leader_k_linear").value)
        self.leader_k_angular = float(self.get_parameter("leader_k_angular").value)
        self.leader_distance_deadband = float(self.get_parameter("leader_distance_deadband").value)
        self.leader_catch_radius = float(self.get_parameter("leader_catch_radius").value)
        self.leader_catch_lookahead = float(self.get_parameter("leader_catch_lookahead").value)
        self.leader_catch_speed = float(self.get_parameter("leader_catch_speed").value)
        self.leader_turn_in_place_thresh = float(self.get_parameter("leader_turn_in_place_thresh").value)
        self.leader_accel_lin = float(self.get_parameter("leader_accel_lin").value)
        self.leader_accel_ang = float(self.get_parameter("leader_accel_ang").value)
        self.leader_initial_phase = str(self.get_parameter("leader_initial_phase").value)
        if self.leader_initial_phase not in ("catch_up", "formation"):
            raise ValueError("leader_initial_phase must be 'catch_up' or 'formation'")
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.publish_zero_when_not_ready = bool(self.get_parameter("publish_zero_when_not_ready").value)
        self.odom_twist_in_body_frame = bool(self.get_parameter("odom_twist_in_body_frame").value)
        self.log_period = 1.0 / max(float(self.get_parameter("log_rate").value), 1e-6)
        self.debug_policy = bool(self.get_parameter("debug_policy").value)
        self.last_log_time = None
        self.leader_loop = Loop(LOOP_CORNERS)
        self.leader_phase = self.leader_initial_phase
        self.leader_prev_lin = 0.0
        self.leader_prev_ang = 0.0
        self.follower_prev_cmds = {
            "go2_2": [0.0, 0.0],
            "go2_3": [0.0, 0.0],
        }
        self.follower_prev_actions = {
            "go2_2": [0.0, 0.0],
            "go2_3": [0.0, 0.0],
        }

        self.robot_names = ("go2_1", "go2_2", "go2_3")
        self.follower_names = ("go2_2", "go2_3")
        self.command_names = ("go2_1", "go2_2", "go2_3") if self.control_leader else self.follower_names
        self.states = {name: EntityState() for name in self.robot_names}
        self.target = EntityState()
        self.last_slots = None

        self.cmd_pubs = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in self.command_names
        }
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot_name=name: self._odom_cb(robot_name, msg),
                10,
            )
        self.create_subscription(Odometry, self.target_odom_topic, self._target_cb, 10)

        self.maddpg = self._load_model()
        rate = float(self.get_parameter("control_rate").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-6), self._timer_cb)
        self.get_logger().info(
            "maddpg_follower_slot_controller started: "
            f"model={self.model_path}, target={self.target_odom_topic}, "
            f"go2_2<=actor0/go2_left, go2_3<=actor1/go3_right, "
            f"side_dist={self.side_dist:.2f}, leader_follow_dist={self.leader_follow_dist:.2f}, "
            f"max_linear={self.max_linear:.2f}, max_angular={self.max_angular:.2f}, "
            f"follower_max_linear={self.follower_max_linear:.2f}, "
            f"follower_max_angular={self.follower_max_angular:.2f}, "
            f"leader_initial_phase={self.leader_initial_phase}, "
            f"control_leader={self.control_leader}, dry_run={self.dry_run}"
        )

    def _load_model(self):
        if not self.maddpg_root.exists():
            raise FileNotFoundError(f"MADDPG root not found: {self.maddpg_root}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"MADDPG model not found: {self.model_path}")

        sys.path.insert(0, str(self.maddpg_root))
        from maddpg import MADDPG

        maddpg = MADDPG(
            state_sizes=[28, 28],
            action_sizes=[2, 2],
            hidden_sizes=(128, 128),
            action_low=-1.0,
            action_high=1.0,
        )
        maddpg.load(str(self.model_path))
        return maddpg

    def _now_age(self, receive_time):
        if receive_time is None:
            return float("inf")
        return (self.get_clock().now() - receive_time).nanoseconds * 1e-9

    def _odom_cb(self, name, msg):
        state = self.states[name]
        state.x = float(msg.pose.pose.position.x)
        state.y = float(msg.pose.pose.position.y)
        state.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        state.vx, state.vy = world_vel_from_odom(state, msg, self.odom_twist_in_body_frame)
        state.wz = float(msg.twist.twist.angular.z)
        state.received = True
        state.receive_time = self.get_clock().now()

    def _target_cb(self, msg):
        self.target.x = float(msg.pose.pose.position.x)
        self.target.y = float(msg.pose.pose.position.y)
        self.target.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        self.target.vx = float(msg.twist.twist.linear.x)
        self.target.vy = float(msg.twist.twist.linear.y)
        self.target.wz = float(msg.twist.twist.angular.z)
        self.target.received = True
        self.target.receive_time = self.get_clock().now()

    def _data_ready(self):
        missing = []
        for name in self.robot_names:
            state = self.states[name]
            if not state.received or self._now_age(state.receive_time) > self.odom_timeout:
                missing.append(name)
        if not self.target.received or self._now_age(self.target.receive_time) > self.target_timeout:
            missing.append("target")
        if missing:
            if self.publish_zero_when_not_ready:
                self._publish_all_zero()
            self.get_logger().info(f"waiting for fresh odom: {', '.join(missing)}")
            return False
        return True

    def _arrays(self):
        leader = self.states["go2_1"]
        go2 = self.states["go2_2"]
        go3 = self.states["go2_3"]
        target_pos = np.array([self.target.x, self.target.y], dtype=np.float32)
        target_vel = np.array([self.target.vx, self.target.vy], dtype=np.float32)
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        leader_vel = np.array([leader.vx, leader.vy], dtype=np.float32)
        follower_pos = np.array([[go2.x, go2.y], [go3.x, go3.y]], dtype=np.float32)
        follower_vel = np.array([[go2.vx, go2.vy], [go3.vx, go3.vy]], dtype=np.float32)
        follower_yaw = np.array([go2.yaw, go3.yaw], dtype=np.float32)
        follower_wz = np.array([go2.wz, go3.wz], dtype=np.float32)
        return target_pos, target_vel, leader_pos, leader_vel, follower_pos, follower_vel, follower_yaw, follower_wz

    def _compute_slots(self, target_pos, leader_pos):
        direction = target_pos - leader_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            forward = unit_from_angle(self.target.yaw)
        else:
            forward = direction / norm
        left = rot90(forward)
        slots = np.zeros((2, 2), dtype=np.float32)
        slots[0] = target_pos + self.side_dist * left
        slots[1] = target_pos - self.side_dist * left
        return slots

    def _build_observations(self):
        target_pos, target_vel, leader_pos, leader_vel, follower_pos, follower_vel, follower_yaw, follower_wz = self._arrays()
        slots = self._compute_slots(target_pos, leader_pos)
        if self.last_slots is None:
            self.last_slots = slots.copy()
        dt = 1.0 / max(float(self.get_parameter("control_rate").value), 1e-6)
        slot_vel = (slots - self.last_slots) / max(dt, 1e-6)
        leader_target_dist = float(np.linalg.norm(target_pos - leader_pos))

        observations = []
        diagnostics = []
        for idx in range(2):
            other_idx = 1 - idx
            yaw = float(follower_yaw[idx])
            self_vel_body = body_frame(yaw, follower_vel[idx]) / VEL_SCALE
            leader_rel = body_frame(yaw, leader_pos - follower_pos[idx]) / POS_SCALE
            leader_rel_vel = body_frame(yaw, leader_vel - follower_vel[idx]) / VEL_SCALE
            target_rel = body_frame(yaw, target_pos - follower_pos[idx]) / POS_SCALE
            target_rel_vel = body_frame(yaw, target_vel - follower_vel[idx]) / VEL_SCALE
            slot_rel = body_frame(yaw, slots[idx] - follower_pos[idx]) / POS_SCALE
            slot_rel_vel = body_frame(yaw, slot_vel[idx] - follower_vel[idx]) / VEL_SCALE
            other_rel = body_frame(yaw, follower_pos[other_idx] - follower_pos[idx]) / POS_SCALE
            role = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float32)
            slot_error = float(np.linalg.norm(slots[idx] - follower_pos[idx]))
            slot_error_norm = np.array([slot_error / POS_SCALE], dtype=np.float32)
            leader_target_dist_norm = np.array([leader_target_dist / POS_SCALE], dtype=np.float32)
            formation_params = np.array(
                [self.side_dist / DIST_PARAM_SCALE, self.leader_follow_dist / DIST_PARAM_SCALE],
                dtype=np.float32,
            )
            robot_name = self.follower_names[idx]
            real_motion_state = np.array(
                [
                    self_vel_body[0],
                    follower_wz[idx] / max(self.follower_max_angular, 1e-6),
                ],
                dtype=np.float32,
            )
            prev_action = np.array(self.follower_prev_actions[robot_name], dtype=np.float32)
            obs = np.concatenate(
                [
                    self_vel_body,
                    np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32),
                    leader_rel,
                    leader_rel_vel,
                    target_rel,
                    target_rel_vel,
                    slot_rel,
                    slot_rel_vel,
                    other_rel,
                    role,
                    slot_error_norm,
                    leader_target_dist_norm,
                    formation_params,
                    real_motion_state,
                    prev_action,
                ]
            ).astype(np.float32)
            observations.append(obs)
            formation_yaw = math.atan2((target_pos - leader_pos)[1], (target_pos - leader_pos)[0])
            diagnostics.append(
                {
                    "slot": slots[idx],
                    "pos": follower_pos[idx],
                    "error_vec": slots[idx] - follower_pos[idx],
                    "slot_error": slot_error,
                    "yaw_error": abs(wrap_angle(formation_yaw - yaw)),
                    "real_v_body_x": float(self_vel_body[0] * self.follower_max_linear),
                    "real_wz": float(follower_wz[idx]),
                }
            )

        self.last_slots = slots.copy()
        return observations, diagnostics

    def _action_to_twist(self, robot_name, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        dt = 1.0 / max(float(self.get_parameter("control_rate").value), 1e-6)
        prev_lin, prev_ang = self.follower_prev_cmds[robot_name]

        if self.follower_action_mode == "accel":
            # New Gazebo-safe policy convention:
            # action[0], action[1] are acceleration-like increments.  We
            # integrate them locally, then clamp the resulting /cmd_vel.
            linear = prev_lin + float(action[0]) * self.follower_accel_lin * dt
            angular = prev_ang + float(action[1]) * self.follower_accel_ang * dt
        else:
            # Backward-compatible mode for older models trained to output
            # direct velocity commands.
            linear = float(action[0]) * self.follower_max_linear
            angular = float(action[1]) * self.follower_max_angular

        min_linear = -self.follower_max_linear if self.follower_allow_reverse else 0.0
        linear = clamp(linear, min_linear, self.follower_max_linear)
        angular = clamp(angular, -self.follower_max_angular, self.follower_max_angular)

        # Quadruped gait is fragile under simultaneous high forward speed and
        # high yaw rate. Slow down while turning sharply to avoid hopping/falls.
        if self.follower_turn_slowdown:
            turn_ratio = min(abs(angular) / max(self.follower_max_angular, 1e-6), 1.0)
            if linear > 0.0:
                linear = min(
                    linear,
                    self.follower_max_linear * max(0.25, 1.0 - 0.65 * turn_ratio),
                )

        if not np.isfinite(linear) or not np.isfinite(angular):
            return Twist()

        if self.follower_action_mode == "velocity":
            linear = clamp(
                linear,
                prev_lin - self.follower_accel_lin * dt,
                prev_lin + self.follower_accel_lin * dt,
            )
            angular = clamp(
                angular,
                prev_ang - self.follower_accel_ang * dt,
                prev_ang + self.follower_accel_ang * dt,
            )
        self.follower_prev_cmds[robot_name] = [linear, angular]
        self.follower_prev_actions[robot_name] = [float(action[0]), float(action[1])]

        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd

    def _leader_tracking_twist(self):
        leader = self.states["go2_1"]
        dx = self.target.x - leader.x
        dy = self.target.y - leader.y
        dist_to_target = math.hypot(dx, dy)

        cmd = Twist()

        # dynamic_encircle-style go2_1 control:
        # catch_up follows the urban loop until go1 is near the pedestrian;
        # formation then faces the pedestrian and maintains a fixed range.
        if self.leader_phase == "catch_up":
            if dist_to_target < self.leader_catch_radius:
                self.leader_phase = "formation"
                self.get_logger().info("go2_1: catch_up -> formation")
            else:
                s_dog = self.leader_loop.project(leader.x, leader.y)
                s_target = self.leader_loop.project(self.target.x, self.target.y)
                delta = self.leader_loop.signed_arc(s_dog, s_target)
                direction = 1.0 if delta >= 0.0 else -1.0
                step = min(self.leader_catch_lookahead, abs(delta))
                gx, gy = self.leader_loop.point_at(s_dog + direction * step)
                yaw_error = wrap_angle(math.atan2(gy - leader.y, gx - leader.x) - leader.yaw)
                if abs(yaw_error) > self.leader_turn_in_place_thresh:
                    lin = 0.0
                else:
                    lin = self.leader_catch_speed
                ang = clamp(self.leader_k_angular * yaw_error, -self.max_angular, self.max_angular)
                cmd.linear.x = clamp(lin, 0.0, self.max_linear)
                cmd.angular.z = ang
                return self._limit_leader_cmd(cmd)

        bearing = math.atan2(dy, dx) if dist_to_target > 1e-6 else leader.yaw
        yaw_error = wrap_angle(bearing - leader.yaw)
        cmd.angular.z = clamp(self.leader_k_angular * yaw_error, -self.max_angular, self.max_angular)

        distance_error = dist_to_target - self.leader_follow_dist
        if abs(distance_error) < self.leader_distance_deadband:
            distance_error = 0.0

        target_feedforward = self.target.vx * math.cos(bearing) + self.target.vy * math.sin(bearing)
        heading_gate = max(math.cos(yaw_error), 0.25)
        cmd.linear.x = clamp(
            (self.leader_k_linear * distance_error + target_feedforward) * heading_gate,
            -0.5 * self.max_linear,
            self.max_linear,
        )
        return self._limit_leader_cmd(cmd)

    def _limit_leader_cmd(self, cmd):
        dt = 1.0 / max(float(self.get_parameter("control_rate").value), 1e-6)
        cmd.linear.x = clamp(
            cmd.linear.x,
            self.leader_prev_lin - self.leader_accel_lin * dt,
            self.leader_prev_lin + self.leader_accel_lin * dt,
        )
        cmd.angular.z = clamp(
            cmd.angular.z,
            self.leader_prev_ang - self.leader_accel_ang * dt,
            self.leader_prev_ang + self.leader_accel_ang * dt,
        )
        self.leader_prev_lin = cmd.linear.x
        self.leader_prev_ang = cmd.angular.z
        return cmd

    def _publish_all_zero(self):
        self.leader_prev_lin = 0.0
        self.leader_prev_ang = 0.0
        for name in self.follower_prev_cmds:
            self.follower_prev_cmds[name] = [0.0, 0.0]
        for pub in self.cmd_pubs.values():
            pub.publish(Twist())

    def _should_log(self):
        now = self.get_clock().now()
        if self.last_log_time is None:
            self.last_log_time = now
            return True
        elapsed = (now - self.last_log_time).nanoseconds * 1e-9
        if elapsed >= self.log_period:
            self.last_log_time = now
            return True
        return False

    def _timer_cb(self):
        if not self._data_ready():
            return

        observations, diagnostics = self._build_observations()
        actions = [np.asarray(action, dtype=np.float32) for action in self.maddpg.act(observations, add_noise=False)]

        if not all(np.all(np.isfinite(action)) for action in actions):
            self.get_logger().error("MADDPG produced NaN or Inf action. Publishing zero velocity.")
            self._publish_all_zero()
            return

        cmds = [
            self._action_to_twist("go2_2", actions[0]),
            self._action_to_twist("go2_3", actions[1]),
        ]
        leader_cmd = self._leader_tracking_twist() if self.control_leader else None
        if not self.dry_run:
            if self.control_leader:
                self.cmd_pubs["go2_1"].publish(leader_cmd)
            self.cmd_pubs["go2_2"].publish(cmds[0])
            self.cmd_pubs["go2_3"].publish(cmds[1])

        if self._should_log():
            leader_text = ""
            if self.control_leader:
                leader_text = (
                    f"go2_1<=leader[{self.leader_phase}] "
                    f"cmd=({leader_cmd.linear.x:+.2f},{leader_cmd.angular.z:+.2f}); "
                )
            if self.debug_policy:
                target_pos, _, leader_pos, _, _, _, _, _ = self._arrays()
                d0, d1 = diagnostics
                self.get_logger().info(
                    "\n"
                    f"[policy_debug] dry_run={self.dry_run}\n"
                    f"  target_pos=({target_pos[0]:+.2f},{target_pos[1]:+.2f}) "
                    f"go1=({leader_pos[0]:+.2f},{leader_pos[1]:+.2f}) "
                    f"{leader_text}\n"
                    f"  expected_slots: "
                    f"go2_left=({d0['slot'][0]:+.2f},{d0['slot'][1]:+.2f}), "
                    f"go3_right=({d1['slot'][0]:+.2f},{d1['slot'][1]:+.2f})\n"
                    f"  go2_2/left actor0: "
                    f"pos=({d0['pos'][0]:+.2f},{d0['pos'][1]:+.2f}) "
                    f"slot=({d0['slot'][0]:+.2f},{d0['slot'][1]:+.2f}) "
                    f"err=({d0['error_vec'][0]:+.2f},{d0['error_vec'][1]:+.2f}) "
                    f"|err|={d0['slot_error']:.2f} yaw_err={d0['yaw_error']:.2f} "
                    f"real=({d0['real_v_body_x']:+.2f},{d0['real_wz']:+.2f}) "
                    f"action=({actions[0][0]:+.3f},{actions[0][1]:+.3f}) "
                    f"cmd_vel=({cmds[0].linear.x:+.2f},{cmds[0].angular.z:+.2f})\n"
                    f"  go2_3/right actor1: "
                    f"pos=({d1['pos'][0]:+.2f},{d1['pos'][1]:+.2f}) "
                    f"slot=({d1['slot'][0]:+.2f},{d1['slot'][1]:+.2f}) "
                    f"err=({d1['error_vec'][0]:+.2f},{d1['error_vec'][1]:+.2f}) "
                    f"|err|={d1['slot_error']:.2f} yaw_err={d1['yaw_error']:.2f} "
                    f"real=({d1['real_v_body_x']:+.2f},{d1['real_wz']:+.2f}) "
                    f"action=({actions[1][0]:+.3f},{actions[1][1]:+.3f}) "
                    f"cmd_vel=({cmds[1].linear.x:+.2f},{cmds[1].angular.z:+.2f})"
                )
            else:
                self.get_logger().info(
                    leader_text +
                    "go2_2<=actor0 "
                    f"a=({actions[0][0]:+.3f},{actions[0][1]:+.3f}) "
                    f"cmd=({cmds[0].linear.x:+.2f},{cmds[0].angular.z:+.2f}) "
                    f"slot_err={diagnostics[0]['slot_error']:.2f} yaw_err={diagnostics[0]['yaw_error']:.2f}; "
                    "go2_3<=actor1 "
                    f"a=({actions[1][0]:+.3f},{actions[1][1]:+.3f}) "
                    f"cmd=({cmds[1].linear.x:+.2f},{cmds[1].angular.z:+.2f}) "
                    f"slot_err={diagnostics[1]['slot_error']:.2f} yaw_err={diagnostics[1]['yaw_error']:.2f}; "
                    f"dry_run={self.dry_run}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = MaddpgFollowerSlotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_all_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
