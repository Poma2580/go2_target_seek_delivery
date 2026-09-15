#!/usr/bin/env python3
"""Runtime Gazebo controller for the 25-D leader-relative MADDPG policy.

This is the test/deployment counterpart of gazebo_leader_slot_train_stage1.py:

- no replay buffer,
- no learning update,
- no target/pedestrian odom,
- go1 follows a predefined straight route,
- go2/go3 use the trained 25-D leader-relative follower policy.
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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String
from std_srvs.srv import SetBool


def find_repo_root():
    env_root = os.environ.get("DELIVERY_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for path in (here, *here.parents):
        for candidate in (path, path.parent):
            if (candidate / "-MADDPG").exists() and (candidate / "go2_ws_v2").exists():
                return candidate.resolve()
    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()
DEFAULT_MADDPG_ROOT = REPO_ROOT / "-MADDPG"
DEFAULT_MODEL_PATH = (
    DEFAULT_MADDPG_ROOT
    / "model"
    / "gazebo_leader_stage1_shared_actor_b256_usteps20_alr2e-05_clr5e-05_20260809_152400"
    / "best_model.pt"
)

POS_SCALE = 8.0
VEL_SCALE = 0.60
DIST_PARAM_SCALE = 3.0
DEFAULT_SIDE_DIST = 1.80
DEFAULT_LEADER_FOLLOW_DIST = 2.70


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


def rot90(v):
    return np.array([-v[1], v[0]], dtype=np.float32)


def body_frame(yaw, vec):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]], dtype=np.float32)


def world_vel_from_odom(state, msg, twist_in_body_frame):
    vx = float(msg.twist.twist.linear.x)
    vy = float(msg.twist.twist.linear.y)
    if not twist_in_body_frame:
        return vx, vy
    c, s = math.cos(state.yaw), math.sin(state.yaw)
    return c * vx - s * vy, s * vx + c * vy


def choose_follower_slots(follower_names, follower_positions, slots, automatic=True):
    """Return a latched robot->slot mapping and both assignment costs."""
    follower_names = tuple(follower_names)
    if len(follower_names) != 2 or len(follower_positions) != 2:
        raise ValueError("exactly two followers are required")
    normal_cost = float(
        np.linalg.norm(follower_positions[0] - slots[0])
        + np.linalg.norm(follower_positions[1] - slots[1])
    )
    swapped_cost = float(
        np.linalg.norm(follower_positions[0] - slots[1])
        + np.linalg.norm(follower_positions[1] - slots[0])
    )
    mapping = (
        {follower_names[0]: 1, follower_names[1]: 0}
        if automatic and swapped_cost < normal_cost
        else {follower_names[0]: 0, follower_names[1]: 1}
    )
    return mapping, normal_cost, swapped_cost


class GazeboLeaderSlotController(Node):
    def __init__(self):
        super().__init__("gazebo_leader_slot_controller")

        self.declare_parameter("maddpg_root", str(DEFAULT_MADDPG_ROOT))
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("odom_twist_in_body_frame", True)
        self.declare_parameter("publish_zero_when_not_ready", True)

        # Load the policy at process startup, but do not publish follower
        # commands until the handoff controller explicitly enables it.
        self.declare_parameter("wait_for_enable", True)
        self.declare_parameter("enable_topic", "/dynamic_encircle/maddpg_enable")
        self.declare_parameter("ready_topic", "/gazebo_leader_slot_controller/ready")
        self.declare_parameter("active_topic", "/gazebo_leader_slot_controller/active")
        self.declare_parameter(
            "set_enabled_service",
            "/gazebo_leader_slot_controller/set_enabled",
        )
        self.declare_parameter("command_topic_suffix", "maddpg_cmd_vel")
        self.declare_parameter("role_topic", "/target_role/perception_robot")
        self.declare_parameter("robot_names", ["go2_1", "go2_2", "go2_3"])

        self.declare_parameter("leader_route_speed", 0.25)
        self.declare_parameter("leader_route_yaw", 0.0)
        self.declare_parameter("leader_open_loop", True)
        self.declare_parameter("leader_yaw_k", 0.0)
        self.declare_parameter("leader_max_angular", 0.30)

        self.declare_parameter("side_dist", DEFAULT_SIDE_DIST)
        self.declare_parameter("leader_follow_dist", DEFAULT_LEADER_FOLLOW_DIST)
        self.declare_parameter("auto_assign_slots_on_enable", True)
        self.declare_parameter("follower_max_linear", 0.45)
        self.declare_parameter("follower_max_angular", 0.45)
        self.declare_parameter("follower_accel_lin", 0.30)
        self.declare_parameter("follower_accel_ang", 0.25)
        self.declare_parameter("follower_turn_slowdown", True)
        self.declare_parameter("near_slot_action_filter", False)
        self.declare_parameter("near_slot_filter_dist", 1.00)
        self.declare_parameter("near_slot_ang_action_limit", 0.35)
        self.declare_parameter("near_slot_yaw_large", 0.80)
        self.declare_parameter("near_slot_yaw_ang_action_limit", 0.55)
        self.declare_parameter("near_slot_action_slew", 0.35)
        self.declare_parameter("follower_yaw_guard_enable", True)
        self.declare_parameter("follower_yaw_guard_limit", 0.70)
        self.declare_parameter("safety_enable", False)
        self.declare_parameter("safety_follower_safe_dist", 0.90)
        self.declare_parameter("safety_follower_hard_dist", 0.65)
        self.declare_parameter("safety_leader_safe_dist", 1.10)
        self.declare_parameter("safety_leader_hard_dist", 0.80)
        self.declare_parameter("safety_leader_soft_slowdown", False)
        self.declare_parameter("safety_max_angular_correction", 0.35)
        self.declare_parameter("safety_hard_angular_correction", 0.55)
        self.declare_parameter("safety_min_linear_scale", 0.20)
        self.declare_parameter("safety_hard_linear", 0.08)

        self.declare_parameter("dry_run", False)
        self.declare_parameter("debug_policy", True)
        self.declare_parameter("log_rate", 2.0)

        self.maddpg_root = Path(self.get_parameter("maddpg_root").value).expanduser().resolve()
        self.model_path = Path(self.get_parameter("model_path").value).expanduser().resolve()
        self.rate_hz = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / max(self.rate_hz, 1e-6)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.odom_twist_in_body_frame = bool(self.get_parameter("odom_twist_in_body_frame").value)
        self.publish_zero_when_not_ready = bool(self.get_parameter("publish_zero_when_not_ready").value)
        self.wait_for_enable = bool(self.get_parameter("wait_for_enable").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)
        self.ready_topic = str(self.get_parameter("ready_topic").value)
        self.active_topic = str(self.get_parameter("active_topic").value)
        self.set_enabled_service = str(self.get_parameter("set_enabled_service").value)
        self.command_topic_suffix = str(
            self.get_parameter("command_topic_suffix").value
        ).strip("/")
        self.role_topic = str(self.get_parameter("role_topic").value)
        self.robot_names = tuple(self.get_parameter("robot_names").value)
        if len(self.robot_names) != 3 or len(set(self.robot_names)) != 3:
            raise ValueError("robot_names must contain three unique names")

        self.leader_route_speed = float(self.get_parameter("leader_route_speed").value)
        self.leader_route_yaw = float(self.get_parameter("leader_route_yaw").value)
        self.leader_open_loop = bool(self.get_parameter("leader_open_loop").value)
        self.leader_yaw_k = float(self.get_parameter("leader_yaw_k").value)
        self.leader_max_angular = float(self.get_parameter("leader_max_angular").value)

        self.side_dist = float(self.get_parameter("side_dist").value)
        self.leader_follow_dist = float(self.get_parameter("leader_follow_dist").value)
        self.auto_assign_slots_on_enable = bool(
            self.get_parameter("auto_assign_slots_on_enable").value
        )
        self.follower_max_linear = float(self.get_parameter("follower_max_linear").value)
        self.follower_max_angular = float(self.get_parameter("follower_max_angular").value)
        self.follower_accel_lin = float(self.get_parameter("follower_accel_lin").value)
        self.follower_accel_ang = float(self.get_parameter("follower_accel_ang").value)
        self.follower_turn_slowdown = bool(self.get_parameter("follower_turn_slowdown").value)
        self.near_slot_action_filter = bool(self.get_parameter("near_slot_action_filter").value)
        self.near_slot_filter_dist = float(self.get_parameter("near_slot_filter_dist").value)
        self.near_slot_ang_action_limit = float(self.get_parameter("near_slot_ang_action_limit").value)
        self.near_slot_yaw_large = float(self.get_parameter("near_slot_yaw_large").value)
        self.near_slot_yaw_ang_action_limit = float(self.get_parameter("near_slot_yaw_ang_action_limit").value)
        self.near_slot_action_slew = float(self.get_parameter("near_slot_action_slew").value)
        self.follower_yaw_guard_enable = bool(self.get_parameter("follower_yaw_guard_enable").value)
        self.follower_yaw_guard_limit = float(self.get_parameter("follower_yaw_guard_limit").value)
        self.safety_enable = bool(self.get_parameter("safety_enable").value)
        self.safety_follower_safe_dist = float(self.get_parameter("safety_follower_safe_dist").value)
        self.safety_follower_hard_dist = float(self.get_parameter("safety_follower_hard_dist").value)
        self.safety_leader_safe_dist = float(self.get_parameter("safety_leader_safe_dist").value)
        self.safety_leader_hard_dist = float(self.get_parameter("safety_leader_hard_dist").value)
        self.safety_leader_soft_slowdown = bool(self.get_parameter("safety_leader_soft_slowdown").value)
        self.safety_max_angular_correction = float(self.get_parameter("safety_max_angular_correction").value)
        self.safety_hard_angular_correction = float(self.get_parameter("safety_hard_angular_correction").value)
        self.safety_min_linear_scale = float(self.get_parameter("safety_min_linear_scale").value)
        self.safety_hard_linear = float(self.get_parameter("safety_hard_linear").value)

        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.debug_policy = bool(self.get_parameter("debug_policy").value)
        self.log_period = 1.0 / max(float(self.get_parameter("log_rate").value), 1e-6)
        self.last_log_time = None
        self.safety_total_ticks = 0
        self.safety_intervention_ticks = 0
        self.last_safety_info = {"intervened": False, "rate": 0.0}
        self.yaw_guard_total_ticks = 0
        self.yaw_guard_intervention_ticks = 0
        self.last_yaw_guard_info = {
            "enabled": bool(self.follower_yaw_guard_enable),
            "intervened": False,
            "rate": 0.0,
            "agents": [
                {"intervened": False, "yaw_rel": 0.0, "limit": self.follower_yaw_guard_limit},
                {"intervened": False, "yaw_rel": 0.0, "limit": self.follower_yaw_guard_limit},
            ],
        }

        self.leader_name = None
        self.follower_names = ()
        self.states = {name: EntityState() for name in self.robot_names}
        self.last_slots = None
        # Canonical slot indices are 0=left and 1=right.  This mapping is
        # selected at each enable edge, then remains latched while active.
        self.slot_index_by_follower = {}
        self.follower_prev_cmds = {
            name: [0.0, 0.0] for name in self.robot_names
        }
        self.follower_prev_actions = {
            name: [0.0, 0.0] for name in self.robot_names
        }

        self.model_loaded = False
        self.ready = False
        self.enable_requested = not self.wait_for_enable
        self.active = False

        self.cmd_pubs = {
            name: self.create_publisher(
                Twist, f"/{name}/{self.command_topic_suffix}", 10
            )
            for name in self.robot_names
        }
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot_name=name: self._odom_cb(robot_name, msg),
                10,
            )

        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.ready_pub = self.create_publisher(Bool, self.ready_topic, status_qos)
        self.active_pub = self.create_publisher(Bool, self.active_topic, status_qos)
        self.enable_sub = self.create_subscription(
            Bool,
            self.enable_topic,
            self._enable_topic_cb,
            status_qos,
        )
        self.role_sub = self.create_subscription(
            String, self.role_topic, self._role_cb, status_qos
        )
        self.enable_service = self.create_service(
            SetBool,
            self.set_enabled_service,
            self._set_enabled_cb,
        )

        try:
            self.maddpg = self._load_model()
            self.model_loaded = True
        except Exception as error:
            self.maddpg = None
            self.get_logger().error(f"MADDPG model unavailable: {error}")
        self._publish_status(force=True)
        self.timer = self.create_timer(self.dt, self._timer_cb)
        self.get_logger().info(
            "gazebo_leader_slot_controller started: "
            f"model={self.model_path}, 25-D leader-relative, "
            "actors=left/right runtime roles, "
            "leader_control=external, "
            f"leader_speed={self.leader_route_speed:.2f}, side_dist={self.side_dist:.2f}, "
            f"leader_follow_dist={self.leader_follow_dist:.2f}, safety_enable={self.safety_enable}, "
            f"near_slot_filter={self.near_slot_action_filter}, "
            f"dry_run={self.dry_run}, wait_for_enable={self.wait_for_enable}, "
            f"enable_topic={self.enable_topic}, set_enabled_service={self.set_enabled_service}, "
            f"command_topic_suffix={self.command_topic_suffix}"
        )

    def _load_model(self):
        if not self.maddpg_root.exists():
            raise FileNotFoundError(f"MADDPG root not found: {self.maddpg_root}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"MADDPG model not found: {self.model_path}")

        sys.path.insert(0, str(self.maddpg_root))
        from maddpg import MADDPGSharedActor

        maddpg = MADDPGSharedActor(
            state_sizes=[25, 25],
            action_sizes=[2, 2],
            hidden_sizes=(128, 128),
            action_low=-1.0,
            action_high=1.0,
        )
        maddpg.load(str(self.model_path))
        return maddpg

    def _missing_fresh_odom(self):
        if self.leader_name is None:
            return ["perception role"]
        missing = []
        for name in self.robot_names:
            state = self.states[name]
            if not state.received or self._now_age(state.receive_time) > self.odom_timeout:
                missing.append(name)
        return missing

    def _publish_status(self, force=False):
        ready_now = (
            self.model_loaded
            and self.leader_name is not None
            and not self._missing_fresh_odom()
        )
        if force or ready_now != self.ready:
            self.ready = ready_now
            self.ready_pub.publish(Bool(data=self.ready))
            self.get_logger().info(f"MADDPG ready={self.ready}")
        self.active_pub.publish(Bool(data=self.active))

    def _reset_policy_state(self):
        self.last_slots = None
        for name in self.follower_prev_cmds:
            self.follower_prev_cmds[name] = [0.0, 0.0]
            self.follower_prev_actions[name] = [0.0, 0.0]

    def _role_cb(self, message):
        selected = message.data.strip("/")
        if selected not in self.robot_names:
            self.get_logger().warning(f"Ignoring unknown perception robot: {selected}")
            return
        if self.leader_name is not None:
            if selected != self.leader_name:
                self.get_logger().warning(
                    f"Ignoring conflicting perception role {selected}"
                )
            return
        self.leader_name = selected
        self.follower_names = tuple(
            name for name in self.robot_names if name != selected
        )
        self.slot_index_by_follower = {
            self.follower_names[0]: 0,
            self.follower_names[1]: 1,
        }
        self._publish_status(force=True)
        self.get_logger().info(
            f"MADDPG role locked: leader={selected}, "
            f"followers={','.join(self.follower_names)}"
        )

    def _assign_slots_for_enable(self):
        """Choose the minimum-distance left/right assignment and latch it."""
        leader = self.states[self.leader_name]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        slots = self._compute_slots(leader_pos, float(leader.yaw))
        follower_positions = [
            np.array(
                [self.states[name].x, self.states[name].y], dtype=np.float32
            )
            for name in self.follower_names
        ]
        (
            self.slot_index_by_follower,
            normal_cost,
            swapped_cost,
        ) = choose_follower_slots(
            self.follower_names,
            follower_positions,
            slots,
            automatic=self.auto_assign_slots_on_enable,
        )
        self.follower_names = tuple(
            sorted(
                self.follower_names,
                key=self.slot_index_by_follower.__getitem__,
            )
        )
        assignment = ", ".join(
            f"{name}->{'left' if index == 0 else 'right'}"
            for name, index in self.slot_index_by_follower.items()
        )
        self.get_logger().info(
            "MADDPG slot roles latched: %s; normal_cost=%.3f m, "
            "swapped_cost=%.3f m"
            % (assignment, normal_cost, swapped_cost)
        )

    def _request_enabled(self, enabled, source):
        if not enabled:
            was_active = self.active
            self.enable_requested = False
            self.active = False
            if was_active:
                self._publish_all_zero()
            self._reset_policy_state()
            self._publish_status(force=True)
            self.get_logger().info(f"MADDPG disabled by {source}")
            return True, "MADDPG disabled"

        self._publish_status()
        if not self.ready:
            missing = self._missing_fresh_odom()
            detail = ", ".join(missing) if missing else "model"
            message = f"MADDPG not ready; waiting for: {detail}"
            self.get_logger().warning(f"Enable rejected from {source}: {message}")
            return False, message

        self._reset_policy_state()
        self._assign_slots_for_enable()
        self.enable_requested = True
        self.active = True
        self._publish_status(force=True)
        self.get_logger().info(f"MADDPG enabled by {source}")
        return True, "MADDPG enabled"

    def _enable_topic_cb(self, message):
        self._request_enabled(bool(message.data), self.enable_topic)

    def _set_enabled_cb(self, request, response):
        response.success, response.message = self._request_enabled(
            bool(request.data),
            self.set_enabled_service,
        )
        return response

    def _odom_cb(self, name, msg):
        state = self.states[name]
        state.x = float(msg.pose.pose.position.x)
        state.y = float(msg.pose.pose.position.y)
        state.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        state.vx, state.vy = world_vel_from_odom(state, msg, self.odom_twist_in_body_frame)
        state.wz = float(msg.twist.twist.angular.z)
        state.received = True
        state.receive_time = self.get_clock().now()

    def _now_age(self, receive_time):
        if receive_time is None:
            return float("inf")
        return (self.get_clock().now() - receive_time).nanoseconds * 1e-9

    def _data_ready(self):
        missing = self._missing_fresh_odom()
        if missing:
            if self.publish_zero_when_not_ready:
                self._publish_all_zero()
            self.get_logger().info(
                f"waiting for fresh odom: {', '.join(missing)}",
                throttle_duration_sec=2.0,
            )
            return False
        return True

    def _compute_slots(self, leader_pos, leader_yaw):
        forward = np.array([math.cos(leader_yaw), math.sin(leader_yaw)], dtype=np.float32)
        left = rot90(forward)
        center = leader_pos + self.leader_follow_dist * forward
        slots = np.zeros((2, 2), dtype=np.float32)
        slots[0] = center + self.side_dist * left
        slots[1] = center - self.side_dist * left
        return slots

    def _build_observations(self):
        leader = self.states[self.leader_name]
        followers = [self.states[name] for name in self.follower_names]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        leader_vel = np.array([leader.vx, leader.vy], dtype=np.float32)
        follower_pos = np.array(
            [[state.x, state.y] for state in followers], dtype=np.float32
        )
        follower_vel = np.array(
            [[state.vx, state.vy] for state in followers], dtype=np.float32
        )
        follower_yaw = np.array(
            [state.yaw for state in followers], dtype=np.float32
        )
        follower_wz = np.array(
            [state.wz for state in followers], dtype=np.float32
        )

        slots = self._compute_slots(leader_pos, float(leader.yaw))
        if self.last_slots is None:
            self.last_slots = slots.copy()
        slot_vel = (slots - self.last_slots) / max(self.dt, 1e-6)

        observations = []
        diagnostics = []
        for idx in range(2):
            other_idx = 1 - idx
            robot_name = self.follower_names[idx]
            # follower_names is canonical left/right order after enable.
            slot_idx = idx
            yaw = float(follower_yaw[idx])
            self_vel_body = body_frame(yaw, follower_vel[idx]) / VEL_SCALE
            leader_rel = body_frame(yaw, leader_pos - follower_pos[idx]) / POS_SCALE
            leader_rel_vel = body_frame(yaw, leader_vel - follower_vel[idx]) / VEL_SCALE
            slot_rel = body_frame(yaw, slots[slot_idx] - follower_pos[idx]) / POS_SCALE
            slot_rel_vel = body_frame(yaw, slot_vel[slot_idx] - follower_vel[idx]) / VEL_SCALE
            other_rel = body_frame(yaw, follower_pos[other_idx] - follower_pos[idx]) / POS_SCALE
            role = np.array(
                [1.0, 0.0] if slot_idx == 0 else [0.0, 1.0],
                dtype=np.float32,
            )
            slot_error = float(
                np.linalg.norm(slots[slot_idx] - follower_pos[idx])
            )
            formation_params = np.array(
                [self.side_dist / DIST_PARAM_SCALE, self.leader_follow_dist / DIST_PARAM_SCALE],
                dtype=np.float32,
            )
            real_motion_state = np.array(
                [self_vel_body[0], follower_wz[idx] / max(self.follower_max_angular, 1e-6)],
                dtype=np.float32,
            )
            prev_action = np.array(self.follower_prev_actions[robot_name], dtype=np.float32)
            leader_yaw_rel = wrap_angle(float(leader.yaw) - yaw)
            leader_yaw_rel_obs = np.array([math.sin(leader_yaw_rel), math.cos(leader_yaw_rel)], dtype=np.float32)

            obs = np.concatenate(
                [
                    self_vel_body,
                    np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32),
                    leader_rel,
                    leader_rel_vel,
                    slot_rel,
                    slot_rel_vel,
                    other_rel,
                    role,
                    np.array([slot_error / POS_SCALE], dtype=np.float32),
                    formation_params,
                    real_motion_state,
                    prev_action,
                    leader_yaw_rel_obs,
                ]
            ).astype(np.float32)
            observations.append(obs)
            diagnostics.append(
                {
                    "slot": slots[slot_idx],
                    "slot_index": slot_idx,
                    "slot_side": "left" if slot_idx == 0 else "right",
                    "pos": follower_pos[idx],
                    "error_vec": slots[slot_idx] - follower_pos[idx],
                    "slot_error": slot_error,
                    "yaw_error": abs(wrap_angle(float(leader.yaw) - yaw)),
                    "real_v_body_x": float(self_vel_body[0] * self.follower_max_linear),
                    "real_wz": float(follower_wz[idx]),
                }
            )
        self.last_slots = slots.copy()
        return observations, diagnostics

    def _leader_twist(self):
        cmd = Twist()
        cmd.linear.x = clamp(self.leader_route_speed, 0.0, 0.60)
        if self.leader_open_loop:
            cmd.angular.z = 0.0
        else:
            leader = self.states[self.leader_name]
            yaw_error = wrap_angle(self.leader_route_yaw - leader.yaw)
            cmd.angular.z = clamp(self.leader_yaw_k * yaw_error, -self.leader_max_angular, self.leader_max_angular)
        return cmd

    def _action_to_twist(self, robot_name, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_lin, prev_ang = self.follower_prev_cmds[robot_name]
        linear = prev_lin + float(action[0]) * self.follower_accel_lin * self.dt
        angular = prev_ang + float(action[1]) * self.follower_accel_ang * self.dt
        linear = clamp(linear, 0.0, self.follower_max_linear)
        angular = clamp(angular, -self.follower_max_angular, self.follower_max_angular)

        if self.follower_turn_slowdown:
            turn_ratio = min(abs(angular) / max(self.follower_max_angular, 1e-6), 1.0)
            linear = min(linear, self.follower_max_linear * max(0.25, 1.0 - 0.65 * turn_ratio))

        self.follower_prev_cmds[robot_name] = [linear, angular]
        self.follower_prev_actions[robot_name] = [float(action[0]), float(action[1])]
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        return cmd

    def _filter_near_slot_actions(self, actions, diagnostics):
        filtered_actions = []
        filter_infos = []
        if not self.near_slot_action_filter:
            for action in actions:
                filtered_actions.append(np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0))
                filter_infos.append({"active": False, "gate": 0.0, "ang_limit": 1.0, "slew": 0.0})
            return filtered_actions, filter_infos

        filter_dist = max(self.near_slot_filter_dist, 1e-6)
        base_ang_limit = clamp(self.near_slot_ang_action_limit, 0.0, 1.0)
        yaw_ang_limit = clamp(self.near_slot_yaw_ang_action_limit, base_ang_limit, 1.0)
        max_slew = max(self.near_slot_action_slew, 0.0)

        for idx, action in enumerate(actions):
            robot_name = self.follower_names[idx]
            raw = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
            slot_error = float(diagnostics[idx]["slot_error"])
            yaw_error = float(diagnostics[idx]["yaw_error"])

            # gate=0: far from slot, keep the original action authority.
            # gate=1: near slot, suppress aggressive angular acceleration.
            near_gate = float(np.clip((filter_dist - slot_error) / filter_dist, 0.0, 1.0))
            near_ang_limit = yaw_ang_limit if yaw_error >= self.near_slot_yaw_large else base_ang_limit
            ang_limit = (1.0 - near_gate) * 1.0 + near_gate * near_ang_limit

            filtered = raw.copy()
            filtered[1] = clamp(float(filtered[1]), -ang_limit, ang_limit)

            # Action slew is also a near-slot stabilizer.  Do not apply it far
            # from the slot; otherwise a valid catch-up command raw=(+1,+1)
            # is artificially weakened at episode start.
            if max_slew > 0.0 and near_gate > 1e-3:
                prev = np.asarray(self.follower_prev_actions[robot_name], dtype=np.float32)
                delta = np.clip(filtered - prev, -max_slew, max_slew)
                filtered = prev + delta

            filtered = np.clip(filtered, -1.0, 1.0).astype(np.float32)
            filter_infos.append(
                {
                    "active": near_gate > 1e-3 or max_slew > 0.0,
                    "gate": near_gate,
                    "ang_limit": ang_limit,
                    "slew": max_slew,
                }
            )
            filtered_actions.append(filtered)
        return filtered_actions, filter_infos

    def _safety_pair_adjustment(self, robot_name, pos, yaw, obstacle_pos, dist, safe_dist, hard_dist, fallback_sign):
        if dist >= safe_dist:
            return 1.0, 0.0, False, 0.0

        denom = max(safe_dist - hard_dist, 1e-6)
        severity = float(np.clip((safe_dist - dist) / denom, 0.0, 1.0))
        hard = dist <= hard_dist

        obstacle_rel_body = body_frame(yaw, obstacle_pos - pos)
        away_body = -obstacle_rel_body
        away_angle = math.atan2(float(away_body[1]), float(away_body[0]))
        steer_unit = clamp(away_angle / (0.5 * math.pi), -1.0, 1.0)
        if abs(steer_unit) < 0.15:
            steer_unit = fallback_sign

        max_corr = self.safety_hard_angular_correction if hard else self.safety_max_angular_correction
        angular_corr = severity * max_corr * steer_unit
        linear_scale = max(self.safety_min_linear_scale, 1.0 - 0.80 * severity)
        return linear_scale, angular_corr, True, severity

    def _apply_safety_layer(self, cmds):
        info = {
            "enabled": bool(self.safety_enable),
            "intervened": False,
            "rate": 0.0,
            "inter_dist": float("inf"),
            "leader_dists": [float("inf"), float("inf")],
            "agents": [
                {"intervened": False, "linear_scale": 1.0, "angular_correction": 0.0, "severity": 0.0},
                {"intervened": False, "linear_scale": 1.0, "angular_correction": 0.0, "severity": 0.0},
            ],
        }
        self.safety_total_ticks += 1
        if not self.safety_enable:
            info["rate"] = self.safety_intervention_ticks / max(self.safety_total_ticks, 1)
            return cmds, info

        leader = self.states[self.leader_name]
        followers = [self.states[name] for name in self.follower_names]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        follower_pos = [
            np.array([state.x, state.y], dtype=np.float32)
            for state in followers
        ]
        follower_yaw = [float(state.yaw) for state in followers]
        inter_dist = float(np.linalg.norm(follower_pos[0] - follower_pos[1]))
        leader_dists = [float(np.linalg.norm(p - leader_pos)) for p in follower_pos]
        info["inter_dist"] = inter_dist
        info["leader_dists"] = leader_dists

        adjusted_cmds = []
        for idx, cmd in enumerate(cmds):
            other_idx = 1 - idx
            fallback_sign = 1.0 if idx == 0 else -1.0
            linear_scale = 1.0
            angular_corr = 0.0
            severity = 0.0
            intervened = False

            scale, corr, active, sev = self._safety_pair_adjustment(
                self.follower_names[idx],
                follower_pos[idx],
                follower_yaw[idx],
                follower_pos[other_idx],
                inter_dist,
                self.safety_follower_safe_dist,
                self.safety_follower_hard_dist,
                fallback_sign,
            )
            if active:
                # go2/go3 slots are in front of go1.  If the leader safety
                # layer slows a follower in the whole soft safety band, that
                # follower can get trapped behind go1 and never pass into its
                # slot.  Keep the soft leader avoidance as steering-only, and
                # reserve speed reduction for the hard-danger zone.
                if not self.safety_leader_soft_slowdown and leader_dists[idx] > self.safety_leader_hard_dist:
                    scale = 1.0
                linear_scale = min(linear_scale, scale)
                angular_corr += corr
                severity = max(severity, sev)
                intervened = True

            scale, corr, active, sev = self._safety_pair_adjustment(
                self.follower_names[idx],
                follower_pos[idx],
                follower_yaw[idx],
                leader_pos,
                leader_dists[idx],
                self.safety_leader_safe_dist,
                self.safety_leader_hard_dist,
                fallback_sign,
            )
            if active:
                linear_scale = min(linear_scale, scale)
                angular_corr += corr
                severity = max(severity, sev)
                intervened = True

            new_cmd = Twist()
            new_cmd.linear.x = clamp(cmd.linear.x * linear_scale, 0.0, self.follower_max_linear)
            new_cmd.angular.z = clamp(
                cmd.angular.z + angular_corr,
                -self.follower_max_angular,
                self.follower_max_angular,
            )
            if severity >= 1.0:
                new_cmd.linear.x = min(new_cmd.linear.x, self.safety_hard_linear)

            if intervened:
                self.follower_prev_cmds[self.follower_names[idx]] = [new_cmd.linear.x, new_cmd.angular.z]
                info["intervened"] = True
            info["agents"][idx] = {
                "intervened": intervened,
                "linear_scale": linear_scale,
                "angular_correction": angular_corr,
                "severity": severity,
            }
            adjusted_cmds.append(new_cmd)

        if info["intervened"]:
            self.safety_intervention_ticks += 1
        info["rate"] = self.safety_intervention_ticks / max(self.safety_total_ticks, 1)
        return adjusted_cmds, info

    def _apply_follower_yaw_guard(self, cmds):
        limit = max(float(self.follower_yaw_guard_limit), 0.0)
        info = {
            "enabled": bool(self.follower_yaw_guard_enable),
            "intervened": False,
            "rate": 0.0,
            "agents": [],
        }

        self.yaw_guard_total_ticks += 1
        if not self.follower_yaw_guard_enable or limit <= 1e-6:
            info["rate"] = self.yaw_guard_intervention_ticks / max(self.yaw_guard_total_ticks, 1)
            for robot_name in self.follower_names:
                yaw_rel = wrap_angle(
                    self.states[robot_name].yaw
                    - self.states[self.leader_name].yaw
                )
                info["agents"].append(
                    {
                        "intervened": False,
                        "yaw_rel": float(yaw_rel),
                        "limit": float(limit),
                        "original_w": 0.0,
                        "guarded_w": 0.0,
                    }
                )
            self.last_yaw_guard_info = info
            return cmds, info

        adjusted_cmds = []
        leader_yaw = self.states[self.leader_name].yaw
        for robot_name, cmd in zip(self.follower_names, cmds):
            yaw_rel = wrap_angle(self.states[robot_name].yaw - leader_yaw)
            original_w = float(cmd.angular.z)
            guarded_w = original_w

            # If already outside +/-limit, only allow corrective angular
            # velocity.  If still inside, clip any command that would cross the
            # limit during the next controller tick.
            if yaw_rel > limit:
                guarded_w = min(guarded_w, 0.0)
            elif yaw_rel < -limit:
                guarded_w = max(guarded_w, 0.0)
            else:
                predicted = yaw_rel + guarded_w * self.dt
                if predicted > limit:
                    guarded_w = min(guarded_w, (limit - yaw_rel) / max(self.dt, 1e-6))
                elif predicted < -limit:
                    guarded_w = max(guarded_w, (-limit - yaw_rel) / max(self.dt, 1e-6))

            new_cmd = Twist()
            new_cmd.linear.x = cmd.linear.x
            new_cmd.angular.z = clamp(guarded_w, -self.follower_max_angular, self.follower_max_angular)

            intervened = abs(new_cmd.angular.z - original_w) > 1e-6
            if intervened:
                info["intervened"] = True
                # Keep the command integrator consistent with the final command.
                self.follower_prev_cmds[robot_name][1] = float(new_cmd.angular.z)

            adjusted_cmds.append(new_cmd)
            info["agents"].append(
                {
                    "intervened": bool(intervened),
                    "yaw_rel": float(yaw_rel),
                    "limit": float(limit),
                    "original_w": original_w,
                    "guarded_w": float(new_cmd.angular.z),
                }
            )

        if info["intervened"]:
            self.yaw_guard_intervention_ticks += 1
        info["rate"] = self.yaw_guard_intervention_ticks / max(self.yaw_guard_total_ticks, 1)
        self.last_yaw_guard_info = info
        return adjusted_cmds, info

    def _publish_all_zero(self):
        for name in self.robot_names:
            self.follower_prev_cmds[name] = [0.0, 0.0]
            self.follower_prev_actions[name] = [0.0, 0.0]
        zero = Twist()
        for name in self.follower_names:
            self.cmd_pubs[name].publish(zero)

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
        self._publish_status()
        if not self.active:
            if self.enable_requested and self.ready:
                self._request_enabled(True, "automatic startup")
            else:
                return
        if not self._data_ready():
            # A stale input during control is a latched safety fault.  Require a
            # new explicit enable request after odometry has recovered.
            self.enable_requested = False
            self.active = False
            self._reset_policy_state()
            self._publish_status(force=True)
            self.get_logger().error("MADDPG disabled because odometry became stale")
            return
        observations, diagnostics = self._build_observations()
        actor_actions = [np.asarray(action, dtype=np.float32) for action in self.maddpg.act(observations, add_noise=False)]
        actions = [
            np.asarray(actor_actions[0], dtype=np.float32),
            np.asarray(actor_actions[1], dtype=np.float32),
        ]
        if not all(np.all(np.isfinite(action)) for action in actions):
            self.get_logger().error("MADDPG produced NaN or Inf action. Publishing zero velocity.")
            self._publish_all_zero()
            self.enable_requested = False
            self.active = False
            self._reset_policy_state()
            self._publish_status(force=True)
            return

        raw_actions = [action.copy() for action in actions]
        actions, action_filter_infos = self._filter_near_slot_actions(actions, diagnostics)
        leader_cmd = None
        prev_cmds_snapshot = {name: list(cmd) for name, cmd in self.follower_prev_cmds.items()}
        prev_actions_snapshot = {name: list(action) for name, action in self.follower_prev_actions.items()}
        safety_total_snapshot = self.safety_total_ticks
        safety_intervention_snapshot = self.safety_intervention_ticks
        safety_info_snapshot = dict(self.last_safety_info)
        yaw_guard_total_snapshot = self.yaw_guard_total_ticks
        yaw_guard_intervention_snapshot = self.yaw_guard_intervention_ticks
        yaw_guard_info_snapshot = dict(self.last_yaw_guard_info)
        cmds = [
            self._action_to_twist(self.follower_names[0], actions[0]),
            self._action_to_twist(self.follower_names[1], actions[1]),
        ]
        cmds, safety = self._apply_safety_layer(cmds)
        cmds, yaw_guard = self._apply_follower_yaw_guard(cmds)

        if self.dry_run:
            # Dry-run is a pure policy/diagnostic mode: show the one-step
            # command that would be produced from the current observation, but
            # do not let internal cmd/action integrators drift over repeated
            # log ticks.  Otherwise a stationary dry-run can falsely appear to
            # saturate cmd_vel only because the controller kept integrating.
            self.follower_prev_cmds = prev_cmds_snapshot
            self.follower_prev_actions = prev_actions_snapshot
            self.safety_total_ticks = safety_total_snapshot
            self.safety_intervention_ticks = safety_intervention_snapshot
            self.last_safety_info = safety_info_snapshot
            self.yaw_guard_total_ticks = yaw_guard_total_snapshot
            self.yaw_guard_intervention_ticks = yaw_guard_intervention_snapshot
            self.last_yaw_guard_info = yaw_guard_info_snapshot

        if not self.dry_run:
            for name, command in zip(self.follower_names, cmds):
                self.cmd_pubs[name].publish(command)

        if self._should_log():
            leader = self.states[self.leader_name]
            follower_lines = []
            for index, (name, diagnostic, command) in enumerate(
                zip(self.follower_names, diagnostics, cmds)
            ):
                follower_lines.append(
                    f"{name}/{diagnostic['slot_side']} actor{index}: "
                    f"error={diagnostic['slot_error']:.2f} "
                    f"yaw_error={diagnostic['yaw_error']:.2f} "
                    f"cmd=({command.linear.x:+.2f},{command.angular.z:+.2f})"
                )
            self.get_logger().info(
                f"[leader_slot_policy] leader={self.leader_name} "
                f"pose=({leader.x:+.2f},{leader.y:+.2f},{leader.yaw:+.2f}) "
                f"safety={safety['intervened']} rate={100.0 * safety['rate']:.1f}% "
                f"yaw_guard={yaw_guard['intervened']} "
                f"yg_rate={100.0 * yaw_guard['rate']:.1f}%\n  "
                + "\n  ".join(follower_lines)
            )


def main(args=None):
    rclpy.init(args=args)
    node = GazeboLeaderSlotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.active:
            node._publish_all_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
