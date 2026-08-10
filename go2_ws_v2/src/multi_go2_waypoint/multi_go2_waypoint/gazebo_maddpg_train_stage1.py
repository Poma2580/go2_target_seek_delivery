#!/usr/bin/env python3
"""Gazebo Stage-1 fine-tuning loop for the 28-D MADDPG follower policy.

This node is intentionally small and explicit:

1. reset Gazebo to a fixed initial state,
2. build the same 28-D observations used in Python pretraining,
3. output acceleration-proportion actions,
4. integrate actions to /cmd_vel with Gazebo-safe limits,
5. compute the same five reward categories,
6. update MADDPG and save model checkpoints.

Do not run this node together with dynamic_encircle or
maddpg_follower_slot_controller because all of them publish /cmd_vel.
"""

import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .gazebo_maddpg_reset import GazeboFixedResetter, yaw_to_quaternion


POS_SCALE = 8.0
VEL_SCALE = 0.60
DIST_PARAM_SCALE = 3.0

SLOT_SUCCESS_THRESHOLD = 0.35
MAX_SLOT_SUCCESS_THRESHOLD = 0.55
YAW_SUCCESS_THRESHOLD = 0.50
SUCCESS_HOLD_STEPS = 50
SAFE_DIST = 0.70

SLOT_ERROR_W = 7.0
SLOT_PROGRESS_W = 8.0
FORMATION_VEL_W = 2.5
FORMATION_YAW_W = 1.5
SAFE_W = 4.0
SMOOTH_W = 0.40
HOLD_GATE_START_ERROR = 0.60
HOLD_GATE_FULL_ERROR = 0.25


def find_repo_root():
    env_root = os.environ.get("DELIVERY_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    here = Path(__file__).resolve()
    for path in (here, *here.parents):
        for candidate in (path, path.parent):
            if (candidate / "三角形MADDPG").exists() and (candidate / "go2_ws_v2").exists():
                return candidate.resolve()
    return Path("/home/wangantong/KD_all/go2_target_seek_delivery").resolve()


REPO_ROOT = find_repo_root()
DEFAULT_MADDPG_ROOT = REPO_ROOT / "三角形MADDPG"
DEFAULT_PRETRAINED_MODEL = (
    DEFAULT_MADDPG_ROOT
    / "runs"
    / "follower_slot_tracking_v0"
    / "MADDPG"
    / "stage5_b512_usteps20_g0.99_t0.005_alr0.0003_clr0.0005_n0.25_minn0.03_h128,128_20260731_111729"
    / "best_model.pt"
)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def body_frame(yaw, vec):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]], dtype=np.float32)


def rot90(v):
    return np.array([-v[1], v[0]], dtype=np.float32)


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_to_rpy(q):
    sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
    cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (q.w * q.y - q.z * q.x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    yaw = quaternion_to_yaw(q)
    return roll, pitch, yaw


def world_vel_from_odom(yaw, msg, twist_in_body_frame):
    vx = float(msg.twist.twist.linear.x)
    vy = float(msg.twist.twist.linear.y)
    if not twist_in_body_frame:
        return vx, vy
    c, s = math.cos(yaw), math.sin(yaw)
    return c * vx - s * vy, s * vx + c * vy


class EntityState:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0
        self.received = False
        self.receive_time = None


class GazeboMaddpgStage1Trainer(Node):
    def __init__(self):
        super().__init__("gazebo_maddpg_train_stage1")

        self.declare_parameter("maddpg_root", str(DEFAULT_MADDPG_ROOT))
        self.declare_parameter("pretrained_model_path", str(DEFAULT_PRETRAINED_MODEL))
        self.declare_parameter("output_root", str(DEFAULT_MADDPG_ROOT / "runs" / "follower_slot_tracking_v0" / "GazeboMADDPG"))
        self.declare_parameter("total_timesteps", 20000)
        self.declare_parameter("max_steps", 250)
        self.declare_parameter("buffer_size", 200000)
        self.declare_parameter("batch_size", 256)
        self.declare_parameter("warmup_steps", 1000)
        self.declare_parameter("update_every", 20)
        self.declare_parameter("updates_per_agent", 1)
        self.declare_parameter("actor_lr", 1e-4)
        self.declare_parameter("critic_lr", 2e-4)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("tau", 0.005)
        self.declare_parameter("hidden_sizes", "128,128")
        self.declare_parameter("noise_scale", 0.12)
        self.declare_parameter("min_noise", 0.02)
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("target_odom_topic", "/walking_target/odom")
        self.declare_parameter("use_internal_training_target", True)
        self.declare_parameter("training_target_odom_topic", "/maddpg_training_target/odom")
        self.declare_parameter("reset_mode", "none")
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("target_timeout", 2.0)
        self.declare_parameter("target_reset_tolerance", 0.5)
        self.declare_parameter("odom_twist_in_body_frame", False)
        self.declare_parameter("wait_stable_timeout", 20.0)
        self.declare_parameter("stable_roll_pitch_limit", 0.45)
        # Many Go2 odom publishers report base_footprint / planar odom with z=0.
        # Keep z-check disabled by default; set stable_min_z > 0 only if the
        # odom topic truly reports body height in the world frame.
        self.declare_parameter("stable_min_z", -1.0)
        self.declare_parameter("stable_max_wz", 1.5)
        self.declare_parameter("side_dist", 1.20)
        self.declare_parameter("leader_follow_dist", 1.80)
        self.declare_parameter("max_linear", 0.60)
        self.declare_parameter("max_angular", 1.00)
        self.declare_parameter("follower_max_linear", 0.60)
        self.declare_parameter("follower_max_angular", 0.80)
        self.declare_parameter("follower_accel_lin", 0.40)
        self.declare_parameter("follower_accel_ang", 0.60)
        self.declare_parameter("leader_k_linear", 0.8)
        self.declare_parameter("leader_k_angular", 0.9)
        self.declare_parameter("leader_distance_deadband", 0.25)
        self.declare_parameter("leader_accel_lin", 0.8)
        self.declare_parameter("leader_accel_ang", 1.2)
        self.declare_parameter("print_interval_steps", 50)
        self.declare_parameter("save_interval_episodes", 5)

        self.maddpg_root = Path(self.get_parameter("maddpg_root").value).expanduser().resolve()
        self.pretrained_model_path = Path(self.get_parameter("pretrained_model_path").value).expanduser().resolve()
        self.output_root = Path(self.get_parameter("output_root").value).expanduser().resolve()
        self.total_timesteps = int(self.get_parameter("total_timesteps").value)
        self.max_steps = int(self.get_parameter("max_steps").value)
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)
        self.update_every = int(self.get_parameter("update_every").value)
        self.updates_per_agent = int(self.get_parameter("updates_per_agent").value)
        self.noise_scale = float(self.get_parameter("noise_scale").value)
        self.min_noise = float(self.get_parameter("min_noise").value)
        self.rate_hz = float(self.get_parameter("control_rate").value)
        self.dt = 1.0 / max(self.rate_hz, 1e-6)
        self.target_odom_topic = str(self.get_parameter("target_odom_topic").value)
        self.use_internal_training_target = bool(self.get_parameter("use_internal_training_target").value)
        self.training_target_odom_topic = str(self.get_parameter("training_target_odom_topic").value)
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.target_reset_tolerance = float(self.get_parameter("target_reset_tolerance").value)
        self.odom_twist_in_body_frame = bool(self.get_parameter("odom_twist_in_body_frame").value)
        self.wait_stable_timeout = float(self.get_parameter("wait_stable_timeout").value)
        self.stable_roll_pitch_limit = float(self.get_parameter("stable_roll_pitch_limit").value)
        self.stable_min_z = float(self.get_parameter("stable_min_z").value)
        self.stable_max_wz = float(self.get_parameter("stable_max_wz").value)
        self.side_dist = float(self.get_parameter("side_dist").value)
        self.leader_follow_dist = float(self.get_parameter("leader_follow_dist").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.follower_max_linear = float(self.get_parameter("follower_max_linear").value)
        self.follower_max_angular = float(self.get_parameter("follower_max_angular").value)
        self.follower_accel_lin = float(self.get_parameter("follower_accel_lin").value)
        self.follower_accel_ang = float(self.get_parameter("follower_accel_ang").value)
        self.leader_k_linear = float(self.get_parameter("leader_k_linear").value)
        self.leader_k_angular = float(self.get_parameter("leader_k_angular").value)
        self.leader_distance_deadband = float(self.get_parameter("leader_distance_deadband").value)
        self.leader_accel_lin = float(self.get_parameter("leader_accel_lin").value)
        self.leader_accel_ang = float(self.get_parameter("leader_accel_ang").value)
        self.print_interval_steps = int(self.get_parameter("print_interval_steps").value)
        self.save_interval_episodes = int(self.get_parameter("save_interval_episodes").value)

        self.agents = ["go2_left", "go3_right"]
        self.robot_names = ("go2_1", "go2_2", "go2_3")
        self.follower_names = ("go2_2", "go2_3")
        self.states = {name: EntityState() for name in self.robot_names}
        self.target = EntityState()

        self.cmd_pubs = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in self.robot_names
        }
        self.target_cmd_pub = self.create_publisher(Twist, "/walking_target/cmd_vel", 10)
        self.training_target_pub = self.create_publisher(Odometry, self.training_target_odom_topic, 10)
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot_name=name: self._odom_cb(robot_name, msg),
                20,
            )
        if not self.use_internal_training_target:
            self.create_subscription(Odometry, self.target_odom_topic, self._target_cb, 20)

        self.resetter = GazeboFixedResetter(self)
        self.last_slots = None
        self.prev_slot_errors = np.zeros(2, dtype=np.float32)
        self.prev_actions = np.zeros((2, 2), dtype=np.float32)
        self.last_actions = np.zeros((2, 2), dtype=np.float32)
        self.follower_prev_cmds = {
            "go2_2": [0.0, 0.0],
            "go2_3": [0.0, 0.0],
        }
        self.leader_prev_lin = 0.0
        self.leader_prev_ang = 0.0
        self.success_hold_count = 0

        self._load_learning_stack()
        if self.use_internal_training_target:
            self.get_logger().info(
                "Using internal resettable training target odom: "
                f"{self.training_target_odom_topic}. Gazebo actor visualization may continue its own script."
            )

    def _load_learning_stack(self):
        sys.path.insert(0, str(self.maddpg_root))
        import torch
        from maddpg import MADDPG, ReplayBuffer

        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        print("=" * 72, flush=True)
        print("Gazebo MADDPG Stage1 fine-tuning", flush=True)
        print(f"torch version: {torch.__version__}", flush=True)
        print(f"cuda available: {torch.cuda.is_available()}", flush=True)
        print(f"device: {device}", flush=True)
        print(f"gpu: {gpu_name}", flush=True)
        print("=" * 72, flush=True)

        hidden_sizes = tuple(int(v) for v in str(self.get_parameter("hidden_sizes").value).split(","))
        self.maddpg = MADDPG(
            state_sizes=[28, 28],
            action_sizes=[2, 2],
            hidden_sizes=hidden_sizes,
            actor_lr=float(self.get_parameter("actor_lr").value),
            critic_lr=float(self.get_parameter("critic_lr").value),
            gamma=float(self.get_parameter("gamma").value),
            tau=float(self.get_parameter("tau").value),
            action_low=-1.0,
            action_high=1.0,
        )
        if self.pretrained_model_path.exists():
            self.maddpg.load(str(self.pretrained_model_path))
            print(f"Loaded pretrained model: {self.pretrained_model_path}", flush=True)
        else:
            print(f"WARNING: pretrained model not found: {self.pretrained_model_path}", flush=True)

        self.buffer = ReplayBuffer(
            buffer_size=int(self.get_parameter("buffer_size").value),
            batch_size=self.batch_size,
            agents=self.agents,
            state_sizes=[28, 28],
            action_sizes=[2, 2],
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_root / (
            f"gazebo_stage1_b{self.batch_size}_usteps{self.update_every}"
            f"_alr{self.get_parameter('actor_lr').value}"
            f"_clr{self.get_parameter('critic_lr').value}_{stamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_dir / "model.pt"
        self.best_model_path = self.output_dir / "best_model.pt"
        print(f"Gazebo fine-tune output: {self.output_dir}", flush=True)

    def _odom_cb(self, name, msg):
        state = self.states[name]
        state.x = float(msg.pose.pose.position.x)
        state.y = float(msg.pose.pose.position.y)
        state.z = float(msg.pose.pose.position.z)
        state.roll, state.pitch, state.yaw = quaternion_to_rpy(msg.pose.pose.orientation)
        state.vx, state.vy = world_vel_from_odom(state.yaw, msg, self.odom_twist_in_body_frame)
        state.wz = float(msg.twist.twist.angular.z)
        state.received = True
        state.receive_time = self.get_clock().now()

    def _target_cb(self, msg):
        if self.use_internal_training_target:
            return
        self.target.x = float(msg.pose.pose.position.x)
        self.target.y = float(msg.pose.pose.position.y)
        self.target.z = float(msg.pose.pose.position.z)
        self.target.roll, self.target.pitch, self.target.yaw = quaternion_to_rpy(msg.pose.pose.orientation)
        self.target.vx = float(msg.twist.twist.linear.x)
        self.target.vy = float(msg.twist.twist.linear.y)
        self.target.wz = float(msg.twist.twist.angular.z)
        self.target.received = True
        self.target.receive_time = self.get_clock().now()

    def _reset_internal_training_target(self):
        poses = self.resetter.fixed_poses()
        pose = poses[self.resetter.target_model]
        speed = float(self.resetter.target_speed)
        self.target.x = float(pose.x)
        self.target.y = float(pose.y)
        self.target.z = float(self.resetter.target_z)
        self.target.roll = 0.0
        self.target.pitch = 0.0
        self.target.yaw = float(pose.yaw)
        self.target.vx = speed * math.cos(self.target.yaw)
        self.target.vy = speed * math.sin(self.target.yaw)
        self.target.wz = 0.0
        self.target.received = True
        self.target.receive_time = self.get_clock().now()
        self._publish_internal_training_target()

    def _publish_internal_training_target(self, advance=False):
        if not self.use_internal_training_target:
            return
        if advance:
            self.target.x += self.target.vx * self.dt
            self.target.y += self.target.vy * self.dt
        self.target.receive_time = self.get_clock().now()

        odom = Odometry()
        odom.header.stamp = self.target.receive_time.to_msg()
        odom.header.frame_id = "world"
        odom.child_frame_id = "maddpg_training_target"
        odom.pose.pose.position.x = float(self.target.x)
        odom.pose.pose.position.y = float(self.target.y)
        odom.pose.pose.position.z = float(self.target.z)
        qx, qy, qz, qw = yaw_to_quaternion(float(self.target.yaw))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = float(self.target.vx)
        odom.twist.twist.linear.y = float(self.target.vy)
        odom.twist.twist.angular.z = float(self.target.wz)
        self.training_target_pub.publish(odom)

    def _now_age(self, receive_time):
        if receive_time is None:
            return float("inf")
        return (self.get_clock().now() - receive_time).nanoseconds * 1e-9

    def _data_ready(self):
        missing = []
        for name in self.robot_names:
            state = self.states[name]
            if not state.received or self._now_age(state.receive_time) > self.odom_timeout:
                age = self._now_age(state.receive_time)
                missing.append(f"{name}(received={state.received}, age={age:.2f})")
        if not self.target.received or self._now_age(self.target.receive_time) > self.target_timeout:
            age = self._now_age(self.target.receive_time)
            missing.append(f"target(received={self.target.received}, age={age:.2f})")
        return len(missing) == 0, missing

    def wait_for_odom(self, timeout_sec=10.0):
        deadline = time.time() + timeout_sec
        last_report = 0.0
        last_missing = []
        while time.time() < deadline:
            self._publish_internal_training_target(advance=False)
            rclpy.spin_once(self, timeout_sec=0.05)
            ready, missing = self._data_ready()
            if ready:
                return True
            last_missing = missing
            if time.time() - last_report > 1.0:
                self.get_logger().info("waiting for fresh odom after reset: " + ", ".join(missing))
                last_report = time.time()
        self.get_logger().error("odom not ready after reset, missing: " + ", ".join(last_missing))
        return False

    def _unstable_reasons(self):
        reasons = []
        for name in self.robot_names:
            state = self.states[name]
            if not state.received:
                reasons.append(f"{name}: no odom")
                continue
            if abs(state.roll) > self.stable_roll_pitch_limit:
                reasons.append(f"{name}: roll={state.roll:+.2f}")
            if abs(state.pitch) > self.stable_roll_pitch_limit:
                reasons.append(f"{name}: pitch={state.pitch:+.2f}")
            if self.stable_min_z >= 0.0 and state.z < self.stable_min_z:
                reasons.append(f"{name}: z={state.z:.2f}")
            if abs(state.wz) > self.stable_max_wz:
                reasons.append(f"{name}: wz={state.wz:+.2f}")
        return reasons

    def wait_for_stable_robots(self):
        deadline = time.time() + self.wait_stable_timeout
        last_report = 0.0
        while time.time() < deadline:
            self._publish_zero_cmds()
            self._publish_internal_training_target(advance=False)
            rclpy.spin_once(self, timeout_sec=0.05)
            ready, _ = self._data_ready()
            reasons = self._unstable_reasons() if ready else ["waiting for odom"]
            if ready and not reasons:
                self.get_logger().info(
                    "Go2 robots are stable enough for training: "
                    + "; ".join(
                        f"{name}(z={self.states[name].z:.2f}, roll={self.states[name].roll:+.2f}, pitch={self.states[name].pitch:+.2f})"
                        for name in self.robot_names
                    )
                )
                return True
            if time.time() - last_report > 1.0:
                self.get_logger().info("waiting for Go2 stable state: " + ", ".join(reasons))
                last_report = time.time()
        self.get_logger().error("Go2 robots not stable; refuse to start training: " + ", ".join(self._unstable_reasons()))
        return False

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
            forward = np.array([math.cos(self.target.yaw), math.sin(self.target.yaw)], dtype=np.float32)
        else:
            forward = direction / norm
        left = rot90(forward)
        slots = np.zeros((2, 2), dtype=np.float32)
        slots[0] = target_pos + self.side_dist * left
        slots[1] = target_pos - self.side_dist * left
        return slots

    def _formation_yaw(self, target_pos, leader_pos):
        direction = target_pos - leader_pos
        if float(np.linalg.norm(direction)) < 1e-6:
            return float(self.target.yaw)
        return math.atan2(float(direction[1]), float(direction[0]))

    def _build_observations(self):
        target_pos, target_vel, leader_pos, leader_vel, follower_pos, follower_vel, follower_yaw, follower_wz = self._arrays()
        slots = self._compute_slots(target_pos, leader_pos)
        if self.last_slots is None:
            self.last_slots = slots.copy()
        slot_vel = (slots - self.last_slots) / max(self.dt, 1e-6)
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
            formation_params = np.array(
                [self.side_dist / DIST_PARAM_SCALE, self.leader_follow_dist / DIST_PARAM_SCALE],
                dtype=np.float32,
            )
            real_motion_state = np.array(
                [
                    self_vel_body[0],
                    follower_wz[idx] / max(self.follower_max_angular, 1e-6),
                ],
                dtype=np.float32,
            )
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
                    np.array([slot_error / POS_SCALE], dtype=np.float32),
                    np.array([leader_target_dist / POS_SCALE], dtype=np.float32),
                    formation_params,
                    real_motion_state,
                    self.prev_actions[idx].astype(np.float32),
                ]
            ).astype(np.float32)
            observations.append(obs)
            diagnostics.append(
                {
                    "slots": slots,
                    "slot": slots[idx],
                    "slot_vel": slot_vel[idx],
                    "pos": follower_pos[idx],
                    "vel": follower_vel[idx],
                    "yaw": yaw,
                    "wz": float(follower_wz[idx]),
                    "slot_error": slot_error,
                    "error_vec": slots[idx] - follower_pos[idx],
                }
            )
        return observations, diagnostics

    def _publish_zero_cmds(self):
        zero = Twist()
        for pub in self.cmd_pubs.values():
            pub.publish(zero)

    def _wait_for_target_reset_pose(self, timeout_sec=3.0):
        if self.use_internal_training_target:
            return True
        expected = self.resetter.fixed_poses()[self.resetter.target_model]
        deadline = time.time() + timeout_sec
        last_error = float("inf")
        while time.time() < deadline:
            if self.target.received:
                dx = self.target.x - expected.x
                dy = self.target.y - expected.y
                last_error = math.hypot(dx, dy)
                if last_error <= self.target_reset_tolerance:
                    self.get_logger().info(
                        "[GazeboResetCheck] target odom reset ok: "
                        f"actual=({self.target.x:+.2f},{self.target.y:+.2f}) "
                        f"expected=({expected.x:+.2f},{expected.y:+.2f}) "
                        f"err={last_error:.3f}m"
                    )
                    return True
            rclpy.spin_once(self, timeout_sec=0.02)
        self.get_logger().error(
            "[GazeboResetCheck] target odom reset mismatch: "
            f"actual=({self.target.x:+.2f},{self.target.y:+.2f}) "
            f"expected=({expected.x:+.2f},{expected.y:+.2f}) "
            f"err={last_error:.3f}m tolerance={self.target_reset_tolerance:.3f}m"
        )
        return False

    def _reset_episode(self):
        self._publish_zero_cmds()
        self.follower_prev_cmds = {"go2_2": [0.0, 0.0], "go2_3": [0.0, 0.0]}
        self.leader_prev_lin = 0.0
        self.leader_prev_ang = 0.0
        self.prev_actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.success_hold_count = 0
        self.last_slots = None
        for state in self.states.values():
            state.received = False
            state.receive_time = None
        self.target.received = False
        self.target.receive_time = None
        self.resetter.reset_fixed_episode()
        if self.use_internal_training_target:
            self._reset_internal_training_target()
        if not self.wait_for_odom(timeout_sec=10.0):
            raise RuntimeError("odom not ready after reset")
        if not self._wait_for_target_reset_pose(timeout_sec=3.0):
            raise RuntimeError("walking_target odom is not at reset pose")
        if not self.wait_for_stable_robots():
            raise RuntimeError("go2 robots not stable after reset")
        obs, diagnostics = self._build_observations()
        self.prev_slot_errors = np.array([d["slot_error"] for d in diagnostics], dtype=np.float32)
        self.last_slots = diagnostics[0]["slots"].copy()
        return obs, diagnostics

    def _leader_tracking_cmd(self):
        leader = self.states["go2_1"]
        dx = self.target.x - leader.x
        dy = self.target.y - leader.y
        dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx) if dist > 1e-6 else leader.yaw
        yaw_error = wrap_angle(bearing - leader.yaw)
        distance_error = dist - self.leader_follow_dist
        if abs(distance_error) < self.leader_distance_deadband:
            distance_error = 0.0

        target_feedforward = self.target.vx * math.cos(bearing) + self.target.vy * math.sin(bearing)
        heading_gate = max(math.cos(yaw_error), 0.25)
        desired_lin = clamp(
            (self.leader_k_linear * distance_error + target_feedforward) * heading_gate,
            -0.5 * self.max_linear,
            self.max_linear,
        )
        desired_ang = clamp(self.leader_k_angular * yaw_error, -self.max_angular, self.max_angular)

        lin = clamp(desired_lin, self.leader_prev_lin - self.leader_accel_lin * self.dt, self.leader_prev_lin + self.leader_accel_lin * self.dt)
        ang = clamp(desired_ang, self.leader_prev_ang - self.leader_accel_ang * self.dt, self.leader_prev_ang + self.leader_accel_ang * self.dt)
        self.leader_prev_lin = lin
        self.leader_prev_ang = ang

        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

    def _action_to_cmd(self, name, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        prev_lin, prev_ang = self.follower_prev_cmds[name]
        lin = prev_lin + float(action[0]) * self.follower_accel_lin * self.dt
        ang = prev_ang + float(action[1]) * self.follower_accel_ang * self.dt
        lin = clamp(lin, 0.0, self.follower_max_linear)
        ang = clamp(ang, -self.follower_max_angular, self.follower_max_angular)

        # Same Gazebo safety convention used by the runtime controller.
        turn_ratio = min(abs(ang) / max(self.follower_max_angular, 1e-6), 1.0)
        lin = min(lin, self.follower_max_linear * max(0.25, 1.0 - 0.65 * turn_ratio))

        self.follower_prev_cmds[name] = [lin, ang]
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

    def _apply_actions_and_wait(self, actions):
        cmd1 = self._leader_tracking_cmd()
        cmd2 = self._action_to_cmd("go2_2", actions[0])
        cmd3 = self._action_to_cmd("go2_3", actions[1])
        self.cmd_pubs["go2_1"].publish(cmd1)
        self.cmd_pubs["go2_2"].publish(cmd2)
        self.cmd_pubs["go2_3"].publish(cmd3)

        # Keep target straight if a target cmd_vel subscriber exists.
        target_cmd = Twist()
        target_cmd.linear.x = float(self.resetter.target_speed)
        self.target_cmd_pub.publish(target_cmd)
        self._publish_internal_training_target(advance=True)

        deadline = time.time() + self.dt
        while time.time() < deadline:
            self._publish_internal_training_target(advance=False)
            rclpy.spin_once(self, timeout_sec=0.01)

    def _compute_rewards(self, diagnostics):
        target_pos, _, leader_pos, _, follower_pos, follower_vel, follower_yaw, _ = self._arrays()
        slots = diagnostics[0]["slots"]
        slot_errors = np.array([d["slot_error"] for d in diagnostics], dtype=np.float32)
        mean_slot_error = float(np.mean(slot_errors))
        max_slot_error = float(np.max(slot_errors))
        formation_yaw = self._formation_yaw(target_pos, leader_pos)
        yaw_errors = np.array([abs(wrap_angle(formation_yaw - float(y))) for y in follower_yaw], dtype=np.float32)
        max_yaw_error = float(np.max(yaw_errors))
        inter_dist = float(np.linalg.norm(follower_pos[0] - follower_pos[1]))
        slot_vel = (slots - self.last_slots) / max(self.dt, 1e-6)

        if (
            mean_slot_error < SLOT_SUCCESS_THRESHOLD
            and max_slot_error < MAX_SLOT_SUCCESS_THRESHOLD
            and max_yaw_error < YAW_SUCCESS_THRESHOLD
        ):
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0

        rewards = np.zeros(2, dtype=np.float32)
        components = []
        for idx in range(2):
            slot_error = float(slot_errors[idx])
            progress = float(self.prev_slot_errors[idx] - slot_error)
            hold_gate = float(np.clip((HOLD_GATE_START_ERROR - slot_error) / max(HOLD_GATE_START_ERROR - HOLD_GATE_FULL_ERROR, 1e-6), 0.0, 1.0))
            vel_match_error = float(np.linalg.norm(slot_vel[idx] - follower_vel[idx]) / max(self.follower_max_linear, 1e-6))
            safe_penalty = float(max(0.0, SAFE_DIST - inter_dist))
            smooth_penalty = float(np.sum((self.last_actions[idx] - self.prev_actions[idx]) ** 2))

            reward_slot = -SLOT_ERROR_W * slot_error
            reward_progress = SLOT_PROGRESS_W * progress
            reward_formation = -hold_gate * (FORMATION_VEL_W * vel_match_error + FORMATION_YAW_W * float(yaw_errors[idx]))
            reward_safe = -SAFE_W * safe_penalty
            reward_smooth = -SMOOTH_W * smooth_penalty
            rewards[idx] = reward_slot + reward_progress + reward_formation + reward_safe + reward_smooth
            components.append((reward_slot, reward_progress, reward_formation, reward_safe, reward_smooth))

        info = {
            "mean_slot_error": mean_slot_error,
            "max_slot_error": max_slot_error,
            "mean_yaw_error": float(np.mean(yaw_errors)),
            "max_yaw_error": max_yaw_error,
            "hold": int(self.success_hold_count),
            "success": bool(self.success_hold_count >= SUCCESS_HOLD_STEPS),
            "inter_follower_dist": inter_dist,
            "reward_components": components,
        }
        return rewards, info

    def train(self):
        print(
            f"Stage1 Gazebo training config: total_timesteps={self.total_timesteps}, "
            f"max_steps={self.max_steps}, batch_size={self.batch_size}, "
            f"warmup_steps={self.warmup_steps}, update_every={self.update_every}, "
            f"noise={self.noise_scale}->{self.min_noise}",
            flush=True,
        )
        tb_writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            tb_writer = SummaryWriter(log_dir=str(self.output_dir))
            print(f"TensorBoard logdir: {self.output_dir}", flush=True)
        except Exception as exc:
            print(f"WARNING: TensorBoard SummaryWriter is unavailable: {exc}", flush=True)

        global_step = 0
        episode = 0
        best_score = -float("inf")
        noise_decay = max(self.noise_scale - self.min_noise, 0.0) / max(self.total_timesteps, 1)

        while global_step < self.total_timesteps and rclpy.ok():
            episode += 1
            observations, diagnostics = self._reset_episode()
            episode_rewards = np.zeros(2, dtype=np.float32)
            episode_info = {}

            for ep_step in range(1, self.max_steps + 1):
                global_step += 1
                states_list = [np.asarray(obs, dtype=np.float32) for obs in observations]
                actions_list = [
                    np.asarray(a, dtype=np.float32)
                    for a in self.maddpg.act(states_list, add_noise=True, noise_scale=self.noise_scale)
                ]
                self.last_actions = np.stack(actions_list).astype(np.float32)
                self._apply_actions_and_wait(actions_list)
                next_observations, next_diagnostics = self._build_observations()
                rewards, episode_info = self._compute_rewards(next_diagnostics)

                done = ep_step >= self.max_steps
                self.buffer.add(
                    states=states_list,
                    actions=actions_list,
                    rewards=rewards,
                    next_states=[np.asarray(obs, dtype=np.float32) for obs in next_observations],
                    dones=np.array([done, done], dtype=np.uint8),
                )
                episode_rewards += rewards

                if len(self.buffer) >= self.batch_size and global_step > self.warmup_steps and global_step % self.update_every == 0:
                    for _ in range(max(self.updates_per_agent, 1)):
                        for agent_idx in range(2):
                            experiences = self.buffer.sample()
                            self.maddpg.learn(experiences, agent_idx)
                        self.maddpg.update_targets()

                self.prev_actions = self.last_actions.copy()
                self.prev_slot_errors = np.array([d["slot_error"] for d in next_diagnostics], dtype=np.float32)
                self.last_slots = next_diagnostics[0]["slots"].copy()
                observations = next_observations

                self.noise_scale = max(self.min_noise, self.noise_scale - noise_decay)

                if global_step % self.print_interval_steps == 0 or ep_step == 1:
                    total_reward = float(np.sum(episode_rewards))
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/running_total_reward", total_reward, global_step)
                        tb_writer.add_scalar("train/mean_slot_error", episode_info.get("mean_slot_error", 0.0), global_step)
                        tb_writer.add_scalar("train/max_slot_error", episode_info.get("max_slot_error", 0.0), global_step)
                        tb_writer.add_scalar("train/mean_yaw_error", episode_info.get("mean_yaw_error", 0.0), global_step)
                        tb_writer.add_scalar("train/inter_follower_dist", episode_info.get("inter_follower_dist", 0.0), global_step)
                        tb_writer.add_scalar("train/success_hold_count", episode_info.get("hold", 0), global_step)
                        tb_writer.add_scalar("train/noise_scale", self.noise_scale, global_step)
                        tb_writer.add_scalar("train/replay_buffer_size", len(self.buffer), global_step)
                        tb_writer.add_scalar("train/safety_rate", episode_info.get("safety_rate", 0.0), global_step)
                        components = episode_info.get("reward_components")
                        if components:
                            comp = np.asarray(components, dtype=np.float32)
                            comp_mean = np.mean(comp, axis=0)
                            if len(comp_mean) == 5:
                                reward_names = ("slot", "progress", "formation", "safe", "smooth")
                            elif len(comp_mean) == 7:
                                reward_names = (
                                    "slot",
                                    "progress",
                                    "formation",
                                    "safe",
                                    "smooth",
                                    "region",
                                    "success",
                                )
                            elif len(comp_mean) == 8:
                                reward_names = (
                                    "slot",
                                    "progress",
                                    "formation",
                                    "near_yaw",
                                    "safe",
                                    "smooth",
                                    "wrong_turn",
                                    "ang_sat",
                                )
                            else:
                                reward_names = (
                                    "slot",
                                    "progress",
                                    "formation",
                                    "near_yaw",
                                    "safe",
                                    "smooth",
                                    "wrong_turn",
                                    "ang_sat",
                                    "team_region",
                                    "success",
                                )
                            for name, value in zip(reward_names, comp_mean):
                                tb_writer.add_scalar(f"train_reward/{name}", float(value), global_step)
                        tb_writer.flush()
                    print(
                        f"[GazeboTrain] step={global_step}/{self.total_timesteps} "
                        f"episode={episode} ep_step={ep_step}/{self.max_steps} "
                        f"reward_sum={total_reward:.2f} "
                        f"mean_slot={episode_info.get('mean_slot_error', 0.0):.3f} "
                        f"max_slot={episode_info.get('max_slot_error', 0.0):.3f} "
                        f"mean_yaw={episode_info.get('mean_yaw_error', 0.0):.3f} "
                        f"inter={episode_info.get('inter_follower_dist', 0.0):.3f} "
                        f"safe={episode_info.get('safety_intervened', False)} "
                        f"safe_rate={100.0 * episode_info.get('safety_rate', 0.0):.1f}% "
                        f"hold={episode_info.get('hold', 0)}/"
                        f"{episode_info.get('success_required_hold_steps', SUCCESS_HOLD_STEPS)} "
                        f"noise={self.noise_scale:.3f} "
                        f"buffer={len(self.buffer)}",
                        flush=True,
                    )

                if global_step >= self.total_timesteps:
                    break

            total_reward = float(np.sum(episode_rewards))
            score = (1000.0 if episode_info.get("success", False) else 0.0) + total_reward
            if tb_writer is not None:
                tb_writer.add_scalar("episode/total_reward", total_reward, global_step)
                tb_writer.add_scalar("episode/score", score, global_step)
                tb_writer.add_scalar("episode/success", 1.0 if episode_info.get("success", False) else 0.0, global_step)
                tb_writer.add_scalar("episode/mean_slot_error", episode_info.get("mean_slot_error", 0.0), global_step)
                tb_writer.add_scalar("episode/max_slot_error", episode_info.get("max_slot_error", 0.0), global_step)
                tb_writer.add_scalar("episode/mean_yaw_error", episode_info.get("mean_yaw_error", 0.0), global_step)
                tb_writer.add_scalar("episode/inter_follower_dist", episode_info.get("inter_follower_dist", 0.0), global_step)
                tb_writer.add_scalar("episode/safety_rate", episode_info.get("safety_rate", 0.0), global_step)
                tb_writer.flush()
            print(
                f"[GazeboEpisode] episode={episode} global_step={global_step} "
                f"total_reward={total_reward:.2f} "
                f"success={episode_info.get('success', False)} "
                f"mean_slot={episode_info.get('mean_slot_error', 0.0):.3f} "
                f"max_slot={episode_info.get('max_slot_error', 0.0):.3f} "
                f"mean_yaw={episode_info.get('mean_yaw_error', 0.0):.3f} "
                f"inter={episode_info.get('inter_follower_dist', 0.0):.3f} "
                f"safe_rate={100.0 * episode_info.get('safety_rate', 0.0):.1f}% "
                f"hold={episode_info.get('hold', 0)}/"
                f"{episode_info.get('success_required_hold_steps', SUCCESS_HOLD_STEPS)}",
                flush=True,
            )

            if score > best_score:
                best_score = score
                self.maddpg.save(str(self.best_model_path))
                print(f"[GazeboTrain] new best model: {self.best_model_path}", flush=True)
            if episode % max(self.save_interval_episodes, 1) == 0:
                self.maddpg.save(str(self.model_path))

            if getattr(self, "early_stop_enable", False):
                if episode_info.get("success", False) and global_step >= getattr(self, "early_stop_min_steps", 3000):
                    self.consecutive_success_episodes = getattr(self, "consecutive_success_episodes", 0) + 1
                else:
                    self.consecutive_success_episodes = 0
                print(
                    f"[GazeboEarlyStop] success={episode_info.get('success', False)} "
                    f"consecutive={self.consecutive_success_episodes}/"
                    f"{getattr(self, 'early_stop_success_episodes', 2)} "
                    f"min_steps={getattr(self, 'early_stop_min_steps', 3000)}",
                    flush=True,
                )
                if self.consecutive_success_episodes >= getattr(self, "early_stop_success_episodes", 2):
                    print(
                        f"[GazeboEarlyStop] stop at step={global_step}; "
                        f"{self.consecutive_success_episodes} consecutive successful episodes.",
                        flush=True,
                    )
                    break

        self._publish_zero_cmds()
        self.maddpg.save(str(self.model_path))
        if tb_writer is not None:
            tb_writer.close()
        print(f"Gazebo Stage1 training finished. final={self.model_path}, best={self.best_model_path}", flush=True)


def main(args=None):
    rclpy.init(args=args)
    node = GazeboMaddpgStage1Trainer()
    try:
        node.train()
    except KeyboardInterrupt:
        node.get_logger().info("Gazebo training interrupted by user.")
    finally:
        node._publish_zero_cmds()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
