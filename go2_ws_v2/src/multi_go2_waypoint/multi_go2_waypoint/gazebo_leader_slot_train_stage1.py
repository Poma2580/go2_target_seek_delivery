#!/usr/bin/env python3
"""Gazebo Stage-1 fine-tuning for the 25-D leader-relative MADDPG policy.

This node does not use a pedestrian/target state.  go1 follows a predefined
straight route, and go2/go3 learn to hold the slots inferred from go1 pose:

    slot = go1_pos + leader_follow_dist * go1_forward +/- side_dist * go1_left

It is intentionally separated from gazebo_maddpg_train_stage1.py, which keeps
the older 28-D target-dependent training loop.
"""

import math
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from geometry_msgs.msg import Twist

from .gazebo_maddpg_train_stage1 import (
    DIST_PARAM_SCALE,
    POS_SCALE,
    VEL_SCALE,
    GazeboMaddpgStage1Trainer,
    body_frame,
    clamp,
    rot90,
    wrap_angle,
)


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
    / "leader_slot_tracking_v0"
    / "MADDPG"
    / "stage5_b512_usteps20_g0.99_t0.005_alr0.0003_clr0.0005_n0.25_minn0.03_h128,128_20260803_104657"
    / "best_model.pt"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_MADDPG_ROOT / "runs" / "leader_slot_tracking_v0" / "GazeboMADDPG"

# Reward/success settings used only by this target-free leader-slot Gazebo trainer.
# Baseline reward used for the stable Gazebo Stage1 fine-tune branch:
# keep the reward compact and close to the original Python/Gazebo slot-tracking
# objective.  Success is still tracked for model selection/logging, but no
# large sparse success bonus is injected into the reward.
SLOT_SUCCESS_THRESHOLD = 1.00
MAX_SLOT_SUCCESS_THRESHOLD = 1.30
YAW_SUCCESS_THRESHOLD = 0.90
SUCCESS_HOLD_STEPS = 30

SLOT_ERROR_W = 7.0
SLOT_PROGRESS_W = 8.0
FORMATION_VEL_W = 2.5
FORMATION_YAW_W = 1.5
SMOOTH_W = 0.40
REGION_BONUS = 0.75
SUCCESS_BONUS = 8.0

SAFE_DIST = 0.70
SAFE_W = 4.0

HOLD_GATE_START_ERROR = 0.60
HOLD_GATE_FULL_ERROR = 0.25


class GazeboLeaderSlotStage1Trainer(GazeboMaddpgStage1Trainer):
    def __init__(self):
        # Patch the base module defaults before its __init__ declares parameters.
        import multi_go2_waypoint.gazebo_maddpg_train_stage1 as base

        base.DEFAULT_PRETRAINED_MODEL = DEFAULT_PRETRAINED_MODEL
        base.DEFAULT_MADDPG_ROOT = DEFAULT_MADDPG_ROOT
        base.SUCCESS_HOLD_STEPS = SUCCESS_HOLD_STEPS
        super().__init__()
        self._declare_safety_params()
        self.safety_enable = bool(self.get_parameter("safety_enable").value)
        self.safety_follower_safe_dist = float(self.get_parameter("safety_follower_safe_dist").value)
        self.safety_follower_hard_dist = float(self.get_parameter("safety_follower_hard_dist").value)
        self.safety_leader_safe_dist = float(self.get_parameter("safety_leader_safe_dist").value)
        self.safety_leader_hard_dist = float(self.get_parameter("safety_leader_hard_dist").value)
        self.safety_max_angular_correction = float(self.get_parameter("safety_max_angular_correction").value)
        self.safety_hard_angular_correction = float(self.get_parameter("safety_hard_angular_correction").value)
        self.safety_min_linear_scale = float(self.get_parameter("safety_min_linear_scale").value)
        self.safety_hard_linear = float(self.get_parameter("safety_hard_linear").value)
        self.safety_total_ticks = 0
        self.safety_intervention_ticks = 0
        self.last_safety_info = {"intervened": False, "rate": 0.0}
        self._declare_curriculum_params()
        self.curriculum_stage = int(self.get_parameter("curriculum_stage").value)
        self.success_mean_slot_threshold = float(self.get_parameter("success_mean_slot_threshold").value)
        self.success_max_slot_threshold = float(self.get_parameter("success_max_slot_threshold").value)
        self.success_yaw_threshold = float(self.get_parameter("success_yaw_threshold").value)
        self.success_required_hold_steps = int(self.get_parameter("success_hold_steps").value)
        self.early_stop_enable = bool(self.get_parameter("early_stop_enable").value)
        self.early_stop_min_steps = int(self.get_parameter("early_stop_min_steps").value)
        self.early_stop_success_episodes = int(self.get_parameter("early_stop_success_episodes").value)
        self.consecutive_success_episodes = 0
        self.get_logger().info(
            "gazebo_leader_slot_train_stage1 started: target-free 25-D leader-relative training. "
            f"curriculum_stage={self.curriculum_stage}, "
            f"go1_route=straight, leader_speed={self.leader_route_speed:.2f} m/s, "
            f"success=(mean<{self.success_mean_slot_threshold:.2f}, "
            f"max<{self.success_max_slot_threshold:.2f}, "
            f"yaw<{self.success_yaw_threshold:.2f}, "
            f"hold={self.success_required_hold_steps}), "
            f"safety_enable={self.safety_enable}"
        )

    def _declare_safety_params(self):
        for name, value in (
            ("safety_enable", True),
            ("safety_follower_safe_dist", 0.90),
            ("safety_follower_hard_dist", 0.65),
            ("safety_leader_safe_dist", 1.10),
            ("safety_leader_hard_dist", 0.80),
            ("safety_max_angular_correction", 0.35),
            ("safety_hard_angular_correction", 0.55),
            ("safety_min_linear_scale", 0.20),
            ("safety_hard_linear", 0.08),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _declare_curriculum_params(self):
        for name, value in (
            ("curriculum_stage", 1),
            ("success_mean_slot_threshold", SLOT_SUCCESS_THRESHOLD),
            ("success_max_slot_threshold", MAX_SLOT_SUCCESS_THRESHOLD),
            ("success_yaw_threshold", YAW_SUCCESS_THRESHOLD),
            ("success_hold_steps", SUCCESS_HOLD_STEPS),
            ("early_stop_enable", False),
            ("early_stop_min_steps", 3000),
            ("early_stop_success_episodes", 2),
        ):
            if not self.has_parameter(name):
                self.declare_parameter(name, value)

    def _load_learning_stack(self):
        sys.path.insert(0, str(self.maddpg_root))
        import torch
        from maddpg import MADDPG, MADDPGSharedActor, ReplayBuffer

        if not self.has_parameter("shared_actor"):
            self.declare_parameter("shared_actor", True)
        self.shared_actor = bool(self.get_parameter("shared_actor").value)
        if not self.has_parameter("curriculum_stage"):
            self.declare_parameter("curriculum_stage", 1)
        self.curriculum_stage = int(self.get_parameter("curriculum_stage").value)

        self.torch = torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
        print("=" * 72, flush=True)
        print("Gazebo Leader-slot Stage1 fine-tuning (25-D, no target)", flush=True)
        print(f"shared actor: {self.shared_actor}", flush=True)
        print(f"torch version: {torch.__version__}", flush=True)
        print(f"cuda available: {torch.cuda.is_available()}", flush=True)
        print(f"device: {device}", flush=True)
        print(f"gpu: {gpu_name}", flush=True)
        print("=" * 72, flush=True)

        hidden_sizes = tuple(int(v) for v in str(self.get_parameter("hidden_sizes").value).split(","))
        maddpg_cls = MADDPGSharedActor if self.shared_actor else MADDPG
        self.maddpg = maddpg_cls(
            state_sizes=[25, 25],
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
            print(f"Loaded pretrained leader-slot model: {self.pretrained_model_path}", flush=True)
        else:
            print(f"WARNING: pretrained model not found: {self.pretrained_model_path}", flush=True)
        if self.shared_actor:
            print("Using one shared actor network for go2_2 and go2_3 during Gazebo fine-tuning.", flush=True)

        self.buffer = ReplayBuffer(
            buffer_size=int(self.get_parameter("buffer_size").value),
            batch_size=self.batch_size,
            agents=self.agents,
            state_sizes=[25, 25],
            action_sizes=[2, 2],
        )
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = DEFAULT_OUTPUT_ROOT
        shared_suffix = "_shared_actor" if self.shared_actor else ""
        self.output_dir = self.output_root / (
            f"gazebo_leader_stage{self.curriculum_stage}{shared_suffix}"
            f"_b{self.batch_size}_usteps{self.update_every}"
            f"_alr{self.get_parameter('actor_lr').value}"
            f"_clr{self.get_parameter('critic_lr').value}_{stamp}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_dir / "model.pt"
        self.best_model_path = self.output_dir / "best_model.pt"
        print(f"Gazebo leader-slot fine-tune output: {self.output_dir}", flush=True)

    def _declare_leader_route_params(self):
        if not self.has_parameter("leader_route_speed"):
            self.declare_parameter("leader_route_speed", 0.25)
        if not self.has_parameter("leader_route_yaw"):
            self.declare_parameter("leader_route_yaw", 0.0)

    @property
    def leader_route_speed(self):
        if not self.has_parameter("leader_route_speed"):
            return 0.25
        return float(self.get_parameter("leader_route_speed").value)

    @property
    def leader_route_yaw(self):
        if not self.has_parameter("leader_route_yaw"):
            return 0.0
        return float(self.get_parameter("leader_route_yaw").value)

    def _data_ready(self):
        missing = []
        for name in self.robot_names:
            state = self.states[name]
            if not state.received or self._now_age(state.receive_time) > self.odom_timeout:
                age = self._now_age(state.receive_time)
                missing.append(f"{name}(received={state.received}, age={age:.2f})")
        return len(missing) == 0, missing

    def _publish_internal_training_target(self, advance=False):
        # No target state in leader-relative training.
        return None

    def _reset_internal_training_target(self):
        return None

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
        self.resetter.reset_fixed_episode()
        if not self.wait_for_odom(timeout_sec=10.0):
            raise RuntimeError("odom not ready after reset")
        if not self.wait_for_stable_robots():
            raise RuntimeError("go2 robots not stable after reset")
        observations, diagnostics = self._build_observations()
        self.prev_slot_errors = np.array([d["slot_error"] for d in diagnostics], dtype=np.float32)
        self.last_slots = diagnostics[0]["slots"].copy()
        return observations, diagnostics

    def _compute_slots(self, leader_pos, leader_yaw):
        forward = np.array([math.cos(leader_yaw), math.sin(leader_yaw)], dtype=np.float32)
        left = rot90(forward)
        center = leader_pos + self.leader_follow_dist * forward
        slots = np.zeros((2, 2), dtype=np.float32)
        slots[0] = center + self.side_dist * left
        slots[1] = center - self.side_dist * left
        return slots

    def _formation_yaw(self, *unused):
        return float(self.states["go2_1"].yaw)

    def _build_observations(self):
        leader = self.states["go2_1"]
        go2 = self.states["go2_2"]
        go3 = self.states["go2_3"]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        leader_vel = np.array([leader.vx, leader.vy], dtype=np.float32)
        follower_pos = np.array([[go2.x, go2.y], [go3.x, go3.y]], dtype=np.float32)
        follower_vel = np.array([[go2.vx, go2.vy], [go3.vx, go3.vy]], dtype=np.float32)
        follower_yaw = np.array([go2.yaw, go3.yaw], dtype=np.float32)
        follower_wz = np.array([go2.wz, go3.wz], dtype=np.float32)

        slots = self._compute_slots(leader_pos, float(leader.yaw))
        if self.last_slots is None:
            self.last_slots = slots.copy()
        slot_vel = (slots - self.last_slots) / max(self.dt, 1e-6)

        observations = []
        diagnostics = []
        for idx in range(2):
            other_idx = 1 - idx
            yaw = float(follower_yaw[idx])
            self_vel_body = body_frame(yaw, follower_vel[idx]) / VEL_SCALE
            leader_rel = body_frame(yaw, leader_pos - follower_pos[idx]) / POS_SCALE
            leader_rel_vel = body_frame(yaw, leader_vel - follower_vel[idx]) / VEL_SCALE
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
                [self_vel_body[0], follower_wz[idx] / max(self.follower_max_angular, 1e-6)],
                dtype=np.float32,
            )
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
                    self.prev_actions[idx].astype(np.float32),
                    leader_yaw_rel_obs,
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

    def _leader_tracking_cmd(self):
        # Stage1 Gazebo curriculum: go1 drives straight at fixed speed.  The
        # launch script already spawns go1 facing +x; if small yaw error exists,
        # gently correct it toward leader_route_yaw.
        leader = self.states["go2_1"]
        desired_yaw = self.leader_route_yaw
        yaw_error = wrap_angle(desired_yaw - leader.yaw)
        desired_lin = clamp(self.leader_route_speed, 0.0, self.max_linear)
        desired_ang = clamp(0.8 * yaw_error, -0.30, 0.30)
        lin = clamp(desired_lin, self.leader_prev_lin - self.leader_accel_lin * self.dt, self.leader_prev_lin + self.leader_accel_lin * self.dt)
        ang = clamp(desired_ang, self.leader_prev_ang - self.leader_accel_ang * self.dt, self.leader_prev_ang + self.leader_accel_ang * self.dt)
        self.leader_prev_lin = lin
        self.leader_prev_ang = ang
        cmd = Twist()
        cmd.linear.x = lin
        cmd.angular.z = ang
        return cmd

    def _safety_pair_adjustment(self, pos, yaw, obstacle_pos, dist, safe_dist, hard_dist, fallback_sign):
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
            self.last_safety_info = info
            return cmds, info

        leader = self.states["go2_1"]
        go2 = self.states["go2_2"]
        go3 = self.states["go2_3"]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        follower_pos = [
            np.array([go2.x, go2.y], dtype=np.float32),
            np.array([go3.x, go3.y], dtype=np.float32),
        ]
        follower_yaw = [float(go2.yaw), float(go3.yaw)]
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
                follower_pos[idx],
                follower_yaw[idx],
                follower_pos[other_idx],
                inter_dist,
                self.safety_follower_safe_dist,
                self.safety_follower_hard_dist,
                fallback_sign,
            )
            if active:
                linear_scale = min(linear_scale, scale)
                angular_corr += corr
                severity = max(severity, sev)
                intervened = True

            scale, corr, active, sev = self._safety_pair_adjustment(
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
        self.last_safety_info = info
        return adjusted_cmds, info

    def _apply_actions_and_wait(self, actions):
        cmd1 = self._leader_tracking_cmd()
        cmd2 = self._action_to_cmd("go2_2", actions[0])
        cmd3 = self._action_to_cmd("go2_3", actions[1])
        (cmd2, cmd3), _ = self._apply_safety_layer([cmd2, cmd3])
        self.cmd_pubs["go2_1"].publish(cmd1)
        self.cmd_pubs["go2_2"].publish(cmd2)
        self.cmd_pubs["go2_3"].publish(cmd3)

        import time
        import rclpy
        deadline = time.time() + self.dt
        while time.time() < deadline:
            rclpy.spin_once(self, timeout_sec=0.01)

    def _compute_rewards(self, diagnostics):
        leader = self.states["go2_1"]
        go2 = self.states["go2_2"]
        go3 = self.states["go2_3"]
        follower_pos = np.array([[go2.x, go2.y], [go3.x, go3.y]], dtype=np.float32)
        follower_vel = np.array([[go2.vx, go2.vy], [go3.vx, go3.vy]], dtype=np.float32)
        follower_yaw = np.array([go2.yaw, go3.yaw], dtype=np.float32)
        slots = diagnostics[0]["slots"]
        slot_errors = np.array([d["slot_error"] for d in diagnostics], dtype=np.float32)
        mean_slot_error = float(np.mean(slot_errors))
        max_slot_error = float(np.max(slot_errors))
        formation_yaw = float(leader.yaw)
        yaw_errors = np.array([abs(wrap_angle(formation_yaw - float(y))) for y in follower_yaw], dtype=np.float32)
        max_yaw_error = float(np.max(yaw_errors))
        inter_dist = float(np.linalg.norm(follower_pos[0] - follower_pos[1]))
        slot_vel = (slots - self.last_slots) / max(self.dt, 1e-6)

        if (
            mean_slot_error < self.success_mean_slot_threshold
            and max_slot_error < self.success_max_slot_threshold
            and max_yaw_error < self.success_yaw_threshold
        ):
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0
        in_success_region = self.success_hold_count > 0
        success = self.success_hold_count >= self.success_required_hold_steps

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
            reward_formation = -hold_gate * (
                FORMATION_VEL_W * vel_match_error
                + FORMATION_YAW_W * float(yaw_errors[idx])
            )
            reward_safe = -SAFE_W * safe_penalty
            reward_smooth = -SMOOTH_W * smooth_penalty
            agent_in_success_region = (
                slot_error < self.success_max_slot_threshold
                and float(yaw_errors[idx]) < self.success_yaw_threshold
            )
            reward_region = REGION_BONUS if agent_in_success_region else 0.0
            reward_success = SUCCESS_BONUS if success else 0.0
            rewards[idx] = (
                reward_slot
                + reward_progress
                + reward_formation
                + reward_safe
                + reward_smooth
                + reward_region
                + reward_success
            )
            components.append(
                (
                    reward_slot,
                    reward_progress,
                    reward_formation,
                    reward_safe,
                    reward_smooth,
                    reward_region,
                    reward_success,
                )
            )

        info = {
            "mean_slot_error": mean_slot_error,
            "max_slot_error": max_slot_error,
            "mean_yaw_error": float(np.mean(yaw_errors)),
            "max_yaw_error": max_yaw_error,
            "hold": int(self.success_hold_count),
            "success": bool(success),
            "in_success_region": bool(in_success_region),
            "success_required_hold_steps": int(self.success_required_hold_steps),
            "success_mean_slot_threshold": float(self.success_mean_slot_threshold),
            "success_max_slot_threshold": float(self.success_max_slot_threshold),
            "success_yaw_threshold": float(self.success_yaw_threshold),
            "inter_follower_dist": inter_dist,
            "safe_penalty": float(max(0.0, SAFE_DIST - inter_dist)),
            "safety_intervened": bool(self.last_safety_info.get("intervened", False)),
            "safety_rate": float(self.last_safety_info.get("rate", 0.0)),
            "leader_follower_dists": self.last_safety_info.get("leader_dists", [float("inf"), float("inf")]),
            "reward_components": components,
        }
        return rewards, info

    def train(self):
        self._declare_leader_route_params()
        print(
            f"Leader-slot Gazebo Stage1 config: total_timesteps={self.total_timesteps}, "
            f"max_steps={self.max_steps}, batch_size={self.batch_size}, "
            f"warmup_steps={self.warmup_steps}, update_every={self.update_every}, "
            f"leader_speed={self.leader_route_speed:.2f}, leader_yaw={self.leader_route_yaw:.2f}, "
            f"noise={self.noise_scale}->{self.min_noise}",
            flush=True,
        )
        return super().train()


def main(args=None):
    import rclpy

    rclpy.init(args=args)
    node = GazeboLeaderSlotStage1Trainer()
    try:
        node.train()
    except KeyboardInterrupt:
        node.get_logger().info("Gazebo leader-slot training interrupted by user.")
    finally:
        node._publish_zero_cmds()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
