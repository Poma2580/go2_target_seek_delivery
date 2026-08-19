"""
Leader-relative two-follower slot tracking environment.

This environment is the target-free Python pretraining version for the Go2
formation task.  The learned followers do not observe the pedestrian target.
Instead, go1/leader defines the formation frame, and follower slots are inferred
from:

    slot = leader_pos + leader_follow_dist * leader_forward +/- side_dist * leader_left

This keeps the follower policy independent from Gazebo actor reset/perception
details.  In deployment, go1 can still track the pedestrian with its own
controller, while go2/go3 only maintain the formation relative to go1.
"""

import math

import numpy as np
from gymnasium import spaces

from envs.follower_slot_tracking_v0 import (
    COLLISION_DIST,
    DIST_PARAM_SCALE,
    DT,
    FOLLOWER_MAX_ANGULAR,
    FOLLOWER_MAX_LINEAR,
    FORMATION_VEL_W,
    FORMATION_YAW_W,
    HOLD_GATE_FULL_ERROR,
    HOLD_GATE_START_ERROR,
    LEADER_FOLLOW_DIST,
    LEADER_MAX_LINEAR,
    MAX_SLOT_SUCCESS_THRESHOLD,
    POS_SCALE,
    SAFE_DIST,
    SAFE_W,
    SIDE_DIST,
    SLOT_SUCCESS_THRESHOLD,
    SLOT_ERROR_W,
    SLOT_PROGRESS_W,
    SMOOTH_W,
    SUCCESS_HOLD_STEPS,
    VEL_SCALE,
    WORLD_BOUND,
    YAW_SUCCESS_THRESHOLD,
    YAW_GATE_FULL_ERROR,
    YAW_GATE_START_ERROR,
    FollowerSlotTrackingEnv,
    _rot90,
    _unit_from_angle,
    _wrap_angle,
)

LARGE_SIDE_DIST = 1.80
LARGE_LEADER_FOLLOW_DIST = 2.70

SLOT_BOUNDARY_MARGIN = 0.80
LEADER_BOUNDARY_MARGIN = 1.30
STAGE_REACH_BONUS = 2.0
STAGE_HOLD_BONUS = 0.08
STAGE_SUCCESS_BONUS = 25.0
WRONG_TURN_W = 1.5
NEAR_YAW_GATE_START_ERROR = 2.0
NEAR_YAW_GATE_FULL_ERROR = 0.5
NEAR_YAW_W = 4.0
NEAR_ANGULAR_ACTION_SAT_W = 3.0
NEAR_ANGULAR_ACTION_SAT_THRESHOLD = 0.50


STAGE_SUCCESS_CONFIGS = {
    # Stage 1: easier gate.  First learn to reach the large Gazebo-like slots
    # and roughly align with go1.
    1: {
        "slot_threshold": 0.75,
        "max_slot_threshold": 1.00,
        "yaw_threshold": 0.65,
        "hold_steps": 35,
    },
    # Stage 2: same motion/initial-state distribution, stricter success gate.
    2: {
        "slot_threshold": 0.55,
        "max_slot_threshold": 0.75,
        "yaw_threshold": 0.40,
        "hold_steps": 60,
    },
    # Stage 3: straight route, harder initial states, stricter gate.
    3: {
        "slot_threshold": 0.55,
        "max_slot_threshold": 0.75,
        "yaw_threshold": 0.40,
        "hold_steps": 60,
    },
    # Stage 4: one slow turn disturbance, easier gate.
    4: {
        "slot_threshold": 0.75,
        "max_slot_threshold": 1.00,
        "yaw_threshold": 0.65,
        "hold_steps": 35,
    },
    # Stage 5: one slow turn disturbance plus formation-size randomization,
    # with the stricter gate.
    5: {
        "slot_threshold": 0.55,
        "max_slot_threshold": 0.75,
        "yaw_threshold": 0.40,
        "hold_steps": 60,
    },
}


STAGE_CONFIGS = {
    # Stage 1: Gazebo-equivalent fixed initial geometry.
    # Python world uses a smaller coordinate range than Gazebo, so this keeps
    # the same relative state instead of the exact Gazebo coordinates:
    #   go2/go3 start in front-left/front-right of go1, all facing +x.
    1: {
        "leader_speed_range": (0.25, 0.25),
        "leader_turn_mode": "straight",
        "side_dist_range": (LARGE_SIDE_DIST, LARGE_SIDE_DIST),
        "leader_follow_dist_range": (LARGE_LEADER_FOLLOW_DIST, LARGE_LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "fixed_gazebo_like_initial": True,
        "fixed_leader_pos": (-4.0, 0.0),
        "fixed_leader_yaw": 0.0,
        "fixed_go2_rel": (1.0, 1.5),
        "fixed_go3_rel": (1.0, -1.5),
        "init_yaw_range": 0.0,
        "obs_noise_std": 0.0,
        "leader_turn_noise_std": 0.0,
        "leader_turn_sine_amp": 0.0,
    },
    # Stage 2: same scene distribution as stage 1; only the success gate is
    # tighter.  This isolates whether the policy can improve steady keeping
    # near the slot without changing the task itself.
    2: {
        "leader_speed_range": (0.25, 0.25),
        "leader_turn_mode": "straight",
        "side_dist_range": (LARGE_SIDE_DIST, LARGE_SIDE_DIST),
        "leader_follow_dist_range": (LARGE_LEADER_FOLLOW_DIST, LARGE_LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "fixed_gazebo_like_initial": True,
        "fixed_leader_pos": (-4.0, 0.0),
        "fixed_leader_yaw": 0.0,
        "fixed_go2_rel": (1.0, 1.5),
        "fixed_go3_rel": (1.0, -1.5),
        "init_yaw_range": 0.0,
        "obs_noise_std": 0.0,
        "leader_turn_noise_std": 0.0,
        "leader_turn_sine_amp": 0.0,
    },
    # Stage 3: go1 straight with stronger follower initial randomization and
    # a tighter slot/yaw gate.
    3: {
        "leader_speed_range": (0.20, 0.28),
        "leader_turn_mode": "straight",
        "side_dist_range": (LARGE_SIDE_DIST, LARGE_SIDE_DIST),
        "leader_follow_dist_range": (LARGE_LEADER_FOLLOW_DIST, LARGE_LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_slot_radius": (1.00, 2.40),
        "min_slot_offset": 0.75,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.0,
        "leader_turn_noise_std": 0.0,
        "leader_turn_sine_amp": 0.0,
    },
    # Stage 4: go1 straight, one slow left/right disturbance in the middle,
    # then straight again.  Success gate is kept easy like stage 1/2.
    4: {
        "leader_speed_range": (0.20, 0.28),
        "leader_turn_mode": ("left_once", "right_once"),
        "side_dist_range": (LARGE_SIDE_DIST, LARGE_SIDE_DIST),
        "leader_follow_dist_range": (LARGE_LEADER_FOLLOW_DIST, LARGE_LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_slot_radius": (1.20, 2.60),
        "min_slot_offset": 0.80,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.002,
        "leader_turn_noise_std": 0.0,
        "leader_turn_sine_amp": 0.0,
        "turn_start_step": 95,
        "turn_duration_steps": 60,
        "turn_rate_range": (0.045, 0.075),
    },
    # Stage 5: same middle slow turn as stage 4, plus formation-size
    # randomization, with the stricter stage-3 success gate.
    5: {
        "leader_speed_range": (0.18, 0.28),
        "leader_turn_mode": ("left_once", "right_once"),
        "side_dist_range": (1.80, 2.40),
        "leader_follow_dist_range": (2.70, 3.60),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_slot_radius": (1.20, 3.20),
        "min_slot_offset": 0.80,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.003,
        "leader_turn_noise_std": 0.0,
        "leader_turn_sine_amp": 0.0,
        "turn_start_step": 95,
        "turn_duration_steps": 60,
        "turn_rate_range": (0.045, 0.075),
    },
}


class LeaderSlotTrackingEnv(FollowerSlotTrackingEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "leader_slot_tracking_v0",
        "is_parallelizable": True,
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_size = 25
        obs_high = np.ones(self.obs_size, dtype=np.float32) * 4.0
        self.observation_spaces = {
            agent: spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.leader_speed = 0.25
        self.leader_turn_mode = "straight"
        self.leader_turn_noise_std = 0.0
        self.leader_turn_sine_amp = 0.0
        self.leader_turn_disturbance = 0.0
        self.turn_start_step = 0
        self.turn_duration_steps = 0
        self.turn_rate = 0.0
        self.success_slot_threshold = SLOT_SUCCESS_THRESHOLD
        self.success_max_slot_threshold = MAX_SLOT_SUCCESS_THRESHOLD
        self.success_yaw_threshold = YAW_SUCCESS_THRESHOLD
        self.success_hold_steps = SUCCESS_HOLD_STEPS

    def reset(self, seed=None, options=None):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.success_hold_count = 0
        self._sample_curriculum(rng)

        if bool(STAGE_CONFIGS[self.training_stage].get("fixed_gazebo_like_initial", False)):
            self.leader_pos, self.leader_yaw = self._fixed_gazebo_like_leader_state()
        else:
            self.leader_pos, self.leader_yaw = self._sample_safe_leader_state(rng)
        self.leader_vel = self.leader_speed * _unit_from_angle(self.leader_yaw)

        # Virtual target is only kept for plotting/backward-compatible info.
        # It is not included in the learned observation.
        self._sync_virtual_target()
        self._update_slots()
        if bool(STAGE_CONFIGS[self.training_stage].get("fixed_gazebo_like_initial", False)):
            self._reset_followers_fixed_gazebo_like()
        else:
            self._reset_followers_near_leader(rng)

        self.follower_pos = np.clip(self.follower_pos, -WORLD_BOUND + 0.5, WORLD_BOUND - 0.5)
        self.follower_vel[:] = 0.0
        self.follower_cmd_linear[:] = 0.0
        self.follower_cmd_angular[:] = 0.0
        self.follower_angular_vel[:] = 0.0
        self.leader_turn_disturbance = 0.0

        yaw_range = float(STAGE_CONFIGS[self.training_stage].get("init_yaw_range", 0.0))
        self.follower_yaw[:] = [
            _wrap_angle(self.leader_yaw + rng.uniform(-yaw_range, yaw_range)),
            _wrap_angle(self.leader_yaw + rng.uniform(-yaw_range, yaw_range)),
        ]
        self.prev_actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_slots = self.slots.copy()
        self.prev_slot_errors = np.linalg.norm(self.follower_pos - self.slots, axis=1).astype(np.float32)

        return self._get_observations(), {agent: {} for agent in self.agents}

    def _fixed_gazebo_like_leader_state(self):
        cfg = STAGE_CONFIGS[self.training_stage]
        leader_pos = np.array(cfg.get("fixed_leader_pos", (-4.0, 0.0)), dtype=np.float32)
        leader_yaw = float(cfg.get("fixed_leader_yaw", 0.0))
        return leader_pos, leader_yaw

    def _reset_followers_fixed_gazebo_like(self):
        cfg = STAGE_CONFIGS[self.training_stage]
        forward = _unit_from_angle(self.leader_yaw)
        left = _rot90(forward)
        rels = [
            cfg.get("fixed_go2_rel", (1.0, 1.5)),
            cfg.get("fixed_go3_rel", (1.0, -1.5)),
        ]
        for idx, rel in enumerate(rels):
            dx, dy = float(rel[0]), float(rel[1])
            self.follower_pos[idx] = self.leader_pos + dx * forward + dy * left

    def _sample_safe_leader_state(self, rng):
        """Sample a leader state whose future slots stay inside the world.

        Large leader-relative formations can put slots outside [-WORLD_BOUND, WORLD_BOUND]
        even when go1 itself is safely inside.  Those episodes teach the policy
        contradictory experiences because follower positions are clipped at the
        boundary.  This sampler rejects such curriculum samples.
        """
        slot_limit = WORLD_BOUND - SLOT_BOUNDARY_MARGIN
        leader_limit = WORLD_BOUND - LEADER_BOUNDARY_MARGIN
        leader_speed = min(float(self.leader_speed), float(self.leader_max_linear))

        for _ in range(100):
            leader_yaw = float(rng.uniform(-math.pi, math.pi))
            pos = np.zeros(2, dtype=np.float32)
            yaw = leader_yaw
            relative_points = []
            leader_points = []

            for step in range(int(self.max_cycles) + 1):
                forward = _unit_from_angle(yaw)
                left = _rot90(forward)
                center = pos + self.leader_follow_dist * forward
                leader_points.append(pos.copy())
                relative_points.extend(
                    [
                        center + self.side_dist * left,
                        center - self.side_dist * left,
                    ]
                )
                yaw = _wrap_angle(yaw + self._leader_turn_at_step(step) * DT)
                pos = pos + leader_speed * _unit_from_angle(yaw) * DT

            slot_points = np.asarray(relative_points, dtype=np.float32)
            leader_points = np.asarray(leader_points, dtype=np.float32)
            slot_low = -slot_limit - slot_points.min(axis=0)
            slot_high = slot_limit - slot_points.max(axis=0)
            leader_low = -leader_limit - leader_points.min(axis=0)
            leader_high = leader_limit - leader_points.max(axis=0)
            low = np.maximum(slot_low, leader_low)
            high = np.minimum(slot_high, leader_high)
            if np.all(low <= high):
                leader_pos = rng.uniform(low=low, high=high).astype(np.float32)
                return leader_pos, leader_yaw

        # Fallback for very large formations/long horizons: use a straight,
        # Gazebo-like route along +x and place go1 near the left side.
        leader_yaw = 0.0
        start_x = -leader_limit + 0.2
        start_y = float(rng.uniform(-slot_limit + self.side_dist + 0.2, slot_limit - self.side_dist - 0.2))
        leader_pos = np.array([start_x, start_y], dtype=np.float32)
        if not self._leader_slots_stay_in_bounds(leader_pos, leader_yaw):
            leader_pos = np.array([-3.5, 0.0], dtype=np.float32)
        return leader_pos, leader_yaw

    def _leader_turn_at_step(self, step):
        mode = self.leader_turn_mode
        if mode == "straight":
            turn = 0.0
        elif mode == "straight_noise":
            turn = 0.0
        elif mode in ("gentle_left_once", "gentle_right_once", "left_once", "right_once"):
            if self.turn_start_step <= step < self.turn_start_step + self.turn_duration_steps:
                sign = 1.0 if "left" in mode else -1.0
                turn = sign * self.turn_rate
            else:
                turn = 0.0
        elif mode == "left_smooth":
            turn = 0.25 * (0.5 + 0.5 * math.sin(0.025 * step + self.turn_phase))
        elif mode == "right_smooth":
            turn = -0.25 * (0.5 + 0.5 * math.sin(0.025 * step + self.turn_phase))
        elif mode == "sine":
            turn = 0.22 * math.sin(0.035 * step * self.turn_rate_scale + self.turn_phase)
        else:
            turn = 0.0
        return float(turn)

    def _leader_slots_stay_in_bounds(self, leader_pos, leader_yaw):
        pos = np.asarray(leader_pos, dtype=np.float32).copy()
        yaw = float(leader_yaw)
        limit = WORLD_BOUND - SLOT_BOUNDARY_MARGIN
        leader_speed = min(float(self.leader_speed), float(self.leader_max_linear))
        for step in range(int(self.max_cycles) + 1):
            forward = _unit_from_angle(yaw)
            left = _rot90(forward)
            center = pos + self.leader_follow_dist * forward
            slots = np.stack(
                [
                    center + self.side_dist * left,
                    center - self.side_dist * left,
                ],
                axis=0,
            )
            all_points = np.vstack([pos, slots])
            if np.any(np.abs(all_points) > limit):
                return False
            yaw = _wrap_angle(yaw + self._leader_turn_at_step(step) * DT)
            pos = pos + leader_speed * _unit_from_angle(yaw) * DT
        return True

    def _reset_followers_near_target(self, rng):
        # Base reset calls this name; keep it aliased for compatibility.
        self._reset_followers_near_leader(rng)

    def _reset_followers_near_leader(self, rng):
        cfg = STAGE_CONFIGS[self.training_stage]
        if self.eval_init_scatter is not None:
            radius_range = (0.4, float(self.eval_init_scatter))
        else:
            radius_range = cfg.get("init_slot_radius", cfg.get("init_leader_radius", (0.70, 1.80)))
        min_slot_offset = float(cfg.get("min_slot_offset", 0.0))
        for idx in range(2):
            for _ in range(100):
                radius = float(rng.uniform(*radius_range))
                angle = float(rng.uniform(-math.pi, math.pi))
                candidate = self.slots[idx] + radius * _unit_from_angle(angle)
                if np.linalg.norm(candidate - self.slots[idx]) >= min_slot_offset:
                    self.follower_pos[idx] = candidate
                    break
            else:
                away = _unit_from_angle(rng.uniform(-math.pi, math.pi))
                norm = float(np.linalg.norm(away))
                if norm < 1e-6:
                    away = _unit_from_angle(rng.uniform(-math.pi, math.pi))
                    norm = 1.0
                self.follower_pos[idx] = self.slots[idx] + max(radius_range[1], min_slot_offset) * away / norm

    def _sample_curriculum(self, rng):
        cfg = STAGE_CONFIGS[self.training_stage]
        self.leader_speed = float(rng.uniform(*cfg["leader_speed_range"]))
        self.target_speed = self.leader_speed
        self.side_dist = float(rng.uniform(*cfg["side_dist_range"]))
        self.leader_follow_dist = float(rng.uniform(*cfg["leader_follow_dist_range"]))
        self.follower_max_linear = float(rng.uniform(*cfg["follower_max_linear_range"]))
        self.follower_max_angular = float(rng.uniform(*cfg["follower_max_angular_range"]))
        self.leader_max_linear = float(rng.uniform(*cfg["leader_max_linear_range"]))
        self.obs_noise_std = float(cfg["obs_noise_std"])
        success_cfg = STAGE_SUCCESS_CONFIGS[self.training_stage]
        self.success_slot_threshold = float(success_cfg["slot_threshold"])
        self.success_max_slot_threshold = float(success_cfg["max_slot_threshold"])
        self.success_yaw_threshold = float(success_cfg["yaw_threshold"])
        self.success_hold_steps = int(success_cfg["hold_steps"])
        self.turn_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        self.turn_rate_scale = float(rng.uniform(0.7, 1.3))
        turn_mode = cfg["leader_turn_mode"]
        if isinstance(turn_mode, (tuple, list)):
            self.leader_turn_mode = str(rng.choice(turn_mode))
        else:
            self.leader_turn_mode = str(turn_mode)
        self.leader_turn_noise_std = float(cfg.get("leader_turn_noise_std", 0.0))
        self.leader_turn_sine_amp = float(cfg.get("leader_turn_sine_amp", 0.0))
        self.turn_start_step = int(cfg.get("turn_start_step", 0))
        self.turn_duration_steps = int(cfg.get("turn_duration_steps", 0))
        turn_rate_range = cfg.get("turn_rate_range", (0.0, 0.0))
        self.turn_rate = float(rng.uniform(*turn_rate_range))

    def _step_target(self):
        # In this target-free environment, this method advances go1 directly.
        turn = self._leader_turn_at_step(self.current_step)
        if self.leader_turn_sine_amp > 0.0:
            turn += self.leader_turn_sine_amp * math.sin(
                0.055 * self.current_step * self.turn_rate_scale + self.turn_phase
            )
        if self.leader_turn_noise_std > 0.0:
            self.leader_turn_disturbance = (
                0.92 * self.leader_turn_disturbance
                + float(self.rng.normal(0.0, self.leader_turn_noise_std))
            )
            turn += self.leader_turn_disturbance

        self.leader_yaw = _wrap_angle(self.leader_yaw + turn * DT)
        leader_speed = min(self.leader_speed, self.leader_max_linear)
        self.leader_vel = leader_speed * _unit_from_angle(self.leader_yaw)
        self.leader_pos = self.leader_pos + self.leader_vel * DT

        for dim in (0, 1):
            if abs(self.leader_pos[dim]) > WORLD_BOUND - 1.0:
                self.leader_yaw = _wrap_angle(math.pi - self.leader_yaw if dim == 0 else -self.leader_yaw)
                self.leader_pos[dim] = np.clip(self.leader_pos[dim], -WORLD_BOUND + 1.0, WORLD_BOUND - 1.0)
                self.leader_vel = leader_speed * _unit_from_angle(self.leader_yaw)

        self._sync_virtual_target()

    def _step_leader(self):
        # go1 has already been advanced by _step_target().
        return None

    def _sync_virtual_target(self):
        forward = _unit_from_angle(self.leader_yaw)
        self.target_pos = self.leader_pos + self.leader_follow_dist * forward
        self.target_heading = self.leader_yaw
        self.target_vel = self.leader_vel.copy()

    def _update_slots(self):
        forward = _unit_from_angle(self.leader_yaw)
        left = _rot90(forward)
        center = self.leader_pos + self.leader_follow_dist * forward
        self.slots[0] = center + self.side_dist * left
        self.slots[1] = center - self.side_dist * left

    def _formation_yaw(self):
        return float(self.leader_yaw)

    def step(self, actions):
        self.current_step += 1
        self.last_slots = self.slots.copy()

        self._step_target()
        self._step_leader()
        self._update_slots()
        self._step_followers(actions)

        slot_errors = np.linalg.norm(self.follower_pos - self.slots, axis=1)
        mean_slot_error = float(np.mean(slot_errors))
        max_slot_error = float(np.max(slot_errors))
        formation_yaw = self._formation_yaw()
        yaw_errors = np.array(
            [abs(_wrap_angle(formation_yaw - float(yaw))) for yaw in self.follower_yaw],
            dtype=np.float32,
        )
        max_yaw_error = float(np.max(yaw_errors))
        inter_follower_dist = float(np.linalg.norm(self.follower_pos[0] - self.follower_pos[1]))
        collision = inter_follower_dist < COLLISION_DIST

        in_success_region = (
            mean_slot_error < self.success_slot_threshold
            and max_slot_error < self.success_max_slot_threshold
            and max_yaw_error < self.success_yaw_threshold
        )
        if in_success_region:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0

        success = self.success_hold_count >= self.success_hold_steps
        truncated = self.current_step >= self.max_cycles

        rewards = {}
        reward_components = {}
        slot_vel = (self.slots - self.last_slots) / DT
        hold_progress = min(self.success_hold_count, self.success_hold_steps) / max(self.success_hold_steps, 1)
        for idx, agent in enumerate(self.agents):
            slot_error = float(slot_errors[idx])
            slot_progress = float(self.prev_slot_errors[idx] - slot_errors[idx])
            action = self.last_actions[idx]
            slot_rel_body = self._body_frame(idx, self.slots[idx] - self.follower_pos[idx])
            slot_rel_body_y = float(slot_rel_body[1])
            yaw_error = float(yaw_errors[idx])
            yaw_gate = float(
                np.clip(
                    (YAW_GATE_START_ERROR - slot_error)
                    / max(YAW_GATE_START_ERROR - YAW_GATE_FULL_ERROR, 1e-6),
                    0.0,
                    1.0,
                )
            )
            hold_gate = float(
                np.clip(
                    (HOLD_GATE_START_ERROR - slot_error)
                    / max(HOLD_GATE_START_ERROR - HOLD_GATE_FULL_ERROR, 1e-6),
                    0.0,
                    1.0,
                )
            )
            near_yaw_gate = float(
                np.clip(
                    (NEAR_YAW_GATE_START_ERROR - slot_error)
                    / max(NEAR_YAW_GATE_START_ERROR - NEAR_YAW_GATE_FULL_ERROR, 1e-6),
                    0.0,
                    1.0,
                )
            )
            vel_match_error = float(
                np.linalg.norm(slot_vel[idx] - self.follower_vel[idx])
                / max(self.follower_max_linear, 1e-6)
            )
            safe_penalty = float(max(0.0, SAFE_DIST - inter_follower_dist))
            action_delta_penalty = float(np.sum((action - self.prev_actions[idx]) ** 2))
            agent_in_success_region = (
                slot_error < self.success_slot_threshold
                and yaw_error < self.success_yaw_threshold
            )

            reward_slot = -SLOT_ERROR_W * slot_error
            reward_progress = SLOT_PROGRESS_W * slot_progress
            reward_formation = -hold_gate * (
                FORMATION_VEL_W * vel_match_error
                + FORMATION_YAW_W * yaw_error
            )
            reward_near_yaw = -NEAR_YAW_W * near_yaw_gate * (1.0 - math.cos(yaw_error))
            reward_safe = -SAFE_W * safe_penalty
            reward_smooth = -SMOOTH_W * action_delta_penalty
            wrong_turn = float(max(0.0, -slot_rel_body_y * float(action[1])))
            angular_action_sat = float(
                max(0.0, abs(float(action[1])) - NEAR_ANGULAR_ACTION_SAT_THRESHOLD) ** 2
            )
            reward_wrong_turn = -WRONG_TURN_W * wrong_turn
            reward_ang_sat = -NEAR_ANGULAR_ACTION_SAT_W * near_yaw_gate * angular_action_sat
            reward_reach = STAGE_REACH_BONUS if agent_in_success_region else 0.0
            reward_hold = STAGE_HOLD_BONUS * self.success_hold_count if in_success_region else 0.0
            reward_success = STAGE_SUCCESS_BONUS if success else 0.0
            reward = (
                reward_slot
                + reward_progress
                + reward_formation
                + reward_near_yaw
                + reward_safe
                + reward_smooth
                + reward_wrong_turn
                + reward_ang_sat
                + reward_reach
                + reward_hold
                + reward_success
            )

            rewards[agent] = float(reward)
            reward_components[agent] = {
                "reward_slot": reward_slot,
                "reward_progress": reward_progress,
                "reward_formation": reward_formation,
                "reward_near_yaw": reward_near_yaw,
                "reward_safe": reward_safe,
                "reward_smooth": reward_smooth,
                "reward_wrong_turn": reward_wrong_turn,
                "reward_ang_sat": reward_ang_sat,
                "reward_reach": reward_reach,
                "reward_hold": reward_hold,
                "reward_success": reward_success,
                "yaw_gate": yaw_gate,
                "hold_gate": hold_gate,
                "near_yaw_gate": near_yaw_gate,
                "vel_match_error": vel_match_error,
                "safe_penalty": safe_penalty,
                "action_delta_penalty": action_delta_penalty,
                "wrong_turn": wrong_turn,
                "angular_action_sat": angular_action_sat,
                "slot_rel_body_y": slot_rel_body_y,
                "in_success_region": float(in_success_region),
                "agent_in_success_region": float(agent_in_success_region),
                "hold_progress": float(hold_progress),
            }

        self.prev_actions = self.last_actions.copy()
        self.prev_slot_errors = slot_errors.astype(np.float32)

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}
        infos = {
            agent: self._info(slot_errors, collision, success, reward_components[agent])
            for agent in self.agents
        }
        return self._get_observations(), rewards, terminations, truncations, infos

    def _get_observations(self):
        observations = {}
        slot_vel = (self.slots - self.last_slots) / DT

        for idx, agent in enumerate(self.agents):
            other_idx = 1 - idx
            vel_scale = max(self.follower_max_linear, 1e-6)
            yaw = float(self.follower_yaw[idx])
            self_vel_body = self._body_frame(idx, self.follower_vel[idx]) / vel_scale
            leader_rel = self._body_frame(idx, self.leader_pos - self.follower_pos[idx]) / POS_SCALE
            leader_rel_vel = self._body_frame(idx, self.leader_vel - self.follower_vel[idx]) / vel_scale
            slot_rel = self._body_frame(idx, self.slots[idx] - self.follower_pos[idx]) / POS_SCALE
            slot_rel_vel = self._body_frame(idx, slot_vel[idx] - self.follower_vel[idx]) / vel_scale
            other_rel = self._body_frame(idx, self.follower_pos[other_idx] - self.follower_pos[idx]) / POS_SCALE
            role = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float32)
            slot_error_norm = np.array(
                [np.linalg.norm(self.slots[idx] - self.follower_pos[idx]) / POS_SCALE],
                dtype=np.float32,
            )
            formation_params = np.array(
                [self.side_dist / DIST_PARAM_SCALE, self.leader_follow_dist / DIST_PARAM_SCALE],
                dtype=np.float32,
            )
            leader_yaw_rel = _wrap_angle(float(self.leader_yaw) - yaw)
            leader_yaw_rel_obs = np.array(
                [math.sin(leader_yaw_rel), math.cos(leader_yaw_rel)],
                dtype=np.float32,
            )
            real_motion_state = np.array(
                [
                    self_vel_body[0],
                    self.follower_angular_vel[idx] / max(self.follower_max_angular, 1e-6),
                ],
                dtype=np.float32,
            )
            prev_action = self.prev_actions[idx].astype(np.float32)

            obs = np.concatenate(
                [
                    self_vel_body,  # 0:2
                    np.array([math.sin(yaw), math.cos(yaw)], dtype=np.float32),  # 2:4
                    leader_rel,  # 4:6
                    leader_rel_vel,  # 6:8
                    slot_rel,  # 8:10
                    slot_rel_vel,  # 10:12
                    other_rel,  # 12:14
                    role,  # 14:16
                    slot_error_norm,  # 16
                    formation_params,  # 17:19
                    real_motion_state,  # 19:21
                    prev_action,  # 21:23
                    leader_yaw_rel_obs,  # 23:25
                ]
            ).astype(np.float32)
            if self.obs_noise_std > 0.0:
                obs = obs + self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
            observations[agent] = obs
        return observations

    def _info(self, slot_errors, collision, success, reward_components):
        yaw_errors = [abs(_wrap_angle(float(self.leader_yaw) - float(yaw))) for yaw in self.follower_yaw]
        return {
            "is_success": float(success),
            "slot_error": float(np.mean(slot_errors)),
            "mean_slot_error": float(np.mean(slot_errors)),
            "max_follower_slot_error": float(np.max(slot_errors)),
            "max_follower_yaw_error": float(np.max(yaw_errors)),
            "mean_follower_yaw_error": float(np.mean(yaw_errors)),
            "formation_error": float(np.mean(slot_errors)),
            "follower_shape_error": float(abs(np.linalg.norm(self.follower_pos[0] - self.follower_pos[1]) - 2 * self.side_dist)),
            "follower_detach_norm": 0.0,
            "formation_hold_count": int(self.success_hold_count),
            "success_hold_count": int(self.success_hold_count),
            "success_required_hold_steps": int(self.success_hold_steps),
            "success_slot_threshold": float(self.success_slot_threshold),
            "success_max_slot_threshold": float(self.success_max_slot_threshold),
            "success_yaw_threshold": float(self.success_yaw_threshold),
            "team_target_distance": 0.0,
            "navigation_active": 1.0,
            "leader_target_distance": float(self.leader_follow_dist),
            "leader_target_progress": 0.0,
            "total_agent_collisions": int(collision),
            "total_obstacle_collisions": 0,
            "any_obstacle_collision": 0.0,
            "min_obstacle_dist": 999.0,
            "target_speed": float(self.leader_speed),
            "leader_speed": float(self.leader_speed),
            "leader_turn_mode": self.leader_turn_mode,
            "side_dist": float(self.side_dist),
            "leader_follow_dist": float(self.leader_follow_dist),
            "follower_max_linear": float(self.follower_max_linear),
            "follower_max_angular": float(self.follower_max_angular),
            "leader_max_linear": float(self.leader_max_linear),
            **reward_components,
        }

    def render(self):
        self.current_episode = self.current_episode
        return super().render()
