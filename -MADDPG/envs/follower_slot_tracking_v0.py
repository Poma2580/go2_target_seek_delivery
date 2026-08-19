"""
Two-follower slot tracking environment for the Go2 dynamic target task.

The pedestrian target moves dynamically. The leader is not learned: it follows
the pedestrian with a simple visual-servo-like controller and stays behind the
target. MADDPG only controls two followers that should hold the left and right
slots around the pedestrian.
"""

import math

import gymnasium
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers


matplotlib.use("Agg")


MAX_CYCLES = 250
DT = 0.10
WORLD_BOUND = 8.0

MAX_LINEAR = 0.60
MAX_ANGULAR = 1.00
FOLLOWER_MAX_LINEAR = 0.60
FOLLOWER_MAX_ANGULAR = 0.80
FOLLOWER_LINEAR_ACCEL = 0.40
FOLLOWER_ANGULAR_ACCEL = 0.60
TARGET_SPEED = 0.25
LEADER_MAX_LINEAR = 0.60
LEADER_MAX_ANGULAR = 1.00

LEADER_FOLLOW_DIST = 1.80
SIDE_DIST = 1.20
SLOT_FORWARD_OFFSET = 0.0

SLOT_SUCCESS_THRESHOLD = 0.35
MAX_SLOT_SUCCESS_THRESHOLD = 0.55
YAW_SUCCESS_THRESHOLD = 0.50
SUCCESS_HOLD_STEPS = 50
COLLISION_DIST = 0.35
SAFE_DIST = 0.70
SLOT_ERROR_W = 7.00
SLOT_PROGRESS_W = 8.00
FORMATION_VEL_W = 2.50
FORMATION_YAW_W = 1.50
SAFE_W = 4.00
SMOOTH_W = 0.40
YAW_GATE_START_ERROR = 0.90
YAW_GATE_FULL_ERROR = 0.35
HOLD_GATE_START_ERROR = 0.60
HOLD_GATE_FULL_ERROR = 0.25
FOLLOWER_TURN_SLOWDOWN = True

POS_SCALE = WORLD_BOUND
VEL_SCALE = FOLLOWER_MAX_LINEAR
DIST_PARAM_SCALE = 3.0

STAGE_CONFIGS = {
    # Stage 1: basic reaching.  The target moves slowly and straight, and the
    # formation size is fixed, so the policy first learns the slot geometry.
    1: {
        "target_speed_range": (0.18, 0.25),
        "turn_mode": "straight",
        "side_dist_range": (SIDE_DIST, SIDE_DIST),
        "leader_follow_dist_range": (LEADER_FOLLOW_DIST, LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_target_radius": (0.45, 1.00),
        "min_slot_offset": 0.55,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.0,
        "leader_lag_steps": 0,
        "target_turn_noise_std": 0.0,
        "target_turn_sine_amp": 0.0,
    },
    # Stage 2: keep the path straight and only increase initial position
    # randomness.  This fixes the previous jump where speed, distance, and
    # target-path disturbance were all increased at once.
    2: {
        "target_speed_range": (0.20, 0.28),
        "turn_mode": "straight",
        "side_dist_range": (SIDE_DIST, SIDE_DIST),
        "leader_follow_dist_range": (LEADER_FOLLOW_DIST, LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_target_radius": (0.55, 1.20),
        "min_slot_offset": 0.60,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.0,
        "leader_lag_steps": 0,
        "target_turn_noise_std": 0.0,
        "target_turn_sine_amp": 0.0,
    },
    # Stage 3: add small smooth pedestrian-path disturbance, while keeping the
    # formation size fixed.  The target is still slow, close to the Gazebo
    # pedestrian speed.
    3: {
        "target_speed_range": (0.20, 0.28),
        "turn_mode": "straight_noise",
        "side_dist_range": (SIDE_DIST, SIDE_DIST),
        "leader_follow_dist_range": (LEADER_FOLLOW_DIST, LEADER_FOLLOW_DIST),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_target_radius": (0.65, 1.40),
        "min_slot_offset": 0.65,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.0,
        "leader_lag_steps": 0,
        "target_turn_noise_std": 0.012,
        "target_turn_sine_amp": 0.025,
    },
    # Stage 4: introduce smooth left turning and mild formation-size
    # randomization.
    4: {
        "target_speed_range": (0.20, 0.30),
        "turn_mode": "left_smooth",
        "side_dist_range": (1.10, 1.30),
        "leader_follow_dist_range": (1.70, 1.90),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_target_radius": (0.70, 1.50),
        "min_slot_offset": 0.70,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.0,
        "leader_lag_steps": 0,
        "target_turn_noise_std": 0.018,
        "target_turn_sine_amp": 0.035,
    },
    # Stage 5: final generalization.  Mix several slow pedestrian trajectory
    # families and randomize the triangle size, but keep the range close to the
    # intended Gazebo formation.
    5: {
        "target_speed_range": (0.18, 0.30),
        "turn_mode": ("straight_noise", "left_smooth", "left90", "sine_random"),
        "side_dist_range": (1.00, 1.40),
        "leader_follow_dist_range": (1.60, 2.10),
        "follower_max_linear_range": (0.60, 0.60),
        "follower_max_angular_range": (0.80, 0.80),
        "leader_max_linear_range": (0.60, 0.60),
        "init_target_radius": (0.75, 1.60),
        "min_slot_offset": 0.70,
        "init_yaw_range": math.pi,
        "obs_noise_std": 0.005,
        "leader_lag_steps": 0,
        "target_turn_noise_std": 0.025,
        "target_turn_sine_amp": 0.045,
    },
}


def env(**kwargs):
    env_instance = FollowerSlotTrackingEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


def _wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _unit_from_angle(theta):
    return np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def _rot90(v):
    return np.array([-v[1], v[0]], dtype=np.float32)


def _yaw_from_vec(v, fallback=0.0):
    norm = float(np.linalg.norm(v))
    if norm < 1e-6:
        return float(fallback)
    return math.atan2(v[1], v[0])


class FollowerSlotTrackingEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "follower_slot_tracking_v0",
        "is_parallelizable": True,
    }

    def __init__(
        self,
        render_mode=None,
        max_cycles=MAX_CYCLES,
        training_stage=1,
        eval_init_scatter=None,
        follower_action_mode="accel",
    ):
        super().__init__()
        if int(training_stage) not in STAGE_CONFIGS:
            raise ValueError("training_stage must be one of {1, 2, 3, 4, 5}")
        if follower_action_mode not in ("accel", "velocity"):
            raise ValueError("follower_action_mode must be 'accel' or 'velocity'")
        self.possible_agents = ["go2_left", "go3_right"]
        self.agents = self.possible_agents.copy()
        self.max_cycles = int(max_cycles)
        self.render_mode = render_mode
        self.training_stage = int(training_stage)
        self.eval_init_scatter = eval_init_scatter
        self.follower_action_mode = follower_action_mode

        # Observation layout:
        # self_vel_body(2), self_yaw_sin/cos(2), leader_rel_body(2),
        # leader_rel_vel_body(2), target_rel_body(2), target_rel_vel_body(2),
        # slot_rel_body(2), slot_rel_vel_body(2), other_follower_rel_body(2),
        # role_left_right(2), slot_error_norm(1), leader_target_dist_norm(1),
        # side_dist_norm(1), leader_follow_dist_norm(1),
        # real_motion_state(2): [real forward body velocity, real yaw rate],
        # previous_action(2).
        #
        # The final four elements are intentionally based on real motion rather
        # than the controller's requested cmd_vel.  In Gazebo, /cmd_vel and odom
        # can differ substantially because of gait dynamics, contact, slip, or
        # roll-over; using real motion here keeps the policy interface aligned
        # between the Python pretraining environment and Gazebo fine-tuning.
        self.obs_size = 28
        obs_high = np.ones(self.obs_size, dtype=np.float32) * 4.0
        self.observation_spaces = {
            agent: spaces.Box(low=-obs_high, high=obs_high, dtype=np.float32)
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        self.current_step = 0
        self.success_hold_count = 0
        self.fig = None
        self.ax = None
        self.current_episode = None

        self.target_pos = np.zeros(2, dtype=np.float32)
        self.target_vel = np.zeros(2, dtype=np.float32)
        self.target_heading = 0.0
        self.leader_pos = np.zeros(2, dtype=np.float32)
        self.leader_vel = np.zeros(2, dtype=np.float32)
        self.leader_yaw = 0.0
        self.follower_pos = np.zeros((2, 2), dtype=np.float32)
        self.follower_vel = np.zeros((2, 2), dtype=np.float32)
        self.follower_yaw = np.zeros(2, dtype=np.float32)
        self.prev_actions = np.zeros((2, 2), dtype=np.float32)
        self.last_actions = np.zeros((2, 2), dtype=np.float32)
        self.prev_slot_errors = np.zeros(2, dtype=np.float32)
        self.last_slots = np.zeros((2, 2), dtype=np.float32)
        self.slots = np.zeros((2, 2), dtype=np.float32)
        self.rng = np.random.RandomState()
        self.target_speed = TARGET_SPEED
        self.side_dist = SIDE_DIST
        self.leader_follow_dist = LEADER_FOLLOW_DIST
        self.follower_max_linear = FOLLOWER_MAX_LINEAR
        self.follower_max_angular = FOLLOWER_MAX_ANGULAR
        self.follower_cmd_linear = np.zeros(2, dtype=np.float32)
        self.follower_cmd_angular = np.zeros(2, dtype=np.float32)
        self.follower_angular_vel = np.zeros(2, dtype=np.float32)
        self.leader_max_linear = LEADER_MAX_LINEAR
        self.leader_lag_steps = 0
        self.obs_noise_std = 0.0
        self.turn_phase = 0.0
        self.turn_rate_scale = 1.0
        self.target_turn_mode = "straight"
        self.target_turn_noise_std = 0.0
        self.target_turn_sine_amp = 0.0
        self.target_turn_disturbance = 0.0
        self.leader_target_history = []

    @property
    def num_agents(self):
        return len(self.possible_agents)

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        rng = np.random.RandomState(seed)
        self.rng = rng
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.success_hold_count = 0
        self._sample_curriculum(rng)

        self.target_pos = rng.uniform(low=[-1.0, -0.5], high=[1.0, 1.0]).astype(np.float32)
        self.target_heading = float(rng.uniform(-math.pi, math.pi))
        self.target_vel = self.target_speed * _unit_from_angle(self.target_heading)

        forward = _unit_from_angle(self.target_heading)
        left = _rot90(forward)
        self.leader_pos = self.target_pos - self.leader_follow_dist * forward
        self.leader_yaw = self.target_heading
        self.leader_vel = self.target_vel.copy()
        self.leader_target_history = [self.target_pos.copy()]

        self._update_slots()
        self._reset_followers_near_target(rng)
        self.follower_pos = np.clip(self.follower_pos, -WORLD_BOUND + 0.5, WORLD_BOUND - 0.5)
        self.follower_vel[:] = 0.0
        self.follower_cmd_linear[:] = 0.0
        self.follower_cmd_angular[:] = 0.0
        self.follower_angular_vel[:] = 0.0
        self.target_turn_disturbance = 0.0
        formation_yaw = _yaw_from_vec(self.target_pos - self.leader_pos, self.target_heading)
        yaw_range = float(STAGE_CONFIGS[self.training_stage].get("init_yaw_range", 0.0))
        self.follower_yaw[:] = [
            _wrap_angle(formation_yaw + rng.uniform(-yaw_range, yaw_range)),
            _wrap_angle(formation_yaw + rng.uniform(-yaw_range, yaw_range)),
        ]
        self.prev_actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_slots = self.slots.copy()
        self.prev_slot_errors = np.linalg.norm(self.follower_pos - self.slots, axis=1).astype(np.float32)

        return self._get_observations(), {agent: {} for agent in self.agents}

    def _reset_followers_near_target(self, rng):
        cfg = STAGE_CONFIGS[self.training_stage]
        if self.eval_init_scatter is not None:
            radius_range = (0.3, float(self.eval_init_scatter))
        else:
            radius_range = cfg.get("init_target_radius", (0.45, 1.20))
        min_slot_offset = float(cfg.get("min_slot_offset", 0.0))
        for idx in range(2):
            for _ in range(100):
                radius = float(rng.uniform(*radius_range))
                angle = float(rng.uniform(-math.pi, math.pi))
                candidate = self.target_pos + radius * _unit_from_angle(angle)
                if np.linalg.norm(candidate - self.slots[idx]) >= min_slot_offset:
                    self.follower_pos[idx] = candidate
                    break
            else:
                away = self.target_pos - self.slots[idx]
                norm = float(np.linalg.norm(away))
                if norm < 1e-6:
                    away = _unit_from_angle(rng.uniform(-math.pi, math.pi))
                    norm = 1.0
                self.follower_pos[idx] = self.target_pos + max(radius_range[1], min_slot_offset) * away / norm

    def _sample_curriculum(self, rng):
        cfg = STAGE_CONFIGS[self.training_stage]
        self.target_speed = float(rng.uniform(*cfg["target_speed_range"]))
        self.side_dist = float(rng.uniform(*cfg["side_dist_range"]))
        self.leader_follow_dist = float(rng.uniform(*cfg["leader_follow_dist_range"]))
        self.follower_max_linear = float(rng.uniform(*cfg["follower_max_linear_range"]))
        self.follower_max_angular = float(rng.uniform(*cfg["follower_max_angular_range"]))
        self.leader_max_linear = float(rng.uniform(*cfg["leader_max_linear_range"]))
        self.leader_lag_steps = int(cfg["leader_lag_steps"])
        self.obs_noise_std = float(cfg["obs_noise_std"])
        self.turn_phase = float(rng.uniform(0.0, 2.0 * math.pi))
        self.turn_rate_scale = float(rng.uniform(0.7, 1.3))
        turn_mode = cfg["turn_mode"]
        if isinstance(turn_mode, (tuple, list)):
            self.target_turn_mode = str(rng.choice(turn_mode))
        else:
            self.target_turn_mode = str(turn_mode)
        self.target_turn_noise_std = float(cfg.get("target_turn_noise_std", 0.0))
        self.target_turn_sine_amp = float(cfg.get("target_turn_sine_amp", 0.0))

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

        if (
            mean_slot_error < SLOT_SUCCESS_THRESHOLD
            and max_slot_error < MAX_SLOT_SUCCESS_THRESHOLD
            and max_yaw_error < YAW_SUCCESS_THRESHOLD
        ):
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0

        success = self.success_hold_count >= SUCCESS_HOLD_STEPS
        truncated = self.current_step >= self.max_cycles

        rewards = {}
        reward_components = {}
        slot_vel = (self.slots - self.last_slots) / DT
        for idx, agent in enumerate(self.agents):
            slot_error = float(slot_errors[idx])
            slot_progress = float(self.prev_slot_errors[idx] - slot_errors[idx])
            action = self.last_actions[idx]
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
            vel_match_error = float(
                np.linalg.norm(slot_vel[idx] - self.follower_vel[idx])
                / max(self.follower_max_linear, 1e-6)
            )
            safe_penalty = float(max(0.0, SAFE_DIST - inter_follower_dist))
            action_delta_penalty = float(np.sum((action - self.prev_actions[idx]) ** 2))

            # Keep the training objective intentionally simple:
            # 1) reach the assigned slot,
            # 2) reward slot-error reduction,
            # 3) after approaching the slot, match slot velocity and face forward,
            # 4) keep followers separated,
            # 5) keep acceleration actions smooth.
            reward_slot = -SLOT_ERROR_W * slot_error
            reward_progress = SLOT_PROGRESS_W * slot_progress
            reward_formation = -hold_gate * (
                FORMATION_VEL_W * vel_match_error
                + FORMATION_YAW_W * yaw_error
            )
            reward_safe = -SAFE_W * safe_penalty
            reward_smooth = -SMOOTH_W * action_delta_penalty
            reward = (
                reward_slot
                + reward_progress
                + reward_formation
                + reward_safe
                + reward_smooth
            )

            rewards[agent] = float(reward)
            reward_components[agent] = {
                "reward_slot": reward_slot,
                "reward_progress": reward_progress,
                "reward_formation": reward_formation,
                "reward_safe": reward_safe,
                "reward_smooth": reward_smooth,
                "yaw_gate": yaw_gate,
                "hold_gate": hold_gate,
                "vel_match_error": vel_match_error,
                "safe_penalty": safe_penalty,
                "action_delta_penalty": action_delta_penalty,
            }

        self.prev_actions = self.last_actions.copy()
        self.prev_slot_errors = slot_errors.astype(np.float32)

        # Do not end the episode immediately after success.  Continuing until
        # max_cycles forces the policy to learn "reach the slot, then keep the
        # moving formation" instead of merely touching the success region.
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: bool(truncated) for agent in self.agents}
        infos = {
            agent: self._info(slot_errors, collision, success, reward_components[agent])
            for agent in self.agents
        }
        return self._get_observations(), rewards, terminations, truncations, infos

    def _step_target(self):
        mode = self.target_turn_mode
        if mode == "straight":
            turn = 0.0
        elif mode == "straight_noise":
            turn = 0.0
        elif mode == "gentle":
            turn = 0.18 * math.sin(0.030 * self.current_step + self.turn_phase)
        elif mode == "left_smooth":
            # Smooth one-direction left turning after a short straight segment.
            if self.current_step < 45:
                turn = 0.0
            else:
                turn = 0.35 * (0.5 + 0.5 * math.sin(0.025 * self.current_step + self.turn_phase))
        elif mode == "left90":
            # Straight first, then a bounded left turn; total heading change is
            # close to 90 degrees over the remaining episode.
            if self.current_step < 70:
                turn = 0.0
            else:
                turn = 0.16
        elif mode == "sine_random":
            slow = 0.22 * math.sin(0.035 * self.current_step * self.turn_rate_scale + self.turn_phase)
            medium = 0.10 * math.sin(0.090 * self.current_step + 0.5 * self.turn_phase)
            turn = slow + medium
        else:
            slow = 0.35 * math.sin(0.045 * self.current_step * self.turn_rate_scale + self.turn_phase)
            medium = 0.14 * math.sin(0.115 * self.current_step + 0.5 * self.turn_phase)
            turn = slow + medium
        if self.target_turn_sine_amp > 0.0:
            turn += self.target_turn_sine_amp * math.sin(
                0.055 * self.current_step * self.turn_rate_scale + self.turn_phase
            )
        if self.target_turn_noise_std > 0.0:
            # Smoothed random disturbance instead of white-noise heading jumps.
            self.target_turn_disturbance = (
                0.92 * self.target_turn_disturbance
                + float(self.rng.normal(0.0, self.target_turn_noise_std))
            )
            turn += self.target_turn_disturbance
        self.target_heading = _wrap_angle(self.target_heading + turn * DT)
        self.target_vel = self.target_speed * _unit_from_angle(self.target_heading)
        self.target_pos = self.target_pos + self.target_vel * DT

        for dim in (0, 1):
            if abs(self.target_pos[dim]) > WORLD_BOUND - 1.0:
                self.target_heading = _wrap_angle(math.pi - self.target_heading if dim == 0 else -self.target_heading)
                self.target_pos[dim] = np.clip(self.target_pos[dim], -WORLD_BOUND + 1.0, WORLD_BOUND - 1.0)
        self.leader_target_history.append(self.target_pos.copy())
        max_history = self.leader_lag_steps + 2
        if len(self.leader_target_history) > max_history:
            self.leader_target_history = self.leader_target_history[-max_history:]

    def _step_leader(self):
        forward = _unit_from_angle(self.target_heading)
        sensed_target = self.target_pos
        if self.leader_lag_steps > 0 and len(self.leader_target_history) > self.leader_lag_steps:
            sensed_target = self.leader_target_history[-1 - self.leader_lag_steps]

        target_vec = sensed_target - self.leader_pos
        target_dist = float(np.linalg.norm(target_vec))
        desired_yaw = math.atan2(target_vec[1], target_vec[0]) if target_dist > 1e-6 else self.leader_yaw
        yaw_error = _wrap_angle(desired_yaw - self.leader_yaw)
        yaw_rate = float(np.clip(2.0 * yaw_error, -LEADER_MAX_ANGULAR, LEADER_MAX_ANGULAR))
        self.leader_yaw = _wrap_angle(self.leader_yaw + yaw_rate * DT)

        distance_error = target_dist - self.leader_follow_dist
        speed = float(np.clip(1.2 * distance_error, -0.5 * self.leader_max_linear, self.leader_max_linear))
        heading_gate = max(math.cos(yaw_error), 0.25)
        self.leader_vel = speed * heading_gate * _unit_from_angle(self.leader_yaw)
        self.leader_pos = self.leader_pos + self.leader_vel * DT

    def _update_slots(self):
        direction = self.target_pos - self.leader_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            forward = _unit_from_angle(self.target_heading)
        else:
            forward = direction / norm
        left = _rot90(forward)

        center = self.target_pos + SLOT_FORWARD_OFFSET * forward
        self.slots[0] = center + self.side_dist * left
        self.slots[1] = center - self.side_dist * left

    def _formation_yaw(self):
        return _yaw_from_vec(self.target_pos - self.leader_pos, self.target_heading)

    def _step_followers(self, actions):
        for idx, agent in enumerate(self.agents):
            action = np.clip(np.asarray(actions[agent], dtype=np.float32), -1.0, 1.0)
            if self.follower_action_mode == "accel":
                # Gazebo/Go2-friendly acceleration action mapping:
                # - policy action[0] changes forward speed; final speed is clamped to [0, max]
                # - policy action[1] changes yaw rate; final yaw rate is clamped to [-max, max]
                # - the policy therefore cannot request instant angular-speed flips.
                linear = float(
                    np.clip(
                        self.follower_cmd_linear[idx] + float(action[0]) * FOLLOWER_LINEAR_ACCEL * DT,
                        0.0,
                        self.follower_max_linear,
                    )
                )
                angular = float(
                    np.clip(
                        self.follower_cmd_angular[idx] + float(action[1]) * FOLLOWER_ANGULAR_ACCEL * DT,
                        -self.follower_max_angular,
                        self.follower_max_angular,
                    )
                )
            else:
                # Velocity-output convention used by the 20260729_210800 model:
                # action[0] maps to forward-only speed, action[1] maps to yaw rate.
                linear = 0.5 * (float(action[0]) + 1.0) * self.follower_max_linear
                angular = float(action[1]) * self.follower_max_angular
            if FOLLOWER_TURN_SLOWDOWN:
                turn_ratio = min(abs(angular) / max(self.follower_max_angular, 1e-6), 1.0)
                linear = min(linear, self.follower_max_linear * max(0.25, 1.0 - 0.65 * turn_ratio))
            self.follower_cmd_linear[idx] = linear
            self.follower_cmd_angular[idx] = angular
            self.follower_angular_vel[idx] = angular
            self.last_actions[idx] = np.array([float(action[0]), float(action[1])], dtype=np.float32)
            self.follower_yaw[idx] = _wrap_angle(float(self.follower_yaw[idx]) + angular * DT)
            self.follower_vel[idx] = linear * _unit_from_angle(float(self.follower_yaw[idx]))
            self.follower_pos[idx] = self.follower_pos[idx] + self.follower_vel[idx] * DT
            self.follower_pos[idx] = np.clip(self.follower_pos[idx], -WORLD_BOUND, WORLD_BOUND)

    def _body_frame(self, idx, vec):
        yaw = float(self.follower_yaw[idx])
        c, s = math.cos(yaw), math.sin(yaw)
        x = c * vec[0] + s * vec[1]
        y = -s * vec[0] + c * vec[1]
        return np.array([x, y], dtype=np.float32)

    def _get_observations(self):
        observations = {}
        slot_vel = (self.slots - self.last_slots) / DT
        leader_target_dist = np.linalg.norm(self.target_pos - self.leader_pos)

        for idx, agent in enumerate(self.agents):
            other_idx = 1 - idx
            vel_scale = max(self.follower_max_linear, 1e-6)
            self_vel_body = self._body_frame(idx, self.follower_vel[idx]) / vel_scale
            leader_rel = self._body_frame(idx, self.leader_pos - self.follower_pos[idx]) / POS_SCALE
            leader_rel_vel = self._body_frame(idx, self.leader_vel - self.follower_vel[idx]) / vel_scale
            target_rel = self._body_frame(idx, self.target_pos - self.follower_pos[idx]) / POS_SCALE
            target_rel_vel = self._body_frame(idx, self.target_vel - self.follower_vel[idx]) / vel_scale
            slot_rel = self._body_frame(idx, self.slots[idx] - self.follower_pos[idx]) / POS_SCALE
            slot_rel_vel = self._body_frame(idx, slot_vel[idx] - self.follower_vel[idx]) / vel_scale
            other_rel = self._body_frame(idx, self.follower_pos[other_idx] - self.follower_pos[idx]) / POS_SCALE
            role = np.array([1.0, 0.0] if idx == 0 else [0.0, 1.0], dtype=np.float32)
            slot_error_norm = np.array(
                [np.linalg.norm(self.slots[idx] - self.follower_pos[idx]) / POS_SCALE],
                dtype=np.float32,
            )
            leader_target_dist_norm = np.array([leader_target_dist / POS_SCALE], dtype=np.float32)
            formation_params = np.array(
                [self.side_dist / DIST_PARAM_SCALE, self.leader_follow_dist / DIST_PARAM_SCALE],
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
                    self_vel_body,
                    np.array([math.sin(self.follower_yaw[idx]), math.cos(self.follower_yaw[idx])], dtype=np.float32),
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
            if self.obs_noise_std > 0.0:
                obs = obs + self.rng.normal(0.0, self.obs_noise_std, size=obs.shape).astype(np.float32)
            observations[agent] = obs
        return observations

    def _info(self, slot_errors, collision, success, reward_components):
        leader_target_distance = float(np.linalg.norm(self.target_pos - self.leader_pos))
        formation_yaw = self._formation_yaw()
        yaw_errors = [abs(_wrap_angle(formation_yaw - float(yaw))) for yaw in self.follower_yaw]
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
            "team_target_distance": leader_target_distance,
            "navigation_active": 1.0,
            "leader_target_distance": leader_target_distance,
            "leader_target_progress": 0.0,
            "total_agent_collisions": int(collision),
            "total_obstacle_collisions": 0,
            "any_obstacle_collision": 0.0,
            "min_obstacle_dist": 999.0,
            "target_speed": float(self.target_speed),
            "side_dist": float(self.side_dist),
            "leader_follow_dist": float(self.leader_follow_dist),
            "follower_max_linear": float(self.follower_max_linear),
            "follower_max_angular": float(self.follower_max_angular),
            "leader_max_linear": float(self.leader_max_linear),
            **reward_components,
        }

    def render(self):
        if self.render_mode not in ("human", "rgb_array"):
            return None

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6), dpi=100)

        self.ax.clear()
        self.ax.set_xlim(-WORLD_BOUND, WORLD_BOUND)
        self.ax.set_ylim(-WORLD_BOUND, WORLD_BOUND)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.25)
        title = f"Follower Slot Tracking step={self.current_step}"
        if self.current_episode is not None:
            title = f"Episode {self.current_episode}  {title}"
        self.ax.set_title(title)

        self.ax.scatter(self.target_pos[0], self.target_pos[1], c="tab:red", s=80, marker="*", label="target")
        self.ax.scatter(self.leader_pos[0], self.leader_pos[1], c="tab:blue", s=70, label="leader/go1")
        self.ax.scatter(self.slots[:, 0], self.slots[:, 1], c="none", edgecolors="tab:green", s=90, label="slots")
        self.ax.scatter(self.follower_pos[0, 0], self.follower_pos[0, 1], c="tab:green", s=60, label="go2 left")
        self.ax.scatter(self.follower_pos[1, 0], self.follower_pos[1, 1], c="tab:orange", s=60, label="go3 right")
        self._draw_heading_arrow(self.target_pos, self.target_heading, "tab:red", 0.50)
        self._draw_heading_arrow(self.leader_pos, self.leader_yaw, "tab:blue", 0.55)
        self._draw_heading_arrow(self.follower_pos[0], float(self.follower_yaw[0]), "tab:green", 0.50)
        self._draw_heading_arrow(self.follower_pos[1], float(self.follower_yaw[1]), "tab:orange", 0.50)
        tri = np.vstack([self.leader_pos, self.follower_pos[0], self.follower_pos[1], self.leader_pos])
        self.ax.plot(tri[:, 0], tri[:, 1], color="0.3", linewidth=1.2, alpha=0.7)
        self.ax.legend(loc="upper right", fontsize=8)

        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        frame = np.asarray(self.fig.canvas.buffer_rgba(), dtype=np.uint8)
        return frame.reshape(height, width, 4)[:, :, :3].copy()

    def _draw_heading_arrow(self, pos, yaw, color, length):
        dx = length * math.cos(yaw)
        dy = length * math.sin(yaw)
        self.ax.arrow(
            pos[0],
            pos[1],
            dx,
            dy,
            width=0.035,
            head_width=0.16,
            head_length=0.20,
            length_includes_head=True,
            color=color,
            alpha=0.9,
            zorder=5,
        )

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None
