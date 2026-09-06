"""Two-agent five-candidate waypoint-selection pretraining environment."""

import math

import numpy as np

from .config import EnvConfig
from .geometry import (
    circle_aabb_clearance,
    rotate,
    segment_segment_distance,
    world_to_body,
)
from .lidar import PlanarLidar, candidate_features


AGENT_NAMES = ("go2", "go3")
DEFAULT_ACTION = 2


class WaypointSelectionEnv:
    """A lightweight Nav2 proxy for discrete MADDPG pretraining.

    go1 travels along +x.  After a clear straight approach, one 1.5 x 1.5 m
    square and one radius-1 m circle are spawned, one in each follower lane.
    Their lane assignment and positions are randomized every episode.  The
    followers must avoid them and then recover their straight default formation.
    Actions select one of five formation-relative Nav2 goals and are held for
    one second while ten 0.1 s motion substeps run.
    """

    def __init__(self, config=None, seed=None, lidar_noise=True):
        self.cfg = config or EnvConfig()
        self.rng = np.random.default_rng(seed)
        self.lidar_noise = bool(lidar_noise)
        self.lidar = PlanarLidar(self.cfg, self.rng)
        self.obstacle_abs_y_range = tuple(self.cfg.obstacle_abs_y_range)
        self.obs_dim = 83
        self.num_agents = self.cfg.num_agents
        self.num_actions = self.cfg.num_actions
        self.reset(seed=seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.lidar = PlanarLidar(self.cfg, self.rng)
        self.step_count = 0
        self.leader_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.leader_yaw = 0.0
        self.leader_velocity = np.array([self.cfg.leader_speed, 0.0], dtype=np.float32)
        self.follower_pos = np.array([[2.0, 2.0], [2.0, -2.0]], dtype=np.float32)
        self.follower_velocity = np.zeros((2, 2), dtype=np.float32)
        self.previous_actions = np.full(2, DEFAULT_ACTION, dtype=np.int64)
        self.current_goals = self._candidate_points()[:, DEFAULT_ACTION].copy()
        self.last_progress = np.zeros(2, dtype=np.float32)
        self.recovery_hold = 0
        # Last non-zero detour side while the default corridor remains blocked.
        # It survives a one-step return through offset zero so a full route-side
        # reversal can receive a stronger penalty than an ordinary switch.
        self.route_directions = np.zeros(2, dtype=np.int8)
        self.default_blocked_latched = np.zeros(2, dtype=bool)
        self.default_clear_counts = np.zeros(2, dtype=np.int64)

        lane_signs = self.rng.permutation(np.asarray([-1.0, 1.0], dtype=np.float32))
        centers = np.stack(
            [
                np.asarray(
                    [
                        self.rng.uniform(*self.cfg.obstacle_spawn_x),
                        lane_signs[index] * self.rng.uniform(*self.obstacle_abs_y_range),
                    ],
                    dtype=np.float32,
                )
                for index in range(2)
            ]
        )
        half = 0.5 * self.cfg.obstacle_size
        self.obstacles = [
            {
                "shape": "square",
                "center": centers[0],
                "size": float(self.cfg.obstacle_size),
                "lower": centers[0] - half,
                "upper": centers[0] + half,
            },
            {
                "shape": "circle",
                "center": centers[1],
                "radius": float(self.cfg.obstacle_circle_radius),
            },
        ]
        # Backward-compatible aliases used by fixed-square scenario tooling.
        self.obstacle_center = self.obstacles[0]["center"]
        self.obstacle_lower = self.obstacles[0]["lower"]
        self.obstacle_upper = self.obstacles[0]["upper"]
        observations, self._candidate_metrics = self._build_observations()
        self._update_default_path_state()
        return observations, self._info(False, False, np.inf, np.inf)

    def set_obstacle_abs_y_range(self, value):
        value = tuple(float(item) for item in value)
        if len(value) != 2 or value[0] < 0.0 or value[1] < value[0]:
            raise ValueError("obstacle_abs_y_range must be nonnegative (min,max)")
        self.obstacle_abs_y_range = value

    def _candidate_points(self):
        forward = np.array([math.cos(self.leader_yaw), math.sin(self.leader_yaw)], dtype=np.float32)
        left = np.array([-forward[1], forward[0]], dtype=np.float32)
        bases = np.stack(
            [
                self.leader_pos + self.cfg.formation_forward * forward + self.cfg.formation_side * left,
                self.leader_pos + self.cfg.formation_forward * forward - self.cfg.formation_side * left,
            ],
            axis=0,
        )
        offsets = np.asarray(self.cfg.candidate_offsets, dtype=np.float32)
        return bases[:, None, :] + offsets[None, :, None] * left[None, None, :]

    def _scan_for_agent(self, index):
        other_index = 1 - index
        other_robots = [
            (self.leader_pos, self.cfg.robot_radius),
            (self.follower_pos[other_index], self.cfg.robot_radius),
        ]
        original_std = self.cfg.lidar_noise_std
        if not self.lidar_noise and original_std:
            # Frozen dataclass: temporarily replace RNG sampling by using a
            # deterministic zero-noise generator through the scan result path.
            state = self.lidar.rng
            self.lidar.rng = _ZeroNoiseRng()
            result = self.lidar.scan(
                self.follower_pos[index],
                self.leader_yaw,
                self.obstacles,
                other_robots,
            )
            self.lidar.rng = state
            return result
        return self.lidar.scan(
            self.follower_pos[index],
            self.leader_yaw,
            self.obstacles,
            other_robots,
        )

    def _build_observations(self):
        candidates = self._candidate_points()
        default_points = candidates[:, DEFAULT_ACTION]
        observations, all_metrics = [], []
        for index in range(2):
            sectors, lidar_points_body = self._scan_for_agent(index)
            candidate_obs, metrics = candidate_features(
                candidates[index],
                self.follower_pos[index],
                self.leader_yaw,
                lidar_points_body,
                self.cfg,
            )
            other = 1 - index
            own_velocity_body = rotate(self.follower_velocity[index], -self.leader_yaw)
            leader_rel = world_to_body(self.leader_pos, self.follower_pos[index], self.leader_yaw)
            leader_vel_body = rotate(self.leader_velocity, -self.leader_yaw)
            teammate_rel = world_to_body(
                self.follower_pos[other], self.follower_pos[index], self.leader_yaw
            )
            teammate_vel = rotate(
                self.follower_velocity[other] - self.follower_velocity[index],
                -self.leader_yaw,
            )
            default_rel = world_to_body(
                default_points[index], self.follower_pos[index], self.leader_yaw
            )
            goal_rel = world_to_body(
                self.current_goals[index], self.follower_pos[index], self.leader_yaw
            )
            role = np.array([1.0, 0.0] if index == 0 else [0.0, 1.0], dtype=np.float32)
            previous_action = np.eye(self.num_actions, dtype=np.float32)[self.previous_actions[index]]
            observation = np.concatenate(
                [
                    sectors,
                    candidate_obs,
                    np.clip(default_rel / 6.0, -1.0, 1.0),
                    np.clip(own_velocity_body / self.cfg.follower_max_speed, -1.0, 1.0),
                    np.clip(leader_rel / 6.0, -1.0, 1.0),
                    np.clip(leader_vel_body / self.cfg.follower_max_speed, -1.0, 1.0),
                    np.clip(teammate_rel / 8.0, -1.0, 1.0),
                    np.clip(teammate_vel / self.cfg.follower_max_speed, -1.0, 1.0),
                    role,
                    previous_action,
                    np.clip(goal_rel / 6.0, -1.0, 1.0),
                    np.array(
                        [np.clip(self.last_progress[index] / self.cfg.follower_max_speed, -1.0, 1.0)],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)
            if observation.shape != (self.obs_dim,):
                raise RuntimeError(f"unexpected observation shape {observation.shape}")
            observations.append(observation)
            all_metrics.append(metrics)
        return np.stack(observations), all_metrics

    @staticmethod
    def _danger(clearance, safe_distance):
        ratio = np.clip(
            (safe_distance - clearance) / max(safe_distance, 1e-6), 0.0, 1.0
        )
        return float(ratio * ratio)

    def _formation_offset_penalties(self, actions):
        """Penalize only displacement beyond the nearest currently safe slot."""
        offsets = np.abs(np.asarray(self.cfg.candidate_offsets, dtype=np.float32))
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        for index in range(self.num_agents):
            safe_actions = [
                action
                for action, metrics in enumerate(self._candidate_metrics[index])
                if not metrics["blocked"]
            ]
            if not safe_actions:
                continue
            minimum_safe_offset = float(np.min(offsets[safe_actions]))
            selected_offset = float(offsets[actions[index]])
            excess_offset = max(0.0, selected_offset - minimum_safe_offset)
            penalties[index] = -self.cfg.formation_excess_offset_weight * excess_offset
        return penalties

    def valid_action_masks(self):
        """Keep clear agents fixed; let blocked agents move adjacently."""
        action_indices = np.arange(self.num_actions, dtype=np.int64)
        masks = (
            np.abs(action_indices[None, :] - self.previous_actions[:, None])
            <= self.cfg.max_action_index_change
        )
        for index in range(self.num_agents):
            if self.default_blocked_latched[index]:
                safe = np.asarray(
                    [not metrics["blocked"] for metrics in self._candidate_metrics[index]],
                    dtype=bool,
                )
                safe_adjacent = masks[index] & safe
                if np.any(safe_adjacent):
                    masks[index] = safe_adjacent
                continue
            masks[index] = False
            previous = int(self.previous_actions[index])
            if previous == DEFAULT_ACTION:
                masks[index, DEFAULT_ACTION] = True
            else:
                step = 1 if previous < DEFAULT_ACTION else -1
                masks[index, previous + step] = True
        return masks

    def _update_default_path_state(self):
        for index in range(self.num_agents):
            default_blocked = self._candidate_metrics[index][DEFAULT_ACTION]["blocked"]
            if default_blocked:
                self.default_blocked_latched[index] = True
                self.default_clear_counts[index] = 0
            elif self.default_blocked_latched[index]:
                self.default_clear_counts[index] += 1
                if (
                    self.default_clear_counts[index]
                    >= self.cfg.default_clear_release_steps
                ):
                    self.default_blocked_latched[index] = False
                    self.default_clear_counts[index] = 0

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        if actions.shape != (2,) or np.any(actions < 0) or np.any(actions >= self.num_actions):
            raise ValueError(f"actions must have shape (2,) with values in [0,{self.num_actions - 1}]")
        valid_masks = self.valid_action_masks()
        if not np.all(valid_masks[np.arange(self.num_agents), actions]):
            raise ValueError(
                "actions may hold or move to an adjacent candidate only; "
                f"previous={self.previous_actions.tolist()}, requested={actions.tolist()}"
            )

        start_positions = self.follower_pos.copy()
        candidates = self._candidate_points()
        goals = np.stack([candidates[i, actions[i]] for i in range(2)])
        selected_metrics = [self._candidate_metrics[i][actions[i]] for i in range(2)]
        formation_offset_penalties = self._formation_offset_penalties(actions)
        switched = actions != self.previous_actions
        switch_penalties = (
            -self.cfg.formation_switch_weight * switched.astype(np.float32)
        )
        reversal_penalties = np.zeros(self.num_agents, dtype=np.float32)
        offsets = np.asarray(self.cfg.candidate_offsets, dtype=np.float32)
        for index in range(self.num_agents):
            default_blocked = self._candidate_metrics[index][DEFAULT_ACTION]["blocked"]
            selected_direction = int(np.sign(offsets[actions[index]]))
            if default_blocked:
                if selected_direction != 0:
                    if (
                        self.route_directions[index] != 0
                        and selected_direction != self.route_directions[index]
                    ):
                        reversal_penalties[index] = (
                            -self.cfg.formation_route_reversal_weight
                        )
                    self.route_directions[index] = selected_direction
            elif actions[index] == DEFAULT_ACTION:
                self.route_directions[index] = 0
        path_pair_distance = segment_segment_distance(
            start_positions[0], goals[0], start_positions[1], goals[1]
        )
        self.current_goals = goals.copy()

        min_obstacle_clearances = np.full(2, np.inf, dtype=np.float32)
        min_pair_distance = float(np.linalg.norm(self.follower_pos[0] - self.follower_pos[1]))
        obstacle_collision = np.zeros(2, dtype=bool)
        obstacle_collision_indices = np.full(2, -1, dtype=np.int64)
        pair_collision = False

        for _ in range(self.cfg.nav_substeps):
            self.leader_pos += self.leader_velocity * self.cfg.nav_dt
            proposed = self.follower_pos.copy()
            proposed_velocities = np.zeros_like(self.follower_velocity)
            for index in range(2):
                delta = goals[index] - self.follower_pos[index]
                distance = float(np.linalg.norm(delta))
                if distance > 1e-8:
                    speed = min(self.cfg.follower_max_speed, distance / self.cfg.nav_dt)
                    proposed_velocities[index] = speed * delta / distance
                    proposed[index] += proposed_velocities[index] * self.cfg.nav_dt

                obstacle_clearances = []
                for obstacle in self.obstacles:
                    if obstacle["shape"] == "square":
                        clearance = circle_aabb_clearance(
                            proposed[index],
                            self.cfg.robot_radius,
                            obstacle["lower"],
                            obstacle["upper"],
                        )
                    else:
                        clearance = float(
                            np.linalg.norm(proposed[index] - obstacle["center"])
                            - self.cfg.robot_radius
                            - obstacle["radius"]
                        )
                    obstacle_clearances.append(clearance)
                clearance = min(obstacle_clearances)
                min_obstacle_clearances[index] = min(min_obstacle_clearances[index], clearance)
                if clearance <= 0.0:
                    obstacle_collision[index] = True
                    obstacle_collision_indices[index] = int(np.argmin(obstacle_clearances))
                    proposed[index] = self.follower_pos[index]
                    proposed_velocities[index] = 0.0

            proposed_pair_distance = float(np.linalg.norm(proposed[0] - proposed[1]))
            min_pair_distance = min(min_pair_distance, proposed_pair_distance)
            if proposed_pair_distance <= self.cfg.pair_collision_distance:
                pair_collision = True
                proposed = self.follower_pos.copy()
                proposed_velocities[:] = 0.0

            self.follower_pos = proposed
            self.follower_velocity = proposed_velocities

        displacement = self.follower_pos - start_positions
        self.last_progress = displacement[:, 0].astype(np.float32)
        self.step_count += 1

        # 1) Obstacle avoidance: actual clearance, selected path clearance,
        # blocked choices, and obstacle collisions form one semantic term.
        obstacle_rewards = np.zeros(2, dtype=np.float32)
        for index in range(2):
            obstacle_rewards[index] -= self.cfg.obstacle_clearance_weight * self._danger(
                float(min_obstacle_clearances[index]), self.cfg.obstacle_safe_clearance
            )
            obstacle_rewards[index] -= self.cfg.selected_path_weight * self._danger(
                selected_metrics[index]["path_clearance"], self.cfg.selected_path_safe_clearance
            )
            if selected_metrics[index]["blocked"]:
                obstacle_rewards[index] -= self.cfg.blocked_path_penalty
        obstacle_reward = float(np.mean(obstacle_rewards))
        if bool(np.any(obstacle_collision)):
            obstacle_reward -= self.cfg.collision_penalty

        # 2) Inter-robot avoidance: keep both the executed motion and the two
        # selected paths separated.  A collision receives one full penalty.
        pair_reward = -self.cfg.pair_clearance_weight * self._danger(
            min_pair_distance, self.cfg.pair_safe_distance
        )
        pair_reward -= self.cfg.pair_path_weight * self._danger(
            path_pair_distance, self.cfg.pair_path_safe_distance
        )
        if pair_collision and not bool(np.any(obstacle_collision)):
            pair_reward -= self.cfg.collision_penalty

        # 3) Formation preservation is evaluated independently per follower.
        # A clear follower stays in its default slot through the action mask;
        # a blocked follower is penalized only for moving farther than its own
        # nearest safe slot, switching, or reversing its avoidance direction.
        # Inter-follower separation is handled exclusively by pair safety.
        goal_pair_distance = float(np.linalg.norm(goals[0] - goals[1]))
        formation_reward = float(
            np.mean(
                formation_offset_penalties
                + switch_penalties
                + reversal_penalties
            )
        )

        # 4) Forward progress prevents a collision-free policy from learning to
        # stand still.  5) The task term charges time and later adds success.
        progress_reward = self.cfg.progress_weight * float(
            np.mean(
                np.clip(
                    self.last_progress / self.cfg.follower_max_speed,
                    -1.0,
                    1.0,
                )
            )
        )
        task_reward = -self.cfg.time_penalty

        team_reward = float(
            task_reward
            + obstacle_reward
            + pair_reward
            + formation_reward
            + progress_reward
        )
        terminated = bool(np.any(obstacle_collision) or pair_collision)

        # The next observation and its action mask must describe the action
        # that was just executed, not the action from two decisions ago.
        self.previous_actions = actions.copy()
        next_observations, self._candidate_metrics = self._build_observations()
        self._update_default_path_state()
        next_defaults = self._candidate_points()[:, DEFAULT_ACTION]
        default_errors = np.linalg.norm(self.follower_pos - next_defaults, axis=1)
        obstacle_end_x = max(
            float(obstacle["upper"][0])
            if obstacle["shape"] == "square"
            else float(obstacle["center"][0] + obstacle["radius"])
            for obstacle in self.obstacles
        )
        obstacle_passed = bool(np.min(self.follower_pos[:, 0]) > obstacle_end_x + 2.0)
        recovered = bool(
            obstacle_passed
            and np.all(actions == DEFAULT_ACTION)
            and np.all(default_errors < self.cfg.recovery_error_tolerance)
        )
        self.recovery_hold = self.recovery_hold + 1 if recovered else 0
        success = self.recovery_hold >= self.cfg.recovery_hold_steps
        if success:
            task_reward += self.cfg.success_bonus
            team_reward += self.cfg.success_bonus
            terminated = True
        truncated = self.step_count >= self.cfg.max_episode_steps

        rewards = np.full(2, team_reward, dtype=np.float32)
        info = self._info(
            success,
            bool(np.any(obstacle_collision) or pair_collision),
            float(np.min(min_obstacle_clearances)),
            min_pair_distance,
        )
        info.update(
            {
                "obstacle_collision_agents": obstacle_collision.copy(),
                "obstacle_collision_indices": obstacle_collision_indices.copy(),
                "pair_collision": bool(pair_collision),
                "actions": actions.copy(),
                "goals": goals.copy(),
                "goal_pair_distance": goal_pair_distance,
                "path_pair_distance": path_pair_distance,
                "default_errors": default_errors.astype(np.float32),
                "minimum_safe_offset_penalties": formation_offset_penalties.copy(),
                "reward_switch_penalties": switch_penalties.copy(),
                "reward_route_reversal_penalties": reversal_penalties.copy(),
                "default_blocked_latched": self.default_blocked_latched.copy(),
                "reward_task": task_reward,
                "reward_obstacle": obstacle_reward,
                "reward_pair": pair_reward,
                "reward_formation": formation_reward,
                "reward_progress": progress_reward,
            }
        )
        return next_observations, rewards, terminated, truncated, info

    def _info(self, success, collision, min_obstacle_clearance, min_pair_distance):
        return {
            "success": bool(success),
            "collision": bool(collision),
            "step": int(self.step_count),
            "leader_position": self.leader_pos.copy(),
            "follower_positions": self.follower_pos.copy(),
            "obstacle_center": self.obstacle_center.copy(),
            "obstacle_size": float(self.cfg.obstacle_size),
            "obstacles": [
                {
                    "shape": obstacle["shape"],
                    "center": obstacle["center"].copy(),
                    **(
                        {"size": float(obstacle["size"])}
                        if obstacle["shape"] == "square"
                        else {"radius": float(obstacle["radius"])}
                    ),
                }
                for obstacle in self.obstacles
            ],
            "min_obstacle_clearance": float(min_obstacle_clearance),
            "min_pair_distance": float(min_pair_distance),
        }


class _ZeroNoiseRng:
    @staticmethod
    def normal(_mean, _std):
        return 0.0
