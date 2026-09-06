"""Configuration for the standalone waypoint-selection experiment.

All distances are metres and all times are seconds.  The physical lidar range
matches the Gazebo Velodyne plugin, while the policy range matches the existing
point-cloud mapping filter so that nearby geometry is not compressed by the
130 m maximum range.
"""

from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass(frozen=True)
class EnvConfig:
    # Formation in the go1 frame: go2=(+2,+2), go3=(+2,-2).
    formation_forward: float = 2.0
    formation_side: float = 2.0
    candidate_offsets: Tuple[float, ...] = (2.0, 1.0, 0.0, -1.0, -2.0)

    # One MARL decision contains ten Nav2-like motion updates.
    marl_dt: float = 1.0
    nav_dt: float = 0.1
    follower_max_speed: float = 0.15
    # A small speed margin is necessary for followers to recover longitudinal
    # error after a lateral detour while respecting their 0.15 m/s limit.
    leader_speed: float = 0.10

    # Gazebo VLP-16 horizontal interface and policy preprocessing.
    lidar_fov: float = 6.283185307179586
    lidar_sim_rays: int = 36
    lidar_observation_size: int = 36
    lidar_update_rate: float = 10.0
    lidar_min_range: float = 0.9
    lidar_physical_max_range: float = 130.0
    lidar_policy_max_range: float = 20.0
    lidar_noise_std: float = 0.008
    lidar_sensor_x: float = 0.20

    # Robot and obstacle proxy geometry.
    robot_radius: float = 0.35
    obstacle_size: float = 1.5
    obstacle_circle_radius: float = 1.0
    # Leave a clear approach before the random obstacle so an episode has
    # three explicit phases: straight formation, avoidance, straight recovery.
    obstacle_spawn_x: Tuple[float, float] = (8.0, 10.0)
    # Absolute obstacle-centre y range.  The sign is sampled independently so
    # one box appears on either side.  Training may expand the upper bound as a
    # curriculum while retaining the original default-lane cases.
    obstacle_abs_y_range: Tuple[float, float] = (1.7, 2.3)
    obstacle_lane_jitter: float = 0.30

    # Candidate/path feature thresholds.
    candidate_clearance_cap: float = 3.0
    candidate_forward_lookahead: float = 3.0
    endpoint_blocked_clearance: float = 0.60
    path_blocked_clearance: float = 0.60
    obstacle_safe_clearance: float = 0.60
    selected_path_safe_clearance: float = 0.80

    # Pair safety is based on centre-to-centre distance.
    desired_pair_distance: float = 4.0
    pair_safe_distance: float = 1.20
    pair_path_safe_distance: float = 1.50
    pair_collision_distance: float = 0.70

    # Five-term reward constants: task, obstacle safety, pair safety,
    # formation preservation, and forward progress.  Success must outweigh the
    # return available by delaying termination near the end of an episode.
    time_penalty: float = 0.02
    success_bonus: float = 50.0
    collision_penalty: float = 100.0
    obstacle_clearance_weight: float = 4.0
    selected_path_weight: float = 3.0
    blocked_path_penalty: float = 5.0
    pair_clearance_weight: float = 8.0
    pair_path_weight: float = 6.0
    formation_excess_offset_weight: float = 0.60
    formation_switch_weight: float = 0.30
    formation_route_reversal_weight: float = 0.50
    progress_weight: float = 0.50

    # A one-second decision may hold its current candidate or move by one
    # adjacent level only: e.g. 0 m -> {-1, 0, +1} m, never directly to +/-2 m.
    max_action_index_change: int = 1
    # Once a default corridor is blocked, require several consecutive clear
    # observations before forcing the agent back toward its default slot.
    default_clear_release_steps: int = 3

    # The farther obstacle still leaves enough time to pass it, travel another
    # two metres, and hold the recovered default formation for three decisions.
    max_episode_steps: int = 130
    recovery_hold_steps: int = 3
    recovery_error_tolerance: float = 0.75

    @property
    def nav_substeps(self) -> int:
        return round(self.marl_dt / self.nav_dt)

    @property
    def num_agents(self) -> int:
        return 2

    @property
    def num_actions(self) -> int:
        return len(self.candidate_offsets)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 7
    total_steps: int = 200_000
    replay_size: int = 100_000
    warmup_steps: int = 5_000
    batch_size: int = 256
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 5e-4
    gamma: float = 0.99
    tau: float = 0.005
    update_every: int = 1
    updates_per_step: int = 1
    initial_epsilon: float = 0.90
    final_epsilon: float = 0.05
    epsilon_decay_steps: int = 150_000
    initial_temperature: float = 1.0
    final_temperature: float = 0.25
    eval_interval: int = 10_000
    eval_episodes: int = 20
    log_interval_episodes: int = 20

    def to_dict(self):
        return asdict(self)
