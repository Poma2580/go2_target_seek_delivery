r"""
Leader-follower 4-agent formation navigation environment.

PettingZoo ParallelEnv interface.
- agent_0 is the leader.
- agent_1/agent_2/agent_3 are followers.
- Followers keep a fixed world-frame triangular formation behind the leader:
      follower_1  follower_2  follower_3
             \       |       /
                    leader
  In coordinates: followers are placed on a horizontal line at y = leader_y - FOLLOWER_BACK_OFFSET.
- All agents keep the same observation dimension. Obstacle observations are appended at the end so
  stage-1 no-obstacle pretraining can be incrementally reused in later stages.

Curriculum / incremental-training stages:
    1: no obstacles, fixed/simple initial formation-following task
    2: fixed obstacles, fixed/simple initial formation-following + obstacle avoidance
    3: fixed obstacles, randomized initial formation center
    4: fixed obstacles, scattered independent initial positions, online formation + navigation
    5: randomized obstacles, scattered independent initial positions
"""

import numpy as np
import gymnasium
from gymnasium import spaces
from pettingzoo import ParallelEnv
from pettingzoo.utils import wrappers
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import font_manager


def setup_chinese_font():
    """Configure Matplotlib font fallback for Chinese labels in rendered GIF frames."""
    preferred_fonts = [
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "SimHei",
        "Microsoft YaHei",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
    for font_name in preferred_fonts:
        if font_name in available_fonts:
            plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
            break
    else:
        plt.rcParams["font.sans-serif"] = preferred_fonts + ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


setup_chinese_font()


# =========================
# Environment configuration
# =========================
MAX_CYCLES = 120
WORLD_BOUND = 3.0
TARGET_POS = np.array([2.20, 2.50], dtype=np.float32)

# Leader-follower triangular formation.
# Three followers stay behind the leader and form one horizontal line in the world frame.
FOLLOWER_LINE_SPACING = 0.60
FOLLOWER_BACK_OFFSET = 0.65
FOLLOWER_OFFSETS = np.array(
    [
        [-FOLLOWER_LINE_SPACING, -FOLLOWER_BACK_OFFSET],
        [0.0, -FOLLOWER_BACK_OFFSET],
        [FOLLOWER_LINE_SPACING, -FOLLOWER_BACK_OFFSET],
    ],
    dtype=np.float32,
)
LEADER_PSEUDO_OFFSET = np.array([0.0, 0.0], dtype=np.float32)
ALL_ROLE_OFFSETS = np.vstack([LEADER_PSEUDO_OFFSET[None, :], FOLLOWER_OFFSETS]).astype(np.float32)

# Action / dynamics
ACTION_SCALE = 0.10
STEP_PENALTY = -0.02

# Obstacles
NUM_OBSTACLES = 3

# Stage-2 fixed obstacles are deliberately small and staggered around the nominal route.
# The agent-agent collision threshold is 0.20, so treating an agent as radius ~= 0.10
# makes OBSTACLE_RADIUS = 0.10 a natural first curriculum setting.
# The soft margin is kept moderate; too large a soft field can teach agents to stop
# in front of an obstacle wall instead of learning a detour.
OBSTACLE_RADIUS = 0.10
OBSTACLE_SAFE_MARGIN = 0.25
SOFT_OBS_DECAY_SCALE = 0.30
FIXED_OBSTACLE_POSITIONS = np.array(
    [
        # upper-side obstacle: visible but not centered on the start-target route
        [-0.10, 0.80],
        # lower-side obstacle: creates a simple detour/gate without closing the path
        [0.90, 0.25],
        # upper-side late obstacle: keeps stage 2 nontrivial near the second half,
        # but leaves the final approach to TARGET_POS open
        [1.65, 1.95],
    ],
    dtype=np.float32,
)
OBSTACLE_MIN_DIST_TO_AGENT_INIT = 0.60
OBSTACLE_MIN_DIST_TO_SLOT = 0.70
OBSTACLE_MIN_DIST_TO_TARGET = 0.90
OBSTACLE_MIN_DIST_BETWEEN = 0.50

# Agent-agent safety
COLLISION_THRESHOLD = 0.20
SOFT_AGENT_DECAY_SCALE = 0.30

# Normalization scales
TARGET_DIST_SCALE = 2 * WORLD_BOUND * np.sqrt(2)
SLOT_ERROR_SCALE = max(FOLLOWER_LINE_SPACING, FOLLOWER_BACK_OFFSET)
SHAPE_ERROR_SCALE = max(FOLLOWER_LINE_SPACING, FOLLOWER_BACK_OFFSET)
MAX_FOLLOWER_DETACH_DIST = 1.60

# Stage 4/5 online-formation gate.
# When followers start scattered, leader should not rush to the target before the team forms.
# The gate is 0 when mean slot error is large and 1 when formation is close enough.
FORMATION_GATE_ERROR_LOW = 0.25
FORMATION_GATE_ERROR_HIGH = 0.90
LEADER_MIN_PROGRESS_SCALE_WHEN_UNFORMED = 0.25
LEADER_WAIT_W = 1.0
FOLLOWER_FORMATION_EXTRA_SCALE_WHEN_UNFORMED = 0.50

# Independent scattered-initialization parameters for stages 4/5.
# Relaxed sampling range: leader is sampled from a wider start region, and each
# follower is independently sampled in a larger box around the leader. We still
# keep only basic physical feasibility constraints: inside boundary, no initial
# overlap between agents, and no initial overlap with fixed obstacles.
SCATTER_LEADER_LOW = np.array([-2.55, -2.45], dtype=np.float32)
SCATTER_LEADER_HIGH = np.array([0.75, 0.85], dtype=np.float32)
SCATTER_FOLLOWER_REL_LOW = np.array([-1.65, -1.55], dtype=np.float32)
SCATTER_FOLLOWER_REL_HIGH = np.array([1.65, 1.45], dtype=np.float32)
SCATTER_MIN_AGENT_DIST = 0.35
SCATTER_MIN_OBS_DIST = 0.45
SCATTER_MAX_ATTEMPTS = 3000

# Success thresholds. Agent-agent collision is intentionally NOT included in success conditions.
SUCCESS_LEADER_TARGET_THRESHOLD = 0.35
SUCCESS_MEAN_FOLLOWER_SLOT_THRESHOLD = 0.25
SUCCESS_MAX_FOLLOWER_SLOT_THRESHOLD = 0.40
SUCCESS_HOLD_STEPS = 1
SUCCESS_BONUS = 100.0
OBSTACLE_COLLISION_TERMINATION_PENALTY = -50.0

# Reward weights.
# Leader reward: navigation + final approach + safety + coordination.
LEADER_GOAL_PROGRESS_W = 8.0
LEADER_TARGET_DIST_W = 0.30
LEADER_SAFETY_W = 3.0
LEADER_DETACH_W = 1.5
# Final approach shaping: only active near the target to help leader finish within the success radius.
FINAL_TARGET_RADIUS = 0.60
LEADER_FINAL_TARGET_W = 3.0
# Follower-slot obstacle risk is folded into the leader safety term.
LEADER_SLOT_OBS_W = 1.5

# Follower reward follows a two-level structure:
#   formation big term = normalized(slot tracking + shape keeping)
#   safety big term    = normalized(obstacle avoidance + agent-agent collision avoidance)
FOLLOWER_FORMATION_W = 5.0
FOLLOWER_SAFETY_W = 3.0
FOLLOWER_TEAM_PROGRESS_W = 1.0

FORMATION_SLOT_COEF = 0.75
FORMATION_SHAPE_COEF = 0.25
SAFETY_OBS_COEF = 0.55
SAFETY_AGENT_COEF = 0.45

# Fixed-start curriculum defaults.
FIXED_LEADER_START = np.array([-1.45, -1.25], dtype=np.float32)
INIT_NOISE_BY_STAGE = {
    1: 0.05,
    2: 0.08,
    3: 0.20,
    4: 0.25,
    5: 0.25,
}


def env(**kwargs):
    env_instance = FormationNavigationEnv(**kwargs)
    env_instance = wrappers.AssertOutOfBoundsWrapper(env_instance)
    env_instance = wrappers.OrderEnforcingWrapper(env_instance)
    return env_instance


class FormationNavigationEnv(ParallelEnv):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "formation_navigation_v0",
        "is_parallelizable": True,
    }

    def __init__(self, render_mode=None, max_cycles=MAX_CYCLES, training_stage=4):
        super().__init__()
        if training_stage not in (1, 2, 3, 4, 5):
            raise ValueError("training_stage must be one of {1, 2, 3, 4, 5}")

        # Keep original names for compatibility with saved MADDPG checkpoints.
        self.possible_agents = [f"agent_{i}" for i in range(4)]
        self.agents = self.possible_agents.copy()
        self.leader_idx = 0
        self.follower_indices = [1, 2, 3]

        self.max_cycles = max_cycles
        self.render_mode = render_mode
        self.training_stage = int(training_stage)
        self.world_bound = WORLD_BOUND

        self.action_spaces = {
            agent: spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # Observation layout, obstacle block intentionally at the end:
        # own_pos(2), own_vel(2), own_slot_rel(2), leader_rel(2), target_rel(2),
        # other_agents_rel(6), role_flag(1), obstacles_rel(2 * NUM_OBSTACLES)
        obs_low = np.concatenate(
            [
                [-self.world_bound] * 2,
                [-1.0] * 2,
                [-2 * self.world_bound] * 2,
                [-2 * self.world_bound] * 2,
                [-2 * self.world_bound] * 2,
                [-2 * self.world_bound] * 6,
                [0.0],
                [-2 * self.world_bound] * (2 * NUM_OBSTACLES),
            ]
        ).astype(np.float32)
        obs_high = np.concatenate(
            [
                [self.world_bound] * 2,
                [1.0] * 2,
                [2 * self.world_bound] * 2,
                [2 * self.world_bound] * 2,
                [2 * self.world_bound] * 2,
                [2 * self.world_bound] * 6,
                [1.0],
                [2 * self.world_bound] * (2 * NUM_OBSTACLES),
            ]
        ).astype(np.float32)
        self.observation_spaces = {
            agent: spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
            for agent in self.possible_agents
        }

        self.current_step = 0
        self.agent_positions = np.zeros((4, 2), dtype=np.float32)
        self.agent_velocities = np.zeros((4, 2), dtype=np.float32)
        self.obstacle_positions = np.zeros((NUM_OBSTACLES, 2), dtype=np.float32)

        self.last_leader_target_distance = 0.0
        self.last_team_target_distance = 0.0
        self.last_slot_errors = np.zeros(4, dtype=np.float32)
        self.last_mean_slot_error = 0.0
        self.last_formation_error = 0.0
        self.last_shape_error = 0.0
        self.success_hold_count = 0
        self.formation_hold_count = 0
        self.navigation_active = True
        self.used_fallback_formation_center = False
        # Optional display-only field set by run.py so GIF titles show the current episode.
        self.current_episode = None

        self.fig, self.ax = None, None

    @property
    def num_agents(self):
        return len(self.agents)

    @staticmethod
    def _clip01(x):
        return float(np.clip(x, 0.0, 1.0))

    @staticmethod
    def _safe_norm(x, scale, low=-1.0, high=1.0):
        if scale <= 1e-8:
            return 0.0
        return float(np.clip(x / scale, low, high))

    def _obstacles_enabled(self):
        return self.training_stage >= 2

    def _random_init_enabled(self):
        return self.training_stage >= 3

    def _scattered_init_enabled(self):
        return self.training_stage >= 4

    def _random_obstacles_enabled(self):
        return self.training_stage >= 5


    def _initial_leader_position(self, np_random):
        if not self._random_init_enabled():
            return FIXED_LEADER_START.copy()

        # Keep enough room for the three fixed-offset followers behind the leader
        # in stage 3, where followers are still initialized near their target slots.
        x_margin = FOLLOWER_LINE_SPACING + 0.25
        y_lower = -self.world_bound - float(np.min(FOLLOWER_OFFSETS[:, 1])) + 0.25
        y_upper = min(0.65, self.world_bound - 0.50)
        x_low = -self.world_bound + x_margin
        x_high = 0.65
        return np_random.uniform(
            low=np.array([x_low, y_lower], dtype=np.float32),
            high=np.array([x_high, y_upper], dtype=np.float32),
        ).astype(np.float32)

    def _valid_scattered_positions(self, positions):
        # Stay inside the world with a small numerical margin.
        if np.any(positions < -self.world_bound + 0.10) or np.any(positions > self.world_bound - 0.10):
            return False

        pairwise = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
        pairwise = pairwise + np.eye(self.num_agents, dtype=np.float32) * 999.0
        if float(np.min(pairwise)) < SCATTER_MIN_AGENT_DIST:
            return False

        # In stage 4, fixed obstacles are already known; avoid invalid starts near them.
        # In stage 5, random obstacles are generated after positions and already avoid agents.
        if self._obstacles_enabled() and not self._random_obstacles_enabled():
            d_obs = np.linalg.norm(positions[:, None, :] - FIXED_OBSTACLE_POSITIONS[None, :, :], axis=-1)
            if float(np.min(d_obs)) < SCATTER_MIN_OBS_DIST:
                return False

        return True

    def _generate_scattered_initial_positions(self, np_random):
        """Generate independent scattered starts for online formation stages 4/5.

        The leader is sampled from a broad start region. Followers are sampled independently
        around the leader, not from their target formation slots. This forces the policy to
        learn forming up while navigating instead of starting already assembled.
        """
        for _ in range(SCATTER_MAX_ATTEMPTS):
            leader_pos = np_random.uniform(SCATTER_LEADER_LOW, SCATTER_LEADER_HIGH).astype(np.float32)
            positions = np.zeros((self.num_agents, 2), dtype=np.float32)
            positions[self.leader_idx] = leader_pos

            for idx in self.follower_indices:
                rel = np_random.uniform(SCATTER_FOLLOWER_REL_LOW, SCATTER_FOLLOWER_REL_HIGH).astype(np.float32)
                positions[idx] = leader_pos + rel

            positions = np.clip(positions, -self.world_bound + 0.10, self.world_bound - 0.10).astype(np.float32)
            if self._valid_scattered_positions(positions):
                return positions

        # Conservative fallback: use a deliberately scattered configuration around a sampled leader.
        leader_pos = np_random.uniform(SCATTER_LEADER_LOW, SCATTER_LEADER_HIGH).astype(np.float32)
        fallback_rel = np.array(
            [
                [0.0, 0.0],
                [-0.90, -0.95],
                [0.85, -0.20],
                [-0.20, 0.85],
            ],
            dtype=np.float32,
        )
        positions = leader_pos[None, :] + fallback_rel
        return np.clip(positions, -self.world_bound + 0.10, self.world_bound - 0.10).astype(np.float32)

    def _ideal_positions_from_leader(self, leader_pos):
        return leader_pos[None, :] + ALL_ROLE_OFFSETS

    def _follower_slot_positions(self):
        leader_pos = self.agent_positions[self.leader_idx]
        return leader_pos[None, :] + FOLLOWER_OFFSETS

    def _slot_obstacle_penalty(self):
        """Penalty for leader when follower target slots pass too close to obstacles.

        This makes the leader plan a path with enough clearance for the whole formation,
        instead of only avoiding obstacles with its own point mass.
        """
        if not self._obstacles_enabled():
            return 0.0, TARGET_DIST_SCALE

        follower_slots = self._follower_slot_positions()
        slot_obs_distances = np.linalg.norm(
            follower_slots[:, None, :] - self.obstacle_positions[None, :, :], axis=-1
        ).astype(np.float32)
        min_slot_obstacle_dist = float(np.min(slot_obs_distances))

        slot_obs_clearance = np.maximum(
            slot_obs_distances - (OBSTACLE_RADIUS + OBSTACLE_SAFE_MARGIN), 0.0
        )
        slot_obs_soft_values = np.exp(-slot_obs_clearance / SOFT_OBS_DECAY_SCALE)
        soft_slot_obs_norm = float(0.70 * np.max(slot_obs_soft_values) + 0.30 * np.mean(slot_obs_soft_values))
        hard_slot_obs_norm = float(
            np.sum(slot_obs_distances < OBSTACLE_RADIUS) / max(1, slot_obs_distances.size)
        )
        slot_obs_penalty_norm = float(0.40 * soft_slot_obs_norm + 0.60 * hard_slot_obs_norm)
        return slot_obs_penalty_norm, min_slot_obstacle_dist

    def _generate_initial_positions(self, np_random):
        if self._scattered_init_enabled():
            return self._generate_scattered_initial_positions(np_random)

        leader_pos = self._initial_leader_position(np_random)
        ideal_positions = self._ideal_positions_from_leader(leader_pos)
        noise_scale = INIT_NOISE_BY_STAGE.get(self.training_stage, 0.10)
        noise = np_random.normal(loc=0.0, scale=noise_scale, size=ideal_positions.shape).astype(np.float32)
        noise[self.leader_idx] *= 0.5
        positions = ideal_positions + noise
        return np.clip(positions, -self.world_bound + 0.10, self.world_bound - 0.10).astype(np.float32)

    def _generate_obstacles(self, np_random):
        if not self._obstacles_enabled():
            return np.zeros((NUM_OBSTACLES, 2), dtype=np.float32)

        if not self._random_obstacles_enabled():
            return FIXED_OBSTACLE_POSITIONS.copy()

        obstacles = []
        # Stage 5: sample obstacle centers across the full bounded map.
        # Keep only physical feasibility constraints below: not on initial agents,
        # not on the initial target slots, not on the target, and not overlapping each other.
        # We intentionally do NOT restrict obstacles to a leader-target route band.
        initial_leader_pos = self.agent_positions[self.leader_idx].copy()
        ideal_slots = self._ideal_positions_from_leader(initial_leader_pos)

        for _ in range(8000):
            if len(obstacles) >= NUM_OBSTACLES:
                break
            pos = np_random.uniform(
                low=-self.world_bound + OBSTACLE_RADIUS,
                high=self.world_bound - OBSTACLE_RADIUS,
                size=2,
            ).astype(np.float32)

            if np.min(np.linalg.norm(self.agent_positions - pos[None, :], axis=1)) < OBSTACLE_MIN_DIST_TO_AGENT_INIT:
                continue
            if np.min(np.linalg.norm(ideal_slots - pos[None, :], axis=1)) < OBSTACLE_MIN_DIST_TO_SLOT:
                continue
            if np.linalg.norm(pos - TARGET_POS) < OBSTACLE_MIN_DIST_TO_TARGET:
                continue
            if obstacles:
                existing = np.array(obstacles, dtype=np.float32)
                if np.min(np.linalg.norm(existing - pos[None, :], axis=1)) < OBSTACLE_MIN_DIST_BETWEEN:
                    continue
            obstacles.append(pos)

        if len(obstacles) < NUM_OBSTACLES:
            # Conservative fallback: use fixed obstacles instead of failing a long run.
            return FIXED_OBSTACLE_POSITIONS.copy()
        return np.array(obstacles, dtype=np.float32)

    def _follower_slot_errors(self):
        follower_slots = self._follower_slot_positions()
        follower_positions = self.agent_positions[self.follower_indices]
        errors = np.linalg.norm(follower_positions - follower_slots, axis=1).astype(np.float32)
        slot_errors = np.zeros(self.num_agents, dtype=np.float32)
        slot_errors[self.follower_indices] = errors
        return slot_errors

    def _shape_error(self):
        # Pairwise shape error using leader + three follower offsets.
        leader_pos = self.agent_positions[self.leader_idx]
        ideal_positions = self._ideal_positions_from_leader(leader_pos)
        diffs = []
        for a in range(self.num_agents):
            for b in range(a + 1, self.num_agents):
                actual = np.linalg.norm(self.agent_positions[a] - self.agent_positions[b])
                ideal = np.linalg.norm(ideal_positions[a] - ideal_positions[b])
                diffs.append(abs(actual - ideal))
        return float(np.mean(diffs)) if diffs else 0.0

    def _distance_matrices(self):
        agent_distances = np.linalg.norm(
            self.agent_positions[:, None, :] - self.agent_positions[None, :, :], axis=-1
        ).astype(np.float32)

        if not self._obstacles_enabled():
            obstacle_distances = np.full((self.num_agents, NUM_OBSTACLES), TARGET_DIST_SCALE, dtype=np.float32)
        else:
            obstacle_distances = np.linalg.norm(
                self.agent_positions[:, None, :] - self.obstacle_positions[None, :, :], axis=-1
            ).astype(np.float32)
        return agent_distances, obstacle_distances

    def _safety_term(self, agent_distances, obstacle_distances, i):
        other_dists = np.delete(agent_distances[i], i)
        hard_agent_count = int(np.sum(other_dists < COLLISION_THRESHOLD))
        hard_agent_norm = hard_agent_count / max(1, self.num_agents - 1)
        agent_clearance = np.maximum(other_dists - COLLISION_THRESHOLD, 0.0)
        soft_agent_norm = float(np.mean(np.exp(-agent_clearance / SOFT_AGENT_DECAY_SCALE)))
        agent_penalty_norm = 0.40 * soft_agent_norm + 0.60 * hard_agent_norm

        if self._obstacles_enabled():
            obs_dists = obstacle_distances[i]
            hard_obs_count = int(np.sum(obs_dists < OBSTACLE_RADIUS))
            hard_obs_norm = hard_obs_count / max(1, NUM_OBSTACLES)
            obs_clearance = np.maximum(obs_dists - (OBSTACLE_RADIUS + OBSTACLE_SAFE_MARGIN), 0.0)
            obs_soft_values = np.exp(-obs_clearance / SOFT_OBS_DECAY_SCALE)
            soft_obs_norm = float(0.70 * np.max(obs_soft_values) + 0.30 * np.mean(obs_soft_values))
            min_obstacle_dist = float(np.min(obs_dists))
        else:
            hard_obs_count = 0
            hard_obs_norm = 0.0
            soft_obs_norm = 0.0
            min_obstacle_dist = TARGET_DIST_SCALE

        obs_penalty_norm = 0.40 * soft_obs_norm + 0.60 * hard_obs_norm
        safety_term = -float(SAFETY_AGENT_COEF * agent_penalty_norm + SAFETY_OBS_COEF * obs_penalty_norm)

        return safety_term, {
            "soft_agent_norm": float(soft_agent_norm),
            "hard_agent_count": int(hard_agent_count),
            "hard_agent_norm": float(hard_agent_norm),
            "agent_penalty_norm": float(agent_penalty_norm),
            "soft_obs_norm": float(soft_obs_norm),
            "hard_obs_count": int(hard_obs_count),
            "hard_obs_norm": float(hard_obs_norm),
            "obs_penalty_norm": float(obs_penalty_norm),
            "min_obstacle_dist": float(min_obstacle_dist),
        }

    def _follower_formation_term(self, slot_error, shape_error):
        slot_norm = self._clip01(slot_error / SLOT_ERROR_SCALE)
        shape_norm = self._clip01(shape_error / SHAPE_ERROR_SCALE)
        term = -(FORMATION_SLOT_COEF * slot_norm + FORMATION_SHAPE_COEF * shape_norm)
        return float(term), float(slot_norm), float(shape_norm)

    def reset(self, seed=None, options=None):
        np_random = np.random.RandomState(seed)
        self.agents = self.possible_agents.copy()
        self.current_step = 0
        self.success_hold_count = 0
        self.formation_hold_count = 0
        self.navigation_active = True
        self.used_fallback_formation_center = False

        self.agent_positions = self._generate_initial_positions(np_random)
        self.agent_velocities = np.zeros((self.num_agents, 2), dtype=np.float32)
        self.obstacle_positions = self._generate_obstacles(np_random)

        slot_errors = self._follower_slot_errors()
        shape_error = self._shape_error()
        mean_follower_slot_error = float(np.mean(slot_errors[self.follower_indices]))

        self.last_slot_errors = slot_errors.copy().astype(np.float32)
        self.last_mean_slot_error = mean_follower_slot_error
        self.last_formation_error = mean_follower_slot_error
        self.last_shape_error = shape_error
        self.last_leader_target_distance = float(np.linalg.norm(self.agent_positions[self.leader_idx] - TARGET_POS))
        self.last_team_target_distance = float(np.linalg.norm(np.mean(self.agent_positions, axis=0) - TARGET_POS))

        return self._get_observations(), {agent: {} for agent in self.agents}

    def step(self, actions):
        self.current_step += 1

        for i, agent in enumerate(self.agents):
            action = np.clip(np.asarray(actions[agent], dtype=np.float32), -1.0, 1.0)
            self.agent_velocities[i] = action * ACTION_SCALE
            self.agent_positions[i] = np.clip(
                self.agent_positions[i] + self.agent_velocities[i],
                -self.world_bound,
                self.world_bound,
            )

        agent_distances, obstacle_distances = self._distance_matrices()
        slot_errors = self._follower_slot_errors()
        follower_slot_errors = slot_errors[self.follower_indices]
        mean_follower_slot_error = float(np.mean(follower_slot_errors))
        max_follower_slot_error = float(np.max(follower_slot_errors))
        shape_error = self._shape_error()

        if self._scattered_init_enabled():
            denom = max(FORMATION_GATE_ERROR_HIGH - FORMATION_GATE_ERROR_LOW, 1e-6)
            formation_gate = float(np.clip((FORMATION_GATE_ERROR_HIGH - mean_follower_slot_error) / denom, 0.0, 1.0))
        else:
            formation_gate = 1.0

        leader_pos = self.agent_positions[self.leader_idx]
        leader_target_distance = float(np.linalg.norm(leader_pos - TARGET_POS))
        leader_target_progress = self.last_leader_target_distance - leader_target_distance
        leader_progress_norm = self._safe_norm(leader_target_progress, ACTION_SCALE, -1.0, 1.0)
        leader_target_dist_norm = self._clip01(leader_target_distance / TARGET_DIST_SCALE)
        final_target_gate = 0.0
        if leader_target_distance < FINAL_TARGET_RADIUS:
            final_target_gate = float(
                np.clip(
                    (FINAL_TARGET_RADIUS - leader_target_distance)
                    / max(FINAL_TARGET_RADIUS - SUCCESS_LEADER_TARGET_THRESHOLD, 1e-6),
                    0.0,
                    1.0,
                )
            )
        final_target_norm = self._clip01(leader_target_distance / FINAL_TARGET_RADIUS)
        slot_obs_penalty_norm, min_slot_obstacle_dist = self._slot_obstacle_penalty()

        team_centroid = np.mean(self.agent_positions, axis=0).astype(np.float32)
        team_target_distance = float(np.linalg.norm(team_centroid - TARGET_POS))

        if self._obstacles_enabled():
            hard_obs_matrix = obstacle_distances < OBSTACLE_RADIUS
            any_obstacle_collision = bool(np.any(hard_obs_matrix))
            total_obstacle_collision_count = int(np.sum(hard_obs_matrix))
            min_obstacle_dist = float(np.min(obstacle_distances))
        else:
            any_obstacle_collision = False
            total_obstacle_collision_count = 0
            min_obstacle_dist = TARGET_DIST_SCALE
        nearest_obs_clearance = max(min_obstacle_dist - OBSTACLE_RADIUS, 0.0)

        follower_dists_to_leader = np.linalg.norm(
            self.agent_positions[self.follower_indices] - leader_pos[None, :], axis=1
        )
        follower_detach_norm = float(
            np.mean(np.maximum(follower_dists_to_leader - MAX_FOLLOWER_DETACH_DIST, 0.0) / MAX_FOLLOWER_DETACH_DIST)
        )

        success_now = (
            leader_target_distance < SUCCESS_LEADER_TARGET_THRESHOLD
            and mean_follower_slot_error < SUCCESS_MEAN_FOLLOWER_SLOT_THRESHOLD
            and max_follower_slot_error < SUCCESS_MAX_FOLLOWER_SLOT_THRESHOLD
            and not any_obstacle_collision
        )
        if success_now:
            self.success_hold_count += 1
        else:
            self.success_hold_count = 0
        success_termination = self.success_hold_count >= SUCCESS_HOLD_STEPS

        if mean_follower_slot_error < SUCCESS_MEAN_FOLLOWER_SLOT_THRESHOLD:
            self.formation_hold_count += 1
        else:
            self.formation_hold_count = 0

        rewards = {agent: 0.0 for agent in self.agents}
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}

        reward_components = {
            "formation": np.zeros(self.num_agents, dtype=np.float32),
            "collision": np.zeros(self.num_agents, dtype=np.float32),
            "goal": np.zeros(self.num_agents, dtype=np.float32),
            "step": np.full(self.num_agents, STEP_PENALTY, dtype=np.float32),
            "success_bonus": np.zeros(self.num_agents, dtype=np.float32),
            "detach": np.zeros(self.num_agents, dtype=np.float32),
            "target_distance": np.zeros(self.num_agents, dtype=np.float32),
            "final_target": np.zeros(self.num_agents, dtype=np.float32),
            "obstacle_collision": np.zeros(self.num_agents, dtype=np.float32),
            "wait": np.zeros(self.num_agents, dtype=np.float32),
        }

        safety_infos = []
        formation_slot_norms = np.zeros(self.num_agents, dtype=np.float32)
        formation_shape_norms = np.zeros(self.num_agents, dtype=np.float32)

        if self._scattered_init_enabled():
            leader_goal_scale_info = LEADER_MIN_PROGRESS_SCALE_WHEN_UNFORMED + (
                1.0 - LEADER_MIN_PROGRESS_SCALE_WHEN_UNFORMED
            ) * formation_gate
            leader_speed_norm_info = float(
                np.clip(
                    np.linalg.norm(self.agent_velocities[self.leader_idx]) / (ACTION_SCALE * np.sqrt(2)),
                    0.0,
                    1.0,
                )
            )
            follower_formation_w_info = FOLLOWER_FORMATION_W * (
                1.0 + FOLLOWER_FORMATION_EXTRA_SCALE_WHEN_UNFORMED * (1.0 - formation_gate)
            )
        else:
            leader_goal_scale_info = 1.0
            leader_speed_norm_info = 0.0
            follower_formation_w_info = FOLLOWER_FORMATION_W

        for i, agent in enumerate(self.agents):
            safety_term, safety_info = self._safety_term(agent_distances, obstacle_distances, i)
            safety_infos.append(safety_info)

            if i == self.leader_idx:
                if self._scattered_init_enabled():
                    leader_goal_scale = LEADER_MIN_PROGRESS_SCALE_WHEN_UNFORMED + (
                        1.0 - LEADER_MIN_PROGRESS_SCALE_WHEN_UNFORMED
                    ) * formation_gate
                    leader_speed_norm = float(
                        np.clip(
                            np.linalg.norm(self.agent_velocities[self.leader_idx]) / (ACTION_SCALE * np.sqrt(2)),
                            0.0,
                            1.0,
                        )
                    )
                    wait_reward = -LEADER_WAIT_W * (1.0 - formation_gate) * leader_speed_norm
                    detach_w = LEADER_DETACH_W * (1.0 + (1.0 - formation_gate))
                else:
                    leader_goal_scale = 1.0
                    leader_speed_norm = 0.0
                    wait_reward = 0.0
                    detach_w = LEADER_DETACH_W

                # Leader reward is organized into four semantic terms:
                #   navigation     = progress + weak target-distance shaping
                #   final approach = near-target convergence shaping
                #   safety         = self safety + follower-slot obstacle risk
                #   coordination   = detach + wait penalties
                progress_reward = LEADER_GOAL_PROGRESS_W * leader_goal_scale * leader_progress_norm
                target_dist_reward = -LEADER_TARGET_DIST_W * leader_target_dist_norm
                final_target_reward = -LEADER_FINAL_TARGET_W * final_target_gate * final_target_norm

                # Fold follower-slot obstacle risk into the existing leader safety term rather than
                # creating an extra standalone reward component. This preserves a compact reward
                # structure while still making the leader leave clearance for the whole formation.
                leader_safety_term = safety_term - (LEADER_SLOT_OBS_W / max(LEADER_SAFETY_W, 1e-8)) * slot_obs_penalty_norm
                safety_reward = LEADER_SAFETY_W * leader_safety_term
                detach_reward = -detach_w * follower_detach_norm
                coordination_reward = detach_reward + wait_reward

                reward_components["goal"][i] = progress_reward
                reward_components["target_distance"][i] = target_dist_reward
                reward_components["final_target"][i] = final_target_reward
                reward_components["collision"][i] = safety_reward
                reward_components["detach"][i] = coordination_reward
            else:
                formation_term, slot_norm, shape_norm = self._follower_formation_term(slot_errors[i], shape_error)
                formation_slot_norms[i] = slot_norm
                formation_shape_norms[i] = shape_norm
                if self._scattered_init_enabled():
                    follower_formation_w = FOLLOWER_FORMATION_W * (
                        1.0 + FOLLOWER_FORMATION_EXTRA_SCALE_WHEN_UNFORMED * (1.0 - formation_gate)
                    )
                else:
                    follower_formation_w = FOLLOWER_FORMATION_W
                # Follower reward is organized into three semantic terms:
                #   formation = slot tracking + shape keeping
                #   safety    = obstacle avoidance + agent-agent collision avoidance
                #   progress  = small team progress signal from the leader
                formation_reward = follower_formation_w * formation_term
                safety_reward = FOLLOWER_SAFETY_W * safety_term
                team_progress_reward = FOLLOWER_TEAM_PROGRESS_W * leader_progress_norm

                reward_components["formation"][i] = formation_reward
                reward_components["collision"][i] = safety_reward
                reward_components["goal"][i] = team_progress_reward

            if safety_info["hard_obs_count"] > 0:
                reward_components["obstacle_collision"][i] = OBSTACLE_COLLISION_TERMINATION_PENALTY

            if success_termination:
                reward_components["success_bonus"][i] = SUCCESS_BONUS

            total_reward = float(sum(component[i] for component in reward_components.values()))
            rewards[agent] = total_reward

        if success_termination:
            terminations = {agent: True for agent in self.agents}
        if any_obstacle_collision:
            terminations = {agent: True for agent in self.agents}
        if self.current_step >= self.max_cycles:
            truncations = {agent: True for agent in self.agents}

        self.last_slot_errors = slot_errors.copy().astype(np.float32)
        self.last_mean_slot_error = mean_follower_slot_error
        self.last_formation_error = mean_follower_slot_error
        self.last_shape_error = shape_error
        self.last_leader_target_distance = leader_target_distance
        self.last_team_target_distance = team_target_distance

        observations = self._get_observations()
        infos = {}
        total_agent_collision_count = 0
        for i in range(self.num_agents):
            total_agent_collision_count += int(safety_infos[i]["hard_agent_count"])
        total_agent_collision_count //= 2

        for i, agent in enumerate(self.agents):
            is_leader = i == self.leader_idx
            info = {
                "role": "leader" if is_leader else "follower",
                "role_id": 1.0 if is_leader else 0.0,
                "training_stage": int(self.training_stage),
                "slot_error": float(slot_errors[i]),
                "mean_slot_error": float(mean_follower_slot_error),
                "formation_error": float(mean_follower_slot_error),
                "nav_slot_error": float(slot_errors[i]),
                "mean_nav_slot_error": float(mean_follower_slot_error),
                "max_follower_slot_error": float(max_follower_slot_error),
                "follower_shape_error": float(shape_error),
                "nav_shape_error": float(shape_error),
                "formation_slot_norm": float(formation_slot_norms[i]),
                "formation_shape_norm": float(formation_shape_norms[i]),
                "formation_hold_count": int(self.formation_hold_count),
                "success_hold_count": int(self.success_hold_count),
                "leader_target_distance": float(leader_target_distance),
                "leader_target_progress": float(leader_target_progress),
                "leader_progress_norm": float(leader_progress_norm),
                "leader_target_dist_norm": float(leader_target_dist_norm),
                "final_target_gate": float(final_target_gate),
                "final_target_norm": float(final_target_norm),
                "slot_obstacle_penalty_norm": float(slot_obs_penalty_norm),
                "min_slot_obstacle_dist": float(min_slot_obstacle_dist),
                "team_target_distance": float(team_target_distance),
                "follower_detach_norm": float(follower_detach_norm),
                "formation_gate": float(formation_gate),
                "leader_goal_scale": float(leader_goal_scale_info),
                "leader_speed_norm": float(leader_speed_norm_info),
                "follower_formation_w": float(follower_formation_w_info),
                "online_formation_stage": 1.0 if self._scattered_init_enabled() else 0.0,
                "navigation_active": 1.0,
                "formation_near_ready": 1.0 if mean_follower_slot_error < SUCCESS_MEAN_FOLLOWER_SLOT_THRESHOLD else 0.0,
                "used_fallback_formation_center": 1.0 if self.used_fallback_formation_center else 0.0,
                "collision_mode": "leader_follower",
                "just_unlocked_navigation": 0.0,
                "reward_nav_unlock_bonus": 0.0,
                "agent_collision_num": int(safety_infos[i]["hard_agent_count"]),
                "total_agent_collisions": int(total_agent_collision_count),
                "obstacle_collision_num": int(safety_infos[i]["hard_obs_count"]),
                "total_obstacle_collisions": int(total_obstacle_collision_count),
                "any_obstacle_collision": 1.0 if any_obstacle_collision else 0.0,
                "min_obstacle_dist": float(min_obstacle_dist),
                "agent_min_obstacle_dist": float(safety_infos[i]["min_obstacle_dist"]),
                "nearest_obs_clearance": float(nearest_obs_clearance),
                "near_obstacle_for_detour": 1.0 if nearest_obs_clearance < (OBSTACLE_SAFE_MARGIN + OBSTACLE_RADIUS) else 0.0,
                "safety_agent_penalty_norm": float(safety_infos[i]["agent_penalty_norm"]),
                "safety_obs_penalty_norm": float(safety_infos[i]["obs_penalty_norm"]),
                "soft_agent_norm": float(safety_infos[i]["soft_agent_norm"]),
                "soft_obs_norm": float(safety_infos[i]["soft_obs_norm"]),
                "hard_agent_norm": float(safety_infos[i]["hard_agent_norm"]),
                "hard_obs_norm": float(safety_infos[i]["hard_obs_norm"]),
                "reward_formation": float(reward_components["formation"][i]),
                "reward_collision": float(reward_components["collision"][i]),
                "reward_safety": float(reward_components["collision"][i]),
                "reward_goal": float(reward_components["goal"][i]),
                "reward_target_distance": float(reward_components["target_distance"][i]),
                "reward_final_target": float(reward_components["final_target"][i]),
                "reward_detach": float(reward_components["detach"][i]),
                "reward_wait": float(reward_components["wait"][i]),
                "reward_step": float(reward_components["step"][i]),
                "reward_obstacle_collision": float(reward_components["obstacle_collision"][i]),
                "reward_success_bonus": float(reward_components["success_bonus"][i]),
                "reward_total": float(rewards[agent]),
                "goal_progress_term": float(leader_progress_norm),
                "goal_stall_term": 0.0,
                "goal_mixed_term": float(leader_progress_norm),
                "reward_goal_progress": float((LEADER_GOAL_PROGRESS_W * leader_goal_scale_info * leader_progress_norm) if is_leader else (FOLLOWER_TEAM_PROGRESS_W * leader_progress_norm)),
                "reward_goal_stall": 0.0,
                "is_success": 1.0 if success_termination else 0.0,
            }
            infos[agent] = info

        if self.render_mode == "human":
            self.render()

        return observations, rewards, terminations, truncations, infos

    def _get_observations(self):
        observations = {}
        leader_pos = self.agent_positions[self.leader_idx]
        follower_slots = self._follower_slot_positions()

        for i, agent in enumerate(self.agents):
            own_pos = self.agent_positions[i]
            own_vel = self.agent_velocities[i]

            if i == self.leader_idx:
                # Pseudo slot for dimensional consistency only; leader reward never uses it.
                own_slot_rel = np.zeros(2, dtype=np.float32)
            else:
                follower_slot_idx = self.follower_indices.index(i)
                own_slot_rel = follower_slots[follower_slot_idx] - own_pos

            leader_rel = leader_pos - own_pos
            target_rel = TARGET_POS - own_pos
            other_positions = np.delete(self.agent_positions, i, axis=0)
            other_agents_rel = (other_positions - own_pos).flatten()
            role_flag = np.array([1.0 if i == self.leader_idx else 0.0], dtype=np.float32)

            if self._obstacles_enabled():
                obstacles_rel = (self.obstacle_positions - own_pos).flatten()
            else:
                obstacles_rel = np.zeros(2 * NUM_OBSTACLES, dtype=np.float32)

            obs = np.concatenate(
                [
                    own_pos,
                    own_vel,
                    own_slot_rel,
                    leader_rel,
                    target_rel,
                    other_agents_rel,
                    role_flag,
                    obstacles_rel,
                ]
            ).astype(np.float32)
            observations[agent] = obs

        return observations

    def render(self):
        if self.fig is None:
            plt.rcParams["figure.figsize"] = (8, 8)
            self.fig, self.ax = plt.subplots(figsize=(8, 8), dpi=100)

        self.ax.clear()
        self.ax.set_xlim(-self.world_bound - 0.5, self.world_bound + 0.5)
        self.ax.set_ylim(-self.world_bound - 0.5, self.world_bound + 0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.3)

        episode_label = getattr(self, "current_episode", None)
        episode_text = f"回合 {episode_label}" if episode_label is not None else "回合 ?"
        self.ax.set_title(
            f"领航-跟随编队（{episode_text}，步数 {self.current_step}） | "
            f"领航者-目标距离: {self.last_leader_target_distance:.3f} | "
            f"期望位置误差: {self.last_mean_slot_error:.3f}"
        )

        self.ax.add_patch(
            patches.Rectangle(
                (-self.world_bound, -self.world_bound),
                2 * self.world_bound,
                2 * self.world_bound,
                linewidth=2,
                edgecolor="black",
                facecolor="none",
            )
        )

        self.ax.scatter(TARGET_POS[0], TARGET_POS[1], s=180, c="red", marker="*", label="目标点")

        if self._obstacles_enabled():
            for j, obs_pos in enumerate(self.obstacle_positions):
                circle = patches.Circle(
                    obs_pos,
                    OBSTACLE_RADIUS,
                    color="darkred",
                    alpha=0.85,
                    label="障碍物" if j == 0 else None,
                )
                self.ax.add_patch(circle)
                soft_circle = patches.Circle(
                    obs_pos,
                    OBSTACLE_RADIUS + OBSTACLE_SAFE_MARGIN,
                    color="darkred",
                    alpha=0.15,
                    fill=False,
                    linestyle="--",
                    label="障碍物安全边界" if j == 0 else None,
                )
                self.ax.add_patch(soft_circle)

        leader_pos = self.agent_positions[self.leader_idx]
        follower_slots = self._follower_slot_positions()
        self.ax.scatter(
            follower_slots[:, 0], follower_slots[:, 1],
            s=90, c="black", marker="s", alpha=0.55, label="跟随者期望位置"
        )
        for slot in follower_slots:
            self.ax.plot([leader_pos[0], slot[0]], [leader_pos[1], slot[1]], color="gray", linestyle="--", linewidth=1)

        colors = ["blue", "green", "orange", "purple"]
        labels = ["领航者", "跟随者1", "跟随者2", "跟随者3"]
        for i, pos in enumerate(self.agent_positions):
            marker = "^" if i == self.leader_idx else "o"
            self.ax.scatter(pos[0], pos[1], s=120, c=colors[i], marker=marker, label=f"智能体{i}（{labels[i]}）")
            vel = self.agent_velocities[i]
            self.ax.arrow(pos[0], pos[1], vel[0], vel[1], head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i])

        self.ax.legend(loc="upper left", fontsize=8)
        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        rgb_array = np.array(rgba[..., :3], copy=True)

        if self.render_mode == "human":
            plt.pause(0.01)
        return rgb_array

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]
