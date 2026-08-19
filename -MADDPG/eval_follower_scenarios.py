"""
Evaluate a trained follower policy in hand-designed scenarios.

Scenarios:
1) Followers start near their slots; target moves straight; leader points to target.
2) Followers start near their slots; target moves straight, then turns left 90 degrees.
3) Followers start farther from their slots; target moves straight, then turns left 90 degrees.
4) Followers start with position and yaw offsets; target moves straight, then turns left 90 degrees.
"""

import argparse
import math
import os
from datetime import datetime

import imageio
import numpy as np

from maddpg import MADDPG
from envs.follower_slot_tracking_v0 import (
    DT,
    FollowerSlotTrackingEnv,
    LEADER_FOLLOW_DIST,
    SIDE_DIST,
    FOLLOWER_MAX_ANGULAR,
    SUCCESS_HOLD_STEPS,
    TARGET_SPEED,
    _unit_from_angle,
    _wrap_angle,
)
from utils.env import get_env_info


SCENARIOS = {
    "scenario1_near_straight": {
        "description": "go2/go3 near slots, target straight, leader faces target",
        "scatter": 0.20,
        "yaw_offsets": (0.0, 0.0),
        "turn_step": None,
        "post_turn_heading": None,
    },
    "scenario2_near_left90": {
        "description": "go2/go3 near slots, target straight then 90 deg left, leader faces target",
        "scatter": 0.20,
        "yaw_offsets": (0.0, 0.0),
        "turn_step": 70,
        "post_turn_heading": math.pi / 2.0,
    },
    "scenario3_far_left90": {
        "description": "go2/go3 farther from slots, target straight then 90 deg left, leader faces target",
        "scatter": 1.20,
        "yaw_offsets": (0.0, 0.0),
        "turn_step": 70,
        "post_turn_heading": math.pi / 2.0,
    },
    "scenario4_offset_yaw_left90": {
        "description": "go2/go3 position offsets and initial yaw offsets, target straight then 90 deg left, leader faces target",
        "scatter": 0.80,
        "yaw_offsets": (math.radians(55.0), math.radians(-55.0)),
        "turn_step": 70,
        "post_turn_heading": math.pi / 2.0,
    },
}


class FixedScenarioFollowerEnv(FollowerSlotTrackingEnv):
    def __init__(self, scenario, **kwargs):
        super().__init__(**kwargs)
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario}")
        self.scenario_name = scenario
        self.scenario_cfg = SCENARIOS[scenario]

    def reset(self, seed=None, options=None):
        obs, infos = super().reset(seed=seed, options=options)

        rng = np.random.RandomState(seed)
        self.target_speed = TARGET_SPEED
        self.side_dist = SIDE_DIST
        self.leader_follow_dist = LEADER_FOLLOW_DIST
        self.follower_max_linear = 0.60
        self.follower_max_angular = FOLLOWER_MAX_ANGULAR
        self.leader_max_linear = 0.60
        self.leader_lag_steps = 0
        self.obs_noise_std = 0.0

        self.target_pos = np.array([-3.0, 0.0], dtype=np.float32)
        self.target_heading = 0.0
        self.target_vel = self.target_speed * _unit_from_angle(self.target_heading)

        forward = _unit_from_angle(self.target_heading)
        self.leader_pos = self.target_pos - self.leader_follow_dist * forward
        self.leader_yaw = self.target_heading
        self.leader_vel = self.target_vel.copy()
        self.leader_target_history = [self.target_pos.copy()]

        self._update_slots()
        scatter_mag = float(self.scenario_cfg["scatter"])
        scatter = rng.uniform(low=-scatter_mag, high=scatter_mag, size=(2, 2)).astype(np.float32)
        self.follower_pos[0] = self.slots[0] + scatter[0]
        self.follower_pos[1] = self.slots[1] + scatter[1]
        self.follower_vel[:] = 0.0
        self.follower_cmd_linear[:] = 0.0
        self.follower_cmd_angular[:] = 0.0
        formation_yaw = self._formation_yaw()
        yaw_offsets = self.scenario_cfg.get("yaw_offsets", (0.0, 0.0))
        self.follower_yaw[0] = _wrap_angle(formation_yaw + float(yaw_offsets[0]))
        self.follower_yaw[1] = _wrap_angle(formation_yaw + float(yaw_offsets[1]))
        self.prev_actions[:] = 0.0
        self.last_actions[:] = 0.0
        self.last_slots = self.slots.copy()

        return self._get_observations(), infos

    def _step_target(self):
        turn_step = self.scenario_cfg["turn_step"]
        if turn_step is not None and self.current_step >= int(turn_step):
            self.target_heading = float(self.scenario_cfg["post_turn_heading"])

        self.target_heading = _wrap_angle(self.target_heading)
        self.target_vel = self.target_speed * _unit_from_angle(self.target_heading)
        self.target_pos = self.target_pos + self.target_vel * DT

        self.leader_target_history.append(self.target_pos.copy())
        self.leader_target_history = self.leader_target_history[-10:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=700)
    parser.add_argument("--training-stage", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--follower-action-mode",
        choices=["velocity", "accel"],
        default="accel",
        help="Use 'velocity' for speed-output models, 'accel' for acceleration-output models.",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Run one fixed scenario or all scenarios.",
    )
    parser.add_argument("--output-dir", default="./gifs/follower_slot_tracking_v0/scenarios")
    return parser.parse_args()


def run_one_scenario(args, scenario_name, agents, action_sizes, action_low, action_high, state_sizes):
    env = FixedScenarioFollowerEnv(
        scenario=scenario_name,
        max_cycles=args.max_steps,
        render_mode="rgb_array",
        training_stage=args.training_stage,
        follower_action_mode=args.follower_action_mode,
    )
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=(128, 128),
        action_low=action_low,
        action_high=action_high,
    )
    maddpg.load(args.model_path)

    obs, _ = env.reset(seed=args.seed)
    frames = [env.render()]
    step = 0
    last_info = {}
    total_reward = np.zeros(len(agents), dtype=np.float64)

    while step < args.max_steps:
        states = [np.asarray(obs[agent], dtype=np.float32) for agent in agents]
        actions_list = maddpg.act(states, add_noise=False)
        actions = {agent: action for agent, action in zip(agents, actions_list)}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        total_reward += np.array([rewards[agent] for agent in agents], dtype=np.float64)
        last_info = infos[agents[0]]
        frames.append(env.render())
        step += 1

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(args.output_dir, f"{scenario_name}_{timestamp}.gif")
    imageio.mimsave(gif_path, frames, duration=0.08)
    env.close()

    print(
        f"{scenario_name}: {SCENARIOS[scenario_name]['description']}\n"
        f"  gif: {gif_path}\n"
        f"  success={bool(last_info.get('is_success', 0.0))} "
        f"steps={step} "
        f"reward_sum={np.sum(total_reward):.2f} "
        f"mean_slot_error={float(last_info.get('mean_slot_error', np.nan)):.3f} "
        f"max_slot_error={float(last_info.get('max_follower_slot_error', np.nan)):.3f} "
        f"mean_yaw_error={float(last_info.get('mean_follower_yaw_error', np.nan)):.3f} "
        f"max_yaw_error={float(last_info.get('max_follower_yaw_error', np.nan)):.3f} "
        f"hold={int(last_info.get('success_hold_count', 0))}/{SUCCESS_HOLD_STEPS}"
    )


def main():
    args = parse_args()
    env_name = "follower_slot_tracking_v0"
    agents, _, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=env_name,
        max_steps=args.max_steps,
        apply_padding=False,
        training_stage=args.training_stage,
    )

    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    for scenario_name in scenario_names:
        run_one_scenario(args, scenario_name, agents, action_sizes, action_low, action_high, state_sizes)


if __name__ == "__main__":
    main()
