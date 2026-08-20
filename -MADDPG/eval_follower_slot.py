"""
Evaluate a trained follower_slot_tracking_v0 policy.
"""

import argparse
import os
from datetime import datetime

import imageio
import numpy as np

from maddpg import MADDPG
from envs.follower_slot_tracking_v0 import FollowerSlotTrackingEnv, SUCCESS_HOLD_STEPS
from utils.env import get_env_info


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--training-stage", type=int, default=5, choices=[1, 2, 3, 4, 5])
    parser.add_argument(
        "--follower-action-mode",
        choices=["velocity", "accel"],
        default="accel",
        help="Use 'velocity' for speed-output models, 'accel' for acceleration-output models.",
    )
    parser.add_argument(
        "--eval-init-scatter",
        type=float,
        default=None,
        help="Override follower initial scatter range for harder visual evaluation.",
    )
    parser.add_argument("--create-gif", action="store_true")
    parser.add_argument("--output-dir", default="./gifs/follower_slot_tracking_v0")
    return parser.parse_args()


def main():
    args = parse_args()
    env_name = "follower_slot_tracking_v0"
    agents, _, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=env_name,
        max_steps=args.max_steps,
        apply_padding=False,
        training_stage=args.training_stage,
    )
    env = FollowerSlotTrackingEnv(
        max_cycles=args.max_steps,
        render_mode="rgb_array" if args.create_gif else None,
        training_stage=args.training_stage,
        eval_init_scatter=args.eval_init_scatter,
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

    episode_rewards = []
    final_mean_errors = []
    final_max_errors = []
    final_mean_yaw_errors = []
    final_max_yaw_errors = []
    final_hold_counts = []
    successes = []
    lengths = []
    all_frames = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = np.zeros(len(agents), dtype=np.float64)
        done = False
        step = 0
        last_info = {}

        while not done and step < args.max_steps:
            states = [np.asarray(obs[agent], dtype=np.float32) for agent in agents]
            actions_list = maddpg.act(states, add_noise=False)
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            total_reward += np.array([rewards[agent] for agent in agents], dtype=np.float64)
            last_info = infos[agents[0]]
            done = any(terminations[agent] or truncations[agent] for agent in agents)
            step += 1

            if args.create_gif and episode < min(args.episodes, 3):
                all_frames.append(env.render())

        episode_rewards.append(total_reward)
        final_mean_errors.append(float(last_info.get("mean_slot_error", np.nan)))
        final_max_errors.append(float(last_info.get("max_follower_slot_error", np.nan)))
        final_mean_yaw_errors.append(float(last_info.get("mean_follower_yaw_error", np.nan)))
        final_max_yaw_errors.append(float(last_info.get("max_follower_yaw_error", np.nan)))
        final_hold_counts.append(float(last_info.get("success_hold_count", np.nan)))
        successes.append(float(last_info.get("is_success", 0.0)))
        lengths.append(step)
        print(
            f"episode={episode + 1:02d} "
            f"success={bool(successes[-1])} "
            f"steps={step:03d} "
            f"reward_sum={np.sum(total_reward):.2f} "
            f"mean_slot_error={final_mean_errors[-1]:.3f} "
            f"max_slot_error={final_max_errors[-1]:.3f} "
            f"mean_yaw_error={final_mean_yaw_errors[-1]:.3f} "
            f"max_yaw_error={final_max_yaw_errors[-1]:.3f} "
            f"hold={final_hold_counts[-1]:.0f}"
        )

    rewards = np.asarray(episode_rewards)
    print("\nSummary")
    print(f"  episodes: {args.episodes}")
    print(f"  training_stage: {args.training_stage}")
    print(f"  eval_init_scatter: {args.eval_init_scatter}")
    print(f"  follower_action_mode: {args.follower_action_mode}")
    print(f"  success_rate: {np.mean(successes) * 100:.1f}%")
    print(f"  mean_total_reward: {np.mean(np.sum(rewards, axis=1)):.2f}")
    print(f"  mean_final_slot_error: {np.nanmean(final_mean_errors):.3f} m")
    print(f"  mean_final_max_slot_error: {np.nanmean(final_max_errors):.3f} m")
    print(f"  mean_final_yaw_error: {np.nanmean(final_mean_yaw_errors):.3f} rad")
    print(f"  mean_final_max_yaw_error: {np.nanmean(final_max_yaw_errors):.3f} rad")
    print(f"  mean_final_success_hold_count: {np.nanmean(final_hold_counts):.1f} / {SUCCESS_HOLD_STEPS}")
    print(f"  mean_episode_length: {np.mean(lengths):.1f} steps")

    if args.create_gif and all_frames:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(args.output_dir, f"eval_{timestamp}.gif")
        imageio.mimsave(gif_path, all_frames, duration=0.08)
        print(f"  gif: {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
