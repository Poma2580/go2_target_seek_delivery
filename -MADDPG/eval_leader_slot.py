"""
Evaluate a trained leader_slot_tracking_v0 policy.
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
    FOLLOWER_ANGULAR_ACCEL,
    FOLLOWER_LINEAR_ACCEL,
    FOLLOWER_TURN_SLOWDOWN,
    SUCCESS_HOLD_STEPS,
)
from envs.leader_slot_tracking_v0 import LeaderSlotTrackingEnv
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
    parser.add_argument("--output-dir", default="./gifs/leader_slot_tracking_v0")
    parser.add_argument("--safety-enable", action="store_true", help="Enable follower collision safety layer during eval.")
    parser.add_argument("--safety-follower-safe-dist", type=float, default=1.20)
    parser.add_argument("--safety-follower-hard-dist", type=float, default=0.85)
    parser.add_argument("--safety-leader-safe-dist", type=float, default=1.35)
    parser.add_argument("--safety-leader-hard-dist", type=float, default=1.00)
    parser.add_argument("--safety-max-angular-correction", type=float, default=0.25)
    parser.add_argument("--safety-hard-angular-correction", type=float, default=0.45)
    parser.add_argument("--safety-min-linear-scale", type=float, default=0.30)
    parser.add_argument("--safety-hard-linear", type=float, default=0.08)
    return parser.parse_args()


def _body_frame(yaw, vec):
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    return np.array([c * vec[0] + s * vec[1], -s * vec[0] + c * vec[1]], dtype=np.float32)


def _safety_pair_adjustment(
    pos,
    yaw,
    obstacle_pos,
    dist,
    safe_dist,
    hard_dist,
    fallback_sign,
    max_angular_correction,
    hard_angular_correction,
    min_linear_scale,
):
    if dist >= safe_dist:
        return 1.0, 0.0, False, 0.0

    denom = max(safe_dist - hard_dist, 1e-6)
    severity = float(np.clip((safe_dist - dist) / denom, 0.0, 1.0))
    hard = dist <= hard_dist

    obstacle_rel_body = _body_frame(yaw, obstacle_pos - pos)
    away_body = -obstacle_rel_body
    away_angle = math.atan2(float(away_body[1]), float(away_body[0]))
    steer_unit = float(np.clip(away_angle / (0.5 * math.pi), -1.0, 1.0))
    if abs(steer_unit) < 0.15:
        steer_unit = fallback_sign

    max_corr = hard_angular_correction if hard else max_angular_correction
    angular_corr = severity * max_corr * steer_unit
    linear_scale = max(min_linear_scale, 1.0 - 0.80 * severity)
    return linear_scale, angular_corr, True, severity


def _action_to_command(env, idx, action, follower_action_mode):
    action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    if follower_action_mode == "accel":
        linear = float(
            np.clip(
                env.follower_cmd_linear[idx] + float(action[0]) * FOLLOWER_LINEAR_ACCEL * DT,
                0.0,
                env.follower_max_linear,
            )
        )
        angular = float(
            np.clip(
                env.follower_cmd_angular[idx] + float(action[1]) * FOLLOWER_ANGULAR_ACCEL * DT,
                -env.follower_max_angular,
                env.follower_max_angular,
            )
        )
    else:
        linear = 0.5 * (float(action[0]) + 1.0) * env.follower_max_linear
        angular = float(action[1]) * env.follower_max_angular

    if FOLLOWER_TURN_SLOWDOWN:
        turn_ratio = min(abs(angular) / max(env.follower_max_angular, 1e-6), 1.0)
        linear = min(linear, env.follower_max_linear * max(0.25, 1.0 - 0.65 * turn_ratio))
    return linear, angular


def _command_to_action(env, idx, linear, angular, follower_action_mode):
    if follower_action_mode == "accel":
        action0 = (float(linear) - float(env.follower_cmd_linear[idx])) / max(FOLLOWER_LINEAR_ACCEL * DT, 1e-6)
        action1 = (float(angular) - float(env.follower_cmd_angular[idx])) / max(FOLLOWER_ANGULAR_ACCEL * DT, 1e-6)
    else:
        action0 = 2.0 * float(linear) / max(env.follower_max_linear, 1e-6) - 1.0
        action1 = float(angular) / max(env.follower_max_angular, 1e-6)
    return np.clip(np.array([action0, action1], dtype=np.float32), -1.0, 1.0)


def _apply_eval_safety_layer(env, actions_list, args):
    info = {
        "intervened": False,
        "inter_dist": float(np.linalg.norm(env.follower_pos[0] - env.follower_pos[1])),
        "leader_dists": [
            float(np.linalg.norm(env.follower_pos[0] - env.leader_pos)),
            float(np.linalg.norm(env.follower_pos[1] - env.leader_pos)),
        ],
        "agent_intervened": [False, False],
    }
    if not args.safety_enable:
        return actions_list, info

    adjusted_actions = []
    for idx, action in enumerate(actions_list):
        other_idx = 1 - idx
        fallback_sign = 1.0 if idx == 0 else -1.0
        linear, angular = _action_to_command(env, idx, action, args.follower_action_mode)
        linear_scale = 1.0
        angular_corr = 0.0
        severity = 0.0
        intervened = False

        scale, corr, active, sev = _safety_pair_adjustment(
            env.follower_pos[idx],
            float(env.follower_yaw[idx]),
            env.follower_pos[other_idx],
            info["inter_dist"],
            args.safety_follower_safe_dist,
            args.safety_follower_hard_dist,
            fallback_sign,
            args.safety_max_angular_correction,
            args.safety_hard_angular_correction,
            args.safety_min_linear_scale,
        )
        if active:
            linear_scale = min(linear_scale, scale)
            angular_corr += corr
            severity = max(severity, sev)
            intervened = True

        scale, corr, active, sev = _safety_pair_adjustment(
            env.follower_pos[idx],
            float(env.follower_yaw[idx]),
            env.leader_pos,
            info["leader_dists"][idx],
            args.safety_leader_safe_dist,
            args.safety_leader_hard_dist,
            fallback_sign,
            args.safety_max_angular_correction,
            args.safety_hard_angular_correction,
            args.safety_min_linear_scale,
        )
        if active:
            linear_scale = min(linear_scale, scale)
            angular_corr += corr
            severity = max(severity, sev)
            intervened = True

        linear = float(np.clip(linear * linear_scale, 0.0, env.follower_max_linear))
        angular = float(np.clip(angular + angular_corr, -env.follower_max_angular, env.follower_max_angular))
        if severity >= 1.0:
            linear = min(linear, args.safety_hard_linear)

        adjusted_actions.append(_command_to_action(env, idx, linear, angular, args.follower_action_mode))
        info["agent_intervened"][idx] = bool(intervened)
        info["intervened"] = info["intervened"] or bool(intervened)

    return adjusted_actions, info


def main():
    args = parse_args()
    env_name = "leader_slot_tracking_v0"
    agents, _, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=env_name,
        max_steps=args.max_steps,
        apply_padding=False,
        training_stage=args.training_stage,
    )
    env = LeaderSlotTrackingEnv(
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
    safety_interventions = []
    min_inter_dists = []
    mean_inter_dists = []
    min_leader_dists = []
    mean_leader_dists = []

    for episode in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        total_reward = np.zeros(len(agents), dtype=np.float64)
        done = False
        step = 0
        last_info = {}
        episode_safety_hits = 0
        episode_inter_dists = []
        episode_leader_min_dists = []
        episode_ever_success = False
        episode_max_hold = 0

        while not done and step < args.max_steps:
            states = [np.asarray(obs[agent], dtype=np.float32) for agent in agents]
            actions_list = maddpg.act(states, add_noise=False)
            actions_list, safety_info = _apply_eval_safety_layer(env, actions_list, args)
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            total_reward += np.array([rewards[agent] for agent in agents], dtype=np.float64)
            last_info = infos[agents[0]]
            done = any(terminations[agent] or truncations[agent] for agent in agents)
            step += 1
            episode_safety_hits += int(safety_info["intervened"])
            episode_inter_dists.append(float(safety_info["inter_dist"]))
            episode_leader_min_dists.append(float(np.min(safety_info["leader_dists"])))
            episode_hold = int(last_info.get("success_hold_count", 0))
            episode_max_hold = max(episode_max_hold, episode_hold)
            episode_ever_success = episode_ever_success or bool(last_info.get("is_success", 0.0))

            if args.create_gif and episode < min(args.episodes, 3):
                all_frames.append(env.render())

        episode_rewards.append(total_reward)
        final_mean_errors.append(float(last_info.get("mean_slot_error", np.nan)))
        final_max_errors.append(float(last_info.get("max_follower_slot_error", np.nan)))
        final_mean_yaw_errors.append(float(last_info.get("mean_follower_yaw_error", np.nan)))
        final_max_yaw_errors.append(float(last_info.get("max_follower_yaw_error", np.nan)))
        final_hold_counts.append(float(episode_max_hold))
        successes.append(float(episode_ever_success))
        lengths.append(step)
        safety_rate = episode_safety_hits / max(step, 1)
        safety_interventions.append(safety_rate)
        min_inter_dists.append(float(np.min(episode_inter_dists)) if episode_inter_dists else np.nan)
        mean_inter_dists.append(float(np.mean(episode_inter_dists)) if episode_inter_dists else np.nan)
        min_leader_dists.append(float(np.min(episode_leader_min_dists)) if episode_leader_min_dists else np.nan)
        mean_leader_dists.append(float(np.mean(episode_leader_min_dists)) if episode_leader_min_dists else np.nan)
        print(
            f"episode={episode + 1:02d} "
            f"success={bool(successes[-1])} "
            f"steps={step:03d} "
            f"reward_sum={np.sum(total_reward):.2f} "
            f"mean_slot_error={final_mean_errors[-1]:.3f} "
            f"max_slot_error={final_max_errors[-1]:.3f} "
            f"mean_yaw_error={final_mean_yaw_errors[-1]:.3f} "
            f"max_yaw_error={final_max_yaw_errors[-1]:.3f} "
            f"max_hold={final_hold_counts[-1]:.0f} "
            f"safety_rate={100.0 * safety_rate:.1f}% "
            f"min_inter={min_inter_dists[-1]:.2f} "
            f"min_follower_leader={min_leader_dists[-1]:.2f}"
        )

    rewards = np.asarray(episode_rewards)
    print("\nSummary")
    print(f"  episodes: {args.episodes}")
    print(f"  training_stage: {args.training_stage}")
    print(f"  eval_init_scatter: {args.eval_init_scatter}")
    print(f"  follower_action_mode: {args.follower_action_mode}")
    print(f"  safety_enable: {args.safety_enable}")
    print(f"  success_rate: {np.mean(successes) * 100:.1f}%")
    print(f"  mean_total_reward: {np.mean(np.sum(rewards, axis=1)):.2f}")
    print(f"  mean_final_slot_error: {np.nanmean(final_mean_errors):.3f} m")
    print(f"  mean_final_max_slot_error: {np.nanmean(final_max_errors):.3f} m")
    print(f"  mean_final_yaw_error: {np.nanmean(final_mean_yaw_errors):.3f} rad")
    print(f"  mean_final_max_yaw_error: {np.nanmean(final_max_yaw_errors):.3f} rad")
    print(f"  mean_max_success_hold_count: {np.nanmean(final_hold_counts):.1f} / {SUCCESS_HOLD_STEPS}")
    print(f"  mean_episode_length: {np.mean(lengths):.1f} steps")
    print(f"  mean_safety_intervention_rate: {np.nanmean(safety_interventions) * 100:.1f}%")
    print(f"  mean_min_go2_go3_dist: {np.nanmean(min_inter_dists):.3f} m")
    print(f"  mean_go2_go3_dist: {np.nanmean(mean_inter_dists):.3f} m")
    print(f"  mean_min_follower_leader_dist: {np.nanmean(min_leader_dists):.3f} m")
    print(f"  mean_follower_leader_dist: {np.nanmean(mean_leader_dists):.3f} m")

    if args.create_gif and all_frames:
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_path = os.path.join(args.output_dir, f"eval_{timestamp}.gif")
        imageio.mimsave(gif_path, all_frames, duration=0.08)
        print(f"  gif: {gif_path}")

    env.close()


if __name__ == "__main__":
    main()
