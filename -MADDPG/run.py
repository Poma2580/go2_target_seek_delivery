"""
Script to run trained agents in various environments
"""


import torch
import numpy as np
import argparse
import os
import random
import imageio
import time
from datetime import datetime
from PIL import Image

from maddpg import MADDPG
from utils.env import create_single_env, ENV_MAP, get_env_info

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                       default="runs/formation_navigation_v0/maddpg/model.pt",
                       help="Path to the trained model file")
    parser.add_argument("--env-name", type=str, default="formation_navigation_v0",
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--algo", type=str, default="MADDPG", choices=["MADDPG"],
                       help="Algorithm to use")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--output-dir", type=str, default="./gifs", help="Directory to save outputs")
    parser.add_argument("--is-parallel", action="store_true", help="Parallel environment")
    parser.add_argument("--create-gif", action="store_true", default=True,
                       help="Create GIF of episodes")
    parser.add_argument("--episode-separator", type=float, default=1.0,
                       help="Duration in seconds for the black frame between episodes")
    parser.add_argument("--training-stage", type=int, default=5, choices=[1, 2, 3, 4, 5],
                       help="Curriculum stage: 1=no obstacles, 2=fixed obstacles, 3=random near-formation init + fixed obstacles, 4=scattered init + fixed obstacles, 5=scattered init + random obstacles")
    parser.add_argument("--seed", type=int, default=None,
                       help="Base random seed for evaluation. If set, episode k uses seed + k - 1.")
    return parser.parse_args()

def resize_frame_to_shape(frame, target_shape):
    """Resize frame to (H, W, C) target shape."""
    target_h, target_w = target_shape[:2]
    img = Image.fromarray(frame)
    img = img.resize((target_w, target_h), Image.BILINEAR)
    return np.array(img)

def set_env_display_episode(env, episode):
    """Set a display-only episode number on an env and common wrapper chains."""
    current = env
    visited = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        try:
            setattr(current, "current_episode", episode)
        except Exception:
            pass

        next_env = getattr(current, "env", None)
        if next_env is None:
            next_env = getattr(current, "aec_env", None)
        if next_env is current:
            break
        current = next_env

def run(args):
    # Set up output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.env_name}/stage{args.training_stage}/{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Set global random seeds for reproducible evaluation.
    # Each episode will also use args.seed + episode_index during env.reset().
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        max_steps=args.max_steps,
        apply_padding=False,  # No padding for single environment
        training_stage=args.training_stage
    )

    # Create environment with appropriate render mode
    render_mode = "rgb_array" if args.create_gif else None
    env = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode=render_mode,
        apply_padding=False,
        training_stage=args.training_stage
    )

    # Create MADDPG agent
    if args.algo == "MADDPG":
        maddpg = MADDPG(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=(128, 128),  # Default hidden sizes
            action_low=action_low,
            action_high=action_high
        )
    else:

        raise ValueError(f"Unknown algorithm: {args.algo}")

    # Load trained model
    if os.path.exists(args.model_path):
        maddpg.load(args.model_path)
        print(f"Loaded model from {args.model_path}")
    else:
        print(f"No model found at {args.model_path}, using random policies")

    # Run episodes

    # 新增：初始化编队指标统计列表
    all_episode_rewards = []
    all_episode_seeds = []
    all_final_mean_slot_errors = []
    all_final_formation_errors = []
    all_final_team_target_distances = []

    all_agent_collisions = []  # 每回合智能体之间的碰撞总数
    success_count = 0  # 成功到达目标的回合数

    all_obstacle_collisions = []
    all_obstacle_terminated = []
    all_min_obstacle_dists = []

    # 新增：按阶段统计障碍物碰撞
    all_form_obstacle_collisions = []
    all_nav_obstacle_collisions = []

    all_form_obstacle_terminated = []
    all_nav_obstacle_terminated = []

    all_form_min_obstacle_dists = []
    all_nav_min_obstacle_dists = []

    # 导航阶段过程指标
    all_nav_mean_formation_errors = []  # 每个episode中，NAV阶段formation_error的平均值
    all_nav_max_formation_errors = []  # 每个episode中，NAV阶段formation_error的最大值
    all_nav_steps = []  # 每个episode进入NAV后的步数
    all_nav_start_steps = []  # 每个episode第一次进入NAV的step；没进入则记None

    all_reward_components = []

    # Decision-time statistics.
    # These timings only wrap maddpg.act(...) policy inference, so they do NOT
    # include env.step(...), env.render(...), GIF creation, or file saving time.
    all_decision_times_ms = []
    all_episode_mean_decision_times_ms = []
    all_episode_max_decision_times_ms = []

    reward_component_keys = [
        "reward_formation",
        "reward_collision",
        "reward_goal",
        "reward_step",
        "reward_nav_unlock_bonus",
        "reward_success_bonus",
    ]

    reward_component_names = {
        "reward_formation": "formation",
        "reward_collision": "collision",
        "reward_goal": "goal",
        "reward_step": "step",
        "reward_success_bonus": "success_bonus",
        "reward_nav_unlock_bonus": "nav_unlock_bonus",
    }

    # For combined GIF creation
    all_frames = [] if args.create_gif else None
    target_frame_shape = None

    for episode in range(1, args.episodes + 1):
        episode_seed = None if args.seed is None else args.seed + episode - 1
        all_episode_seeds.append(episode_seed)

        if episode_seed is None:
            observations, _ = env.reset()
        else:
            try:
                observations, _ = env.reset(seed=episode_seed)
            except TypeError:
                # Fallback for older env wrappers that do not expose reset(seed=...).
                # The global NumPy seed is still updated so environments using np.random
                # directly can remain reproducible.
                np.random.seed(episode_seed)
                observations, _ = env.reset()

        episode_rewards = np.zeros(len(agents))

        episode_reward_components = {
            key: np.zeros(len(agents), dtype=np.float64)
            for key in reward_component_keys
        }

        # 新增：分别初始化两种碰撞的统计
        episode_agent_collisions = 0
        done = False
        step = 0
        episode_obstacle_collisions = 0
        episode_obstacle_terminated = 0
        episode_min_obstacle_dist = float("inf")
        # 新增：分阶段障碍物碰撞统计
        episode_form_obstacle_collisions = 0
        episode_nav_obstacle_collisions = 0

        episode_form_obstacle_terminated = 0
        episode_nav_obstacle_terminated = 0

        episode_form_min_obstacle_dist = float("inf")
        episode_nav_min_obstacle_dist = float("inf")
        final_team_target_distance = 0.0
        final_formation_error = 0.0

        # For individual episode frames
        episode_frames = [] if args.create_gif else None
        final_mean_slot_error = 0.0

        episode_frames = [] if args.create_gif else None
        final_mean_slot_error = 0.0

        episode_nav_formation_errors = []
        episode_nav_steps = 0
        episode_nav_start_step = None
        episode_decision_times_ms = []

        while not done and step < args.max_steps:
            # Get states for all agents
            states = [np.array(observations[agent], dtype=np.float32) for agent in agents]

            # Get actions for all agents (no noise for evaluation).
            # This is the measured single-step policy decision time.
            # It excludes env.step(...), env.render(...), GIF creation, and file saving.
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            decision_t0 = time.perf_counter()
            actions_list = maddpg.act(states, add_noise=False)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            decision_time_ms = (time.perf_counter() - decision_t0) * 1000.0
            episode_decision_times_ms.append(decision_time_ms)
            all_decision_times_ms.append(decision_time_ms)

            # Convert actions to dictionary for environment
            actions = {agent: action for agent, action in zip(agents, actions_list)}

            # Take a step in the environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            for agent_idx, agent_name in enumerate(agents):
                agent_info = infos.get(agent_name, {})
                for key in reward_component_keys:
                    episode_reward_components[key][agent_idx] += float(agent_info.get(key, 0.0))

            # 读取环境实际输出的info字段
            current_agent_collision = infos[agents[0]]["agent_collision_num"]
            current_obstacle_collision = infos[agents[0]].get("total_obstacle_collisions", 0)
            current_any_obstacle_collision = infos[agents[0]].get("any_obstacle_collision", 0.0)
            current_min_obstacle_dist = infos[agents[0]].get("min_obstacle_dist", float("inf"))
            current_mean_slot_error = infos[agents[0]]["mean_slot_error"]
            current_hold_count = infos[agents[0]]["success_hold_count"]
            current_team_target_distance = infos[agents[0]]["team_target_distance"]
            current_navigation_active = infos[agents[0]]["navigation_active"]
            is_success = infos[agents[0]]["is_success"]
            current_formation_error = infos[agents[0]].get("formation_error", current_mean_slot_error)
            final_team_target_distance = current_team_target_distance
            final_formation_error = current_formation_error
            # 记录导航阶段过程指标
            if current_navigation_active > 0.5:
                episode_nav_formation_errors.append(float(current_formation_error))
                episode_nav_steps += 1

                if episode_nav_start_step is None:
                    episode_nav_start_step = step
            episode_agent_collisions += abs(current_agent_collision)
            episode_obstacle_collisions += abs(current_obstacle_collision)
            episode_obstacle_terminated = max(
                episode_obstacle_terminated,
                int(current_any_obstacle_collision > 0.5)
            )
            episode_min_obstacle_dist = min(
                episode_min_obstacle_dist,
                float(current_min_obstacle_dist)
            )
            # 新增：按当前阶段统计障碍物碰撞
            if current_navigation_active > 0.5:
                episode_nav_obstacle_collisions += abs(current_obstacle_collision)
                episode_nav_obstacle_terminated = max(
                    episode_nav_obstacle_terminated,
                    int(current_any_obstacle_collision > 0.5)
                )
                episode_nav_min_obstacle_dist = min(
                    episode_nav_min_obstacle_dist,
                    float(current_min_obstacle_dist)
                )
            else:
                episode_form_obstacle_collisions += abs(current_obstacle_collision)
                episode_form_obstacle_terminated = max(
                    episode_form_obstacle_terminated,
                    int(current_any_obstacle_collision > 0.5)
                )
                episode_form_min_obstacle_dist = min(
                    episode_form_min_obstacle_dist,
                    float(current_min_obstacle_dist)
                )
            final_mean_slot_error = current_mean_slot_error

            # Check if episode is done
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)

            # Update observations
            observations = next_observations

            # Update rewards
            episode_rewards += np.array(list(rewards.values()))

            # Capture frame for GIF if needed
            if args.create_gif:
                set_env_display_episode(env, episode)
                frame = env.render()

                # GIF 中图像上方的标题已经包含关键状态信息，
                # 这里不再叠加左上角白色信息框，避免遮挡画面。
                labeled_frame = frame

                if target_frame_shape is None:
                    target_frame_shape = labeled_frame.shape
                elif labeled_frame.shape != target_frame_shape:
                    labeled_frame = resize_frame_to_shape(labeled_frame, target_frame_shape)

                episode_frames.append(labeled_frame)
                all_frames.append(labeled_frame)

            step += 1

        # Save episode statistics
        all_episode_rewards.append(episode_rewards)
        all_reward_components.append({
            key: values.copy()
            for key, values in episode_reward_components.items()
        })
        all_final_mean_slot_errors.append(final_mean_slot_error)
        all_final_formation_errors.append(final_formation_error)
        all_final_team_target_distances.append(final_team_target_distance)

        all_agent_collisions.append(episode_agent_collisions)
        all_obstacle_collisions.append(episode_obstacle_collisions)
        all_obstacle_terminated.append(episode_obstacle_terminated)
        all_min_obstacle_dists.append(episode_min_obstacle_dist)
        all_form_obstacle_collisions.append(episode_form_obstacle_collisions)
        all_nav_obstacle_collisions.append(episode_nav_obstacle_collisions)

        all_form_obstacle_terminated.append(episode_form_obstacle_terminated)
        all_nav_obstacle_terminated.append(episode_nav_obstacle_terminated)

        all_form_min_obstacle_dists.append(episode_form_min_obstacle_dist)
        all_nav_min_obstacle_dists.append(episode_nav_min_obstacle_dist)
        # 保存导航阶段过程指标
        if episode_nav_formation_errors:
            all_nav_mean_formation_errors.append(float(np.mean(episode_nav_formation_errors)))
            all_nav_max_formation_errors.append(float(np.max(episode_nav_formation_errors)))
        else:
            all_nav_mean_formation_errors.append(np.nan)
            all_nav_max_formation_errors.append(np.nan)

        all_nav_steps.append(int(episode_nav_steps))
        all_nav_start_steps.append(episode_nav_start_step)

        if episode_decision_times_ms:
            episode_mean_decision_time_ms = float(np.mean(episode_decision_times_ms))
            episode_max_decision_time_ms = float(np.max(episode_decision_times_ms))
        else:
            episode_mean_decision_time_ms = 0.0
            episode_max_decision_time_ms = 0.0

        all_episode_mean_decision_times_ms.append(episode_mean_decision_time_ms)
        all_episode_max_decision_times_ms.append(episode_max_decision_time_ms)

        # 新增：判定是否成功
        if is_success == 1.0:
            success_count += 1
            success_tag = " [SUCCESS]"
        else:
            success_tag = ""

        # Print episode results (新增编队指标)
        seed_text = "" if episode_seed is None else f" seed={episode_seed}"
        print(f"Episode {episode}{seed_text}{success_tag}, "
              f"Rewards: {episode_rewards}, Total: {np.sum(episode_rewards):.1f}, "
              f"Final Formation Error: {final_formation_error:.3f}, "
              f"Final Mean Slot Error: {final_mean_slot_error:.3f}, "
              f"Final Team Target Dist: {final_team_target_distance:.3f}, "
              f"Agent Collisions: {episode_agent_collisions}, "
              f"Decision Time: mean={episode_mean_decision_time_ms:.3f} ms, "
              f"max={episode_max_decision_time_ms:.3f} ms")

        # Print episode results
        #print(f"Episode {episode}, Rewards: {episode_rewards}, Total: {np.sum(episode_rewards)}")

        # Add a simple separator between episodes
        if args.create_gif and episode < args.episodes and episode_frames:
            frame_shape = target_frame_shape if target_frame_shape is not None else (480, 640, 3)
            black_frame = np.zeros(frame_shape, dtype=np.uint8)

            # 使用纯黑帧作为 episode 分隔，不再叠加左上角白色文字框。
            next_frame = black_frame

            # Add separator frames
            separator_frames = int(args.episode_separator * 10)  # 10 frames per second
            for _ in range(separator_frames):
                all_frames.append(next_frame)

    # Save combined GIF with all episodes
    if args.create_gif and all_frames:
        combined_gif_path = os.path.join(output_dir, f"{args.algo}_all_episodes.gif")
        try:
            imageio.mimsave(combined_gif_path, all_frames, duration=0.1)  # 100ms per frame
            print(f"Saved combined GIF with all episodes to {combined_gif_path}")
        except Exception as e:
            print(f"Error saving combined GIF: {e}")

    # Print summary statistics
    if all_episode_rewards:
        avg_rewards = np.mean(all_episode_rewards, axis=0)
        avg_reward_components = {}
        avg_team_reward_components = {}

        if all_reward_components:
            for key in reward_component_keys:
                component_values = np.array([
                    episode_components[key]
                    for episode_components in all_reward_components
                ])
                avg_reward_components[key] = np.mean(component_values, axis=0)
                avg_team_reward_components[key] = float(np.sum(avg_reward_components[key]))

        avg_final_mean_slot_error = np.mean(all_final_mean_slot_errors)
        avg_final_formation_error = np.mean(all_final_formation_errors)
        avg_final_team_target_distance = np.mean(all_final_team_target_distances)

        avg_agent_collision = np.mean(all_agent_collisions) if all_agent_collisions else 0.0
        success_rate = success_count / args.episodes * 100

        avg_obstacle_collision = np.mean(all_obstacle_collisions) if all_obstacle_collisions else 0.0
        obstacle_termination_rate = (
            np.mean(all_obstacle_terminated) * 100
            if all_obstacle_terminated else 0.0
        )
        avg_min_obstacle_dist = np.mean(all_min_obstacle_dists) if all_min_obstacle_dists else 0.0

        avg_form_obstacle_collision = (
            float(np.mean(all_form_obstacle_collisions))
            if all_form_obstacle_collisions else 0.0
        )

        avg_nav_obstacle_collision = (
            float(np.mean(all_nav_obstacle_collisions))
            if all_nav_obstacle_collisions else 0.0
        )

        form_obstacle_termination_rate = (
            float(np.mean(all_form_obstacle_terminated) * 100)
            if all_form_obstacle_terminated else 0.0
        )

        nav_obstacle_termination_rate = (
            float(np.mean(all_nav_obstacle_terminated) * 100)
            if all_nav_obstacle_terminated else 0.0
        )

        valid_form_min_obs = [
            x for x in all_form_min_obstacle_dists
            if np.isfinite(x)
        ]

        valid_nav_min_obs = [
            x for x in all_nav_min_obstacle_dists
            if np.isfinite(x)
        ]

        avg_form_min_obstacle_dist = (
            float(np.mean(valid_form_min_obs))
            if valid_form_min_obs else 0.0
        )

        avg_nav_min_obstacle_dist = (
            float(np.mean(valid_nav_min_obs))
            if valid_nav_min_obs else 0.0
        )

        # 导航阶段过程指标
        valid_nav_mean_errors = [
            x for x in all_nav_mean_formation_errors
            if not np.isnan(x)
        ]
        valid_nav_max_errors = [
            x for x in all_nav_max_formation_errors
            if not np.isnan(x)
        ]
        valid_nav_start_steps = [
            x for x in all_nav_start_steps
            if x is not None
        ]

        avg_nav_mean_formation_error = (
            float(np.mean(valid_nav_mean_errors))
            if valid_nav_mean_errors else 0.0
        )

        avg_nav_max_formation_error = (
            float(np.mean(valid_nav_max_errors))
            if valid_nav_max_errors else 0.0
        )

        avg_nav_steps = float(np.mean(all_nav_steps)) if all_nav_steps else 0.0

        nav_activation_rate = (
                sum(1 for x in all_nav_steps if x > 0) / args.episodes * 100
        )

        avg_nav_start_step = (
            float(np.mean(valid_nav_start_steps))
            if valid_nav_start_steps else -1.0
        )

        avg_decision_time_ms = (
            float(np.mean(all_decision_times_ms))
            if all_decision_times_ms else 0.0
        )
        max_decision_time_ms = (
            float(np.max(all_decision_times_ms))
            if all_decision_times_ms else 0.0
        )
        std_decision_time_ms = (
            float(np.std(all_decision_times_ms))
            if all_decision_times_ms else 0.0
        )

        print("\n" + "=" * 50)
        print("Formation Navigation Evaluation Summary")
        print("=" * 50)
        print(f"Base seed: {args.seed}")
        if args.seed is not None:
            print(f"Episode seeds: {args.seed} to {args.seed + args.episodes - 1}")
        for i, agent_name in enumerate(agents):
            print(f"{agent_name} average reward: {avg_rewards[i]:.2f}")
        print(f"\nTeam Metrics:")
        print(f"  Average total reward:     {np.sum(avg_rewards):.2f}")
        print(f"  Average final mean slot error:{avg_final_mean_slot_error:.4f}")
        print(f"  Average final formation error:{avg_final_formation_error:.4f}")
        print(f"  Average final team target dist:{avg_final_team_target_distance:.4f}")
        print(f"  Average agent collisions:  {avg_agent_collision:.2f}")
        print(f"  Success rate:             {success_rate:.1f}% ({success_count}/{args.episodes})")

        print("\nNavigation Phase Metrics:")
        print(f"  Navigation activation rate:       {nav_activation_rate:.1f}%")
        print(f"  Average navigation start step:    {avg_nav_start_step:.2f}")
        print(f"  Average navigation steps:         {avg_nav_steps:.2f}")
        print(f"  Average NAV formation error:      {avg_nav_mean_formation_error:.4f}")
        print(f"  Average max NAV formation error:  {avg_nav_max_formation_error:.4f}")
        print("\nReward Component Averages (weighted, episode cumulative):")
        for key in reward_component_keys:
            component_name = reward_component_names[key]
            if key in avg_reward_components:
                per_agent_str = np.array2string(
                    avg_reward_components[key],
                    precision=2,
                    suppress_small=True
                )
                print(
                    f"  {component_name:<14} "
                    f"Team: {avg_team_reward_components[key]:>8.2f} | "
                    f"Per-agent: {per_agent_str}"
                )

        if avg_team_reward_components:
            recomposed_total = sum(avg_team_reward_components.values())
            print(f"  {'recomposed_total':<14} Team: {recomposed_total:>8.2f}")
        print("\nDecision Time Metrics:")
        print("  Measured scope: maddpg.act(...) only; excludes env.step/render/GIF/file saving")
        print(f"  Average decision time per step:     {avg_decision_time_ms:.3f} ms")
        print(f"  Max decision time per step:         {max_decision_time_ms:.3f} ms")
        print(f"  Std decision time per step:         {std_decision_time_ms:.3f} ms")

        print("\nObstacle Metrics:")
        print(f"  Average obstacle collisions:        {avg_obstacle_collision:.2f}")
        print(f"  Obstacle termination rate:          {obstacle_termination_rate:.1f}%")
        print(f"  Average min obstacle dist:          {avg_min_obstacle_dist:.4f}")

        print(f"  FORM obstacle collisions:           {avg_form_obstacle_collision:.2f}")
        print(f"  FORM obstacle termination rate:     {form_obstacle_termination_rate:.1f}%")
        print(f"  FORM min obstacle dist:             {avg_form_min_obstacle_dist:.4f}")

        print(f"  NAV obstacle collisions:            {avg_nav_obstacle_collision:.2f}")
        print(f"  NAV obstacle termination rate:      {nav_obstacle_termination_rate:.1f}%")
        print(f"  NAV min obstacle dist:              {avg_nav_min_obstacle_dist:.4f}")
        print("=" * 50)

        # 新增：保存评估结果到文件（便于后续plot.py使用）
        results = {
            "agent_names": agents,
            "avg_agent_rewards": avg_rewards.tolist(),
            "avg_total_reward": float(np.sum(avg_rewards)),
            "avg_final_mean_slot_error": float(avg_final_mean_slot_error),
            "avg_agent_collision": float(avg_agent_collision),
            "success_rate": float(success_rate),
            "success_count": int(success_count),
            "total_episodes": int(args.episodes),
            "all_episode_rewards": np.array(all_episode_rewards).tolist(),
            "all_final_mean_slot_errors": all_final_mean_slot_errors,
            "all_agent_collisions": all_agent_collisions,
            "model_path": args.model_path,
            "env_name": args.env_name,
            "timestamp": timestamp,
            "seed": args.seed,
            "episode_seeds": all_episode_seeds,
            "avg_final_team_target_distance": float(avg_final_team_target_distance),
            "all_final_team_target_distances": all_final_team_target_distances,
            "avg_final_formation_error": float(avg_final_formation_error),
            "all_final_formation_errors": all_final_formation_errors,
            "nav_activation_rate": float(nav_activation_rate),
            "avg_nav_start_step": float(avg_nav_start_step),
            "avg_nav_steps": float(avg_nav_steps),
            "avg_nav_mean_formation_error": float(avg_nav_mean_formation_error),
            "avg_nav_max_formation_error": float(avg_nav_max_formation_error),
            "avg_decision_time_ms": float(avg_decision_time_ms),
            "max_decision_time_ms": float(max_decision_time_ms),
            "std_decision_time_ms": float(std_decision_time_ms),
            "all_decision_times_ms": all_decision_times_ms,
            "all_episode_mean_decision_times_ms": all_episode_mean_decision_times_ms,
            "all_episode_max_decision_times_ms": all_episode_max_decision_times_ms,
            "decision_time_scope": "maddpg.act(...) only; excludes env.step/render/GIF/file saving",

            "all_nav_mean_formation_errors": all_nav_mean_formation_errors,
            "all_nav_max_formation_errors": all_nav_max_formation_errors,
            "all_nav_steps": all_nav_steps,
            "all_nav_start_steps": all_nav_start_steps,
            "avg_obstacle_collision": float(avg_obstacle_collision),
            "obstacle_termination_rate": float(obstacle_termination_rate),
            "avg_min_obstacle_dist": float(avg_min_obstacle_dist),

            "avg_form_obstacle_collision": float(avg_form_obstacle_collision),
            "avg_nav_obstacle_collision": float(avg_nav_obstacle_collision),
            "form_obstacle_termination_rate": float(form_obstacle_termination_rate),
            "nav_obstacle_termination_rate": float(nav_obstacle_termination_rate),
            "avg_form_min_obstacle_dist": float(avg_form_min_obstacle_dist),
            "avg_nav_min_obstacle_dist": float(avg_nav_min_obstacle_dist),

            "all_obstacle_collisions": all_obstacle_collisions,
            "all_obstacle_terminated": all_obstacle_terminated,
            "all_min_obstacle_dists": all_min_obstacle_dists,

            "all_form_obstacle_collisions": all_form_obstacle_collisions,
            "all_nav_obstacle_collisions": all_nav_obstacle_collisions,
            "all_form_obstacle_terminated": all_form_obstacle_terminated,
            "all_nav_obstacle_terminated": all_nav_obstacle_terminated,
            "all_form_min_obstacle_dists": all_form_min_obstacle_dists,
            "all_nav_min_obstacle_dists": all_nav_min_obstacle_dists,
        }

        # 保存为npy文件（便于程序读取）
        np.save(os.path.join(output_dir, "evaluation_results.npy"), results)

        # 保存为txt文件（便于人工查看）
        with open(os.path.join(output_dir, "evaluation_summary.txt"), "w", encoding="utf-8") as f:
            f.write("=" * 50 + "\n")
            f.write("Formation Navigation Evaluation Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Model Path: {args.model_path}\n")
            f.write(f"Environment: {args.env_name}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Episodes: {args.episodes}\n")
            f.write(f"Base seed: {args.seed}\n")
            if args.seed is not None:
                f.write(f"Episode seeds: {args.seed} to {args.seed + args.episodes - 1}\n")
            f.write("\n")
            for i, agent_name in enumerate(agents):
                f.write(f"{agent_name} average reward: {avg_rewards[i]:.2f}\n")
            f.write("\nTeam Metrics:\n")
            f.write(f"  Average total reward:     {np.sum(avg_rewards):.2f}\n")
            f.write(f"  Average final mean slot error:{avg_final_mean_slot_error:.4f}\n")
            f.write(f"  Average final formation error:{avg_final_formation_error:.4f}\n")
            f.write(f"  Average final team target dist:{avg_final_team_target_distance:.4f}\n")
            f.write(f"  Average agent collisions:  {avg_agent_collision:.2f}\n")
            f.write(f"  Success rate:             {success_rate:.1f}% ({success_count}/{args.episodes})\n")

            f.write("\nNavigation Phase Metrics:\n")
            f.write(f"  Navigation activation rate:       {nav_activation_rate:.1f}%\n")
            f.write(f"  Average navigation start step:    {avg_nav_start_step:.2f}\n")
            f.write(f"  Average navigation steps:         {avg_nav_steps:.2f}\n")
            f.write(f"  Average NAV formation error:      {avg_nav_mean_formation_error:.4f}\n")
            f.write(f"  Average max NAV formation error:  {avg_nav_max_formation_error:.4f}\n")
            f.write("\nReward Component Averages (weighted, episode cumulative):\n")
            for key in reward_component_keys:
                component_name = reward_component_names[key]
                if key in avg_reward_components:
                    per_agent_str = np.array2string(
                        avg_reward_components[key],
                        precision=2,
                        suppress_small=True
                    )
                    f.write(
                        f"  {component_name:<14} "
                        f"Team: {avg_team_reward_components[key]:>8.2f} | "
                        f"Per-agent: {per_agent_str}\n"
                    )

            if avg_team_reward_components:
                recomposed_total = sum(avg_team_reward_components.values())
                f.write(f"  {'recomposed_total':<14} Team: {recomposed_total:>8.2f}\n")

            f.write("\nDecision Time Metrics:\n")
            f.write("  Measured scope: maddpg.act(...) only; excludes env.step/render/GIF/file saving\n")
            f.write(f"  Average decision time per step:     {avg_decision_time_ms:.3f} ms\n")
            f.write(f"  Max decision time per step:         {max_decision_time_ms:.3f} ms\n")
            f.write(f"  Std decision time per step:         {std_decision_time_ms:.3f} ms\n")

            f.write("\nObstacle Metrics:\n")
            f.write(f"  Average obstacle collisions:        {avg_obstacle_collision:.2f}\n")
            f.write(f"  Obstacle termination rate:          {obstacle_termination_rate:.1f}%\n")
            f.write(f"  Average min obstacle dist:          {avg_min_obstacle_dist:.4f}\n")

            f.write(f"  FORM obstacle collisions:           {avg_form_obstacle_collision:.2f}\n")
            f.write(f"  FORM obstacle termination rate:     {form_obstacle_termination_rate:.1f}%\n")
            f.write(f"  FORM min obstacle dist:             {avg_form_min_obstacle_dist:.4f}\n")

            f.write(f"  NAV obstacle collisions:            {avg_nav_obstacle_collision:.2f}\n")
            f.write(f"  NAV obstacle termination rate:      {nav_obstacle_termination_rate:.1f}%\n")
            f.write(f"  NAV min obstacle dist:              {avg_nav_min_obstacle_dist:.4f}\n")
            f.write("=" * 50 + "\n")

        print(f"\nEvaluation results saved to: {output_dir}")

    # Close environment
    env.close()

if __name__ == "__main__":
    args = parse_args()
    run(args)


