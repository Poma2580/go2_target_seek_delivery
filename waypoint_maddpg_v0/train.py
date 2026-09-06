"""Train the standalone five-candidate discrete MADDPG policy."""

import argparse
import csv
import json
import random
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig, TrainConfig
from .discrete_maddpg import DiscreteMADDPG
from .environment import WaypointSelectionEnv
from .replay_buffer import ReplayBuffer


def linear_schedule(start, end, step, duration):
    fraction = min(max(step / max(duration, 1), 0.0), 1.0)
    return start + fraction * (end - start)


def evaluate(policy, env_config, episodes, seed):
    rewards, successes, collisions, min_clearances, min_pair_distances = [], [], [], [], []
    obstacle_collisions = np.zeros(env_config.num_agents, dtype=np.int64)
    obstacle_shape_collisions = {"square": 0, "circle": 0}
    pair_collisions = 0
    action_counts = np.zeros(env_config.num_actions, dtype=np.int64)
    for episode in range(episodes):
        env = WaypointSelectionEnv(env_config, seed=seed + episode, lidar_noise=True)
        observations, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_min_clearance = np.inf
        episode_min_pair = np.inf
        final_info = {}
        while True:
            actions = policy.act(
                observations,
                action_masks=env.valid_action_masks(),
                deterministic=True,
            )
            for action in actions:
                action_counts[action] += 1
            observations, reward, terminated, truncated, final_info = env.step(actions)
            episode_reward += float(reward[0])
            episode_min_clearance = min(
                episode_min_clearance, final_info["min_obstacle_clearance"]
            )
            episode_min_pair = min(episode_min_pair, final_info["min_pair_distance"])
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        successes.append(float(final_info.get("success", False)))
        collisions.append(float(final_info.get("collision", False)))
        obstacle_collisions += np.asarray(
            final_info.get("obstacle_collision_agents", [False, False]), dtype=np.int64
        )
        for obstacle_index in final_info.get("obstacle_collision_indices", []):
            if int(obstacle_index) >= 0:
                shape = final_info["obstacles"][int(obstacle_index)]["shape"]
                obstacle_shape_collisions[shape] += 1
        pair_collisions += int(final_info.get("pair_collision", False))
        min_clearances.append(episode_min_clearance)
        min_pair_distances.append(episode_min_pair)
    return {
        "reward": float(np.mean(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "obstacle_collision_counts": obstacle_collisions.tolist(),
        "obstacle_shape_collision_counts": obstacle_shape_collisions,
        "pair_collision_count": int(pair_collisions),
        "min_obstacle_clearance": float(np.mean(min_clearances)),
        "min_pair_distance": float(np.mean(min_pair_distances)),
        "action_counts": action_counts.tolist(),
    }


def parse_args():
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--replay-size", type=int, default=defaults.replay_size)
    parser.add_argument("--hidden-dim", type=int, default=defaults.hidden_dim)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--eval-episodes", type=int, default=defaults.eval_episodes)
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--initial-epsilon", type=float, default=defaults.initial_epsilon)
    parser.add_argument("--final-epsilon", type=float, default=defaults.final_epsilon)
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps
    )
    parser.add_argument("--leader-speed", type=float, default=EnvConfig().leader_speed)
    parser.add_argument("--sim-rays", type=int, choices=(36, 72, 108), default=36)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--shared-actor",
        action="store_true",
        help="Use one actor network and one averaged actor update for both followers.",
    )
    parser.add_argument(
        "--max-obstacle-abs-y",
        type=float,
        default=EnvConfig().obstacle_abs_y_range[1],
    )
    parser.add_argument(
        "--curriculum-steps",
        type=int,
        default=0,
        help="Steps used to expand max |obstacle y| from 2.3 m to the requested maximum.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny integration training job instead of a real experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    train_cfg = replace(
        TrainConfig(),
        seed=args.seed,
        total_steps=args.total_steps,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        replay_size=args.replay_size,
        hidden_dim=args.hidden_dim,
        eval_interval=args.eval_interval,
        eval_episodes=args.eval_episodes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        initial_epsilon=args.initial_epsilon,
        final_epsilon=args.final_epsilon,
        epsilon_decay_steps=args.epsilon_decay_steps,
    )
    if args.smoke:
        train_cfg = replace(
            train_cfg,
            total_steps=200,
            replay_size=2_000,
            warmup_steps=32,
            batch_size=32,
            hidden_dim=64,
            eval_interval=100,
            eval_episodes=2,
            log_interval_episodes=2,
        )
    env_cfg = replace(
        EnvConfig(),
        leader_speed=args.leader_speed,
        lidar_sim_rays=args.sim_rays,
        obstacle_abs_y_range=(
            EnvConfig().obstacle_abs_y_range[0],
            args.max_obstacle_abs_y,
        ),
    )
    if args.max_obstacle_abs_y < EnvConfig().obstacle_abs_y_range[1]:
        raise ValueError("--max-obstacle-abs-y must be at least 2.3")

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_cfg.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or (
        Path(__file__).resolve().parent / "runs" / f"five_candidate_maddpg_{timestamp}"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / "config.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "environment": env_cfg.to_dict(),
                "training": train_cfg.to_dict(),
                "initialized_from": str(args.init_checkpoint) if args.init_checkpoint else None,
                "curriculum": {
                    "initial_obstacle_abs_y": list(EnvConfig().obstacle_abs_y_range),
                    "final_obstacle_abs_y": list(env_cfg.obstacle_abs_y_range),
                    "steps": args.curriculum_steps,
                },
                "shared_actor": args.shared_actor,
            },
            stream,
            indent=2,
        )

    env = WaypointSelectionEnv(env_cfg, seed=train_cfg.seed, lidar_noise=True)
    policy = DiscreteMADDPG(
        env.num_agents,
        env.obs_dim,
        env.num_actions,
        hidden_dim=train_cfg.hidden_dim,
        actor_lr=train_cfg.actor_lr,
        critic_lr=train_cfg.critic_lr,
        gamma=train_cfg.gamma,
        tau=train_cfg.tau,
        device=device,
        shared_actor=args.shared_actor,
    )
    if args.init_checkpoint is not None:
        policy.load(args.init_checkpoint)
    replay = ReplayBuffer(
        train_cfg.replay_size,
        env.num_agents,
        env.obs_dim,
        env.num_actions,
        device,
    )

    metrics_path = run_dir / "episodes.csv"
    metrics_file = metrics_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(
        metrics_file,
        fieldnames=(
            "global_step",
            "episode",
            "length",
            "reward",
            "success",
            "collision",
            "epsilon",
            "temperature",
        ),
    )
    writer.writeheader()

    initial_y_range = EnvConfig().obstacle_abs_y_range
    env.set_obstacle_abs_y_range(initial_y_range)
    observations, _ = env.reset(seed=train_cfg.seed)
    episode_reward, episode_length, episode_index = 0.0, 0, 0
    recent_rewards, recent_successes, recent_collisions = [], [], []
    best_score = -float("inf")
    last_losses = {"actor_loss": float("nan"), "critic_loss": float("nan")}
    print(
        f"run={run_dir}\ndevice={device} obs={env.obs_dim} actions={env.num_actions} "
        f"lidar={env_cfg.lidar_sim_rays}->{env_cfg.lidar_observation_size}\n"
        f"initialized_from={args.init_checkpoint}\n"
        f"shared_actor={args.shared_actor}\n"
        f"obstacle_abs_y={initial_y_range}->{env_cfg.obstacle_abs_y_range} "
        f"curriculum_steps={args.curriculum_steps}"
    )

    try:
        for global_step in range(1, train_cfg.total_steps + 1):
            epsilon = linear_schedule(
                train_cfg.initial_epsilon,
                train_cfg.final_epsilon,
                global_step,
                train_cfg.epsilon_decay_steps,
            )
            temperature = linear_schedule(
                train_cfg.initial_temperature,
                train_cfg.final_temperature,
                global_step,
                train_cfg.epsilon_decay_steps,
            )
            if global_step <= train_cfg.warmup_steps:
                action_masks = env.valid_action_masks()
                actions = np.asarray(
                    [
                        np.random.choice(np.flatnonzero(action_masks[index]))
                        for index in range(env.num_agents)
                    ],
                    dtype=np.int64,
                )
            else:
                action_masks = env.valid_action_masks()
                actions = policy.act(
                    observations,
                    action_masks=action_masks,
                    epsilon=epsilon,
                )
            actions_one_hot = np.eye(env.num_actions, dtype=np.float32)[actions]
            next_observations, rewards, terminated, truncated, info = env.step(actions)
            next_action_masks = env.valid_action_masks()
            done = terminated or truncated
            replay.add(
                observations,
                actions_one_hot,
                action_masks,
                rewards,
                next_observations,
                next_action_masks,
                done,
            )
            observations = next_observations
            episode_reward += float(rewards[0])
            episode_length += 1

            if (
                global_step > train_cfg.warmup_steps
                and len(replay) >= train_cfg.batch_size
                and global_step % train_cfg.update_every == 0
            ):
                for _ in range(train_cfg.updates_per_step):
                    last_losses = policy.update(
                        replay.sample(train_cfg.batch_size), temperature=temperature
                    )

            if done:
                episode_index += 1
                success = float(info.get("success", False))
                collision = float(info.get("collision", False))
                recent_rewards.append(episode_reward)
                recent_successes.append(success)
                recent_collisions.append(collision)
                writer.writerow(
                    {
                        "global_step": global_step,
                        "episode": episode_index,
                        "length": episode_length,
                        "reward": episode_reward,
                        "success": success,
                        "collision": collision,
                        "epsilon": epsilon,
                        "temperature": temperature,
                    }
                )
                metrics_file.flush()
                curriculum_fraction = min(
                    global_step / max(args.curriculum_steps, 1), 1.0
                ) if args.curriculum_steps > 0 else 1.0
                curriculum_max_y = initial_y_range[1] + curriculum_fraction * (
                    env_cfg.obstacle_abs_y_range[1] - initial_y_range[1]
                )
                env.set_obstacle_abs_y_range(
                    (initial_y_range[0], curriculum_max_y)
                )
                observations, _ = env.reset(seed=train_cfg.seed + episode_index)
                episode_reward, episode_length = 0.0, 0

                if episode_index % train_cfg.log_interval_episodes == 0:
                    print(
                        f"step={global_step} episode={episode_index} "
                        f"reward={np.mean(recent_rewards[-train_cfg.log_interval_episodes:]):.2f} "
                        f"success={np.mean(recent_successes[-train_cfg.log_interval_episodes:]):.2f} "
                        f"collision={np.mean(recent_collisions[-train_cfg.log_interval_episodes:]):.2f} "
                        f"actor_loss={last_losses['actor_loss']:.4f} "
                        f"critic_loss={last_losses['critic_loss']:.4f}"
                    )

            if global_step % train_cfg.eval_interval == 0 or global_step == train_cfg.total_steps:
                eval_metrics = evaluate(
                    policy,
                    env_cfg,
                    train_cfg.eval_episodes,
                    seed=train_cfg.seed + 100_000 + global_step,
                )
                score = (
                    eval_metrics["success_rate"]
                    - eval_metrics["collision_rate"]
                    + 0.001 * eval_metrics["reward"]
                )
                print(f"evaluation step={global_step}: {eval_metrics}")
                metadata = {
                    "global_step": global_step,
                    "environment": env_cfg.to_dict(),
                    "training": train_cfg.to_dict(),
                    "initialized_from": str(args.init_checkpoint) if args.init_checkpoint else None,
                    "curriculum": {
                        "initial_obstacle_abs_y": list(initial_y_range),
                        "final_obstacle_abs_y": list(env_cfg.obstacle_abs_y_range),
                        "steps": args.curriculum_steps,
                    },
                    "shared_actor": args.shared_actor,
                    "evaluation": eval_metrics,
                }
                policy.save(run_dir / "latest_model.pt", metadata=metadata)
                if score > best_score:
                    best_score = score
                    policy.save(run_dir / "best_model.pt", metadata=metadata)
    finally:
        metrics_file.close()

    print(f"training complete: {run_dir}")


if __name__ == "__main__":
    main()
