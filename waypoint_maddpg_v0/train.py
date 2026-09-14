"""Train the standalone five-candidate discrete MADDPG policy."""

import argparse
import csv
import json
import random
from collections import deque
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


def evaluation_score(metrics):
    """Rank safety first, then clear-lane discipline and action stability."""
    return (
        2.0 * metrics["success_rate"]
        - 2.0 * metrics["collision_rate"]
        + 0.75 * metrics["clear_default_action_rate"]
        - 0.50 * metrics["action_switch_rate"]
        - 0.75 * metrics["action_oscillation_rate"]
        + 0.25 * metrics["empty_success_rate"]
        + 0.25 * metrics["obstacle_success_rate"]
        + 0.001 * metrics["reward"]
    )


def evaluate(policy, env_config, episodes, seed):
    rewards, successes, collisions, lengths = [], [], [], []
    min_clearances, min_pair_distances = [], []
    obstacle_collisions = np.zeros(env_config.num_agents, dtype=np.int64)
    obstacle_shape_collisions = {"square": 0, "circle": 0}
    pair_collisions = 0
    action_counts = np.zeros(env_config.num_actions, dtype=np.int64)
    action_switches = 0
    action_oscillations = 0
    agent_decisions = 0
    clear_default_actions = 0
    clear_default_opportunities = 0
    empty_successes, obstacle_successes = [], []
    for episode in range(episodes):
        env = WaypointSelectionEnv(env_config, seed=seed + episode, lidar_noise=True)
        observations, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        episode_min_clearance = np.inf
        episode_min_pair = np.inf
        final_info = {}
        empty_episode = not bool(env.obstacles)
        while True:
            actions = policy.act(
                observations,
                action_masks=env.valid_action_masks(),
                deterministic=True,
            )
            for action in actions:
                action_counts[action] += 1
            previous = env.previous_actions.copy()
            previous_previous = env.previous_previous_actions.copy()
            action_switches += int(np.count_nonzero(actions != previous))
            action_oscillations += int(
                np.count_nonzero(
                    (actions != previous) & (actions == previous_previous)
                )
            )
            agent_decisions += env_config.num_agents
            for index in range(env_config.num_agents):
                if not env._candidate_metrics[index][2]["blocked"]:
                    clear_default_opportunities += 1
                    clear_default_actions += int(actions[index] == 2)
            observations, reward, terminated, truncated, final_info = env.step(actions)
            episode_reward += float(reward[0])
            episode_min_clearance = min(
                episode_min_clearance, final_info["min_obstacle_clearance"]
            )
            episode_min_pair = min(episode_min_pair, final_info["min_pair_distance"])
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        lengths.append(final_info.get("step", 0))
        successes.append(float(final_info.get("success", False)))
        if empty_episode:
            empty_successes.append(float(final_info.get("success", False)))
        else:
            obstacle_successes.append(float(final_info.get("success", False)))
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
        "reward_std": float(np.std(rewards)),
        "success_rate": float(np.mean(successes)),
        "collision_rate": float(np.mean(collisions)),
        "episode_length": float(np.mean(lengths)),
        "obstacle_collision_counts": obstacle_collisions.tolist(),
        "obstacle_shape_collision_counts": obstacle_shape_collisions,
        "pair_collision_count": int(pair_collisions),
        "pair_collision_rate": float(pair_collisions / max(episodes, 1)),
        "min_obstacle_clearance": float(np.mean(min_clearances)),
        "min_pair_distance": float(np.mean(min_pair_distances)),
        "action_counts": action_counts.tolist(),
        "action_switch_rate": float(action_switches / max(agent_decisions, 1)),
        "action_oscillation_rate": float(
            action_oscillations / max(agent_decisions, 1)
        ),
        "clear_default_action_rate": float(
            clear_default_actions / max(clear_default_opportunities, 1)
        ),
        "empty_episode_count": len(empty_successes),
        "empty_success_rate": float(np.mean(empty_successes)) if empty_successes else 0.0,
        "obstacle_episode_count": len(obstacle_successes),
        "obstacle_success_rate": (
            float(np.mean(obstacle_successes)) if obstacle_successes else 0.0
        ),
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
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=10_000,
        help="First seed of the fixed evaluation set (identical at every evaluation).",
    )
    parser.add_argument("--actor-lr", type=float, default=defaults.actor_lr)
    parser.add_argument("--critic-lr", type=float, default=defaults.critic_lr)
    parser.add_argument("--initial-epsilon", type=float, default=defaults.initial_epsilon)
    parser.add_argument("--final-epsilon", type=float, default=defaults.final_epsilon)
    parser.add_argument(
        "--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps
    )
    parser.add_argument("--leader-speed", type=float, default=EnvConfig().leader_speed)
    parser.add_argument("--sim-rays", type=int, choices=(36, 72, 108), default=36)
    parser.add_argument("--obstacle-count", type=int, choices=(1, 2), default=2)
    parser.add_argument("--empty-episode-probability", type=float, default=0.0)
    parser.add_argument("--clear-default-offset-weight", type=float, default=0.60)
    parser.add_argument("--blocked-default-offset-weight", type=float, default=0.60)
    parser.add_argument("--switch-weight", type=float, default=0.30)
    parser.add_argument("--oscillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--adjacent-only-mask",
        action="store_true",
        help="Do not filter blocked actions or force default/return actions.",
    )
    parser.add_argument(
        "--direct-offset-formation-reward",
        action="store_true",
        help="Penalize actual lateral offset instead of using the nearest safe slot.",
    )
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
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard event logging (enabled by default).",
    )
    parser.add_argument(
        "--min-steps-before-stop",
        type=int,
        default=0,
        help="Enable evaluation-based early stopping after this many steps (0 disables it).",
    )
    parser.add_argument("--early-stop-success-rate", type=float, default=0.95)
    parser.add_argument("--early-stop-max-collision-rate", type=float, default=0.02)
    parser.add_argument("--early-stop-max-pair-collision-rate", type=float, default=1.0)
    parser.add_argument("--early-stop-min-clear-default-rate", type=float, default=0.0)
    parser.add_argument("--early-stop-min-empty-success-rate", type=float, default=0.0)
    parser.add_argument("--early-stop-min-obstacle-success-rate", type=float, default=0.0)
    parser.add_argument("--early-stop-max-switch-rate", type=float, default=1.0)
    parser.add_argument("--early-stop-max-oscillation-rate", type=float, default=1.0)
    parser.add_argument("--early-stop-consecutive-evals", type=int, default=3)
    parser.add_argument(
        "--require-convergence",
        action="store_true",
        help="Fail instead of accepting max-step completion when the criterion is unmet.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny integration training job instead of a real experiment.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.early_stop_success_rate <= 1.0:
        raise ValueError("--early-stop-success-rate must be in [0,1]")
    if not 0.0 <= args.early_stop_max_collision_rate <= 1.0:
        raise ValueError("--early-stop-max-collision-rate must be in [0,1]")
    for name in (
        "empty_episode_probability",
        "early_stop_min_clear_default_rate",
        "early_stop_min_empty_success_rate",
        "early_stop_min_obstacle_success_rate",
        "early_stop_max_switch_rate",
        "early_stop_max_oscillation_rate",
        "early_stop_max_pair_collision_rate",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0,1]")
    for name in (
        "clear_default_offset_weight",
        "blocked_default_offset_weight",
        "switch_weight",
        "oscillation_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    if args.early_stop_consecutive_evals < 1:
        raise ValueError("--early-stop-consecutive-evals must be at least 1")
    if args.min_steps_before_stop < 0:
        raise ValueError("--min-steps-before-stop must be nonnegative")
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
        obstacle_count=args.obstacle_count,
        empty_episode_probability=args.empty_episode_probability,
        clear_default_offset_weight=args.clear_default_offset_weight,
        blocked_default_offset_weight=args.blocked_default_offset_weight,
        formation_switch_weight=args.switch_weight,
        formation_oscillation_weight=args.oscillation_weight,
        heuristic_action_mask=not args.adjacent_only_mask,
        nearest_safe_offset_reward=not args.direct_offset_formation_reward,
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
    tensorboard_writer = None
    if not args.no_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError(
                "TensorBoard is enabled but not installed; run "
                "'python3 -m pip install tensorboard' or pass --no-tensorboard"
            ) from error
        tensorboard_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
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
                "fixed_evaluation_seed": args.eval_seed,
                "tensorboard": not args.no_tensorboard,
                "early_stopping": {
                    "min_steps": args.min_steps_before_stop,
                    "success_rate": args.early_stop_success_rate,
                    "max_collision_rate": args.early_stop_max_collision_rate,
                    "max_pair_collision_rate": args.early_stop_max_pair_collision_rate,
                    "min_clear_default_action_rate": args.early_stop_min_clear_default_rate,
                    "min_empty_success_rate": args.early_stop_min_empty_success_rate,
                    "min_obstacle_success_rate": args.early_stop_min_obstacle_success_rate,
                    "max_action_switch_rate": args.early_stop_max_switch_rate,
                    "max_action_oscillation_rate": args.early_stop_max_oscillation_rate,
                    "consecutive_evaluations": args.early_stop_consecutive_evals,
                    "required": args.require_convergence,
                },
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
    eval_metrics_file = (run_dir / "evaluations.csv").open(
        "w", newline="", encoding="utf-8"
    )
    eval_writer = csv.DictWriter(
        eval_metrics_file,
        fieldnames=(
            "global_step",
            "reward",
            "reward_std",
            "success_rate",
            "collision_rate",
            "pair_collision_rate",
            "episode_length",
            "min_obstacle_clearance",
            "min_pair_distance",
            "clear_default_action_rate",
            "action_switch_rate",
            "action_oscillation_rate",
            "empty_success_rate",
            "obstacle_success_rate",
        ),
    )
    eval_writer.writeheader()

    initial_y_range = (
        EnvConfig().obstacle_abs_y_range
        if args.curriculum_steps > 0
        else env_cfg.obstacle_abs_y_range
    )
    env.set_obstacle_abs_y_range(initial_y_range)
    observations, _ = env.reset(seed=train_cfg.seed)
    episode_reward, episode_length, episode_index = 0.0, 0, 0
    reward_component_keys = (
        "reward_task",
        "reward_obstacle",
        "reward_pair",
        "reward_formation",
        "reward_progress",
    )
    episode_components = {key: 0.0 for key in reward_component_keys}
    recent_rewards, recent_successes, recent_collisions = [], [], []
    moving_rewards = deque(maxlen=50)
    moving_successes = deque(maxlen=50)
    moving_collisions = deque(maxlen=50)
    best_score = -float("inf")
    convergence_streak = 0
    convergence_met = False
    completed_step = 0
    final_eval_metrics = None
    last_losses = {"actor_loss": float("nan"), "critic_loss": float("nan")}
    print(
        f"run={run_dir}\ndevice={device} obs={env.obs_dim} actions={env.num_actions} "
        f"lidar={env_cfg.lidar_sim_rays}->{env_cfg.lidar_observation_size}\n"
        f"initialized_from={args.init_checkpoint}\n"
        f"shared_actor={args.shared_actor}\n"
        f"obstacles={env_cfg.obstacle_count} eval_seed={args.eval_seed}\n"
        f"obstacle_abs_y={initial_y_range}->{env_cfg.obstacle_abs_y_range} "
        f"curriculum_steps={args.curriculum_steps}"
    )

    try:
        # Establish the untrained/warm-start baseline before any replay or
        # gradient update. This makes actual learning visible in TensorBoard.
        initial_eval_metrics = evaluate(
            policy,
            env_cfg,
            train_cfg.eval_episodes,
            seed=args.eval_seed,
        )
        final_eval_metrics = initial_eval_metrics
        print(f"evaluation step=0: {initial_eval_metrics}")
        eval_writer.writerow(
            {
                key: initial_eval_metrics[key]
                for key in eval_writer.fieldnames
                if key != "global_step"
            }
            | {"global_step": 0}
        )
        eval_metrics_file.flush()
        if tensorboard_writer is not None:
            for key in (
                "reward",
                "reward_std",
                "success_rate",
                "collision_rate",
                "pair_collision_rate",
                "episode_length",
                "min_obstacle_clearance",
                "min_pair_distance",
                "clear_default_action_rate",
                "action_switch_rate",
                "action_oscillation_rate",
                "empty_success_rate",
                "obstacle_success_rate",
            ):
                tensorboard_writer.add_scalar(
                    f"eval/{key}", initial_eval_metrics[key], 0
                )
            action_total = max(sum(initial_eval_metrics["action_counts"]), 1)
            for action, count in enumerate(initial_eval_metrics["action_counts"]):
                tensorboard_writer.add_scalar(
                    f"eval_action_fraction/action_{action}", count / action_total, 0
                )
            tensorboard_writer.flush()
        initial_metadata = {
            "global_step": 0,
            "environment": env_cfg.to_dict(),
            "training": train_cfg.to_dict(),
            "initialized_from": str(args.init_checkpoint) if args.init_checkpoint else None,
            "curriculum": {
                "initial_obstacle_abs_y": list(initial_y_range),
                "final_obstacle_abs_y": list(env_cfg.obstacle_abs_y_range),
                "steps": args.curriculum_steps,
            },
            "shared_actor": args.shared_actor,
            "fixed_evaluation_seed": args.eval_seed,
            "evaluation": initial_eval_metrics,
        }
        best_score = evaluation_score(initial_eval_metrics)
        policy.save(run_dir / "checkpoint_step_0000000.pt", metadata=initial_metadata)
        policy.save(run_dir / "best_model.pt", metadata=initial_metadata)

        for global_step in range(1, train_cfg.total_steps + 1):
            completed_step = global_step
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
            for key in reward_component_keys:
                episode_components[key] += float(info[key])

            if (
                global_step > train_cfg.warmup_steps
                and len(replay) >= train_cfg.batch_size
                and global_step % train_cfg.update_every == 0
            ):
                for _ in range(train_cfg.updates_per_step):
                    last_losses = policy.update(
                        replay.sample(train_cfg.batch_size), temperature=temperature
                    )
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar(
                            "loss/actor", last_losses["actor_loss"], global_step
                        )
                        tensorboard_writer.add_scalar(
                            "loss/critic", last_losses["critic_loss"], global_step
                        )

            if done:
                episode_index += 1
                success = float(info.get("success", False))
                collision = float(info.get("collision", False))
                recent_rewards.append(episode_reward)
                recent_successes.append(success)
                recent_collisions.append(collision)
                moving_rewards.append(episode_reward)
                moving_successes.append(success)
                moving_collisions.append(collision)
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
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        "train/episode_reward", episode_reward, global_step
                    )
                    tensorboard_writer.add_scalar(
                        "train/episode_length", episode_length, global_step
                    )
                    tensorboard_writer.add_scalar(
                        "train/reward_moving_average_50",
                        float(np.mean(moving_rewards)),
                        global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "train/success_rate_50",
                        float(np.mean(moving_successes)),
                        global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "train/collision_rate_50",
                        float(np.mean(moving_collisions)),
                        global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "exploration/epsilon", epsilon, global_step
                    )
                    tensorboard_writer.add_scalar(
                        "exploration/temperature", temperature, global_step
                    )
                    for key, value in episode_components.items():
                        tensorboard_writer.add_scalar(
                            f"train_reward_components/{key.removeprefix('reward_')}",
                            value,
                            global_step,
                        )
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
                episode_components = {key: 0.0 for key in reward_component_keys}

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
                    seed=args.eval_seed,
                )
                score = evaluation_score(eval_metrics)
                final_eval_metrics = eval_metrics
                print(f"evaluation step={global_step}: {eval_metrics}")
                eval_writer.writerow(
                    {
                        key: eval_metrics[key]
                        for key in eval_writer.fieldnames
                        if key != "global_step"
                    }
                    | {"global_step": global_step}
                )
                eval_metrics_file.flush()
                if tensorboard_writer is not None:
                    for key in (
                        "reward",
                        "reward_std",
                        "success_rate",
                        "collision_rate",
                        "pair_collision_rate",
                        "episode_length",
                        "min_obstacle_clearance",
                        "min_pair_distance",
                        "clear_default_action_rate",
                        "action_switch_rate",
                        "action_oscillation_rate",
                        "empty_success_rate",
                        "obstacle_success_rate",
                    ):
                        tensorboard_writer.add_scalar(
                            f"eval/{key}", eval_metrics[key], global_step
                        )
                    action_total = max(sum(eval_metrics["action_counts"]), 1)
                    for action, count in enumerate(eval_metrics["action_counts"]):
                        tensorboard_writer.add_scalar(
                            f"eval_action_fraction/action_{action}",
                            count / action_total,
                            global_step,
                        )
                    tensorboard_writer.flush()
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
                    "fixed_evaluation_seed": args.eval_seed,
                    "evaluation": eval_metrics,
                }
                policy.save(run_dir / "latest_model.pt", metadata=metadata)
                policy.save(
                    run_dir / f"checkpoint_step_{global_step:07d}.pt",
                    metadata=metadata,
                )
                if score > best_score:
                    best_score = score
                    policy.save(run_dir / "best_model.pt", metadata=metadata)
                criteria_enabled = args.min_steps_before_stop > 0
                evaluation_passed = bool(
                    global_step >= args.min_steps_before_stop
                    and eval_metrics["success_rate"] >= args.early_stop_success_rate
                    and eval_metrics["collision_rate"]
                    <= args.early_stop_max_collision_rate
                    and eval_metrics["pair_collision_rate"]
                    <= args.early_stop_max_pair_collision_rate
                    and eval_metrics["clear_default_action_rate"]
                    >= args.early_stop_min_clear_default_rate
                    and eval_metrics["empty_success_rate"]
                    >= args.early_stop_min_empty_success_rate
                    and eval_metrics["obstacle_success_rate"]
                    >= args.early_stop_min_obstacle_success_rate
                    and eval_metrics["action_switch_rate"]
                    <= args.early_stop_max_switch_rate
                    and eval_metrics["action_oscillation_rate"]
                    <= args.early_stop_max_oscillation_rate
                )
                if criteria_enabled:
                    convergence_streak = convergence_streak + 1 if evaluation_passed else 0
                    if tensorboard_writer is not None:
                        tensorboard_writer.add_scalar(
                            "eval/convergence_streak", convergence_streak, global_step
                        )
                    if convergence_streak >= args.early_stop_consecutive_evals:
                        convergence_met = True
                        policy.save(run_dir / "converged_model.pt", metadata=metadata)
                        print(
                            "convergence criterion met: "
                            f"step={global_step} consecutive={convergence_streak} "
                            f"success={eval_metrics['success_rate']:.3f} "
                            f"collision={eval_metrics['collision_rate']:.3f}"
                        )
                        break
    finally:
        metrics_file.close()
        eval_metrics_file.close()
        if tensorboard_writer is not None:
            tensorboard_writer.close()

    summary = {
        "completed_step": completed_step,
        "maximum_steps": train_cfg.total_steps,
        "convergence_criterion_enabled": args.min_steps_before_stop > 0,
        "convergence_criterion_met": convergence_met,
        "consecutive_passing_evaluations": convergence_streak,
        "final_evaluation": final_eval_metrics,
    }
    with (run_dir / "training_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    if args.require_convergence and not convergence_met:
        raise RuntimeError(
            "maximum training steps reached before the required convergence "
            f"criterion was met; see {run_dir / 'training_summary.json'}"
        )
    print(f"training complete: {run_dir}")


if __name__ == "__main__":
    main()
