"""
Training script for MADDPG with PettingZoo
"""
import torch
import numpy as np
import os
import argparse
from tqdm import tqdm
from datetime import datetime


from maddpg import MADDPG, MADDPGSharedActor, ReplayBuffer
from utils.env import get_env_info, ENV_MAP, create_single_env
from utils.logger import Logger
from utils.utils import evaluate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", type=str, default="formation_navigation_v0",
                       choices=list(ENV_MAP.keys()),
                       help="Name of the environment to use")
    parser.add_argument("--algo", type=str, default="MADDPG", choices=["MADDPG",],
                       help="Algorithm to use")
    parser.add_argument("--total-timesteps", type=int, default=int(2e6), help="Total timesteps")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="Replay buffer size")
    parser.add_argument("--warmup-steps", type=int, default=50000, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--max-steps", type=int, default=250, help="Maximum steps per episode")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft update parameter")
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=5e-4, help="Critic learning rate")
    parser.add_argument("--hidden-sizes", type=str, default="128,128", help="Hidden layer sizes (comma-separated)")
    parser.add_argument("--update-every", type=int, default=20, help="Update networks every n steps")
    parser.add_argument("--noise-scale", type=float, default=0.4, help="Initial noise scale")
    parser.add_argument("--min-noise", type=float, default=0.05, help="Minimum noise scale")
    parser.add_argument("--noise-decay-steps", type=int, 
                        default=int(2e6),
                        help="Number of step to decay noise to min_noise default: 300k")
    parser.add_argument("--use-noise-decay", action="store_true", help="Use noise decay")
    parser.add_argument("--render-mode", type=str, default=None, choices=[None, "human", "rgb_array"], 
                       help="Render mode for visualization")
    parser.add_argument("--create-gif", action="store_true", help="Create GIF of episodes")
    parser.add_argument("--eval-interval", type=int, default=10000, help="Evaluate every n steps")
    parser.add_argument("--pretrained-model-path", type=str, default=None,
                       help="Path to pretrained model for incremental training")
    parser.add_argument("--training-stage", type=int, default=4, choices=[1, 2, 3, 4, 5],
                       help=("Curriculum stage for leader-follower environment: "
                             "1=no obstacles, 2=fixed obstacles, "
                             "3=random near-formation init + fixed obstacles, "
                             "4=scattered init + fixed obstacles, "
                             "5=scattered init + random obstacles"))
    parser.add_argument("--shared-actor", action="store_true",
                       help="Use one shared actor policy for all follower agents")
    parser.add_argument("--early-stop-success-rate", type=float, default=None,
                       help="Stop training early after evaluation success rate reaches this value")
    parser.add_argument("--early-stop-patience", type=int, default=2,
                       help="Number of consecutive successful evaluations required for early stop")
    parser.add_argument("--early-stop-min-steps", type=int, default=3000,
                       help="Minimum training steps before early stop can trigger")

    return parser.parse_args()

def train(args):

    # Add timestamp to experiment name for uniqueness
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = (
        f"stage{args.training_stage}"
        f"{'_shared_actor' if getattr(args, 'shared_actor', False) else ''}"
        f"_b{args.batch_size}"
        f"_usteps{args.update_every}"
        f"_g{args.gamma}"
        f"_t{args.tau}"
        f"_alr{args.actor_lr}"
        f"_clr{args.critic_lr}"
        f"_n{args.noise_scale}"
        f"_minn{args.min_noise}"
        f"_h{args.hidden_sizes}"
        f"_{timestamp}")

    logger = Logger(
        run_name=experiment_name,
        folder="runs",
        algo=args.algo,
        env=args.env_name
    )
    logger.log_all_hyperparameters(vars(args))


    # Get environment information
    agents, num_agents, action_sizes, action_low, action_high, state_sizes = get_env_info(
        env_name=args.env_name,
        max_steps=args.max_steps,
        apply_padding=False,
        training_stage=args.training_stage
    )

    # Create environment with appropriate render mode
    env = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode=args.render_mode,
        apply_padding=False,
        training_stage=args.training_stage
    )

    # Create evaluation environment
    env_evaluate = create_single_env(
        env_name=args.env_name,
        max_steps=args.max_steps,
        render_mode="rgb_array",
        apply_padding=False,
        training_stage=args.training_stage
    )

    # Model path
    model_path = os.path.join(logger.dir_name, "model.pt")
    best_model_path = os.path.join(logger.dir_name, "best_model.pt")
    best_score = -float('inf')
    episode_count = 0
    consecutive_success_evals = 0

    # Parse hidden sizes
    hidden_sizes = tuple(map(int, args.hidden_sizes.split(',')))

    # Create MADDPG agent
    if args.algo == "MADDPG":
        maddpg_cls = MADDPGSharedActor if getattr(args, "shared_actor", False) else MADDPG
        maddpg = maddpg_cls(
            state_sizes=state_sizes,
            action_sizes=action_sizes,
            hidden_sizes=hidden_sizes,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            gamma=args.gamma,
            tau=args.tau,
            action_low=action_low,
            action_high=action_high
        )
        if getattr(args, "shared_actor", False):
            print("Using shared actor policy for all follower agents.")

    # 新增：加载预训练模型
    if args.pretrained_model_path is not None and os.path.exists(args.pretrained_model_path):
        maddpg.load_with_obs_padding(args.pretrained_model_path)
        print(f"Loaded pretrained model with observation padding from {args.pretrained_model_path}")

    # Create replay buffer with the correct dimensions
    buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        agents=agents,
        state_sizes=state_sizes,
        action_sizes=action_sizes
    )

    # Training loop
    noise_scale = args.noise_scale
    noise_decay = (args.noise_scale - args.min_noise) / min(args.noise_decay_steps, args.total_timesteps)
    print(f"Using linear noise decay: {args.noise_scale} to {args.min_noise} over {args.noise_decay_steps} steps")
    print(f"Noise will decrease by {noise_decay:.6f} per step")

    _, _ = evaluate(env_evaluate, maddpg, logger, record_gif=args.create_gif, num_eval_episodes=10, global_step=0)

    # For tracking agent-specific rewards
    agent_rewards = [[] for _ in range(len(agents))]
    episode_rewards = np.zeros(len(agents))

    # Reset environment and agents
    observations, _ = env.reset()

    for global_step in tqdm(range(1, args.total_timesteps + 1), desc="Training"):

        # Get states for all agents
        states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]

        # Get actions for all agents
        actions_list = maddpg.act(states_list, add_noise=True, noise_scale=noise_scale)

        # Convert actions to dictionary for environment
        actions = {agent: action for agent, action in zip(agents, actions_list)}

        # Take a step in the environment
        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        # 【新增】从 infos 中提取编队指标（所有智能体的 info 一致，取第一个即可）
        # 只有当 infos 不为空且包含我们需要的键时才读取（兼容旧环境）
        slot_error = 0.0
        mean_slot_error = 0.0
        formation_hold_count = 0
        formation_error = 0.0

        team_target_distance = 0.0
        success_hold_count = 0
        navigation_active = 0.0
        total_obstacle_collisions = 0
        any_obstacle_collision = 0.0
        min_obstacle_dist = 0.0
        leader_target_distance = 0.0
        leader_target_progress = 0.0
        max_follower_slot_error = 0.0
        follower_shape_error = 0.0
        follower_detach_norm = 0.0
        total_agent_collisions = 0

        if agents and agents[0] in infos:
            info = infos[agents[0]]
            if "slot_error" in info:
                slot_error = info["slot_error"]
            if "mean_slot_error" in info:
                mean_slot_error = info["mean_slot_error"]
            if "formation_hold_count" in info:
                formation_hold_count = info["formation_hold_count"]
            if "success_hold_count" in info:
                success_hold_count = info["success_hold_count"]
            if "team_target_distance" in info:
                team_target_distance = info["team_target_distance"]
            if "navigation_active" in info:
                navigation_active = info["navigation_active"]
            if "formation_error" in info:
                formation_error = info["formation_error"]
            if "total_obstacle_collisions" in info:
                total_obstacle_collisions = info["total_obstacle_collisions"]
            if "any_obstacle_collision" in info:
                any_obstacle_collision = info["any_obstacle_collision"]
            if "min_obstacle_dist" in info:
                min_obstacle_dist = info["min_obstacle_dist"]
            if "leader_target_distance" in info:
                leader_target_distance = info["leader_target_distance"]
            if "leader_target_progress" in info:
                leader_target_progress = info["leader_target_progress"]
            if "max_follower_slot_error" in info:
                max_follower_slot_error = info["max_follower_slot_error"]
            if "follower_shape_error" in info:
                follower_shape_error = info["follower_shape_error"]
            if "follower_detach_norm" in info:
                follower_detach_norm = info["follower_detach_norm"]
            if "total_agent_collisions" in info:
                total_agent_collisions = info["total_agent_collisions"]

        # Check if episode is done
        dones = [terminations[agent] or truncations[agent] for agent in agents]
        done = any(dones)

        # Prepare data for buffer (convert to NumPy once)
        rewards_array = np.array([rewards[agent] for agent in agents], dtype=np.float32)
        next_states_list = [np.array(next_observations[agent], dtype=np.float32) for agent in agents]
        # we care about the termination of the episode
        terminations_array = np.array([terminations[agent] for agent in agents], dtype=np.uint8)

        # Store experience in replay buffer
        buffer.add(
            states=states_list,
            actions=actions_list,
            rewards=rewards_array,
            next_states=next_states_list,
            dones=terminations_array
        )

        # Update observations and rewards
        observations = next_observations
        episode_rewards += np.array(list(rewards.values()))

        # Learn only after the replay buffer has enough samples for one full batch.
        # This is important when warmup_steps is 0 or smaller than batch_size; otherwise
        # ReplayBuffer.sample() will try to sample more items than currently stored.
        if (
            len(buffer) >= args.batch_size
            and global_step > args.warmup_steps
            and global_step % args.update_every == 0
        ):
            for i in range(len(agents)):
                experiences = buffer.sample()  # Now returns pre-combined states
                critic_loss, actor_loss = maddpg.learn(experiences, i)

                # Log losses to TensorBoard
                logger.add_scalar(f'{agents[i]}/critic_loss', critic_loss, global_step)
                logger.add_scalar(f'{agents[i]}/actor_loss', actor_loss, global_step)

            maddpg.update_targets()

        # Update noise scale based on iteration number
        if global_step > args.warmup_steps and args.use_noise_decay:
            noise_scale = max(
                args.min_noise,
                noise_scale - noise_decay
            )

        # Handle episode end
        if done or (global_step % args.max_steps == 0):  # Reset after max_steps if not done
            episode_count += 1
            for i, reward in enumerate(episode_rewards):
                agent_rewards[i].append(reward)
                logger.add_scalar(f"{agents[i]}/episode_reward", reward, global_step)
            train_total_reward = float(np.sum(episode_rewards))
            logger.add_scalar('train/total_reward', train_total_reward, global_step)
            if episode_count % 10 == 0:
                print(
                    f"Episode {episode_count}: "
                    f"step={global_step}, "
                    f"train_total_reward={train_total_reward:.2f}, "
                    f"mean_agent_reward={train_total_reward / max(len(agents), 1):.2f}, "
                    f"slot_error={mean_slot_error:.3f}, "
                    f"max_slot_error={max_follower_slot_error:.3f}, "
                    f"hold={success_hold_count}"
                )
            logger.add_scalar(f"noise/scale", noise_scale, global_step)
            # 在evaluate函数调用后，从infos中读取指标，写入logger
            logger.add_scalar('train/slot_error', slot_error, global_step)
            logger.add_scalar('train/mean_slot_error', mean_slot_error, global_step)
            logger.add_scalar('train/formation_hold_count', formation_hold_count, global_step)
            logger.add_scalar('train/team_target_distance', team_target_distance, global_step)
            logger.add_scalar('train/formation_error', formation_error, global_step)
            logger.add_scalar('train/success_hold_count', success_hold_count, global_step)
            logger.add_scalar('train/navigation_active', navigation_active, global_step)
            logger.add_scalar('train/total_obstacle_collisions', total_obstacle_collisions, global_step)
            logger.add_scalar('train/any_obstacle_collision', any_obstacle_collision, global_step)
            logger.add_scalar('train/min_obstacle_dist', min_obstacle_dist, global_step)
            logger.add_scalar('train/leader_target_distance', leader_target_distance, global_step)
            logger.add_scalar('train/leader_target_progress', leader_target_progress, global_step)
            logger.add_scalar('train/max_follower_slot_error', max_follower_slot_error, global_step)
            logger.add_scalar('train/follower_shape_error', follower_shape_error, global_step)
            logger.add_scalar('train/follower_detach_norm', follower_detach_norm, global_step)
            logger.add_scalar('train/total_agent_collisions', total_agent_collisions, global_step)
            observations, _ = env.reset()
            episode_rewards = np.zeros(len(agents))
        
        # Evaluate and save
        if global_step % args.eval_interval == 0 or global_step == args.total_timesteps:
            maddpg.save(model_path)
            avg_eval_rewards, eval_success_rate = evaluate(env_evaluate, maddpg, logger,
                    num_eval_episodes=20, record_gif=args.create_gif, global_step=global_step)
            eval_total_reward = float(np.sum(avg_eval_rewards))
            print(
                f"Eval step={global_step}: "
                f"success_rate={eval_success_rate * 100:.1f}%, "
                f"avg_total_reward={eval_total_reward:.2f}, "
                f"avg_agent_reward={eval_total_reward / max(len(agents), 1):.2f}"
            )
            np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
            # 用成功率+平均奖励综合打分，优先保存能成功到达终点的模型
            #success_rate = np.sum([1 for r in avg_eval_rewards if r > 0]) / len(avg_eval_rewards)
            score = eval_success_rate * 1000 + np.sum(avg_eval_rewards)  # 成功率权重远高于平均奖励
            if score > best_score:
                best_score = score
                maddpg.save(best_model_path)
                print(
                    f"New best model saved! Success rate: {eval_success_rate * 100:.1f}%, Avg reward: {np.sum(avg_eval_rewards):.2f}")
            if args.early_stop_success_rate is not None:
                if (
                    global_step >= args.early_stop_min_steps
                    and eval_success_rate >= args.early_stop_success_rate
                ):
                    consecutive_success_evals += 1
                else:
                    consecutive_success_evals = 0
                print(
                    f"Early-stop check: success_rate={eval_success_rate * 100:.1f}% "
                    f">= {args.early_stop_success_rate * 100:.1f}% "
                    f"for {consecutive_success_evals}/{args.early_stop_patience} evals, "
                    f"min_steps={args.early_stop_min_steps}"
                )
                if consecutive_success_evals >= args.early_stop_patience:
                    print(
                        f"Early stopping at step {global_step}: "
                        f"{consecutive_success_evals} consecutive successful evaluations."
                    )
                    break
    
    # Save final models
    maddpg.save(model_path)
    np.save(os.path.join(logger.dir_name, "agent_rewards.npy"), agent_rewards)
    
    # Close environment and TensorBoard writer
    env.close()
    env_evaluate.close()
    logger.close()
    
    # Return both the agent rewards and the experiment name
    return agent_rewards, experiment_name

if __name__ == "__main__":
    args = parse_args()
    train(args)
