"""
Utility functions.
"""

import os
import imageio
import matplotlib.pyplot as plt
import numpy as np

def needs_padding(sizes):
    """
    Check if padding is needed by determining if all elements in the list are identical.
    
    Args:
        sizes: List of sizes (action or observation)
        
    Returns:
        bool: True if padding is needed (sizes are not all identical), False otherwise
    """
    return len(set(sizes)) > 1

def save_gif(frames, dir, iteration):
    gif_path = os.path.join(dir, f"batch_{iteration}.gif")
    try:
        # Save GIF with appropriate duration (slower for better viewing)
        imageio.mimsave(gif_path, frames, duration=0.1)  # 100ms per frame
        print(f"Saved GIF for iteration {iteration} to {gif_path}")
    except Exception as e:
        print(f"Error saving GIF: {e}")


def evaluate(env, maddpg, logger, record_gif=False, num_eval_episodes=10, end_episode=0, global_step=0):
    eval_rewards = []
    frames = [] if record_gif else None
    success_count = 0  # 新增：统计成功回合数
    last_info_for_logging = {}

    test_seeds = list(range(num_eval_episodes))

    for episode in range(num_eval_episodes):
        seed = test_seeds[episode]
        observations, _ = env.reset(seed=seed)
        episode_rewards = np.zeros(len(env.agents))
        done = False
        step = 0
        episode_success = False

        while not done and step < env.max_cycles:
            states = [np.array(observations[agent], dtype=np.float32) for agent in env.agents]
            actions_list = maddpg.act(states, add_noise=False)
            actions = {agent: action for agent, action in zip(env.agents, actions_list)}
            next_observations, rewards, terminations, truncations, infos = env.step(actions)

            # 提取成功标志和 leader-follower 指标
            if env.agents[0] in infos:
                leader_info = infos[env.agents[0]]
                last_info_for_logging = leader_info
                if leader_info.get("is_success", 0.0) == 1.0:
                    episode_success = True

            dones = [terminations[agent] or truncations[agent] for agent in env.agents]
            done = any(dones)
            observations = next_observations
            episode_rewards += np.array(list(rewards.values()))

            if record_gif:
                frame = env.render()
                frames.append(frame)
            step += 1

        eval_rewards.append(episode_rewards)
        if episode_success:
            success_count += 1

    # 计算平均奖励和成功率
    avg_rewards = np.mean(eval_rewards, axis=0)
    success_rate = success_count / num_eval_episodes

    # 记录到TensorBoard
    logger.add_scalar('eval/avg_total_reward', np.sum(avg_rewards), global_step)
    logger.add_scalar('eval/success_rate', success_rate, global_step)
    for key in [
        "leader_target_distance",
        "mean_slot_error",
        "max_follower_slot_error",
        "follower_shape_error",
        "follower_detach_norm",
        "min_obstacle_dist",
        "total_agent_collisions",
        "total_obstacle_collisions",
    ]:
        if key in last_info_for_logging:
            logger.add_scalar(f'eval/{key}', float(last_info_for_logging[key]), global_step)

    # 保存GIF
    if record_gif and frames:
        # 这里保留你原来的GIF保存逻辑
        pass

    # 【修改】返回平均奖励和成功率
    return avg_rewards, success_rate

def evaluate_ddpg(env, ddpg_agents, logger, record_gif=False, num_eval_episodes=10, end_episode=0, global_step=0):
    """Run evaluation episodes and return average rewards."""
    eval_rewards = [] 
    frames = []  if record_gif else None
    for episode in range(num_eval_episodes):
        observations, _ = env.reset()
        agents = env.agents
        done = False 
        episode_rewards = np.zeros(len(agents))
        while not done:
            states_list = [np.array(observations[agent], dtype=np.float32) for agent in agents]
            actions_list = [ddpg_agents[i].act(states_list[i], add_noise=False) for i in range(len(agents))]
            actions = {agent: action for agent, action in zip(agents, actions_list)}
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            episode_rewards += np.array(list(rewards.values()))
            dones = [terminations[agent] or truncations[agent] for agent in agents]
            done = any(dones)
            if record_gif and episode == num_eval_episodes - 1: 
                frames.append(env.render())
            observations = next_observations
        eval_rewards.append(episode_rewards) # (num_eval_episodes, num_envs)
    avg_eval_rewards = np.mean(eval_rewards, axis=0) # (num_envs,)
    
    for i, avg_reward in enumerate(avg_eval_rewards):
        logger.add_scalar(f'{agents[i]}/eval_reward', avg_reward, end_episode if end_episode > 0 else global_step) 

    total_eval_reward = np.sum(eval_rewards) / num_eval_episodes
    logger.add_scalar('eval/total_reward', total_eval_reward, global_step) 
    print(f"Step {global_step}, Eval rewards: {avg_eval_rewards}, Sum: {total_eval_reward}")    
    
    if frames:
        save_gif(frames, logger.dir_name, end_episode if end_episode > 0 else global_step)           
    
    return avg_eval_rewards
