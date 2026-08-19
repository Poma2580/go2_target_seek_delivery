import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MaxNLocator


def setup_chinese_font():
    """Configure Matplotlib font fallback for Chinese plot labels."""
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


def display_agent_name(agent_name):
    """Translate default agent labels such as agent_0 to Chinese for plot legends."""
    if isinstance(agent_name, str) and agent_name.startswith("agent_"):
        suffix = agent_name.split("agent_", 1)[1]
        if suffix.isdigit():
            return f"智能体{suffix}"
    return agent_name


def display_algorithm_name(algo_name):
    """Translate known algorithm display names while preserving MADDPG."""
    return {
        "MADDPG-Approx": "MADDPG-近似版",
    }.get(algo_name, algo_name)


setup_chinese_font()

def running_average(x, window_size):
    """计算滑动平均"""
    if window_size == 0:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / window_size

def plot_rewards_single_env(agents, agent_rewards, output_dir, env_name, target_score=None, window_size=100):
    num_agents = len(agents)
    rewards_array = np.array(agent_rewards)  # (num_agents, num_episodes)

    # 创建总奖励曲线
    total_rewards = np.sum(rewards_array, axis=0)

    # 创建绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    fig.suptitle(f'{env_name} - MADDPG 训练奖励', fontsize=16)

    # 子图1：每个智能体的奖励
    ax1.set_title('各智能体单回合奖励')
    ax1.set_xlabel('回合')
    ax1.set_ylabel('奖励')
    ax1.grid(True, alpha=0.3)

    for i, agent in enumerate(agents):
        rewards = rewards_array[i]
        # 绘制原始奖励
        ax1.plot(rewards, alpha=0.3, label=f'{display_agent_name(agent)}（原始值）')
        # 绘制滑动平均
        if len(rewards) > window_size:
            avg_rewards = running_average(rewards, window_size)
            ax1.plot(range(window_size - 1, len(rewards)), avg_rewards, label=f'{display_agent_name(agent)}（平均值 {window_size}）')

    if target_score is not None:
        ax1.axhline(y=target_score, color='red', linestyle='--', label=f'目标分数: {target_score}')

    ax1.legend(fontsize=8)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 子图2：团队总奖励
    ax2.set_title('团队总回合奖励')
    ax2.set_xlabel('回合')
    ax2.set_ylabel('总奖励')
    ax2.grid(True, alpha=0.3)

    # 绘制原始总奖励
    ax2.plot(total_rewards, alpha=0.3, label='总奖励（原始值）')
    # 绘制滑动平均
    if len(total_rewards) > window_size:
        avg_total = running_average(total_rewards, window_size)
        ax2.plot(range(window_size - 1, len(total_rewards)), avg_total,
                 color='red', label=f'总奖励（平均值 {window_size}）')

    if target_score is not None:
        ax2.axhline(y=target_score * num_agents, color='green', linestyle='--',
                    label=f'团队目标总分: {target_score * num_agents}')

    ax2.legend(fontsize=8)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存图片
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'training_rewards.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved reward plot to {plot_path}")

    # 额外绘制编队指标（如果有）
    # （如果train.py保存了formation_error等指标，可以在这里补充）

def compare_algorithms(env_name, algo_paths, output_dir, window_size=100):
    """
        对比不同算法的奖励曲线
        """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle(f'{env_name} - 算法对比', fontsize=16)

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    color_idx = 0

    for algo_name, rewards_path in algo_paths.items():
        if not os.path.exists(rewards_path):
            print(f"Warning: Rewards file {rewards_path} not found for {algo_name}")
            continue

        # 加载奖励数据
        agent_rewards = np.load(rewards_path, allow_pickle=True)
        total_rewards = np.sum(agent_rewards, axis=0)

        # 绘制原始奖励
        ax.plot(total_rewards, alpha=0.3, color=colors[color_idx])
        # 绘制滑动平均
        if len(total_rewards) > window_size:
            avg_total = running_average(total_rewards, window_size)
            ax.plot(range(window_size - 1, len(total_rewards)), avg_total,
                    color=colors[color_idx], label=f'{display_algorithm_name(algo_name)}（平均值 {window_size}）')

        color_idx = (color_idx + 1) % len(colors)

    ax.set_xlabel('回合')
    ax.set_ylabel('团队总奖励')
    ax.set_title('算法对比 - 总奖励')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 保存图片
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'algorithm_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_path}")





