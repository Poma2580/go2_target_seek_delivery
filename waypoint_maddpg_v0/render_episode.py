"""Render one deterministic-policy episode from a trained checkpoint as a GIF."""

import argparse
from dataclasses import fields
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from .config import EnvConfig
from .discrete_maddpg import DiscreteMADDPG
from .environment import WaypointSelectionEnv


def _environment_from_checkpoint(payload):
    checkpoint_env = payload.get("metadata", {}).get("environment", {})
    valid_fields = {field.name for field in fields(EnvConfig)}
    values = {key: value for key, value in checkpoint_env.items() if key in valid_fields}
    if "candidate_offsets" in values:
        values["candidate_offsets"] = tuple(values["candidate_offsets"])
    if "obstacle_spawn_x" in values:
        values["obstacle_spawn_x"] = tuple(values["obstacle_spawn_x"])
    if "obstacle_abs_y_range" in values:
        values["obstacle_abs_y_range"] = tuple(values["obstacle_abs_y_range"])
    return EnvConfig(**values)


def collect_episode(checkpoint, seed, obstacle_x=None, obstacle_y=None):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    env_cfg = _environment_from_checkpoint(payload)
    env = WaypointSelectionEnv(env_cfg, seed=seed, lidar_noise=True)
    policy = DiscreteMADDPG(
        env.num_agents,
        env.obs_dim,
        env.num_actions,
        hidden_dim=int(payload.get("hidden_dim", 256)),
        device=torch.device("cpu"),
        shared_actor=bool(payload.get("shared_actor", False)),
    )
    policy.load(checkpoint)

    observations, initial_info = env.reset(seed=seed)
    if obstacle_x is not None or obstacle_y is not None:
        obstacle_center = env.obstacle_center.copy()
        if obstacle_x is not None:
            obstacle_center[0] = float(obstacle_x)
        if obstacle_y is not None:
            obstacle_center[1] = float(obstacle_y)
        env.obstacle_center = obstacle_center
        half = 0.5 * env.cfg.obstacle_size
        env.obstacle_lower = env.obstacle_center - half
        env.obstacle_upper = env.obstacle_center + half
        env.obstacles[0].update(
            center=env.obstacle_center,
            lower=env.obstacle_lower,
            upper=env.obstacle_upper,
        )
        observations, env._candidate_metrics = env._build_observations()
        env.default_blocked_latched.fill(False)
        env.default_clear_counts.fill(0)
        env._update_default_path_state()
        initial_info = env._info(False, False, np.inf, np.inf)
    frames = [
        {
            "leader": initial_info["leader_position"].copy(),
            "followers": initial_info["follower_positions"].copy(),
            "goals": env.current_goals.copy(),
            "actions": env.previous_actions.copy(),
            "reward": 0.0,
            "info": initial_info,
        }
    ]
    total_reward = 0.0
    while True:
        actions = policy.act(
            observations,
            action_masks=env.valid_action_masks(),
            deterministic=True,
        )
        observations, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(rewards[0])
        frames.append(
            {
                "leader": info["leader_position"].copy(),
                "followers": info["follower_positions"].copy(),
                "goals": info["goals"].copy(),
                "actions": actions.copy(),
                "reward": total_reward,
                "info": info,
            }
        )
        if terminated or truncated:
            break
    return env_cfg, frames


def render_gif(checkpoint, output, seed, fps, obstacle_x=None, obstacle_y=None):
    cfg, frames = collect_episode(checkpoint, seed, obstacle_x, obstacle_y)
    obstacles = frames[0]["info"]["obstacles"]

    all_x = np.concatenate(
        [np.asarray([frame["leader"] for frame in frames])[:, 0]]
        + [np.asarray([frame["followers"][i] for frame in frames])[:, 0] for i in range(2)]
    )
    x_min = min(-0.75, float(np.min(all_x)) - 0.75)
    obstacle_x_max = max(
        float(obstacle["center"][0])
        + (obstacle["size"] / 2.0 if obstacle["shape"] == "square" else obstacle["radius"])
        for obstacle in obstacles
    )
    obstacle_y_extent = max(
        abs(float(obstacle["center"][1]))
        + (obstacle["size"] / 2.0 if obstacle["shape"] == "square" else obstacle["radius"])
        for obstacle in obstacles
    )
    x_max = max(float(np.max(all_x)) + 1.5, obstacle_x_max + 3.0)
    y_extent = max(5.0, obstacle_y_extent + 1.5)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=100)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.25)
    ax.axhline(cfg.formation_side, color="#4c78a8", alpha=0.12, linestyle="--")
    ax.axhline(-cfg.formation_side, color="#f58518", alpha=0.12, linestyle="--")
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"])
        if obstacle["shape"] == "square":
            half = obstacle["size"] / 2.0
            patch = Rectangle(
                center - half,
                obstacle["size"],
                obstacle["size"],
                facecolor="#d62728",
                edgecolor="#8b0000",
                alpha=0.72,
                label="1.5 m square",
            )
        else:
            patch = Circle(
                center,
                obstacle["radius"],
                facecolor="#e45756",
                edgecolor="#8b0000",
                alpha=0.72,
                label="radius 1 m circle",
            )
        ax.add_patch(patch)

    colors = ("#4c78a8", "#f58518")
    leader_circle = Circle((0, 0), cfg.robot_radius, color="#54a24b", zorder=5)
    follower_circles = [Circle((0, 0), cfg.robot_radius, color=color, zorder=5) for color in colors]
    ax.add_patch(leader_circle)
    for circle in follower_circles:
        ax.add_patch(circle)

    leader_trail, = ax.plot([], [], color="#54a24b", linewidth=2, label="go1 leader")
    follower_trails = [
        ax.plot([], [], color=color, linewidth=2, label=f"go{i + 2} trajectory")[0]
        for i, color in enumerate(colors)
    ]
    initial_defaults = np.asarray(
        [
            frames[0]["leader"] + [cfg.formation_forward, cfg.formation_side],
            frames[0]["leader"] + [cfg.formation_forward, -cfg.formation_side],
        ]
    )
    default_markers = ax.scatter(
        initial_defaults[:, 0],
        initial_defaults[:, 1],
        marker="+",
        s=90,
        color=colors,
        linewidth=2,
        zorder=6,
    )
    initial_goals = frames[0]["goals"]
    goal_markers = ax.scatter(
        initial_goals[:, 0],
        initial_goals[:, 1],
        marker="x",
        s=60,
        color=colors,
        linewidth=2,
        zorder=6,
    )
    goal_lines = [ax.plot([], [], color=color, alpha=0.45, linestyle=":")[0] for color in colors]
    status = ax.text(
        0.01,
        0.99,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
    )
    ax.legend(loc="lower left", fontsize=8)

    leader_history = np.asarray([frame["leader"] for frame in frames])
    follower_history = np.asarray([frame["followers"] for frame in frames])

    def update(frame_index):
        frame = frames[frame_index]
        leader = frame["leader"]
        followers = frame["followers"]
        goals = frame["goals"]
        defaults = np.asarray(
            [
                leader + [cfg.formation_forward, cfg.formation_side],
                leader + [cfg.formation_forward, -cfg.formation_side],
            ]
        )

        leader_circle.center = leader
        leader_trail.set_data(leader_history[: frame_index + 1, 0], leader_history[: frame_index + 1, 1])
        default_markers.set_offsets(defaults)
        goal_markers.set_offsets(goals)
        for index in range(2):
            follower_circles[index].center = followers[index]
            follower_trails[index].set_data(
                follower_history[: frame_index + 1, index, 0],
                follower_history[: frame_index + 1, index, 1],
            )
            goal_lines[index].set_data(
                [followers[index, 0], goals[index, 0]],
                [followers[index, 1], goals[index, 1]],
            )

        info = frame["info"]
        outcome = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "RUNNING"
        status.set_text(
            f"seed={seed}  t={info['step'] * cfg.marl_dt:.0f}s  "
            f"step={info['step']:02d}/{cfg.max_episode_steps}  {outcome}\n"
            f"actions={frame['actions'].tolist()}  return={frame['reward']:.2f}"
        )
        return (
            leader_circle,
            leader_trail,
            *follower_circles,
            *follower_trails,
            default_markers,
            goal_markers,
            *goal_lines,
            status,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat_delay=1200,
    )
    movie.save(output, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    final_info = frames[-1]["info"]
    return len(frames), frames[-1]["reward"], final_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=Path("maddpg_episode.gif"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--obstacle-x", type=float, default=None)
    parser.add_argument("--obstacle-y", type=float, default=None)
    args = parser.parse_args()
    frames, reward, info = render_gif(
        args.checkpoint,
        args.output,
        args.seed,
        args.fps,
        obstacle_x=args.obstacle_x,
        obstacle_y=args.obstacle_y,
    )
    print(f"saved: {args.output}")
    print(f"frames: {frames}")
    print(f"reward: {reward}")
    print(f"success: {info.get('success', False)}")
    print(f"collision: {info.get('collision', False)}")


if __name__ == "__main__":
    main()
