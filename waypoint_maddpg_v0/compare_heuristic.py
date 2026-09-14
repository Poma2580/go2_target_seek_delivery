"""Compare a trained policy with a local nearest-safe adjacent heuristic."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np
import torch

from .discrete_maddpg import DiscreteMADDPG
from .environment import WaypointSelectionEnv
from .render_episode import _environment_from_checkpoint


def heuristic_actions(env):
    """Pick the smallest-offset clear adjacent action independently per agent."""
    masks = env.valid_action_masks()
    offsets = np.abs(np.asarray(env.cfg.candidate_offsets, dtype=np.float32))
    actions = []
    for index in range(env.num_agents):
        valid = np.flatnonzero(masks[index]).tolist()
        metrics = env._candidate_metrics[index]
        clear = [action for action in valid if not metrics[action]["blocked"]]
        if clear:
            action = min(
                clear,
                key=lambda value: (
                    offsets[value],
                    -metrics[value]["path_clearance"],
                    -metrics[value]["endpoint_clearance"],
                    value,
                ),
            )
        else:
            action = max(
                valid,
                key=lambda value: (
                    metrics[value]["path_clearance"],
                    metrics[value]["endpoint_clearance"],
                    -offsets[value],
                ),
            )
        actions.append(action)
    return np.asarray(actions, dtype=np.int64)


def collect(env_config, seed, controller):
    env = WaypointSelectionEnv(env_config, seed=seed, lidar_noise=True)
    observations, initial_info = env.reset(seed=seed)
    frames = [{
        "leader": initial_info["leader_position"].copy(),
        "followers": initial_info["follower_positions"].copy(),
        "goals": env.current_goals.copy(),
        "actions": env.previous_actions.copy(),
        "reward": 0.0,
        "info": initial_info,
    }]
    total_reward = 0.0
    while True:
        actions = controller(env, observations)
        observations, rewards, terminated, truncated, info = env.step(actions)
        total_reward += float(rewards[0])
        frames.append({
            "leader": info["leader_position"].copy(),
            "followers": info["follower_positions"].copy(),
            "goals": info["goals"].copy(),
            "actions": actions.copy(),
            "reward": total_reward,
            "info": info,
        })
        if terminated or truncated:
            return frames


def summarize(frames):
    final = frames[-1]["info"]
    actions = np.asarray([frame["actions"] for frame in frames])
    return {
        "success": bool(final.get("success", False)),
        "collision": bool(final.get("collision", False)),
        "steps": int(final["step"]),
        "total_reward": float(frames[-1]["reward"]),
        "action_switches": int(np.sum(actions[1:] != actions[:-1])),
        "min_obstacle_clearance": float(min(
            frame["info"].get("min_obstacle_clearance", np.inf)
            for frame in frames[1:]
        )),
        "min_pair_distance": float(min(
            frame["info"].get("min_pair_distance", np.inf)
            for frame in frames[1:]
        )),
        "final_actions": actions[-1].tolist(),
    }


def add_scene(ax, config, obstacles, title, x_limits, y_extent):
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*x_limits)
    ax.set_ylim(-y_extent, y_extent)
    ax.set_xlabel("x (m)")
    ax.grid(True, alpha=0.25)
    ax.axhline(config.formation_side, color="#4c78a8", alpha=0.12, linestyle="--")
    ax.axhline(-config.formation_side, color="#f58518", alpha=0.12, linestyle="--")
    for obstacle in obstacles:
        center = np.asarray(obstacle["center"])
        if obstacle["shape"] == "square":
            half = obstacle["size"] / 2.0
            patch = Rectangle(
                center - half, obstacle["size"], obstacle["size"],
                facecolor="#d62728", edgecolor="#8b0000", alpha=0.72,
            )
        else:
            patch = Circle(
                center, obstacle["radius"], facecolor="#e45756",
                edgecolor="#8b0000", alpha=0.72,
            )
        ax.add_patch(patch)


def render_comparison(config, seed, rl_frames, heuristic_frames, output, fps):
    histories = (rl_frames, heuristic_frames)
    all_x = np.concatenate([
        np.asarray([frame["leader"] for frame in frames])[:, 0]
        for frames in histories
    ] + [
        np.asarray([frame["followers"] for frame in frames])[:, :, 0].reshape(-1)
        for frames in histories
    ])
    obstacles = rl_frames[0]["info"]["obstacles"]
    obstacle_x_max = max(
        float(item["center"][0])
        + (item["size"] / 2 if item["shape"] == "square" else item["radius"])
        for item in obstacles
    )
    y_extent = max(5.0, max(
        abs(float(item["center"][1]))
        + (item["size"] / 2 if item["shape"] == "square" else item["radius"])
        for item in obstacles
    ) + 1.5)
    x_limits = (min(-0.75, float(all_x.min()) - 0.75), max(float(all_x.max()) + 1.5, obstacle_x_max + 3.0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=100, sharey=True)
    axes[0].set_ylabel("y (m)")
    titles = ("RL (joint learned policy)", "Heuristic (local nearest-safe)")
    colors = ("#4c78a8", "#f58518")
    artists = []
    for ax, title, frames in zip(axes, titles, histories):
        add_scene(ax, config, obstacles, title, x_limits, y_extent)
        leader = Circle((0, 0), config.robot_radius, color="#54a24b", zorder=5)
        followers = [Circle((0, 0), config.robot_radius, color=color, zorder=5) for color in colors]
        ax.add_patch(leader)
        for item in followers:
            ax.add_patch(item)
        leader_trail, = ax.plot([], [], color="#54a24b", linewidth=2)
        follower_trails = [ax.plot([], [], color=color, linewidth=2)[0] for color in colors]
        initial_goals = frames[0]["goals"]
        goals = ax.scatter(
            initial_goals[:, 0], initial_goals[:, 1], marker="x", s=55,
            color=colors, linewidth=2, zorder=6,
        )
        status = ax.text(
            0.01, 0.99, "", transform=ax.transAxes, va="top", family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        artists.append((leader, followers, leader_trail, follower_trails, goals, status, frames))

    def update(frame_index):
        changed = []
        for leader, followers, leader_trail, follower_trails, goals, status, frames in artists:
            index = min(frame_index, len(frames) - 1)
            frame = frames[index]
            leader_history = np.asarray([item["leader"] for item in frames[: index + 1]])
            follower_history = np.asarray([item["followers"] for item in frames[: index + 1]])
            leader.center = frame["leader"]
            leader_trail.set_data(leader_history[:, 0], leader_history[:, 1])
            goals.set_offsets(frame["goals"])
            for agent in range(2):
                followers[agent].center = frame["followers"][agent]
                follower_trails[agent].set_data(
                    follower_history[:, agent, 0], follower_history[:, agent, 1]
                )
            info = frame["info"]
            outcome = "SUCCESS" if info.get("success") else "COLLISION" if info.get("collision") else "RUNNING"
            status.set_text(
                f"seed={seed} step={info['step']:03d} {outcome}\n"
                f"actions={frame['actions'].tolist()} return={frame['reward']:.1f}"
            )
            changed.extend([leader, *followers, leader_trail, *follower_trails, goals, status])
        return changed

    fig.suptitle("Same obstacle layout: learned MARL policy vs local greedy heuristic")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        fig, update, frames=max(map(len, histories)), interval=1000 / fps,
        blit=False, repeat_delay=1200,
    )
    movie.save(output, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=20027)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("rl_vs_heuristic.gif"))
    parser.add_argument("--report", type=Path, default=Path("rl_vs_heuristic.json"))
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = _environment_from_checkpoint(payload)
    policy = DiscreteMADDPG(
        config.num_agents, int(payload["obs_dim"]), config.num_actions,
        hidden_dim=int(payload["hidden_dim"]), device=torch.device("cpu"),
        shared_actor=bool(payload.get("shared_actor", False)),
    )
    policy.load(args.checkpoint)

    rl_frames = collect(
        config, args.seed,
        lambda env, observations: policy.act(
            observations, action_masks=env.valid_action_masks(), deterministic=True
        ),
    )
    heuristic_frames = collect(
        config, args.seed, lambda env, _observations: heuristic_actions(env)
    )
    report = {
        "checkpoint": str(args.checkpoint),
        "seed": args.seed,
        "baseline": "independent nearest-clear adjacent action; max path clearance fallback",
        "obstacles": [
            {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in item.items()}
            for item in rl_frames[0]["info"]["obstacles"]
        ],
        "rl": summarize(rl_frames),
        "heuristic": summarize(heuristic_frames),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    render_comparison(config, args.seed, rl_frames, heuristic_frames, args.output, args.fps)
    print(json.dumps(report, indent=2))
    print(f"saved GIF: {args.output}")
    print(f"saved report: {args.report}")


if __name__ == "__main__":
    main()
