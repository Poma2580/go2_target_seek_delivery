"""Render five deterministic two-obstacle checkpoint tests in one GIF."""

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


def collect(policy, config, seed):
    env = WaypointSelectionEnv(config, seed=seed, lidar_noise=True)
    observations, initial_info = env.reset(seed=seed)
    if len(env.obstacles) != 2:
        return None

    frames = [
        {
            "leader": initial_info["leader_position"].copy(),
            "followers": initial_info["follower_positions"].copy(),
            "goals": env.current_goals.copy(),
            "actions": env.previous_actions.copy(),
            "step": 0,
            "outcome": "RUNNING",
            "reward": 0.0,
        }
    ]
    total_reward = 0.0
    action_history = []
    final_info = initial_info
    while True:
        actions = policy.act(
            observations,
            action_masks=env.valid_action_masks(),
            deterministic=True,
        )
        observations, rewards, terminated, truncated, final_info = env.step(actions)
        total_reward += float(rewards[0])
        action_history.append(actions.tolist())
        outcome = (
            "SUCCESS"
            if final_info.get("success")
            else "COLLISION"
            if final_info.get("collision")
            else "TIMEOUT"
            if truncated
            else "RUNNING"
        )
        frames.append(
            {
                "leader": final_info["leader_position"].copy(),
                "followers": final_info["follower_positions"].copy(),
                "goals": final_info["goals"].copy(),
                "actions": actions.copy(),
                "step": int(final_info["step"]),
                "outcome": outcome,
                "reward": total_reward,
            }
        )
        if terminated or truncated:
            break

    result = {
        "seed": seed,
        "obstacles": [
            {
                "shape": obstacle["shape"],
                "center": [float(value) for value in obstacle["center"]],
                **(
                    {"size": float(obstacle["size"])}
                    if obstacle["shape"] == "square"
                    else {"radius": float(obstacle["radius"])}
                ),
            }
            for obstacle in env.obstacles
        ],
        "success": bool(final_info.get("success", False)),
        "collision": bool(final_info.get("collision", False)),
        "steps": int(final_info.get("step", 0)),
        "total_reward": total_reward,
        "min_obstacle_clearance": float(final_info.get("min_obstacle_clearance", np.inf)),
        "min_pair_distance": float(final_info.get("min_pair_distance", np.inf)),
        "final_actions": action_history[-1] if action_history else [2, 2],
        "action_history": action_history,
    }
    return frames, result


def render(checkpoint, output, report, first_seed, scenario_count, fps):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = _environment_from_checkpoint(payload)
    if config.obstacle_count != 2:
        raise ValueError("checkpoint environment must use two obstacles")
    policy = DiscreteMADDPG(
        config.num_agents,
        int(payload["obs_dim"]),
        config.num_actions,
        hidden_dim=int(payload["hidden_dim"]),
        device=torch.device("cpu"),
        shared_actor=bool(payload.get("shared_actor", False)),
    )
    policy.load(checkpoint)

    scenarios = []
    seed = first_seed
    while len(scenarios) < scenario_count:
        item = collect(policy, config, seed)
        if item is not None:
            scenarios.append(item)
        seed += 1

    frame_sets = [item[0] for item in scenarios]
    results = [item[1] for item in scenarios]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=90)
    axes = axes.flat
    artist_sets = []

    for index, ((frames, result), axis) in enumerate(zip(scenarios, axes)):
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(-0.75, 15.0)
        axis.set_ylim(-5.25, 5.25)
        axis.grid(True, alpha=0.2)
        axis.axhline(2.0, color="#4c78a8", linestyle="--", alpha=0.25)
        axis.axhline(-2.0, color="#f58518", linestyle="--", alpha=0.25)
        axis.set_title(f"Scenario {index + 1} · seed {result['seed']}")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        for obstacle in result["obstacles"]:
            center = np.asarray(obstacle["center"])
            if obstacle["shape"] == "square":
                half = obstacle["size"] / 2.0
                patch = Rectangle(
                    center - half,
                    obstacle["size"],
                    obstacle["size"],
                    facecolor="#d62728",
                    edgecolor="#7f0000",
                    alpha=0.72,
                )
            else:
                patch = Circle(
                    center,
                    obstacle["radius"],
                    facecolor="#e45756",
                    edgecolor="#7f0000",
                    alpha=0.72,
                )
            axis.add_patch(patch)

        leader = Circle((0, 0), config.robot_radius, color="#54a24b", zorder=5)
        followers = [
            Circle((0, 0), config.robot_radius, color=color, zorder=5)
            for color in ("#4c78a8", "#f58518")
        ]
        axis.add_patch(leader)
        for follower in followers:
            axis.add_patch(follower)
        leader_trail, = axis.plot([], [], color="#54a24b", linewidth=1.5)
        follower_trails = [
            axis.plot([], [], color=color, linewidth=2)[0]
            for color in ("#4c78a8", "#f58518")
        ]
        goals = axis.scatter(
            [0.0, 0.0],
            [0.0, 0.0],
            marker="x",
            s=55,
            color=("#4c78a8", "#f58518"),
            linewidth=2,
            zorder=6,
        )
        status = axis.text(
            0.02,
            0.98,
            "",
            transform=axis.transAxes,
            va="top",
            family="monospace",
            fontsize=8,
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        artist_sets.append(
            (leader, followers, leader_trail, follower_trails, goals, status)
        )

    axes[scenario_count].axis("off")
    axes[scenario_count].text(
        0.05,
        0.95,
        "Legend\n\nGreen: go1\nBlue: go2\nOrange: go3\nRed square/circle: obstacles\n×: selected waypoint",
        va="top",
        fontsize=12,
    )
    fig.suptitle(
        f"Five two-obstacle deterministic tests · {fps}× playback",
        fontsize=14,
    )
    max_frames = max(len(frames) for frames in frame_sets)

    def update(frame_index):
        changed = []
        for frames, artists in zip(frame_sets, artist_sets):
            active_index = min(frame_index, len(frames) - 1)
            frame = frames[active_index]
            history = frames[: active_index + 1]
            leader, followers, leader_trail, follower_trails, goals, status = artists
            leader.center = frame["leader"]
            leader_trail.set_data(
                [item["leader"][0] for item in history],
                [item["leader"][1] for item in history],
            )
            for robot_index in range(2):
                followers[robot_index].center = frame["followers"][robot_index]
                follower_trails[robot_index].set_data(
                    [item["followers"][robot_index, 0] for item in history],
                    [item["followers"][robot_index, 1] for item in history],
                )
            goals.set_offsets(frame["goals"])
            status.set_text(
                f"t={frame['step']:03d}s  actions={frame['actions'].tolist()}\n"
                f"{frame['outcome']}  return={frame['reward']:.1f}"
            )
            changed.extend(
                [leader, *followers, leader_trail, *follower_trails, goals, status]
            )
        return changed

    output.parent.mkdir(parents=True, exist_ok=True)
    movie = animation.FuncAnimation(
        fig,
        update,
        frames=max_frames,
        interval=1000 / fps,
        blit=False,
        repeat_delay=1500,
    )
    movie.save(output, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)

    aggregate = {
        "checkpoint": str(checkpoint),
        "playback_speed": fps,
        "scenario_count": scenario_count,
        "successes": sum(item["success"] for item in results),
        "collisions": sum(item["collision"] for item in results),
        "scenarios": results,
    }
    with report.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=51000)
    parser.add_argument("--scenario-count", type=int, default=5)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    result = render(
        args.checkpoint,
        args.output,
        args.report,
        args.first_seed,
        args.scenario_count,
        args.fps,
    )
    print(json.dumps({key: result[key] for key in ("successes", "collisions")}, indent=2))
    print(f"saved_gif={args.output}")
    print(f"saved_report={args.report}")


if __name__ == "__main__":
    main()
