"""Render ten targeted Go3 action-1-obstacle policy tests as one GIF."""

import argparse
import json
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


DEFAULT_CHECKPOINT = Path(
    "waypoint_maddpg_v0/runs/three_seed_obstacle_curriculum_20260909_091240/"
    "seed_27/stage2_two_obstacles/best_model.pt"
)


def environment_from_checkpoint(payload):
    valid = {field.name for field in fields(EnvConfig)}
    values = {
        key: value
        for key, value in payload.get("metadata", {}).get("environment", {}).items()
        if key in valid
    }
    for name in (
        "candidate_offsets",
        "obstacle_spawn_x",
        "obstacle_abs_y_range",
        "obstacle_y_range",
    ):
        if name in values and values[name] is not None:
            values[name] = tuple(values[name])
    return EnvConfig(**values)


def install_square(env, center):
    center = np.asarray(center, dtype=np.float32)
    half = 0.5 * env.cfg.obstacle_size
    env.obstacles = [
        {
            "shape": "square",
            "center": center,
            "size": float(env.cfg.obstacle_size),
            "lower": center - half,
            "upper": center + half,
        }
    ]
    env.obstacle_center = center
    env.obstacle_lower = center - half
    env.obstacle_upper = center + half
    observations, env._candidate_metrics = env._build_observations()
    env.default_blocked_latched.fill(False)
    env.default_clear_counts.fill(0)
    env._update_default_path_state()
    return observations


def collect_scenario(policy, config, center, seed, max_steps):
    env = WaypointSelectionEnv(config, seed=seed, lidar_noise=False)
    observations, _ = env.reset(seed=seed)
    observations = install_square(env, center)
    frames = []
    action_history = []
    final_info = {"collision": False, "success": False}

    for _ in range(max_steps):
        candidates = env._candidate_points()
        masks = env.valid_action_masks()
        previous = env.previous_actions.copy()
        actions = policy.act(
            observations,
            action_masks=masks,
            deterministic=True,
        )
        action_history.append(int(actions[1]))
        frames.append(
            {
                "step": int(env.step_count),
                "leader": env.leader_pos.copy(),
                "followers": env.follower_pos.copy(),
                "candidates": candidates[1].copy(),
                "blocked": np.asarray(
                    [metric["blocked"] for metric in env._candidate_metrics[1]],
                    dtype=bool,
                ),
                "mask": masks[1].copy(),
                "previous": int(previous[1]),
                "action": int(actions[1]),
                "collision": False,
                "success": False,
            }
        )
        observations, _, terminated, truncated, final_info = env.step(actions)
        if terminated or truncated:
            frames[-1]["collision"] = bool(final_info.get("collision", False))
            frames[-1]["success"] = bool(final_info.get("success", False))
            break

    initial = frames[0]
    if not initial["blocked"][1] or initial["blocked"][2] or initial["blocked"][3]:
        raise RuntimeError(
            f"invalid targeted scenario {center}: blocked={initial['blocked'].tolist()}"
        )
    first_action_zero = next(
        (index for index, action in enumerate(action_history) if action == 0), None
    )
    return frames, {
        "obstacle_center": [float(center[0]), float(center[1])],
        "initial_blocked_actions": np.flatnonzero(initial["blocked"]).tolist(),
        "initial_valid_actions": np.flatnonzero(initial["mask"]).tolist(),
        "initial_action": action_history[0],
        "action_history": action_history,
        "selected_action_0": first_action_zero is not None,
        "first_action_0_step": first_action_zero,
        "collision": bool(final_info.get("collision", False)),
        "success": bool(final_info.get("success", False)),
        "steps": len(action_history),
    }


def render(checkpoint, output, summary_output, fps, max_steps):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = environment_from_checkpoint(payload)
    policy = DiscreteMADDPG(
        config.num_agents,
        int(payload["obs_dim"]),
        config.num_actions,
        hidden_dim=int(payload.get("hidden_dim", 256)),
        device=torch.device("cpu"),
        shared_actor=bool(payload.get("shared_actor", False)),
    )
    policy.load(checkpoint)

    centers = [(float(x), -0.5) for x in np.arange(3.0, 5.26, 0.25)]
    collected = [
        collect_scenario(policy, config, center, 31000 + index, max_steps)
        for index, center in enumerate(centers)
    ]
    frame_sets = [item[0] for item in collected]
    results = [item[1] for item in collected]

    fig, axes = plt.subplots(2, 5, figsize=(15, 7.4), dpi=90)
    artists = []
    for index, (axis, center) in enumerate(zip(axes.flat, centers)):
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlim(0.5, 8.5)
        axis.set_ylim(-5.0, 2.0)
        axis.grid(True, alpha=0.2)
        axis.axhline(-config.formation_side, color="#999999", linestyle="--", alpha=0.5)
        axis.set_title(f"Scene {index + 1}: obstacle=({center[0]:.2f}, {center[1]:.2f})", fontsize=9)
        if index >= 5:
            axis.set_xlabel("x (m)")
        if index % 5 == 0:
            axis.set_ylabel("y (m)")
        half = 0.5 * config.obstacle_size
        axis.add_patch(
            Rectangle(
                (center[0] - half, center[1] - half),
                config.obstacle_size,
                config.obstacle_size,
                facecolor="#d62728",
                edgecolor="#7f0000",
                alpha=0.7,
            )
        )
        go3 = Circle((0, 0), config.robot_radius, color="#f58518", zorder=5)
        leader = Circle((0, 0), config.robot_radius, color="#54a24b", alpha=0.65, zorder=4)
        axis.add_patch(go3)
        axis.add_patch(leader)
        trail, = axis.plot([], [], color="#f58518", linewidth=2)
        candidate_scatter = axis.scatter([], [], s=42, marker="x", linewidth=2, zorder=6)
        selected_scatter = axis.scatter(
            [], [], s=125, facecolors="none", edgecolors="#ffbf00", linewidth=2.2, zorder=7
        )
        status = axis.text(
            0.02,
            0.98,
            "",
            transform=axis.transAxes,
            va="top",
            fontsize=8,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none"},
        )
        artists.append((go3, leader, trail, candidate_scatter, selected_scatter, status))

    fig.suptitle(
        "Go3 targeted test: action 1 blocked, actions 2 and 3 clear\n"
        "candidate color: red=blocked, blue=clear; yellow ring=selected",
        fontsize=13,
    )
    max_frames = max(len(frames) for frames in frame_sets)

    def update(frame_index):
        changed = []
        for scenario_index, (frames, artist_set) in enumerate(zip(frame_sets, artists)):
            frame = frames[min(frame_index, len(frames) - 1)]
            go3, leader, trail, candidates, selected, status = artist_set
            go3.center = frame["followers"][1]
            leader.center = frame["leader"]
            history = frames[: min(frame_index + 1, len(frames))]
            trail.set_data(
                [item["followers"][1, 0] for item in history],
                [item["followers"][1, 1] for item in history],
            )
            candidates.set_offsets(frame["candidates"])
            candidates.set_color(
                ["#d62728" if blocked else "#1f77b4" for blocked in frame["blocked"]]
            )
            selected.set_offsets(frame["candidates"][[frame["action"]]])
            outcome = " COLLISION" if frame["collision"] else " SUCCESS" if frame["success"] else ""
            status.set_text(
                f"t={frame['step']:02d} prev={frame['previous']} -> a={frame['action']}{outcome}\n"
                f"blocked={np.flatnonzero(frame['blocked']).tolist()}"
            )
            changed.extend(artist_set)
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
        "scenario_contract": {
            "go3_initial_position": [2.0, -2.0],
            "action_1_blocked": True,
            "action_2_clear": True,
            "action_3_clear": True,
            "lidar_noise": False,
            "max_steps": max_steps,
        },
        "counts": {
            "initial_actions": {
                str(action): sum(result["initial_action"] == action for result in results)
                for action in range(config.num_actions)
            },
            "scenarios_selecting_action_0": sum(
                result["selected_action_0"] for result in results
            ),
            "collisions": sum(result["collision"] for result in results),
            "successes": sum(result["success"] for result in results),
        },
        "scenarios": results,
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    with summary_output.open("w", encoding="utf-8") as stream:
        json.dump(aggregate, stream, indent=2)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=45)
    args = parser.parse_args()
    result = render(
        args.checkpoint,
        args.output,
        args.summary_output,
        args.fps,
        args.max_steps,
    )
    print(json.dumps(result["counts"], indent=2))
    print(f"saved_gif={args.output}")
    print(f"saved_summary={args.summary_output}")


if __name__ == "__main__":
    main()
