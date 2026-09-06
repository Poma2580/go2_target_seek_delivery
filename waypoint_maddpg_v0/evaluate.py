"""Evaluate a trained five-candidate discrete MADDPG checkpoint."""

import argparse
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import torch

from .config import EnvConfig
from .discrete_maddpg import DiscreteMADDPG
from .environment import WaypointSelectionEnv
from .train import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--sim-rays", type=int, choices=(36, 72, 108), default=None)
    parser.add_argument("--path-blocked-clearance", type=float, default=None)
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_env = payload.get("metadata", {}).get("environment", {})
    valid_fields = {field.name for field in fields(EnvConfig)}
    environment_values = {
        key: value for key, value in checkpoint_env.items() if key in valid_fields
    }
    if "candidate_offsets" in environment_values:
        environment_values["candidate_offsets"] = tuple(environment_values["candidate_offsets"])
    if "obstacle_abs_y_range" in environment_values:
        environment_values["obstacle_abs_y_range"] = tuple(
            environment_values["obstacle_abs_y_range"]
        )
    env_cfg = EnvConfig(**environment_values)
    if args.sim_rays is not None:
        env_cfg = replace(env_cfg, lidar_sim_rays=args.sim_rays)
    if args.path_blocked_clearance is not None:
        env_cfg = replace(
            env_cfg, path_blocked_clearance=args.path_blocked_clearance
        )
    env = WaypointSelectionEnv(env_cfg, seed=args.seed)
    policy = DiscreteMADDPG(
        env.num_agents,
        env.obs_dim,
        env.num_actions,
        hidden_dim=int(payload.get("hidden_dim", 256)),
        shared_actor=bool(payload.get("shared_actor", False)),
    )
    metadata = policy.load(args.checkpoint)
    metrics = evaluate(policy, env_cfg, args.episodes, args.seed)
    print("Evaluation")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"  action_distribution: {np.asarray(metrics['action_counts']) / max(sum(metrics['action_counts']), 1)}")


if __name__ == "__main__":
    main()
