#!/usr/bin/env python3
"""Offline MADDPG inference smoke test.

This script loads the 4-agent formation-navigation MADDPG checkpoint and runs
one policy inference step with four 23-dimensional observations. It does not
start ROS2 or Gazebo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MADDPG_ROOT = REPO_ROOT / "三角形MADDPG"
DEFAULT_MODEL_PATH = (
    MADDPG_ROOT
    / "runs"
    / "stage4_b512_usteps20_g0.99_t0.005_alr5e-05_clr0.0005_n0.14_minn0.02_h128,128_20260430_132728"
    / "best_model.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load best_model.pt and print one 4-agent MADDPG action batch."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the MADDPG checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --random-observations is enabled.",
    )
    parser.add_argument(
        "--random-observations",
        action="store_true",
        help="Use small random observations instead of deterministic test observations.",
    )
    return parser.parse_args()


def build_test_observations(random_observations: bool, seed: int) -> list[np.ndarray]:
    """Return four observations matching the training layout and shape.

    Observation layout:
    own_pos(2), own_vel(2), own_slot_rel(2), leader_rel(2), target_rel(2),
    other_agents_rel(6), role_flag(1), obstacles_rel(6)
    """
    if random_observations:
        rng = np.random.default_rng(seed)
        obs = rng.normal(loc=0.0, scale=0.1, size=(4, 23)).astype(np.float32)
    else:
        obs = np.zeros((4, 23), dtype=np.float32)

        # A simple leader-follower formation-shaped test state.
        positions = np.array(
            [
                [0.0, 0.0],
                [-0.60, -0.65],
                [0.0, -0.65],
                [0.60, -0.65],
            ],
            dtype=np.float32,
        )
        target = np.array([2.20, 2.50], dtype=np.float32)
        leader = positions[0]
        follower_offsets = {
            1: np.array([-0.60, -0.65], dtype=np.float32),
            2: np.array([0.0, -0.65], dtype=np.float32),
            3: np.array([0.60, -0.65], dtype=np.float32),
        }

        for i in range(4):
            own = positions[i]
            own_vel = np.zeros(2, dtype=np.float32)
            if i == 0:
                own_slot_rel = np.zeros(2, dtype=np.float32)
                role_flag = 1.0
            else:
                own_slot_rel = leader + follower_offsets[i] - own
                role_flag = 0.0

            other_positions = np.delete(positions, i, axis=0)
            obs[i] = np.concatenate(
                [
                    own,
                    own_vel,
                    own_slot_rel,
                    leader - own,
                    target - own,
                    (other_positions - own).reshape(-1),
                    np.array([role_flag], dtype=np.float32),
                    np.zeros(6, dtype=np.float32),
                ]
            ).astype(np.float32)

    return [row.astype(np.float32) for row in obs]


def main() -> int:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()

    if not model_path.exists():
        print(f"ERROR: model not found: {model_path}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(MADDPG_ROOT))
    from maddpg import MADDPG

    state_sizes = [23, 23, 23, 23]
    action_sizes = [2, 2, 2, 2]
    maddpg = MADDPG(
        state_sizes=state_sizes,
        action_sizes=action_sizes,
        hidden_sizes=(128, 128),
        action_low=-1.0,
        action_high=1.0,
    )
    maddpg.load(str(model_path))

    observations = build_test_observations(args.random_observations, args.seed)
    actions = maddpg.act(observations, add_noise=False)

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    print(f"model: {model_path}")
    print("observation_shapes:", [tuple(obs.shape) for obs in observations])
    print("action_shapes:", [tuple(np.asarray(action).shape) for action in actions])
    print("actions:")
    for i, action in enumerate(actions):
        print(f"  agent_{i}: {np.asarray(action, dtype=np.float32)}")

    if not all(np.all(np.isfinite(action)) for action in actions):
        print("ERROR: model produced NaN or Inf action.", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
