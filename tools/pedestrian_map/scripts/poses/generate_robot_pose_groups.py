#!/usr/bin/env python3
"""Command entry point for deterministic Go2 spawn-pose generation."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pedestrian_map.robot_pose_generation import main


if __name__ == "__main__":
    raise SystemExit(main())
