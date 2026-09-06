"""Standalone discrete-MADDPG waypoint-selection training package."""

from .config import EnvConfig, TrainConfig
from .environment import WaypointSelectionEnv

__all__ = ["EnvConfig", "TrainConfig", "WaypointSelectionEnv"]
