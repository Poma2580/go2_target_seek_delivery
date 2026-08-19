"""Data-only state shared by dynamic encirclement components."""

from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class DogState:
    """Latest odometry and command history for one robot."""

    name: str
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    frame_id: str = ""
    last_stamp: Any = None
    received: bool = False
    previous_linear: float = 0.0
    previous_angular: float = 0.0

    def __post_init__(self):
        """没有显式 frame 时使用该机器人的 odom frame。"""
        if not self.frame_id:
            self.frame_id = f"{self.name}/odom"


@dataclass(frozen=True)
class TargetSample:
    """One target observation timestamped at local receipt time."""

    x: float
    y: float
    vx: float
    vy: float
    frame_id: str
    received_at: float


@dataclass(frozen=True)
class ResolvedTarget:
    """A usable measured or short-term predicted target state."""

    x: float
    y: float
    vx: float
    vy: float
    frame_id: str
    predicted: bool


@dataclass(frozen=True)
class ControlCommand:
    """Planar velocity command produced by the perception controller."""

    linear: float
    angular: float


@dataclass(frozen=True)
class FormationPlan:
    """Latest fixed-slot formation plan for the Nav2-controlled robots."""

    slots: Dict[str, Tuple[float, float, float]]
    route_heading: float
    slot_indices: Dict[str, int]
    completed: bool
    assignment_created: bool = False
