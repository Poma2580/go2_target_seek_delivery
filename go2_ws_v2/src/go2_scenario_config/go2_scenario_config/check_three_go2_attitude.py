#!/usr/bin/env python3
"""Check whether any spawned Go2 has remained rolled over for several frames."""

import math
import sys
import time
from dataclasses import dataclass
from typing import Sequence

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Quaternion
from rclpy.node import Node


EXIT_OK = 0
EXIT_FALLEN = 10
EXIT_ERROR = 20


@dataclass(frozen=True)
class EulerDegrees:
    roll: float
    pitch: float
    yaw: float


def quaternion_to_euler_degrees(quaternion: Quaternion) -> EulerDegrees:
    """Convert a geometry quaternion to roll, pitch and yaw in degrees."""
    x = quaternion.x
    y = quaternion.y
    z = quaternion.z
    w = quaternion.w

    roll = math.atan2(
        2.0 * (w * x + y * z),
        1.0 - 2.0 * (x * x + y * y),
    )
    pitch_sine = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, pitch_sine)))
    yaw = math.atan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )
    return EulerDegrees(
        roll=math.degrees(roll),
        pitch=math.degrees(pitch),
        yaw=math.degrees(yaw),
    )


class ConsecutiveRollChecker:
    """Track per-robot consecutive roll-limit violations."""

    def __init__(
        self,
        robot_names: Sequence[str],
        roll_limit_deg: float,
        required_frames: int,
    ) -> None:
        self.robot_names = tuple(robot_names)
        self.roll_limit_deg = roll_limit_deg
        self.required_frames = required_frames
        self.valid_frames = 0
        self.consecutive_counts = {name: 0 for name in self.robot_names}
        self.fallen_robots: set[str] = set()

    def add_frame(self, attitudes: dict[str, EulerDegrees]) -> None:
        """Add one complete frame and update the per-robot counters."""
        missing = set(self.robot_names) - attitudes.keys()
        if missing:
            raise ValueError(f"incomplete attitude frame: missing {sorted(missing)}")

        self.valid_frames += 1
        for name in self.robot_names:
            if abs(attitudes[name].roll) > self.roll_limit_deg:
                self.consecutive_counts[name] += 1
            else:
                self.consecutive_counts[name] = 0
            if self.consecutive_counts[name] >= self.required_frames:
                self.fallen_robots.add(name)

    @property
    def complete(self) -> bool:
        return self.valid_frames >= self.required_frames


class ThreeGo2AttitudeNode(Node):
    """Collect a fixed number of complete Gazebo model-state frames."""

    def __init__(self) -> None:
        super().__init__("check_three_go2_attitude")
        self.declare_parameter("model_states_topic", "/gazebo/model_states")
        self.declare_parameter("robot_names", ["go2_1", "go2_2", "go2_3"])
        self.declare_parameter("roll_limit_deg", 90.0)
        self.declare_parameter("required_frames", 3)
        self.declare_parameter("timeout_seconds", 10.0)

        self.model_states_topic = str(
            self.get_parameter("model_states_topic").value
        )
        self.robot_names = tuple(self.get_parameter("robot_names").value)
        self.roll_limit_deg = float(
            self.get_parameter("roll_limit_deg").value
        )
        self.required_frames = int(self.get_parameter("required_frames").value)
        self.timeout_seconds = float(self.get_parameter("timeout_seconds").value)
        self._validate_parameters()

        self.checker = ConsecutiveRollChecker(
            self.robot_names,
            self.roll_limit_deg,
            self.required_frames,
        )
        self.last_missing_names: tuple[str, ...] = self.robot_names
        self.create_subscription(
            ModelStates,
            self.model_states_topic,
            self._model_states_callback,
            10,
        )

    def _validate_parameters(self) -> None:
        if not self.robot_names or any(not name for name in self.robot_names):
            raise ValueError("robot_names must contain at least one non-empty name")
        if len(set(self.robot_names)) != len(self.robot_names):
            raise ValueError("robot_names must not contain duplicates")
        if not math.isfinite(self.roll_limit_deg) or self.roll_limit_deg <= 0.0:
            raise ValueError("roll_limit_deg must be a positive finite number")
        if self.required_frames <= 0:
            raise ValueError("required_frames must be greater than zero")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be a positive finite number")

    def _model_states_callback(self, message: ModelStates) -> None:
        if self.checker.complete:
            return

        pose_by_name = dict(zip(message.name, message.pose))
        missing = tuple(name for name in self.robot_names if name not in pose_by_name)
        if missing:
            self.last_missing_names = missing
            return

        self.last_missing_names = ()
        attitudes = {
            name: quaternion_to_euler_degrees(pose_by_name[name].orientation)
            for name in self.robot_names
        }
        self.checker.add_frame(attitudes)

        values = "; ".join(
            f"{name}: roll={attitudes[name].roll:.2f} deg, "
            f"pitch={attitudes[name].pitch:.2f} deg, "
            f"yaw={attitudes[name].yaw:.2f} deg"
            for name in self.robot_names
        )
        self.get_logger().info(
            f"Attitude frame {self.checker.valid_frames}/"
            f"{self.required_frames}: {values}"
        )


def run_check() -> int:
    node = ThreeGo2AttitudeNode()
    deadline = time.monotonic() + node.timeout_seconds
    try:
        while rclpy.ok() and not node.checker.complete:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                missing_text = ", ".join(node.last_missing_names) or "none"
                node.get_logger().error(
                    "Timed out waiting for complete model-state frames; "
                    f"last missing models: {missing_text}"
                )
                return EXIT_ERROR
            rclpy.spin_once(node, timeout_sec=min(0.2, remaining))

        if not rclpy.ok():
            node.get_logger().error("ROS shut down before the attitude check completed")
            return EXIT_ERROR
        if node.checker.fallen_robots:
            fallen = ", ".join(sorted(node.checker.fallen_robots))
            node.get_logger().error(
                f"Confirmed fallen robot(s): {fallen}; "
                f"abs(roll) exceeded {node.roll_limit_deg:.2f} deg for "
                f"{node.required_frames} consecutive frames"
            )
            return EXIT_FALLEN

        node.get_logger().info(
            f"All {len(node.robot_names)} Go2 robots passed the attitude check"
        )
        return EXIT_OK
    finally:
        node.destroy_node()


def main() -> int:
    rclpy.init()
    try:
        return run_check()
    except Exception as error:  # Keep the shell-facing exit contract stable.
        print(f"Attitude checker failed: {error}", file=sys.stderr)
        return EXIT_ERROR
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main())
