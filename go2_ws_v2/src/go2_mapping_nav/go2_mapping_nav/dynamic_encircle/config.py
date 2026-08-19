"""ROS parameter loading and validation for dynamic encirclement."""

import math
from dataclasses import dataclass, fields


DOG_NAMES = ("go2_1", "go2_2", "go2_3")
LOOP_CORNERS = ((41.0, 4.0), (41.0, 36.0), (-13.0, 36.0), (-13.0, 4.0))


@dataclass(frozen=True)
class EncircleConfig:
    """Validated configuration shared by all encirclement components."""

    formation_radius: float = 2.0
    success_tolerance: float = 2.0
    success_yaw_tolerance: float = math.radians(60.0)
    control_rate: float = 20.0
    encircle_update_rate: float = 1.0
    nav_goal_update_rate: float = 0.2
    target_timeout: float = 5.0
    target_hold: float = 8.0
    odom_timeout: float = 0.5
    max_linear: float = 0.65
    max_angular: float = 0.9
    max_coast_speed: float = 0.65
    position_deadband: float = 0.25
    k_linear: float = 0.8
    k_angular: float = 0.9
    turn_in_place_thresh: float = 1.2
    accel_lin: float = 1.0
    accel_ang: float = 3.0
    catch_lookahead: float = 1.5
    catch_speed: float = 0.6
    catch_radius: float = 3.5
    tf_timeout: float = 0.1
    robot_names: tuple = DOG_NAMES
    perception_robot_topic: str = "/target_role/perception_robot"
    global_frame: str = "merged_map"

    @classmethod
    def declare_and_load(cls, node):
        """Declare the existing ROS parameters and return validated values."""
        defaults = cls()
        numeric_names = tuple(
            field.name
            for field in fields(cls)
            if field.name
            not in ("robot_names", "perception_robot_topic", "global_frame")
        )
        for name in numeric_names:
            node.declare_parameter(name, getattr(defaults, name))
        node.declare_parameter("robot_names", list(defaults.robot_names))
        node.declare_parameter(
            "perception_robot_topic", defaults.perception_robot_topic
        )
        node.declare_parameter("global_frame", defaults.global_frame)

        values = {
            name: float(node.get_parameter(name).value)
            for name in numeric_names
        }
        values.update(
            robot_names=tuple(node.get_parameter("robot_names").value),
            perception_robot_topic=node.get_parameter(
                "perception_robot_topic"
            ).value,
            global_frame=node.get_parameter("global_frame").value,
        )
        config = cls(**values)
        config.validate()
        return config

    def validate(self):
        """Apply the validation rules used by the original single-file node."""
        positive = (
            "formation_radius",
            "success_tolerance",
            "success_yaw_tolerance",
            "control_rate",
            "encircle_update_rate",
            "nav_goal_update_rate",
            "target_timeout",
            "target_hold",
            "odom_timeout",
            "max_linear",
            "max_angular",
            "max_coast_speed",
            "k_linear",
            "k_angular",
            "turn_in_place_thresh",
            "accel_lin",
            "accel_ang",
            "catch_lookahead",
            "catch_speed",
            "catch_radius",
            "tf_timeout",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.success_yaw_tolerance > math.pi:
            raise ValueError("success_yaw_tolerance must be no greater than pi")
        if not math.isfinite(self.position_deadband) or self.position_deadband < 0.0:
            raise ValueError("position_deadband must be finite and non-negative")
        if self.target_hold < self.target_timeout:
            raise ValueError("target_hold must be greater than or equal to target_timeout")
        if not all(isinstance(name, str) and name for name in self.robot_names):
            raise ValueError("robot_names entries must be non-empty strings")
        if len(self.robot_names) != 3 or len(set(self.robot_names)) != 3:
            raise ValueError("robot_names must contain exactly three unique names")
        if not isinstance(self.perception_robot_topic, str) or not self.perception_robot_topic:
            raise ValueError("perception_robot_topic must be a non-empty string")
        if not isinstance(self.global_frame, str) or not self.global_frame:
            raise ValueError("global_frame must be a non-empty string")
