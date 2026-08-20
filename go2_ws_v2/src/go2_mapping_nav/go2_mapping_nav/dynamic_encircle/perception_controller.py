"""Direct catch-up and formation controller for the perception robot."""

import math

from .geometry import clamp, normalize_angle
from .models import ControlCommand


class PerceptionController:
    """Compute acceleration-limited direct velocity commands without ROS I/O."""

    def __init__(self, config, loop):
        """使用共享配置和闭合路线初始化直接控制器。"""
        self.config = config
        self.loop = loop
        self.phase = None

    def start(self):
        """Start a newly elected perception robot in catch-up mode."""
        self.phase = "catch_up"

    def reset_command_history(self, dog):
        """Reset acceleration history when a stop command is emitted."""
        dog.previous_linear = 0.0
        dog.previous_angular = 0.0

    def compute(self, dog, target, dt):
        """Compute one catch-up or formation command and update command history."""
        distance_to_target = math.hypot(target.x - dog.x, target.y - dog.y)

        if self.phase == "catch_up":
            if distance_to_target < self.config.catch_radius:
                self.phase = "formation"
            else:
                command = self._catch_up_command(dog, target)
                return self._limit_acceleration(dog, command, dt)

        command = self._formation_command(dog, target, distance_to_target)
        return self._limit_acceleration(dog, command, dt)

    def _catch_up_command(self, dog, target):
        """Follow the shortest direction around the configured closed loop."""
        dog_arc = self.loop.project(dog.x, dog.y)
        target_arc = self.loop.project(target.x, target.y)
        delta = self.loop.signed_arc(dog_arc, target_arc)
        direction = 1.0 if delta >= 0.0 else -1.0
        step = min(self.config.catch_lookahead, abs(delta))
        control_x, control_y = self.loop.point_at(dog_arc + direction * step)
        yaw_error = normalize_angle(
            math.atan2(control_y - dog.y, control_x - dog.x) - dog.yaw
        )
        linear = (
            0.0
            if abs(yaw_error) > self.config.turn_in_place_thresh
            else self.config.catch_speed
        )
        angular = clamp(
            self.config.k_angular * yaw_error,
            -self.config.max_angular,
            self.config.max_angular,
        )
        return ControlCommand(linear, angular)

    def _formation_command(self, dog, target, distance_to_target):
        """Track target range with velocity feed-forward and heading gating."""
        bearing = math.atan2(target.y - dog.y, target.x - dog.x)
        yaw_error = normalize_angle(bearing - dog.yaw)
        angular = clamp(
            self.config.k_angular * yaw_error,
            -self.config.max_angular,
            self.config.max_angular,
        )
        range_error = distance_to_target - self.config.formation_radius
        if abs(range_error) < self.config.position_deadband:
            range_error = 0.0
        feed_forward = target.vx * math.cos(bearing) + target.vy * math.sin(bearing)
        heading_gate = max(math.cos(yaw_error), 0.25)
        linear = clamp(
            (self.config.k_linear * range_error + feed_forward) * heading_gate,
            -0.5 * self.config.max_linear,
            self.config.max_linear,
        )
        return ControlCommand(linear, angular)

    def _limit_acceleration(self, dog, command, dt):
        """Limit per-cycle velocity change exactly as the original publisher did."""
        linear = clamp(
            command.linear,
            dog.previous_linear - self.config.accel_lin * dt,
            dog.previous_linear + self.config.accel_lin * dt,
        )
        angular = clamp(
            command.angular,
            dog.previous_angular - self.config.accel_ang * dt,
            dog.previous_angular + self.config.accel_ang * dt,
        )
        dog.previous_linear = linear
        dog.previous_angular = angular
        return ControlCommand(linear, angular)
