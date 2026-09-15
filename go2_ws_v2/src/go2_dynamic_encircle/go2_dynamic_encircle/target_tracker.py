"""Target caching, activation, prediction, and expiry."""

import math

from .models import ResolvedTarget, TargetSample


class TargetTracker:
    """Track the elected perception robot's latest usable target estimate."""

    def __init__(self, target_timeout, target_hold, max_coast_speed):
        """配置目标新鲜期、外推保持期和最大外推速度。"""
        self.target_timeout = target_timeout
        self.target_hold = target_hold
        self.max_coast_speed = max_coast_speed
        self.active_robot = None
        self.cached = {}
        self.latest = None

    def cache(self, robot_name, sample):
        """Cache the most recent sample for a robot before or after election."""
        self.cached[robot_name] = sample

    def activate(self, robot_name, now):
        """Lock the active source and adopt a fresh, correctly framed cache."""
        self.active_robot = robot_name
        sample = self.cached.get(robot_name)
        if sample is None:
            return False
        if now - sample.received_at > self.target_timeout:
            return False
        if sample.frame_id != f"{robot_name}/odom":
            return False
        self.latest = sample
        return True

    def update_active(self, robot_name, sample):
        """Cache every sample but accept only a correctly framed active source."""
        self.cache(robot_name, sample)
        if robot_name != self.active_robot:
            return False
        if sample.frame_id != f"{robot_name}/odom":
            return False
        self.latest = sample
        return True

    def resolve(self, now):
        """Return a measured, predicted, or expired target for the given time."""
        if self.latest is None:
            return None
        age = now - self.latest.received_at
        if age <= self.target_timeout:
            return ResolvedTarget(
                self.latest.x,
                self.latest.y,
                self.latest.vx,
                self.latest.vy,
                self.latest.frame_id,
                predicted=False,
            )
        if age > self.target_hold:
            return None

        velocity_x = self.latest.vx
        velocity_y = self.latest.vy
        speed = math.hypot(velocity_x, velocity_y)
        if speed > self.max_coast_speed and speed > 1e-6:
            scale = self.max_coast_speed / speed
            velocity_x *= scale
            velocity_y *= scale
        return ResolvedTarget(
            self.latest.x + velocity_x * age,
            self.latest.y + velocity_y * age,
            velocity_x,
            velocity_y,
            self.latest.frame_id,
            predicted=True,
        )
