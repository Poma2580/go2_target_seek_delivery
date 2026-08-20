#!/usr/bin/env python3
"""Select and latch the first Go2 with a stable target estimate."""

import math
from collections import deque
from functools import partial

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String


DEFAULT_ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


def role_qos_profile():
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )


def stamp_to_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def valid_target_message(message, expected_frame, now_seconds, max_message_age):
    """Return whether an estimate is fresh, finite and in the expected odom frame."""
    if message.header.frame_id != expected_frame:
        return False
    values = (
        message.pose.pose.position.x,
        message.pose.pose.position.y,
        message.pose.pose.position.z,
        message.twist.twist.linear.x,
        message.twist.twist.linear.y,
        message.twist.twist.linear.z,
    )
    if not all(math.isfinite(value) for value in values):
        return False
    age = now_seconds - stamp_to_seconds(message.header.stamp)
    return -1e-3 <= age <= max_message_age


class RoleElectionState:
    """Pure first-to-N election state using a per-robot confirmation window."""

    def __init__(self, robot_names, confirmation_count, confirmation_window):
        names = tuple(robot_names)
        if not names or len(set(names)) != len(names):
            raise ValueError("robot_names must be non-empty and unique")
        if confirmation_count < 1:
            raise ValueError("confirmation_count must be greater than zero")
        if not math.isfinite(confirmation_window) or confirmation_window <= 0.0:
            raise ValueError("confirmation_window must be finite and positive")
        self.robot_names = names
        self.confirmation_count = confirmation_count
        self.confirmation_window = confirmation_window
        self.samples = {name: deque() for name in names}
        self.selected = None

    def reject(self, name):
        if name not in self.samples:
            raise ValueError(f"unknown robot: {name}")
        self.samples[name].clear()

    def observe(self, name, now_seconds):
        if name not in self.samples:
            raise ValueError(f"unknown robot: {name}")
        if self.selected is not None:
            return self.selected
        samples = self.samples[name]
        cutoff = now_seconds - self.confirmation_window
        while samples and samples[0] < cutoff:
            samples.popleft()
        samples.append(now_seconds)
        if len(samples) >= self.confirmation_count:
            self.selected = name
        return self.selected


class TargetRoleSelector(Node):
    def __init__(self):
        super().__init__("target_role_selector")
        self.declare_parameter("robot_names", list(DEFAULT_ROBOT_NAMES))
        self.declare_parameter("confirmation_count", 3)
        self.declare_parameter("confirmation_window", 1.0)
        self.declare_parameter("max_message_age", 0.5)
        self.declare_parameter("role_topic", "/target_role/perception_robot")

        self.robot_names = tuple(self.get_parameter("robot_names").value)
        confirmation_count = int(self.get_parameter("confirmation_count").value)
        confirmation_window = float(
            self.get_parameter("confirmation_window").value
        )
        self.max_message_age = float(self.get_parameter("max_message_age").value)
        self.role_topic = self.get_parameter("role_topic").value
        if not math.isfinite(self.max_message_age) or self.max_message_age <= 0.0:
            raise ValueError("max_message_age must be finite and positive")
        if not isinstance(self.role_topic, str) or not self.role_topic:
            raise ValueError("role_topic must be a non-empty string")

        self.election = RoleElectionState(
            self.robot_names, confirmation_count, confirmation_window
        )
        role_qos = role_qos_profile()
        self.role_publisher = self.create_publisher(String, self.role_topic, role_qos)
        self.target_subscriptions = [
            self.create_subscription(
                Odometry,
                f"/{name}/target_estimated/odom",
                partial(self._target_callback, name),
                10,
            )
            for name in self.robot_names
        ]
        self.get_logger().info(
            "target_role_selector started: robots=%s confirmation=%d/%.2fs role=%s"
            % (
                ",".join(self.robot_names),
                confirmation_count,
                confirmation_window,
                self.role_topic,
            )
        )

    def _target_callback(self, name, message):
        if self.election.selected is not None:
            return
        now_seconds = self.get_clock().now().nanoseconds * 1e-9
        expected_frame = f"{name}/odom"
        if not valid_target_message(
            message, expected_frame, now_seconds, self.max_message_age
        ):
            self.election.reject(name)
            self.get_logger().warning(
                f"Rejected invalid target estimate from {name}",
                throttle_duration_sec=2.0,
            )
            return
        selected = self.election.observe(name, now_seconds)
        if selected is None:
            return
        self.role_publisher.publish(String(data=selected))
        self.get_logger().info(
            f"Perception role locked for this run: {selected}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = TargetRoleSelector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
