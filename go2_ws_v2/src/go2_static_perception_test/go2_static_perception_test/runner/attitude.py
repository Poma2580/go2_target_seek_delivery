#!/usr/bin/env python3
"""Check go2_1 roll over a bounded ModelStates frame window."""

import math
import sys
import time

import rclpy
from gazebo_msgs.msg import ModelStates
from rclpy.node import Node


EXIT_FALLEN = 10
EXIT_ERROR = 20


def roll_degrees(q):
    return math.degrees(math.atan2(2.0 * (q.w*q.x + q.y*q.z),
                                   1.0 - 2.0 * (q.x*q.x + q.y*q.y)))


class AttitudeNode(Node):
    def __init__(self):
        super().__init__("check_static_go2_attitude")
        self.declare_parameter("roll_limit_deg", 90.0)
        self.declare_parameter("sample_frames", 5)
        self.declare_parameter("timeout_sec", 10.0)
        self.limit = float(self.get_parameter("roll_limit_deg").value)
        self.required = int(self.get_parameter("sample_frames").value)
        self.timeout = float(self.get_parameter("timeout_sec").value)
        if not math.isfinite(self.limit) or self.limit <= 0 or self.required <= 0 or self.timeout <= 0:
            raise ValueError("invalid attitude checker parameters")
        self.frames, self.fallen = 0, False
        self.create_subscription(ModelStates, "/gazebo/model_states", self.callback, 10)

    def callback(self, message):
        if self.frames >= self.required or self.fallen or "go2_1" not in message.name: return
        roll = roll_degrees(message.pose[message.name.index("go2_1")].orientation)
        self.frames += 1
        self.fallen = abs(roll) > self.limit
        self.get_logger().info(f"attitude frame {self.frames}/{self.required}: roll={roll:.2f} deg")


def main(args=None):
    rclpy.init(args=args)
    try:
        node = AttitudeNode(); deadline = time.monotonic() + node.timeout
        while rclpy.ok() and not node.fallen and node.frames < node.required and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.2)
        result = EXIT_FALLEN if node.fallen else (0 if node.frames >= node.required else EXIT_ERROR)
        node.destroy_node(); return result
    except Exception as error:
        print(f"static attitude checker failed: {error}", file=sys.stderr); return EXIT_ERROR
    finally:
        if rclpy.ok(): rclpy.shutdown()
