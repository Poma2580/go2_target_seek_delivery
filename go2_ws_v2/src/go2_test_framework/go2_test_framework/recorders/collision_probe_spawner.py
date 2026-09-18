#!/usr/bin/env python3
"""Spawn a small static box at a live trunk pose for Phase 6 smoke only."""

import time

from gazebo_msgs.srv import SpawnEntity
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node


BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="phase6_test_box">
    <static>true</static>
    <link name="link">
      <collision name="collision">
        <geometry><box><size>0.10 0.10 0.08</size></box></geometry>
      </collision>
      <visual name="visual">
        <geometry><box><size>0.10 0.10 0.08</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>"""


class CollisionProbeSpawner(Node):
    def __init__(self):
        super().__init__("collision_probe_spawner")
        self.declare_parameter("robot", "go2_3")
        self.declare_parameter("model_name", "phase6_test_box")
        self.robot = str(self.get_parameter("robot").value)
        self.model_name = str(self.get_parameter("model_name").value)
        self.pose = None
        self.subscription = self.create_subscription(
            Odometry, f"/{self.robot}/odom/ground_truth", self._pose_cb, 10
        )
        self.client = self.create_client(SpawnEntity, "/spawn_entity")

    def _pose_cb(self, message):
        self.pose = message.pose.pose

    def spawn(self, timeout_sec=20.0):
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and self.pose is None and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.pose is None or not self.client.wait_for_service(timeout_sec=2.0):
            return False
        request = SpawnEntity.Request()
        request.name = self.model_name
        request.xml = BOX_SDF.replace("phase6_test_box", self.model_name)
        request.initial_pose = self.pose
        request.reference_frame = "world"
        future = self.client.call_async(request)
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        return bool(
            future.done() and future.result() is not None
            and future.result().success
        )


def main(args=None):
    rclpy.init(args=args)
    node = CollisionProbeSpawner()
    success = False
    try:
        success = node.spawn()
        if success:
            node.get_logger().info(
                f"spawned {node.model_name} at live {node.robot} trunk pose"
            )
        else:
            node.get_logger().error("failed to spawn collision probe")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
