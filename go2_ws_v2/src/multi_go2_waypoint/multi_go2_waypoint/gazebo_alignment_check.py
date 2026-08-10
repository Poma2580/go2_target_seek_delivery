#!/usr/bin/env python3
"""Print Gazebo model positions relevant to target/leader alignment."""

import math

import rclpy
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from rclpy.node import Node


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class GazeboAlignmentCheck(Node):
    def __init__(self):
        super().__init__("gazebo_alignment_check")
        self.declare_parameter("model_states_topic", "/gazebo/model_states")
        self.declare_parameter("target_model", "walking_target")
        self.declare_parameter("leader_model", "go2_1")
        self.declare_parameter("extra_models", "person_standing,person_standing_0")
        self.declare_parameter("target_odom_topic", "/walking_target/odom")
        self.declare_parameter("leader_odom_topic", "/go2_1/odom")
        self.declare_parameter("print_once", True)
        self.declare_parameter("wait_seconds", 2.0)

        self.target_model = str(self.get_parameter("target_model").value)
        self.leader_model = str(self.get_parameter("leader_model").value)
        extra = str(self.get_parameter("extra_models").value)
        self.extra_models = [name.strip() for name in extra.split(",") if name.strip()]
        self.print_once = bool(self.get_parameter("print_once").value)
        self.wait_seconds = float(self.get_parameter("wait_seconds").value)
        self.target_odom = None
        self.leader_odom = None
        self.model_states = None
        self.start_time = self.get_clock().now()

        self.create_subscription(
            Odometry,
            str(self.get_parameter("target_odom_topic").value),
            self._target_odom_cb,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("leader_odom_topic").value),
            self._leader_odom_cb,
            10,
        )
        self.create_subscription(
            ModelStates,
            str(self.get_parameter("model_states_topic").value),
            self._model_states_cb,
            10,
        )
        self.timer = self.create_timer(0.2, self._timer_cb)

    def _target_odom_cb(self, msg):
        self.target_odom = msg

    def _leader_odom_cb(self, msg):
        self.leader_odom = msg

    def _model_states_cb(self, msg):
        self.model_states = msg

    def _timer_cb(self):
        elapsed = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        if elapsed < self.wait_seconds or self.model_states is None:
            return
        msg = self.model_states
        names = [self.target_model, self.leader_model] + self.extra_models
        print("\n[GazeboAlignmentCheck]", flush=True)
        for name in names:
            if name not in msg.name:
                print(f"  model_states[{name}]: NOT FOUND", flush=True)
                continue
            idx = msg.name.index(name)
            pose = msg.pose[idx]
            yaw = yaw_from_quaternion(pose.orientation)
            print(
                f"  model_states[{name}]: "
                f"x={pose.position.x:+.3f}, y={pose.position.y:+.3f}, "
                f"z={pose.position.z:+.3f}, yaw={yaw:+.3f}",
                flush=True,
            )

        if self.target_odom is not None:
            p = self.target_odom.pose.pose.position
            print(f"  /walking_target/odom: x={p.x:+.3f}, y={p.y:+.3f}, z={p.z:+.3f}", flush=True)
        else:
            print("  /walking_target/odom: NOT RECEIVED", flush=True)

        if self.leader_odom is not None:
            p = self.leader_odom.pose.pose.position
            print(f"  /go2_1/odom:         x={p.x:+.3f}, y={p.y:+.3f}, z={p.z:+.3f}", flush=True)
        else:
            print("  /go2_1/odom:         NOT RECEIVED", flush=True)

        if self.target_odom is not None and self.leader_odom is not None:
            tp = self.target_odom.pose.pose.position
            lp = self.leader_odom.pose.pose.position
            dx = tp.x - lp.x
            dy = tp.y - lp.y
            print(
                f"  target - go2_1: dx={dx:+.3f}, dy={dy:+.3f}, dist={math.hypot(dx, dy):.3f}",
                flush=True,
            )
        if self.print_once:
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = GazeboAlignmentCheck()
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
