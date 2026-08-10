#!/usr/bin/env python3
"""Resettable Gazebo actor controller for the walking target.

The original ``walking_target`` in ``QY_MODEL/target_seek`` was driven by an
SDF ``<script><trajectory>``.  That makes episode reset unreliable: calling
``/gazebo/set_entity_state`` can move the actor for one frame, then the actor
script snaps it back to the trajectory time position.

This node assumes the actor still exists visually, but its SDF trajectory has
been removed.  It owns the actor state by repeatedly calling
``/gazebo/set_entity_state``.  The target therefore remains the same YOLO-visible
actor, while training can reset it deterministically every episode.
"""

import math
import time

import rclpy
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_srvs.srv import Empty


def yaw_to_quaternion(yaw):
    qz = math.sin(0.5 * yaw)
    qw = math.cos(0.5 * yaw)
    return 0.0, 0.0, qz, qw


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class WalkingTargetActorController(Node):
    def __init__(self):
        super().__init__("walking_target_actor_controller")

        self.declare_parameter("model_name", "walking_target")
        self.declare_parameter("set_entity_service", "/gazebo/set_entity_state")
        self.declare_parameter("cmd_vel_topic", "/walking_target/cmd_vel")
        self.declare_parameter("reset_service", "/walking_target/reset")
        self.declare_parameter("start_x", -8.0)
        self.declare_parameter("start_y", 4.0)
        self.declare_parameter("start_z", 0.0)
        self.declare_parameter("start_yaw", 6.2832)
        # Same speed as QY_MODEL/target_seek: (-13,4) -> (81,4) in 464 s.
        self.declare_parameter("default_speed", 94.0 / 464.0)
        self.declare_parameter("default_angular", 0.0)
        self.declare_parameter("auto_start", False)
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("command_timeout", 1.0)
        self.declare_parameter("initial_reset", True)
        self.declare_parameter("debug_log_interval", 2.0)

        self.model_name = str(self.get_parameter("model_name").value)
        self.set_entity_service = str(self.get_parameter("set_entity_service").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.reset_service = str(self.get_parameter("reset_service").value)
        self.start_x = float(self.get_parameter("start_x").value)
        self.start_y = float(self.get_parameter("start_y").value)
        self.start_z = float(self.get_parameter("start_z").value)
        self.start_yaw = float(self.get_parameter("start_yaw").value)
        self.default_speed = float(self.get_parameter("default_speed").value)
        self.default_angular = float(self.get_parameter("default_angular").value)
        self.auto_start = bool(self.get_parameter("auto_start").value)
        self.control_rate = float(self.get_parameter("control_rate").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        self.initial_reset = bool(self.get_parameter("initial_reset").value)
        self.debug_log_interval = float(self.get_parameter("debug_log_interval").value)

        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        self.yaw = self.start_yaw
        self.v = self.default_speed if self.auto_start else 0.0
        self.w = self.default_angular if self.auto_start else 0.0
        self.last_cmd_wall_time = None
        self.last_update_time = None
        self.pending_future = None
        self.initial_reset_done = False
        self.last_debug_log_time = 0.0

        self.set_entity_client = self.create_client(SetEntityState, self.set_entity_service)
        self.create_subscription(Twist, self.cmd_vel_topic, self._cmd_cb, 10)
        self.create_service(Empty, self.reset_service, self._reset_cb)
        self.timer = self.create_timer(1.0 / max(self.control_rate, 1.0), self._timer_cb)

        self.get_logger().info(
            "walking_target_actor_controller started: "
            f"model={self.model_name}, start=({self.start_x:.2f},{self.start_y:.2f},{self.start_yaw:.2f}), "
            f"default_speed={self.default_speed:.3f}, cmd={self.cmd_vel_topic}, reset={self.reset_service}"
        )

    def _cmd_cb(self, msg):
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)
        self.last_cmd_wall_time = time.time()

    def _reset_pose(self):
        self.x = self.start_x
        self.y = self.start_y
        self.z = self.start_z
        self.yaw = self.start_yaw
        self.v = 0.0
        self.w = 0.0
        self.last_cmd_wall_time = None
        self.last_update_time = None
        self._send_state(vx=0.0, vy=0.0, wz=0.0, force=True)

    def _reset_cb(self, request, response):
        del request
        self._reset_pose()
        self.get_logger().info(
            f"walking_target reset to ({self.x:.2f}, {self.y:.2f}, yaw={self.yaw:.2f}), "
            f"speed={self.v:.3f}"
        )
        return response

    def _send_state(self, vx=0.0, vy=0.0, wz=0.0, force=False):
        if not self.set_entity_client.service_is_ready():
            self.set_entity_client.wait_for_service(timeout_sec=0.0)
            return
        if self.pending_future is not None and not self.pending_future.done() and not force:
            return

        req = SetEntityState.Request()
        req.state.name = self.model_name
        req.state.reference_frame = "world"
        req.state.pose.position.x = float(self.x)
        req.state.pose.position.y = float(self.y)
        req.state.pose.position.z = float(self.z)
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        req.state.twist.linear.x = float(vx)
        req.state.twist.linear.y = float(vy)
        req.state.twist.angular.z = float(wz)
        self.pending_future = self.set_entity_client.call_async(req)
        self.pending_future.add_done_callback(self._set_entity_done_cb)

    def _set_entity_done_cb(self, future):
        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"set_entity_state({self.model_name}) exception: {exc}")
            return
        if result is None:
            self.get_logger().error(f"set_entity_state({self.model_name}) returned no result")
            return
        if hasattr(result, "success") and not result.success:
            self.get_logger().error(
                f"set_entity_state({self.model_name}) failed: {getattr(result, 'status_message', '')}"
            )
        elif self.debug_log_interval > 0:
            self.get_logger().debug(f"set_entity_state({self.model_name}) ok")

    def _send_state_sync(self, vx=0.0, vy=0.0, wz=0.0):
        if not self.set_entity_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"service not available: {self.set_entity_service}")

        req = SetEntityState.Request()
        req.state.name = self.model_name
        req.state.reference_frame = "world"
        req.state.pose.position.x = float(self.x)
        req.state.pose.position.y = float(self.y)
        req.state.pose.position.z = float(self.z)
        qx, qy, qz, qw = yaw_to_quaternion(self.yaw)
        req.state.pose.orientation.x = qx
        req.state.pose.orientation.y = qy
        req.state.pose.orientation.z = qz
        req.state.pose.orientation.w = qw
        req.state.twist.linear.x = float(vx)
        req.state.twist.linear.y = float(vy)
        req.state.twist.angular.z = float(wz)
        future = self.set_entity_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)
        result = future.result()
        if result is None:
            raise RuntimeError(f"failed to set entity state: {self.model_name}")
        if hasattr(result, "success") and not result.success:
            raise RuntimeError(f"set_entity_state({self.model_name}) failed: {result.status_message}")
        self.pending_future = None

    def _timer_cb(self):
        if self.initial_reset and not self.initial_reset_done:
            if not self.set_entity_client.service_is_ready():
                self.set_entity_client.wait_for_service(timeout_sec=0.0)
                return
            self._reset_pose()
            self.initial_reset_done = True
            self.get_logger().info(
                f"initial reset sent: ({self.x:.2f}, {self.y:.2f}, yaw={self.yaw:.2f})"
            )

        now = self.get_clock().now()
        if self.last_update_time is None:
            dt = 1.0 / max(self.control_rate, 1.0)
        else:
            dt = (now - self.last_update_time).nanoseconds * 1e-9
            if dt <= 0.0 or dt > 1.0:
                dt = 1.0 / max(self.control_rate, 1.0)
        self.last_update_time = now

        # If no recent command exists, keep the original straight-walking motion.
        if self.last_cmd_wall_time is None or time.time() - self.last_cmd_wall_time > self.command_timeout:
            if self.auto_start:
                self.v = self.default_speed
                self.w = self.default_angular
            else:
                self.v = 0.0
                self.w = 0.0

        self.yaw = normalize_angle(self.yaw + self.w * dt)
        self.x += self.v * math.cos(self.yaw) * dt
        self.y += self.v * math.sin(self.yaw) * dt
        self._send_state(vx=self.v * math.cos(self.yaw), vy=self.v * math.sin(self.yaw), wz=self.w)

        wall_now = time.time()
        if self.debug_log_interval > 0 and wall_now - self.last_debug_log_time >= self.debug_log_interval:
            self.last_debug_log_time = wall_now
            self.get_logger().info(
                f"commanding {self.model_name}: pos=({self.x:+.2f},{self.y:+.2f}), "
                f"yaw={self.yaw:+.2f}, v={self.v:+.3f}, w={self.w:+.3f}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = WalkingTargetActorController()
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
