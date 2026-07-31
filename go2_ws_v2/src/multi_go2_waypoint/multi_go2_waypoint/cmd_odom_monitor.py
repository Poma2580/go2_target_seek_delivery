#!/usr/bin/env python3
"""Monitor commanded velocity versus odometry feedback for Go2 robots."""

import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def world_to_body_vx(yaw, vx, vy):
    """Project world-frame velocity onto robot forward axis."""
    return math.cos(yaw) * vx + math.sin(yaw) * vy


@dataclass
class RobotSample:
    cmd: Twist | None = None
    odom: Odometry | None = None
    cmd_time = None
    odom_time = None


class CmdOdomMonitor(Node):
    def __init__(self):
        super().__init__("cmd_odom_monitor")
        self.declare_parameter("robots", ["go2_2", "go2_3"])
        self.declare_parameter("log_rate", 2.0)
        self.declare_parameter("stale_timeout", 1.0)

        robots_param = self.get_parameter("robots").value
        self.robots = [str(name) for name in robots_param]
        self.stale_timeout = float(self.get_parameter("stale_timeout").value)
        self.samples = {name: RobotSample() for name in self.robots}

        for name in self.robots:
            self.create_subscription(Twist, f"/{name}/cmd_vel", self._cmd_cb(name), 20)
            self.create_subscription(Odometry, f"/{name}/odom", self._odom_cb(name), 20)

        rate = float(self.get_parameter("log_rate").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-6), self._timer_cb)
        self.get_logger().info(
            f"cmd_odom_monitor started for {', '.join(self.robots)} "
            f"(log_rate={rate:.2f} Hz)"
        )

    def _cmd_cb(self, name):
        def callback(msg):
            sample = self.samples[name]
            sample.cmd = msg
            sample.cmd_time = self.get_clock().now()

        return callback

    def _odom_cb(self, name):
        def callback(msg):
            sample = self.samples[name]
            sample.odom = msg
            sample.odom_time = self.get_clock().now()

        return callback

    def _age(self, stamp):
        if stamp is None:
            return float("inf")
        return (self.get_clock().now() - stamp).nanoseconds * 1e-9

    def _timer_cb(self):
        lines = ["\n[cmd_odom_monitor]"]
        for name in self.robots:
            sample = self.samples[name]
            cmd_age = self._age(sample.cmd_time)
            odom_age = self._age(sample.odom_time)
            if (
                sample.cmd is None
                or sample.odom is None
                or cmd_age > self.stale_timeout
                or odom_age > self.stale_timeout
            ):
                lines.append(
                    f"  {name}: waiting/freshness cmd_age={cmd_age:.2f}s "
                    f"odom_age={odom_age:.2f}s"
                )
                continue

            cmd_v = float(sample.cmd.linear.x)
            cmd_w = float(sample.cmd.angular.z)
            odom = sample.odom
            yaw = quaternion_to_yaw(odom.pose.pose.orientation)
            odom_vx_world = float(odom.twist.twist.linear.x)
            odom_vy_world = float(odom.twist.twist.linear.y)
            odom_v_forward = world_to_body_vx(yaw, odom_vx_world, odom_vy_world)
            odom_speed = math.hypot(odom_vx_world, odom_vy_world)
            odom_w = float(odom.twist.twist.angular.z)
            err_v = cmd_v - odom_v_forward
            err_w = cmd_w - odom_w

            lines.append(
                f"  {name}: "
                f"cmd=(v={cmd_v:+.3f}, w={cmd_w:+.3f}) "
                f"odom=(v_body_x={odom_v_forward:+.3f}, speed={odom_speed:.3f}, w={odom_w:+.3f}) "
                f"err=(v={err_v:+.3f}, w={err_w:+.3f}) "
                f"age=(cmd={cmd_age:.2f}s, odom={odom_age:.2f}s)"
            )

        self.get_logger().info("\n".join(lines))


def main(args=None):
    rclpy.init(args=args)
    node = CmdOdomMonitor()
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
