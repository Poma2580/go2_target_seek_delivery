#!/usr/bin/env python3
"""Role-aware exclusive Nav2/MADDPG command selector for three Go2 robots."""

from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String


ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


def navigation_dogs(robot_names, perception_dog):
    names = tuple(robot_names)
    if perception_dog not in names:
        raise ValueError("perception dog must be configured")
    return tuple(name for name in names if name != perception_dog)


@dataclass
class CommandSample:
    message: Twist = None
    receive_time: object = None


class FollowerCmdVelMux(Node):
    """Forward exactly one command source and fail closed on stale input."""

    def __init__(self):
        super().__init__("follower_cmd_vel_mux")
        self.declare_parameter("select_topic", "/dynamic_encircle/use_maddpg")
        self.declare_parameter("role_topic", "/target_role/perception_robot")
        self.declare_parameter("robot_names", list(ROBOT_NAMES))
        self.declare_parameter("nav_topic_suffix", "nav_cmd_vel")
        self.declare_parameter("maddpg_topic_suffix", "maddpg_cmd_vel")
        self.declare_parameter("output_topic_suffix", "cmd_vel")
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("command_timeout", 0.5)

        self.select_topic = str(self.get_parameter("select_topic").value)
        self.role_topic = str(self.get_parameter("role_topic").value)
        self.robot_names = tuple(self.get_parameter("robot_names").value)
        if len(self.robot_names) != 3 or len(set(self.robot_names)) != 3:
            raise ValueError("robot_names must contain three unique names")
        self.nav_suffix = str(
            self.get_parameter("nav_topic_suffix").value
        ).strip("/")
        self.maddpg_suffix = str(
            self.get_parameter("maddpg_topic_suffix").value
        ).strip("/")
        self.output_suffix = str(
            self.get_parameter("output_topic_suffix").value
        ).strip("/")
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.command_timeout = float(self.get_parameter("command_timeout").value)
        if publish_rate <= 0.0 or self.command_timeout <= 0.0:
            raise ValueError("publish_rate and command_timeout must be positive")

        self.use_maddpg = False
        self.perception_dog = None
        self.navigation_dogs = ()
        self.samples = {
            source: {name: CommandSample() for name in self.robot_names}
            for source in ("nav", "maddpg")
        }
        self.output_pubs = {
            name: self.create_publisher(
                Twist, f"/{name}/{self.output_suffix}", 10
            )
            for name in self.robot_names
        }
        for name in self.robot_names:
            self.create_subscription(
                Twist,
                f"/{name}/{self.nav_suffix}",
                lambda msg, robot=name: self._command_cb("nav", robot, msg),
                10,
            )
            self.create_subscription(
                Twist,
                f"/{name}/{self.maddpg_suffix}",
                lambda msg, robot=name: self._command_cb(
                    "maddpg", robot, msg
                ),
                10,
            )

        selector_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(
            Bool, self.select_topic, self._select_cb, selector_qos
        )
        self.create_subscription(
            String, self.role_topic, self._role_cb, selector_qos
        )
        self.timer = self.create_timer(1.0 / publish_rate, self._timer_cb)
        self.get_logger().info(
            "Follower cmd_vel mux started in Nav2 mode: "
            f"selector={self.select_topic}, timeout={self.command_timeout:.2f}s"
        )

    def _command_cb(self, source, name, message):
        sample = self.samples[source][name]
        sample.message = message
        sample.receive_time = self.get_clock().now()

    def _select_cb(self, message):
        requested = bool(message.data)
        if requested == self.use_maddpg:
            return
        self.use_maddpg = requested
        # A zero at the edge prevents a command cached from the previous mode
        # from leaking across the ownership transition.
        for name in self.navigation_dogs:
            self.output_pubs[name].publish(Twist())
        selected = "MADDPG" if self.use_maddpg else "Nav2"
        self.get_logger().warning(f"Follower command ownership -> {selected}")

    def _role_cb(self, message):
        selected = message.data.strip("/")
        if selected not in self.robot_names:
            self.get_logger().warning(f"Ignoring unknown perception robot: {selected}")
            return
        if self.perception_dog is not None:
            if selected != self.perception_dog:
                self.get_logger().warning(
                    f"Ignoring conflicting perception role {selected}"
                )
            return
        self.perception_dog = selected
        self.navigation_dogs = navigation_dogs(self.robot_names, selected)
        for name in self.navigation_dogs:
            self.output_pubs[name].publish(Twist())
        self.get_logger().info(
            f"Mux role locked: perception={selected}, "
            f"navigation={','.join(self.navigation_dogs)}"
        )

    def _timer_cb(self):
        source = "maddpg" if self.use_maddpg else "nav"
        now = self.get_clock().now()
        names = self.navigation_dogs or self.robot_names
        for name in names:
            sample = self.samples[source][name]
            fresh = (
                sample.receive_time is not None
                and (now - sample.receive_time).nanoseconds * 1e-9
                <= self.command_timeout
            )
            self.output_pubs[name].publish(
                sample.message if fresh and sample.message is not None else Twist()
            )

    def stop(self):
        names = self.navigation_dogs or self.robot_names
        for name in names:
            self.output_pubs[name].publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = FollowerCmdVelMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
