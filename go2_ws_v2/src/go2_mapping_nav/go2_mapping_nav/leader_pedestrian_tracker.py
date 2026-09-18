"""Track the Gazebo walking actor with GO1 without commanding GO2 or GO3."""

from types import SimpleNamespace

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger

from .dynamic_encircle.config import EncircleConfig, LOOP_CORNERS
from .dynamic_encircle.geometry import Loop, quaternion_to_yaw
from .dynamic_encircle.models import DogState
from .dynamic_encircle.perception_controller import PerceptionController


class LeaderPedestrianTracker(Node):
    """Run only the former perception-dog controller, fixed to GO1."""

    def __init__(self):
        super().__init__("leader_pedestrian_tracker")
        self.declare_parameter("leader_name", "go2_1")
        self.declare_parameter("target_topic", "/walking_target/odom")
        self.declare_parameter("control_rate", 20.0)
        self.declare_parameter("target_timeout", 1.0)
        self.declare_parameter("formation_radius", 2.0)
        self.declare_parameter("catch_radius", 3.5)
        self.declare_parameter("catch_speed", 0.10)
        self.declare_parameter("max_linear", 0.10)
        self.declare_parameter("max_angular", 0.9)
        self.declare_parameter("auto_start_target", True)
        self.declare_parameter("formation_ready_topic", "/maddpg_waypoint/ready")

        self.leader_name = str(self.get_parameter("leader_name").value)
        target_topic = str(self.get_parameter("target_topic").value)
        control_rate = float(self.get_parameter("control_rate").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.auto_start_target = bool(
            self.get_parameter("auto_start_target").value
        )
        if control_rate <= 0.0 or self.target_timeout <= 0.0:
            raise ValueError("control_rate and target_timeout must be positive")

        config = EncircleConfig(
            control_rate=control_rate,
            target_timeout=self.target_timeout,
            formation_radius=float(self.get_parameter("formation_radius").value),
            catch_radius=float(self.get_parameter("catch_radius").value),
            catch_speed=float(self.get_parameter("catch_speed").value),
            max_linear=float(self.get_parameter("max_linear").value),
            max_angular=float(self.get_parameter("max_angular").value),
        )
        config.validate()
        self.controller = PerceptionController(config, Loop(LOOP_CORNERS))
        self.controller.start()
        self.control_dt = 1.0 / control_rate
        self.dog = DogState(name=self.leader_name)
        self.target = None
        self.target_received_at = None
        self.last_phase = self.controller.phase
        self.formation_ready = False

        self.command_publisher = self.create_publisher(
            Twist, f"/{self.leader_name}/cmd_vel", 10
        )
        self.create_subscription(
            Odometry, f"/{self.leader_name}/odom", self._odom_callback, 20
        )
        self.create_subscription(Odometry, target_topic, self._target_callback, 20)
        self.create_subscription(
            Bool,
            str(self.get_parameter("formation_ready_topic").value),
            self._formation_ready_callback,
            10,
        )
        self.create_timer(self.control_dt, self._control_callback)

        self.start_client = self.create_client(Trigger, "/walking_target/start")
        self.pause_client = self.create_client(Trigger, "/walking_target/pause")
        self.start_future = None
        self.pause_future = None
        self.initial_pause_complete = not self.auto_start_target
        self.target_started = not self.auto_start_target
        if self.auto_start_target:
            self.create_timer(0.5, self._manage_target_motion)

        self.get_logger().warning(
            "GO1 waits for the initial formation, then tracks the pedestrian: "
            "catch_speed=%.2f m/s, max_linear=%.2f m/s. GO2/GO3 are not "
            "commanded by this node."
            % (config.catch_speed, config.max_linear)
        )

    def _clock_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _odom_callback(self, message):
        self.dog.x = float(message.pose.pose.position.x)
        self.dog.y = float(message.pose.pose.position.y)
        self.dog.yaw = quaternion_to_yaw(message.pose.pose.orientation)
        self.dog.received = True
        self.dog.last_stamp = message.header.stamp

    def _target_callback(self, message):
        self.target = SimpleNamespace(
            x=float(message.pose.pose.position.x),
            y=float(message.pose.pose.position.y),
            vx=float(message.twist.twist.linear.x),
            vy=float(message.twist.twist.linear.y),
        )
        self.target_received_at = self._clock_seconds()

    def _formation_ready_callback(self, message):
        if bool(message.data) and not self.formation_ready:
            self.formation_ready = True
            self.get_logger().info(
                "Initial GO2/GO3 formation confirmed; starting pedestrian and GO1"
            )

    def _manage_target_motion(self):
        if not self.initial_pause_complete:
            self._pause_target_if_ready()
            return
        if self.formation_ready:
            self._start_target_if_ready()

    def _pause_target_if_ready(self):
        if self.pause_future is not None or not self.pause_client.service_is_ready():
            return
        self.pause_future = self.pause_client.call_async(Trigger.Request())
        self.pause_future.add_done_callback(self._pause_target_done)

    def _pause_target_done(self, future):
        self.pause_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"walking target pause failed: {error}")
            return
        self.initial_pause_complete = bool(response.success)
        self.target_started = False
        if response.success:
            self.get_logger().info(
                "Walking target paused until the initial formation is ready"
            )
        else:
            self.get_logger().warning(response.message)

    def _start_target_if_ready(self):
        if (
            not self.formation_ready
            or self.target_started
            or self.start_future is not None
        ):
            return
        if not self.start_client.service_is_ready():
            return
        self.start_future = self.start_client.call_async(Trigger.Request())
        self.start_future.add_done_callback(self._start_target_done)

    def _start_target_done(self, future):
        self.start_future = None
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().warning(f"walking target start failed: {error}")
            return
        self.target_started = bool(response.success)
        if response.success:
            self.get_logger().info(response.message)
        else:
            self.get_logger().warning(response.message)

    def _publish_stop(self):
        self.controller.reset_command_history(self.dog)
        self.command_publisher.publish(Twist())

    def _control_callback(self):
        if (
            not self.formation_ready
            or not self.target_started
            or not self.dog.received
            or self.target is None
            or self.target_received_at is None
            or self._clock_seconds() - self.target_received_at > self.target_timeout
        ):
            self._publish_stop()
            return
        command = self.controller.compute(self.dog, self.target, self.control_dt)
        message = Twist()
        message.linear.x = float(command.linear)
        message.angular.z = float(command.angular)
        self.command_publisher.publish(message)
        if self.controller.phase != self.last_phase:
            self.last_phase = self.controller.phase
            self.get_logger().info(
                "GO1 caught the pedestrian; switching to 2 m formation tracking"
            )

    def stop(self):
        self._publish_stop()


def main(args=None):
    rclpy.init(args=args)
    node = LeaderPedestrianTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
