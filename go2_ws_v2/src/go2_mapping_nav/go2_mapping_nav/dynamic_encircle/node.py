"""ROS orchestration node for modular dynamic encirclement components."""

import math
from functools import partial

from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

# Importing this module registers geometry_msgs conversions with tf2_ros.
import tf2_geometry_msgs  # noqa: F401,E402

from .config import EncircleConfig, LOOP_CORNERS
from .formation_planner import FormationPlanner
from .geometry import (
    Loop,
    navigation_dog_names,
    quaternion_to_yaw,
    yaw_to_quaternion_components,
)
from .models import DogState, TargetSample
from .nav_goal_manager import NavGoalManager
from .perception_controller import PerceptionController
from .target_tracker import TargetTracker


class DynamicEncircle(Node):
    """Coordinate target tracking, direct control, formation, and Nav2 goals."""

    def __init__(self):
        """创建配置、功能组件、ROS 通信对象、TF 和三个定时器。"""
        super().__init__("nav2_dynamic_encircle")
        self.config = EncircleConfig.declare_and_load(self)
        self.control_dt = 1.0 / self.config.control_rate

        loop = Loop(LOOP_CORNERS)
        self.target_tracker = TargetTracker(
            self.config.target_timeout,
            self.config.target_hold,
            self.config.max_coast_speed,
        )
        self.perception_controller = PerceptionController(self.config, loop)
        self.formation_planner = FormationPlanner(self.config, loop)
        self.nav_goal_manager = NavGoalManager(
            self,
            self.config.robot_names,
            self.config.global_frame,
            1.0 / self.config.nav_goal_update_rate,
        )

        self.perception_dog = None
        self.navigation_dogs = ()
        self._target_lost_logged = False
        self.dogs = {
            name: DogState(name=name) for name in self.config.robot_names
        }
        self.command_publishers = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in self.config.robot_names
        }
        self.odom_subscriptions = [
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                partial(self._odom_callback, name),
                10,
            )
            for name in self.config.robot_names
        ]
        self.target_subscriptions = [
            self.create_subscription(
                Odometry,
                f"/{name}/target_estimated/odom",
                partial(self._target_callback, name),
                10,
            )
            for name in self.config.robot_names
        ]
        role_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.role_subscription = self.create_subscription(
            String,
            self.config.perception_robot_topic,
            self._role_callback,
            role_qos,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.control_timer = self.create_timer(
            self.control_dt,
            self._control_timer_callback,
        )
        self.encircle_timer = self.create_timer(
            1.0 / self.config.encircle_update_rate,
            self._geometry_timer_callback,
        )
        self.nav_goal_timer = self.create_timer(
            1.0 / self.config.nav_goal_update_rate,
            self._nav_goal_timer_callback,
        )

        self.get_logger().info(
            "[role] dynamic encircle started: topic=%s robots=%s geometry=%.2f Hz "
            "Nav2 goals=%.2f Hz radius=%.2f m tolerance=%.2f m/%.1f deg"
            % (
                self.config.perception_robot_topic,
                ",".join(self.config.robot_names),
                self.config.encircle_update_rate,
                self.config.nav_goal_update_rate,
                self.config.formation_radius,
                self.config.success_tolerance,
                math.degrees(self.config.success_yaw_tolerance),
            )
        )

    def _clock_seconds(self):
        """Return the current ROS clock as floating-point seconds."""
        return self.get_clock().now().nanoseconds * 1e-9

    def _message_age(self, stamp):
        """Return the age of a ROS message timestamp in seconds."""
        if stamp is None:
            return float("inf")
        return (self.get_clock().now() - Time.from_msg(stamp)).nanoseconds * 1e-9

    def _odom_callback(self, name, message):
        """Update one data-only robot state from odometry."""
        dog = self.dogs[name]
        dog.x = message.pose.pose.position.x
        dog.y = message.pose.pose.position.y
        dog.yaw = quaternion_to_yaw(message.pose.pose.orientation)
        dog.frame_id = message.header.frame_id or f"{name}/odom"
        dog.last_stamp = message.header.stamp
        dog.received = True

    def _role_callback(self, message):
        """Lock the first valid perception role and configure both controllers."""
        selected = message.data.strip("/")
        if selected not in self.config.robot_names:
            self.get_logger().warning(f"[role] Ignoring unknown robot: {selected}")
            return
        if self.perception_dog is not None:
            if selected != self.perception_dog:
                self.get_logger().warning(
                    f"[role] Ignoring conflicting role {selected}; "
                    f"already locked to {self.perception_dog}"
                )
            return

        self.perception_dog = selected
        self.navigation_dogs = navigation_dog_names(
            self.config.robot_names,
            selected,
        )
        self.perception_controller.start()
        self.nav_goal_manager.set_navigation_dogs(self.navigation_dogs)
        self.target_tracker.activate(selected, self._clock_seconds())
        self.get_logger().info(
            "[role] locked: perception=%s navigation=%s"
            % (selected, ",".join(self.navigation_dogs))
        )

    def _target_callback(self, name, message):
        """Cache all target samples and validate the elected source frame."""
        sample = TargetSample(
            x=message.pose.pose.position.x,
            y=message.pose.pose.position.y,
            vx=message.twist.twist.linear.x,
            vy=message.twist.twist.linear.y,
            frame_id=message.header.frame_id,
            received_at=self._clock_seconds(),
        )
        accepted = self.target_tracker.update_active(name, sample)
        if name != self.perception_dog or accepted:
            return
        self.get_logger().warning(
            f"[target] Ignoring {name} target in frame {sample.frame_id!r}; "
            f"expected {name + '/odom'!r}",
            throttle_duration_sec=2.0,
        )

    def _publish_command(self, name, linear, angular):
        """Publish one planar velocity command."""
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        self.command_publishers[name].publish(command)

    def _publish_zero(self, name):
        """Reset controller history and command one robot to stop."""
        self.perception_controller.reset_command_history(self.dogs[name])
        self.command_publishers[name].publish(Twist())

    def _control_timer_callback(self):
        """Resolve target state and publish direct perception-robot control."""
        if self.perception_dog is None:
            return
        dog = self.dogs[self.perception_dog]
        if not dog.received or self._message_age(dog.last_stamp) > self.config.odom_timeout:
            self._publish_zero(self.perception_dog)
            return

        target = self.target_tracker.resolve(self._clock_seconds())
        if target is None:
            self._handle_target_loss()
            return

        self._target_lost_logged = False
        previous_phase = self.perception_controller.phase
        command = self.perception_controller.compute(dog, target, self.control_dt)
        if previous_phase != self.perception_controller.phase:
            self.get_logger().info(
                f"[perception] {self.perception_dog} caught target -> formation"
            )
        self._publish_command(self.perception_dog, command.linear, command.angular)

    def _handle_target_loss(self):
        """Stop direct control and suspend Nav2 after the target hold expires."""
        self._publish_zero(self.perception_dog)
        if not self._target_lost_logged:
            self.get_logger().warning(
                "[target] estimate exceeded %.1f s hold; stopping %s and Nav2 dogs"
                % (self.config.target_hold, self.perception_dog)
            )
            self._target_lost_logged = True
        self.nav_goal_manager.suspend("target estimate lost")

    def _transform_xy(self, x, y, source_frame):
        """Transform a target point into the configured global frame."""
        if source_frame == self.config.global_frame:
            return x, y
        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = Time().to_msg()
        point.point.x = x
        point.point.y = y
        transformed = self.tf_buffer.transform(
            point,
            self.config.global_frame,
            timeout=Duration(seconds=self.config.tf_timeout),
        )
        return transformed.point.x, transformed.point.y

    def _transform_dog_pose(self, dog):
        """Transform one robot pose into the configured global frame."""
        if dog.frame_id == self.config.global_frame:
            return dog.x, dog.y, dog.yaw
        pose = PoseStamped()
        pose.header.frame_id = dog.frame_id
        pose.header.stamp = Time().to_msg()
        pose.pose.position.x = dog.x
        pose.pose.position.y = dog.y
        pose.pose.orientation.z, pose.pose.orientation.w = (
            yaw_to_quaternion_components(dog.yaw)
        )
        transformed = self.tf_buffer.transform(
            pose,
            self.config.global_frame,
            timeout=Duration(seconds=self.config.tf_timeout),
        )
        return (
            transformed.pose.position.x,
            transformed.pose.position.y,
            quaternion_to_yaw(transformed.pose.orientation),
        )

    def _global_geometry(self, target):
        """Collect fresh robot poses and transform the complete scene."""
        if any(
            not dog.received
            or self._message_age(dog.last_stamp) > self.config.odom_timeout
            for dog in self.dogs.values()
        ):
            return None
        try:
            target_xy = self._transform_xy(target.x, target.y, target.frame_id)
            dog_poses = {
                name: self._transform_dog_pose(dog)
                for name, dog in self.dogs.items()
            }
            return target_xy, dog_poses
        except Exception as error:  # tf2 exception classes vary by ROS release
            self.get_logger().warning(
                f"[geometry] Cannot transform scene to "
                f"{self.config.global_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None

    def _geometry_timer_callback(self):
        """Update fixed slots, completion, and the plan held by Nav2."""
        if self.perception_dog is None:
            return
        target = self.target_tracker.resolve(self._clock_seconds())
        if target is None:
            self.nav_goal_manager.suspend("target estimate lost")
            return
        if self.nav_goal_manager.resume():
            self.get_logger().info("[target] estimate recovered; Nav2 goals resumed")

        transformed = self._global_geometry(target)
        if transformed is None:
            return
        target_xy, dog_poses = transformed
        plan = self.formation_planner.update(
            target_xy,
            dog_poses,
            self.perception_dog,
            self.navigation_dogs,
        )
        self.nav_goal_manager.set_plan(plan)
        if plan.assignment_created:
            self.get_logger().info(
                "[geometry] Fixed Nav2 slot assignment: "
                + ", ".join(
                    f"{name}->slot{index}"
                    for name, index in plan.slot_indices.items()
                )
            )
        if plan.completed:
            self._complete_encirclement()
            return
        if self.nav_goal_manager.last_dispatch is None:
            if self.nav_goal_manager.dispatch_if_due(self._clock_seconds()):
                self.nav_goal_timer.reset()

    def _nav_goal_timer_callback(self):
        """Dispatch the latest plan while the target remains valid."""
        now = self._clock_seconds()
        if self.target_tracker.resolve(now) is None:
            return
        self.nav_goal_manager.dispatch_if_due(now)

    def _complete_encirclement(self):
        """Latch Nav2 completion while leaving direct target tracking active."""
        if not self.nav_goal_manager.complete():
            return
        self.get_logger().info(
            f"[completion] {' and '.join(self.navigation_dogs)} are simultaneously "
            f"within {self.config.success_tolerance:.2f} m of fixed slots and "
            f"within {math.degrees(self.config.success_yaw_tolerance):.1f} deg "
            f"of route heading; Nav2 updates stopped while "
            f"{self.perception_dog} continues tracking"
        )

    def stop(self):
        """Stop direct control and cancel Nav2 goals before shutdown."""
        if self.perception_dog is not None:
            self._publish_zero(self.perception_dog)
        self.nav_goal_manager.shutdown()
        self.get_logger().info("[shutdown] dynamic encircle stopped")
