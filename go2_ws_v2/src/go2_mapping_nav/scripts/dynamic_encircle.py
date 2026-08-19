#!/usr/bin/env python3
"""Three-Go2 dynamic encirclement using direct control and Nav2 goals.

go2_1 keeps the original camera-facing catch-up/formation controller.  The
other two robots receive fixed-slot NavigateToPose goals around the perceived
target.  Slot geometry and Nav2 goal publication deliberately run at separate
rates so a noisy perception stream does not continuously reset navigation.
"""

import itertools
import math
from functools import partial

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time
from std_msgs.msg import Bool, String
from tf2_ros import Buffer, TransformListener

# Importing this module registers geometry_msgs conversions with tf2_ros.
import tf2_geometry_msgs  # noqa: F401,E402


DOG_NAMES = ("go2_1", "go2_2", "go2_3")
NAV_DOG_NAMES = ("go2_2", "go2_3")
LOOP_CORNERS = ((41.0, 4.0), (41.0, 36.0), (-13.0, 36.0), (-13.0, 4.0))

NAV2_ACTIVE = "NAV2_ACTIVE"
ARRIVAL_HOLD = "ARRIVAL_HOLD"
NAV2_CANCELLING = "NAV2_CANCELLING"
WAITING_FOR_STOP = "WAITING_FOR_STOP"
WAITING_FOR_MADDPG_READY = "WAITING_FOR_MADDPG_READY"
ENABLING_MADDPG = "ENABLING_MADDPG"
MADDPG_ACTIVE = "MADDPG_ACTIVE"
HANDOFF_FAILED = "HANDOFF_FAILED"


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def quaternion_to_yaw(quaternion):
    siny_cosp = 2.0 * (
        quaternion.w * quaternion.z + quaternion.x * quaternion.y
    )
    cosy_cosp = 1.0 - 2.0 * (
        quaternion.y * quaternion.y + quaternion.z * quaternion.z
    )
    return math.atan2(siny_cosp, cosy_cosp)


def solve_encircle_points(target_x, target_y, radius, num_dogs, start_angle):
    """Return uniformly spaced (x, y, yaw) slots anchored at start_angle."""
    if num_dogs < 1:
        raise ValueError("num_dogs must be greater than zero")
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("formation_radius must be finite and greater than zero")
    values = (target_x, target_y, start_angle)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("encircle geometry inputs must be finite")

    points = []
    for index in range(num_dogs):
        angle = normalize_angle(start_angle + 2.0 * math.pi * index / num_dogs)
        points.append(
            (
                target_x + radius * math.cos(angle),
                target_y + radius * math.sin(angle),
                normalize_angle(angle + math.pi),
            )
        )
    return points


def assign_remaining_slots(dog_positions, points):
    """Assign point indices 1..N to navigation dogs with minimum total distance."""
    names = tuple(dog_positions)
    candidate_indices = tuple(range(1, len(points)))
    if len(names) != len(candidate_indices):
        raise ValueError("navigation dog count must match remaining slot count")

    best_indices = None
    best_cost = float("inf")
    for permutation in itertools.permutations(candidate_indices):
        cost = sum(
            math.hypot(
                points[point_index][0] - dog_positions[name][0],
                points[point_index][1] - dog_positions[name][1],
            )
            for name, point_index in zip(names, permutation)
        )
        if cost < best_cost:
            best_cost = cost
            best_indices = permutation
    return dict(zip(names, best_indices))


def update_assigned_slots(slot_indices, points):
    """Update slot coordinates while preserving the initial robot/index mapping."""
    if any(index <= 0 or index >= len(points) for index in slot_indices.values()):
        raise ValueError("assigned slot index is outside the remaining points")
    return {name: points[index] for name, index in slot_indices.items()}


def encircle_reached(dog_positions, assigned_points, tolerance):
    """Return true only when all dogs are simultaneously inside tolerance."""
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("success_tolerance must be finite and greater than zero")
    if set(dog_positions) != set(assigned_points):
        raise ValueError("dog positions and assigned points must have matching names")
    return all(
        math.hypot(
            dog_positions[name][0] - assigned_points[name][0],
            dog_positions[name][1] - assigned_points[name][1],
        )
        <= tolerance
        for name in dog_positions
    )


class GoalUpdateState:
    """Pure state for throttling goals and rejecting stale action callbacks."""

    def __init__(self, period):
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("goal update period must be finite and greater than zero")
        self.period = period
        self.last_dispatch = None
        self.generation = 0
        self.suspended = False
        self.completed = False

    def due(self, now):
        if self.suspended or self.completed:
            return False
        return (
            self.last_dispatch is None
            or now - self.last_dispatch >= self.period - 1e-6
        )

    def mark_dispatched(self, now):
        if not self.due(now):
            raise RuntimeError("goal dispatch is not due")
        self.last_dispatch = now
        self.generation += 1
        return self.generation

    def suspend(self):
        if self.completed or self.suspended:
            return False
        self.suspended = True
        self.generation += 1
        return True

    def resume(self):
        if self.completed or not self.suspended:
            return False
        self.suspended = False
        self.last_dispatch = None
        return True

    def complete(self):
        if self.completed:
            return False
        self.completed = True
        self.suspended = False
        self.generation += 1
        return True

    def is_current(self, generation):
        return (
            generation == self.generation
            and not self.suspended
            and not self.completed
        )


class Loop:
    """Closed polygon represented by arc length, retained from the old controller."""

    def __init__(self, corners):
        self.corners = tuple(corners)
        self.edges = []
        self.cumulative = [0.0]
        for index, (x0, y0) in enumerate(self.corners):
            x1, y1 = self.corners[(index + 1) % len(self.corners)]
            length = math.hypot(x1 - x0, y1 - y0)
            self.edges.append((x0, y0, x1, y1, length))
            self.cumulative.append(self.cumulative[-1] + length)
        self.length = self.cumulative[-1]

    def project(self, px, py):
        best_s = 0.0
        best_distance_squared = float("inf")
        for index, (x0, y0, x1, y1, length) in enumerate(self.edges):
            if length < 1e-9:
                continue
            fraction = (
                (px - x0) * (x1 - x0) + (py - y0) * (y1 - y0)
            ) / (length * length)
            fraction = clamp(fraction, 0.0, 1.0)
            closest_x = x0 + fraction * (x1 - x0)
            closest_y = y0 + fraction * (y1 - y0)
            distance_squared = (px - closest_x) ** 2 + (py - closest_y) ** 2
            if distance_squared < best_distance_squared:
                best_distance_squared = distance_squared
                best_s = self.cumulative[index] + fraction * length
        return best_s

    def point_at(self, arc_length):
        arc_length %= self.length
        for index, (x0, y0, x1, y1, length) in enumerate(self.edges):
            if arc_length <= self.cumulative[index + 1] or index == len(self.edges) - 1:
                fraction = (
                    (arc_length - self.cumulative[index]) / length
                    if length > 1e-9
                    else 0.0
                )
                return (
                    x0 + fraction * (x1 - x0),
                    y0 + fraction * (y1 - y0),
                )
        return self.corners[0]

    def signed_arc(self, start, end):
        distance = (end - start) % self.length
        if distance > self.length / 2.0:
            distance -= self.length
        return distance


class DogState:
    def __init__(self, node, name, publish_commands=False):
        self.name = name
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.frame_id = f"{name}/odom"
        self.last_stamp = None
        self.received = False
        self.previous_linear = 0.0
        self.previous_angular = 0.0
        self.linear_speed = 0.0
        self.angular_speed = 0.0
        self.command_publisher = (
            node.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            if publish_commands
            else None
        )
        self.subscription = node.create_subscription(
            Odometry, f"/{name}/odom", self._odom_callback, 10
        )

    def _odom_callback(self, message):
        self.x = message.pose.pose.position.x
        self.y = message.pose.pose.position.y
        self.yaw = quaternion_to_yaw(message.pose.pose.orientation)
        self.linear_speed = math.hypot(
            message.twist.twist.linear.x, message.twist.twist.linear.y
        )
        self.angular_speed = abs(message.twist.twist.angular.z)
        self.frame_id = message.header.frame_id or f"{self.name}/odom"
        self.last_stamp = message.header.stamp
        self.received = True

    def publish_zero(self):
        if self.command_publisher is None:
            return
        self.previous_linear = 0.0
        self.previous_angular = 0.0
        self.command_publisher.publish(Twist())


class DynamicEncircle(Node):
    def __init__(self):
        super().__init__("nav2_dynamic_encircle")

        for name, default in (
            ("formation_radius", 2.0),
            ("success_tolerance", 2.0),
            ("control_rate", 20.0),
            ("encircle_update_rate", 1.0),
            ("nav_goal_update_rate", 0.2),
            ("target_timeout", 5.0),
            ("target_hold", 8.0),
            ("odom_timeout", 0.5),
            ("max_linear", 0.65),
            ("max_angular", 0.9),
            ("max_coast_speed", 0.65),
            ("position_deadband", 0.25),
            ("k_linear", 0.8),
            ("k_angular", 0.9),
            ("turn_in_place_thresh", 1.2),
            ("accel_lin", 1.0),
            ("accel_ang", 3.0),
            ("catch_lookahead", 1.5),
            ("catch_speed", 0.6),
            ("catch_radius", 3.5),
            ("tf_timeout", 0.1),
            ("arrival_hold_duration", 1.0),
            ("stopped_hold_duration", 0.5),
            ("stop_linear_threshold", 0.08),
            ("stop_angular_threshold", 0.12),
            ("cancel_timeout", 10.0),
            ("stop_timeout", 10.0),
            ("maddpg_ready_timeout", 30.0),
            ("maddpg_enable_timeout", 5.0),
            ("handoff_update_rate", 10.0),
        ):
            self.declare_parameter(name, default)
        self.declare_parameter(
            "target_odom_topic", "/go2_1/target_estimated/odom"
        )
        self.declare_parameter("global_frame", "merged_map")
        self.declare_parameter(
            "maddpg_ready_topic", "/gazebo_leader_slot_controller/ready"
        )
        self.declare_parameter(
            "maddpg_active_topic", "/gazebo_leader_slot_controller/active"
        )
        self.declare_parameter(
            "maddpg_enable_topic", "/dynamic_encircle/maddpg_enable"
        )
        self.declare_parameter(
            "cmd_mux_select_topic", "/dynamic_encircle/use_maddpg"
        )
        self.declare_parameter(
            "handoff_state_topic", "/dynamic_encircle/handoff_state"
        )

        numeric_names = (
            "formation_radius",
            "success_tolerance",
            "control_rate",
            "encircle_update_rate",
            "nav_goal_update_rate",
            "target_timeout",
            "target_hold",
            "odom_timeout",
            "max_linear",
            "max_angular",
            "max_coast_speed",
            "position_deadband",
            "k_linear",
            "k_angular",
            "turn_in_place_thresh",
            "accel_lin",
            "accel_ang",
            "catch_lookahead",
            "catch_speed",
            "catch_radius",
            "tf_timeout",
            "arrival_hold_duration",
            "stopped_hold_duration",
            "stop_linear_threshold",
            "stop_angular_threshold",
            "cancel_timeout",
            "stop_timeout",
            "maddpg_ready_timeout",
            "maddpg_enable_timeout",
            "handoff_update_rate",
        )
        for name in numeric_names:
            setattr(self, name, float(self.get_parameter(name).value))
        self.target_odom_topic = self.get_parameter("target_odom_topic").value
        self.global_frame = self.get_parameter("global_frame").value
        self.maddpg_ready_topic = self.get_parameter("maddpg_ready_topic").value
        self.maddpg_active_topic = self.get_parameter("maddpg_active_topic").value
        self.maddpg_enable_topic = self.get_parameter("maddpg_enable_topic").value
        self.cmd_mux_select_topic = self.get_parameter("cmd_mux_select_topic").value
        self.handoff_state_topic = self.get_parameter("handoff_state_topic").value
        self._validate_parameters()

        self.control_dt = 1.0 / self.control_rate
        self.loop = Loop(LOOP_CORNERS)
        self.go2_1_phase = "catch_up"

        self.target_received = False
        self.target_frame = "go2_1/odom"
        self.last_good_x = 0.0
        self.last_good_y = 0.0
        self.last_good_vx = 0.0
        self.last_good_vy = 0.0
        self.last_good_time = None
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_vx = 0.0
        self.target_vy = 0.0
        self.target_ok = False
        self._target_lost_logged = False

        self.dogs = {
            name: DogState(self, name, publish_commands=(name == "go2_1"))
            for name in DOG_NAMES
        }
        self.target_subscription = self.create_subscription(
            Odometry, self.target_odom_topic, self._target_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.action_clients = {
            name: ActionClient(self, NavigateToPose, f"/{name}/navigate_to_pose")
            for name in NAV_DOG_NAMES
        }
        self.active_goal_handles = {name: None for name in NAV_DOG_NAMES}
        self.pending_goal_sends = {name: False for name in NAV_DOG_NAMES}
        self.slot_indices = None
        self.latest_slots = None
        self.goal_state = GoalUpdateState(1.0 / self.nav_goal_update_rate)

        self.handoff_state = NAV2_ACTIVE
        self.state_since = self._clock_seconds()
        self.arrival_since = None
        self.stopped_since = None
        self.cancel_done = {name: False for name in NAV_DOG_NAMES}
        self.cancel_inflight = set()
        self.maddpg_ready = False
        self.maddpg_active = False
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.maddpg_enable_pub = self.create_publisher(
            Bool, self.maddpg_enable_topic, status_qos
        )
        self.cmd_mux_select_pub = self.create_publisher(
            Bool, self.cmd_mux_select_topic, status_qos
        )
        self.handoff_state_pub = self.create_publisher(
            String, self.handoff_state_topic, status_qos
        )
        self.create_subscription(
            Bool, self.maddpg_ready_topic, self._maddpg_ready_callback, status_qos
        )
        self.create_subscription(
            Bool, self.maddpg_active_topic, self._maddpg_active_callback, status_qos
        )
        self.maddpg_enable_pub.publish(Bool(data=False))
        self.cmd_mux_select_pub.publish(Bool(data=False))
        self._publish_handoff_state()

        self.control_timer = self.create_timer(self.control_dt, self._control_go2_1)
        self.encircle_timer = self.create_timer(
            1.0 / self.encircle_update_rate, self._update_encircle_geometry
        )
        self.nav_goal_timer = self.create_timer(
            1.0 / self.nav_goal_update_rate, self._nav_goal_timer_callback
        )
        self.handoff_timer = self.create_timer(
            1.0 / self.handoff_update_rate, self._handoff_timer_callback
        )

        self.get_logger().info(
            "nav2_dynamic_encircle started: target=%s geometry=%.2f Hz "
            "Nav2 goals=%.2f Hz radius=%.2f m tolerance=%.2f m"
            % (
                self.target_odom_topic,
                self.encircle_update_rate,
                self.nav_goal_update_rate,
                self.formation_radius,
                self.success_tolerance,
            )
        )

    def _validate_parameters(self):
        positive = (
            "formation_radius",
            "success_tolerance",
            "control_rate",
            "encircle_update_rate",
            "nav_goal_update_rate",
            "target_timeout",
            "target_hold",
            "odom_timeout",
            "max_linear",
            "max_angular",
            "max_coast_speed",
            "k_linear",
            "k_angular",
            "turn_in_place_thresh",
            "accel_lin",
            "accel_ang",
            "catch_lookahead",
            "catch_speed",
            "catch_radius",
            "tf_timeout",
            "arrival_hold_duration",
            "stopped_hold_duration",
            "stop_linear_threshold",
            "stop_angular_threshold",
            "cancel_timeout",
            "stop_timeout",
            "maddpg_ready_timeout",
            "maddpg_enable_timeout",
            "handoff_update_rate",
        )
        for name in positive:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if not math.isfinite(self.position_deadband) or self.position_deadband < 0.0:
            raise ValueError("position_deadband must be finite and non-negative")
        if self.target_hold < self.target_timeout:
            raise ValueError("target_hold must be greater than or equal to target_timeout")
        if not isinstance(self.target_odom_topic, str) or not self.target_odom_topic:
            raise ValueError("target_odom_topic must be a non-empty string")
        if not isinstance(self.global_frame, str) or not self.global_frame:
            raise ValueError("global_frame must be a non-empty string")

    def _publish_handoff_state(self):
        self.handoff_state_pub.publish(String(data=self.handoff_state))

    def _set_handoff_state(self, state, message):
        if state == self.handoff_state:
            return
        self.handoff_state = state
        self.state_since = self._clock_seconds()
        self._publish_handoff_state()
        self.get_logger().info(f"Handoff state -> {state}: {message}")

    def _maddpg_ready_callback(self, message):
        self.maddpg_ready = bool(message.data)

    def _maddpg_active_callback(self, message):
        self.maddpg_active = bool(message.data)

    def _target_callback(self, message):
        self.last_good_x = message.pose.pose.position.x
        self.last_good_y = message.pose.pose.position.y
        self.last_good_vx = message.twist.twist.linear.x
        self.last_good_vy = message.twist.twist.linear.y
        self.target_frame = message.header.frame_id or "go2_1/odom"
        self.last_good_time = self.get_clock().now()
        self.target_received = True

    def _message_age(self, stamp):
        if stamp is None:
            return float("inf")
        return (
            self.get_clock().now() - Time.from_msg(stamp)
        ).nanoseconds * 1e-9

    def _resolve_target(self):
        if not self.target_received or self.last_good_time is None:
            self.target_ok = False
            return False

        age = (self.get_clock().now() - self.last_good_time).nanoseconds * 1e-9
        if age <= self.target_timeout:
            self.target_x = self.last_good_x
            self.target_y = self.last_good_y
            self.target_vx = self.last_good_vx
            self.target_vy = self.last_good_vy
            self.target_ok = True
            return True

        if age <= self.target_hold:
            velocity_x = self.last_good_vx
            velocity_y = self.last_good_vy
            speed = math.hypot(velocity_x, velocity_y)
            if speed > self.max_coast_speed and speed > 1e-6:
                scale = self.max_coast_speed / speed
                velocity_x *= scale
                velocity_y *= scale
            self.target_x = self.last_good_x + velocity_x * age
            self.target_y = self.last_good_y + velocity_y * age
            self.target_vx = velocity_x
            self.target_vy = velocity_y
            self.target_ok = True
            return False

        self.target_ok = False
        return False

    def _emit_go2_1(self, linear, angular):
        dog = self.dogs["go2_1"]
        linear = clamp(
            linear,
            dog.previous_linear - self.accel_lin * self.control_dt,
            dog.previous_linear + self.accel_lin * self.control_dt,
        )
        angular = clamp(
            angular,
            dog.previous_angular - self.accel_ang * self.control_dt,
            dog.previous_angular + self.accel_ang * self.control_dt,
        )
        dog.previous_linear = linear
        dog.previous_angular = angular
        command = Twist()
        command.linear.x = linear
        command.angular.z = angular
        dog.command_publisher.publish(command)

    def _control_go2_1(self):
        self._resolve_target()
        dog = self.dogs["go2_1"]
        if not dog.received or self._message_age(dog.last_stamp) > self.odom_timeout:
            dog.publish_zero()
            return
        if not self.target_ok:
            dog.publish_zero()
            if not self._target_lost_logged:
                self.get_logger().warning(
                    "Target estimate exceeded %.1f s hold; stopping go2_1 and Nav2 dogs"
                    % self.target_hold
                )
                self._target_lost_logged = True
            self._suspend_navigation("target estimate lost")
            return

        self._target_lost_logged = False
        distance_to_target = math.hypot(self.target_x - dog.x, self.target_y - dog.y)

        if self.go2_1_phase == "catch_up":
            if distance_to_target < self.catch_radius:
                self.go2_1_phase = "formation"
                self.get_logger().info("go2_1 caught target -> formation")
            else:
                dog_arc = self.loop.project(dog.x, dog.y)
                target_arc = self.loop.project(self.target_x, self.target_y)
                delta = self.loop.signed_arc(dog_arc, target_arc)
                direction = 1.0 if delta >= 0.0 else -1.0
                step = min(self.catch_lookahead, abs(delta))
                control_x, control_y = self.loop.point_at(
                    dog_arc + direction * step
                )
                yaw_error = normalize_angle(
                    math.atan2(control_y - dog.y, control_x - dog.x) - dog.yaw
                )
                linear = (
                    0.0
                    if abs(yaw_error) > self.turn_in_place_thresh
                    else self.catch_speed
                )
                angular = clamp(
                    self.k_angular * yaw_error,
                    -self.max_angular,
                    self.max_angular,
                )
                self._emit_go2_1(linear, angular)
                return

        bearing = math.atan2(self.target_y - dog.y, self.target_x - dog.x)
        yaw_error = normalize_angle(bearing - dog.yaw)
        angular = clamp(
            self.k_angular * yaw_error, -self.max_angular, self.max_angular
        )
        range_error = distance_to_target - self.formation_radius
        if abs(range_error) < self.position_deadband:
            range_error = 0.0
        feed_forward = (
            self.target_vx * math.cos(bearing)
            + self.target_vy * math.sin(bearing)
        )
        heading_gate = max(math.cos(yaw_error), 0.25)
        linear = clamp(
            (self.k_linear * range_error + feed_forward) * heading_gate,
            -0.5 * self.max_linear,
            self.max_linear,
        )
        self._emit_go2_1(linear, angular)

    def _transform_xy(self, x, y, source_frame):
        if source_frame == self.global_frame:
            return (x, y)
        point = PointStamped()
        point.header.frame_id = source_frame
        point.header.stamp = Time().to_msg()
        point.point.x = x
        point.point.y = y
        transformed = self.tf_buffer.transform(
            point,
            self.global_frame,
            timeout=Duration(seconds=self.tf_timeout),
        )
        return (transformed.point.x, transformed.point.y)

    def _global_positions(self):
        if any(
            not dog.received or self._message_age(dog.last_stamp) > self.odom_timeout
            for dog in self.dogs.values()
        ):
            return None
        try:
            target = self._transform_xy(
                self.target_x, self.target_y, self.target_frame
            )
            dogs = {
                name: self._transform_xy(dog.x, dog.y, dog.frame_id)
                for name, dog in self.dogs.items()
            }
            return target, dogs
        except Exception as error:  # tf2 exception types vary across ROS releases
            self.get_logger().warning(
                f"Cannot transform encircle geometry to {self.global_frame}: {error}",
                throttle_duration_sec=2.0,
            )
            return None

    def _update_encircle_geometry(self):
        self._resolve_target()
        if not self.target_ok:
            self._suspend_navigation("target estimate lost")
            return
        if (
            self.handoff_state in (NAV2_ACTIVE, ARRIVAL_HOLD)
            and self.goal_state.suspended
        ):
            self.goal_state.resume()
            self.get_logger().info("Target estimate recovered; Nav2 goals resumed")

        if self.handoff_state not in (NAV2_ACTIVE, ARRIVAL_HOLD):
            return

        transformed = self._global_positions()
        if transformed is None:
            return
        target, dog_positions = transformed
        start_angle = math.atan2(
            dog_positions["go2_1"][1] - target[1],
            dog_positions["go2_1"][0] - target[0],
        )
        points = solve_encircle_points(
            target[0],
            target[1],
            self.formation_radius,
            len(DOG_NAMES),
            start_angle,
        )

        if self.slot_indices is None:
            self.slot_indices = assign_remaining_slots(
                {name: dog_positions[name] for name in NAV_DOG_NAMES}, points
            )
            self.get_logger().info(
                "Fixed Nav2 slot assignment: "
                + ", ".join(
                    f"{name}->slot{index}"
                    for name, index in self.slot_indices.items()
                )
            )

        self.latest_slots = update_assigned_slots(self.slot_indices, points)
        reached = encircle_reached(
            {name: dog_positions[name] for name in NAV_DOG_NAMES},
            self.latest_slots,
            self.success_tolerance,
        )
        now = self._clock_seconds()
        if reached:
            if self.handoff_state == NAV2_ACTIVE:
                self.arrival_since = now
                self._set_handoff_state(
                    ARRIVAL_HOLD,
                    "both followers entered their Nav2 slot tolerances",
                )
        elif self.handoff_state == ARRIVAL_HOLD:
            self.arrival_since = None
            self._set_handoff_state(
                NAV2_ACTIVE,
                "arrival condition broke before the hold duration elapsed",
            )

        if self.goal_state.last_dispatch is None:
            if self._dispatch_latest_goals():
                self.nav_goal_timer.reset()

    def _clock_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _nav_goal_timer_callback(self):
        if (
            self.handoff_state not in (NAV2_ACTIVE, ARRIVAL_HOLD)
            or self.latest_slots is None
            or not self.target_ok
        ):
            return
        self._dispatch_latest_goals()

    def _goal_message(self, point):
        x, y, yaw = point
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z = math.sin(yaw * 0.5)
        pose.pose.orientation.w = math.cos(yaw * 0.5)
        goal.pose = pose
        return goal

    def _dispatch_latest_goals(self):
        now = self._clock_seconds()
        if self.latest_slots is None or not self.goal_state.due(now):
            return False
        if not all(client.server_is_ready() for client in self.action_clients.values()):
            self.get_logger().warning(
                "Waiting for both NavigateToPose action servers",
                throttle_duration_sec=2.0,
            )
            return False

        generation = self.goal_state.mark_dispatched(now)
        for name in NAV_DOG_NAMES:
            point = self.latest_slots[name]
            self.pending_goal_sends[name] = True
            future = self.action_clients[name].send_goal_async(
                self._goal_message(point)
            )
            future.add_done_callback(
                partial(self._goal_response_callback, name, generation, point)
            )
        self.get_logger().info(
            "Nav2 goal generation %d: %s"
            % (
                generation,
                "; ".join(
                    "%s=(%.2f, %.2f)" % (
                        name,
                        self.latest_slots[name][0],
                        self.latest_slots[name][1],
                    )
                    for name in NAV_DOG_NAMES
                ),
            )
        )
        return True

    def _goal_response_callback(self, name, generation, point, future):
        self.pending_goal_sends[name] = False
        try:
            goal_handle = future.result()
        except Exception as error:
            if self.handoff_state == NAV2_CANCELLING:
                self.cancel_done[name] = True
            if self.goal_state.is_current(generation):
                self.get_logger().error(f"{name} goal send failed: {error}")
            return

        if not self.goal_state.is_current(generation):
            if goal_handle.accepted:
                if self.handoff_state == NAV2_CANCELLING:
                    self.active_goal_handles[name] = goal_handle
                    self.cancel_done[name] = False
                    self._request_goal_cancel(name, goal_handle)
                else:
                    goal_handle.cancel_goal_async()
            elif self.handoff_state == NAV2_CANCELLING:
                self.cancel_done[name] = True
            return
        if not goal_handle.accepted:
            self.get_logger().warning(
                "%s rejected goal generation %d at (%.2f, %.2f)"
                % (name, generation, point[0], point[1])
            )
            return

        self.active_goal_handles[name] = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._goal_result_callback, name, generation)
        )

    def _goal_result_callback(self, name, generation, future):
        if self.handoff_state == NAV2_CANCELLING:
            self.active_goal_handles[name] = None
        if not self.goal_state.is_current(generation):
            return
        self.active_goal_handles[name] = None
        try:
            status = future.result().status
        except Exception as error:
            self.get_logger().error(f"{name} Nav2 result failed: {error}")
            return
        label = (
            "SUCCEEDED"
            if status == GoalStatus.STATUS_SUCCEEDED
            else f"status={status}"
        )
        self.get_logger().info(
            f"{name} Nav2 generation {generation} finished: {label}; "
            "overall success still uses simultaneous slot distance"
        )

    def _request_goal_cancel(self, name, goal_handle):
        if name in self.cancel_inflight:
            return
        self.cancel_inflight.add(name)
        self.get_logger().warning(f"Cancelling {name} Nav2 goal for handoff")
        future = goal_handle.cancel_goal_async()
        future.add_done_callback(partial(self._cancel_result_callback, name))

    def _cancel_result_callback(self, name, future):
        self.cancel_inflight.discard(name)
        try:
            response = future.result()
            count = len(response.goals_canceling)
            self.get_logger().info(
                f"{name} Nav2 cancel response received: goals_canceling={count}"
            )
        except Exception as error:
            self.get_logger().error(f"{name} Nav2 cancel request failed: {error}")
            self._fail_handoff(f"{name} cancellation failed")
            return
        self.cancel_done[name] = True
        self.active_goal_handles[name] = None

    def _cancel_active_goals(self, reason, wait_for_results=False):
        for name, goal_handle in self.active_goal_handles.items():
            if goal_handle is not None and goal_handle.accepted:
                self.get_logger().warning(f"Cancelling {name} Nav2 goal: {reason}")
                if wait_for_results:
                    self._request_goal_cancel(name, goal_handle)
                else:
                    goal_handle.cancel_goal_async()
                    self.active_goal_handles[name] = None
            elif wait_for_results and not self.pending_goal_sends[name]:
                self.cancel_done[name] = True
            else:
                self.active_goal_handles[name] = None

    def _suspend_navigation(self, reason):
        if self.handoff_state not in (NAV2_ACTIVE, ARRIVAL_HOLD):
            return
        if self.goal_state.suspend():
            self._cancel_active_goals(reason)

    def _begin_handoff(self):
        if self.handoff_state != ARRIVAL_HOLD:
            return
        self.goal_state.suspend()
        self.cancel_done = {name: False for name in NAV_DOG_NAMES}
        self.cancel_inflight.clear()
        self._set_handoff_state(
            NAV2_CANCELLING,
            "arrival stayed valid for %.2f s; Nav2 goal updates frozen"
            % self.arrival_hold_duration,
        )
        self._cancel_active_goals("MADDPG handoff", wait_for_results=True)

    def _followers_stopped(self):
        for name in NAV_DOG_NAMES:
            dog = self.dogs[name]
            if (
                not dog.received
                or self._message_age(dog.last_stamp) > self.odom_timeout
                or dog.linear_speed > self.stop_linear_threshold
                or dog.angular_speed > self.stop_angular_threshold
            ):
                return False
        return True

    def _current_slots_reached(self):
        if self.latest_slots is None:
            return False
        transformed = self._global_positions()
        if transformed is None:
            return False
        _, dog_positions = transformed
        return encircle_reached(
            {name: dog_positions[name] for name in NAV_DOG_NAMES},
            self.latest_slots,
            self.success_tolerance,
        )

    def _fail_handoff(self, reason):
        if self.handoff_state == HANDOFF_FAILED:
            return
        self.maddpg_enable_pub.publish(Bool(data=False))
        self.cmd_mux_select_pub.publish(Bool(data=False))
        self._set_handoff_state(HANDOFF_FAILED, reason)
        self.get_logger().error(
            f"MADDPG handoff failed safely; mux remains on Nav2: {reason}"
        )

    def _handoff_timer_callback(self):
        now = self._clock_seconds()
        elapsed = now - self.state_since

        if self.handoff_state == ARRIVAL_HOLD:
            if not self._current_slots_reached():
                self.arrival_since = None
                self._set_handoff_state(
                    NAV2_ACTIVE,
                    "arrival condition broke before the hold duration elapsed",
                )
            elif (
                self.arrival_since is not None
                and now - self.arrival_since >= self.arrival_hold_duration
            ):
                self._begin_handoff()

        elif self.handoff_state == NAV2_CANCELLING:
            if all(self.cancel_done.values()):
                self.stopped_since = None
                self._set_handoff_state(
                    WAITING_FOR_STOP,
                    "cancel responses settled for go2_2 and go2_3",
                )
            elif elapsed > self.cancel_timeout:
                self._fail_handoff("timed out waiting for Nav2 cancel responses")

        elif self.handoff_state == WAITING_FOR_STOP:
            if self._followers_stopped():
                if self.stopped_since is None:
                    self.stopped_since = now
                elif now - self.stopped_since >= self.stopped_hold_duration:
                    self._set_handoff_state(
                        WAITING_FOR_MADDPG_READY,
                        "both followers are measurably stopped",
                    )
            else:
                self.stopped_since = None
            if elapsed > self.stop_timeout:
                self._fail_handoff("timed out waiting for followers to stop")

        elif self.handoff_state == WAITING_FOR_MADDPG_READY:
            if self.maddpg_ready:
                self.maddpg_enable_pub.publish(Bool(data=True))
                self._set_handoff_state(
                    ENABLING_MADDPG,
                    "MADDPG ready received; enable request published",
                )
            elif elapsed > self.maddpg_ready_timeout:
                self._fail_handoff("timed out waiting for MADDPG ready")

        elif self.handoff_state == ENABLING_MADDPG:
            if self.maddpg_active:
                # MADDPG has accepted the enable request and is publishing only
                # to its private mux inputs.  Ownership changes atomically here.
                self.cmd_mux_select_pub.publish(Bool(data=True))
                self.goal_state.complete()
                self._set_handoff_state(
                    MADDPG_ACTIVE,
                    "MADDPG active acknowledged; follower mux switched",
                )
            elif elapsed > self.maddpg_enable_timeout:
                self._fail_handoff("timed out waiting for MADDPG active")

        elif self.handoff_state == MADDPG_ACTIVE and not self.maddpg_active:
            self._fail_handoff(
                "MADDPG active status dropped; follower mux returned to Nav2/zero"
            )

    def stop(self):
        self.dogs["go2_1"].publish_zero()
        self.maddpg_enable_pub.publish(Bool(data=False))
        self.cmd_mux_select_pub.publish(Bool(data=False))
        if not self.goal_state.completed:
            self.goal_state.complete()
        self._cancel_active_goals("node shutdown")


def main(args=None):
    # Keep the context valid through KeyboardInterrupt so stop() can publish
    # go2_1's final zero command and request cancellation of active Nav2 goals.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = None
    try:
        node = DynamicEncircle()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
