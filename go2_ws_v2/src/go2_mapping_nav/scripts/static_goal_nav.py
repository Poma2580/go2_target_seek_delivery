#!/usr/bin/env python3
"""Navigate to a static goal through map-bounded rolling subgoals."""

import math
import sys
import time
from dataclasses import dataclass

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


@dataclass(frozen=True)
class Subgoal:
    x: float
    y: float
    yaw: float
    state: str
    forward_distance: float
    lateral_offset: float
    is_final: bool


class StaticGoalNavigator(Node):
    """Choose safe map-local subgoals until the requested goal is reached."""

    _LATERAL_OFFSETS = (0.0, 0.5, -0.5, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0)

    def __init__(self):
        super().__init__("static_goal_nav")

        numeric_parameter = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter("action_name", "/go2_1/navigate_to_pose")
        self.declare_parameter("map_topic", "/go2_1/map")
        self.declare_parameter("global_frame", "go2_1/map")
        self.declare_parameter("robot_frame", "go2_1/base_link")
        for name, default in (
            ("goal_x", 0.0),
            ("goal_y", 0.0),
            ("goal_yaw", 0.0),
            ("segment_length", 5.0),
            ("min_segment_length", 0.5),
            ("segment_length_step", 0.25),
            ("map_boundary_margin", 1.0),
            ("subgoal_clearance_radius", 0.45),
            ("free_threshold", 30),
            ("occupied_threshold", 65),
            ("map_timeout_sec", 15.0),
            ("tf_timeout_sec", 3.0),
            ("server_timeout_sec", 30.0),
            ("navigation_timeout_sec", 180.0),
            ("map_update_wait_sec", 2.0),
            ("final_goal_tolerance", 0.8),
            ("minimum_progress", 0.3),
            ("max_segments", 30),
            ("max_segment_failures", 4),
            ("max_no_progress_segments", 3),
        ):
            self.declare_parameter(name, default, numeric_parameter)
        self.declare_parameter("allow_unknown_subgoal", True)

        self.action_name = self._string_parameter("action_name")
        self.map_topic = self._string_parameter("map_topic")
        self.global_frame = self._string_parameter("global_frame")
        self.robot_frame = self._string_parameter("robot_frame")
        for name in (
            "goal_x",
            "goal_y",
            "goal_yaw",
            "segment_length",
            "min_segment_length",
            "segment_length_step",
            "map_boundary_margin",
            "subgoal_clearance_radius",
            "free_threshold",
            "occupied_threshold",
            "map_timeout_sec",
            "tf_timeout_sec",
            "server_timeout_sec",
            "navigation_timeout_sec",
            "map_update_wait_sec",
            "final_goal_tolerance",
            "minimum_progress",
        ):
            setattr(self, name, float(self.get_parameter(name).value))
        for name in (
            "max_segments",
            "max_segment_failures",
            "max_no_progress_segments",
        ):
            setattr(self, name, self._integer_parameter(name))
        self.allow_unknown_subgoal = self.get_parameter("allow_unknown_subgoal").value
        self._validate_parameters()

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.latest_map = None
        self.map_sequence = 0
        self.create_subscription(OccupancyGrid, self.map_topic, self._map_callback, map_qos)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.client = ActionClient(self, NavigateToPose, self.action_name)

        self.goal_handle = None
        self.goal_response_received = False
        self.result_status = None
        self.result_received = False
        self.callback_error = None
        self.last_feedback_log = 0.0
        self.sent_goal_keys = set()

    def _string_parameter(self, name):
        value = self.get_parameter(name).value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _integer_parameter(self, name):
        value = self.get_parameter(name).value
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        return int(value)

    def _validate_parameters(self):
        numeric_values = {
            name: getattr(self, name)
            for name in (
                "goal_x",
                "goal_y",
                "goal_yaw",
                "segment_length",
                "min_segment_length",
                "segment_length_step",
                "map_boundary_margin",
                "subgoal_clearance_radius",
                "free_threshold",
                "occupied_threshold",
                "map_timeout_sec",
                "tf_timeout_sec",
                "server_timeout_sec",
                "navigation_timeout_sec",
                "map_update_wait_sec",
                "final_goal_tolerance",
                "minimum_progress",
            )
        }
        for name, value in numeric_values.items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.min_segment_length <= 0.0 or self.segment_length < self.min_segment_length:
            raise ValueError("segment_length must be at least min_segment_length > 0")
        if self.segment_length_step <= 0.0:
            raise ValueError("segment_length_step must be greater than zero")
        if self.map_boundary_margin < 0.0 or self.subgoal_clearance_radius < 0.0:
            raise ValueError("map margins and clearance radius must be non-negative")
        if not 0.0 < self.free_threshold < self.occupied_threshold <= 100.0:
            raise ValueError("thresholds must satisfy 0 < free < occupied <= 100")
        for name in (
            "map_timeout_sec",
            "tf_timeout_sec",
            "server_timeout_sec",
            "navigation_timeout_sec",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")
        if self.map_update_wait_sec < 0.0:
            raise ValueError("map_update_wait_sec must be non-negative")
        if self.final_goal_tolerance < 0.0 or self.minimum_progress < 0.0:
            raise ValueError("goal tolerance and minimum progress must be non-negative")
        for name in (
            "max_segments",
            "max_segment_failures",
            "max_no_progress_segments",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if not isinstance(self.allow_unknown_subgoal, bool):
            raise ValueError("allow_unknown_subgoal must be a boolean")

    def _map_callback(self, message):
        self.latest_map = message
        self.map_sequence += 1

    def _map_is_valid(self, grid=None):
        grid = self.latest_map if grid is None else grid
        if grid is None:
            return False
        info = grid.info
        return (
            info.width > 0
            and info.height > 0
            and math.isfinite(info.resolution)
            and info.resolution > 0.0
            and len(grid.data) >= info.width * info.height
        )

    @staticmethod
    def _yaw_from_quaternion(quaternion):
        return math.atan2(
            2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
            1.0 - 2.0 * (quaternion.y * quaternion.y + quaternion.z * quaternion.z),
        )

    def _world_to_local(self, wx, wy, grid=None):
        grid = self.latest_map if grid is None else grid
        if not self._map_is_valid(grid):
            return None
        origin = grid.info.origin
        yaw = self._yaw_from_quaternion(origin.orientation)
        dx = wx - origin.position.x
        dy = wy - origin.position.y
        return (
            math.cos(yaw) * dx + math.sin(yaw) * dy,
            -math.sin(yaw) * dx + math.cos(yaw) * dy,
        )

    def _local_to_world(self, lx, ly, grid=None):
        grid = self.latest_map if grid is None else grid
        origin = grid.info.origin
        yaw = self._yaw_from_quaternion(origin.orientation)
        return (
            origin.position.x + math.cos(yaw) * lx - math.sin(yaw) * ly,
            origin.position.y + math.sin(yaw) * lx + math.cos(yaw) * ly,
        )

    def world_to_grid(self, wx, wy, grid=None):
        """Return (column, row), or None when the point lies outside the map."""
        grid = self.latest_map if grid is None else grid
        local = self._world_to_local(wx, wy, grid)
        if local is None:
            return None
        local_x, local_y = local
        resolution = grid.info.resolution
        if (
            local_x < 0.0
            or local_y < 0.0
            or local_x >= grid.info.width * resolution
            or local_y >= grid.info.height * resolution
        ):
            return None
        return (int(math.floor(local_x / resolution)), int(math.floor(local_y / resolution)))

    def _cell_state(self, column, row, grid=None):
        grid = self.latest_map if grid is None else grid
        if (
            not self._map_is_valid(grid)
            or column < 0
            or row < 0
            or column >= grid.info.width
            or row >= grid.info.height
        ):
            return "outside"
        value = grid.data[row * grid.info.width + column]
        if value == -1:
            return "unknown"
        if 0 <= value < self.free_threshold:
            return "free"
        if value >= self.occupied_threshold:
            return "occupied"
        return "intermediate"

    def _has_boundary_margin(self, wx, wy):
        local = self._world_to_local(wx, wy)
        if local is None:
            return False
        local_x, local_y = local
        info = self.latest_map.info
        return min(
            local_x,
            local_y,
            info.width * info.resolution - local_x,
            info.height * info.resolution - local_y,
        ) >= self.map_boundary_margin

    def _clear_of_occupied(self, wx, wy):
        cell = self.world_to_grid(wx, wy)
        if cell is None:
            return False
        column, row = cell
        info = self.latest_map.info
        radius_cells = int(math.ceil(self.subgoal_clearance_radius / info.resolution))
        local = self._world_to_local(wx, wy)
        for check_row in range(max(0, row - radius_cells), min(info.height, row + radius_cells + 1)):
            for check_column in range(
                max(0, column - radius_cells), min(info.width, column + radius_cells + 1)
            ):
                cell_x = (check_column + 0.5) * info.resolution
                cell_y = (check_row + 0.5) * info.resolution
                if math.hypot(cell_x - local[0], cell_y - local[1]) <= self.subgoal_clearance_radius:
                    if self._cell_state(check_column, check_row) == "occupied":
                        return False
        return True

    def _map_bounds(self, grid=None):
        grid = self.latest_map if grid is None else grid
        if not self._map_is_valid(grid):
            return None
        width = grid.info.width * grid.info.resolution
        height = grid.info.height * grid.info.resolution
        corners = (
            self._local_to_world(0.0, 0.0, grid),
            self._local_to_world(width, 0.0, grid),
            self._local_to_world(width, height, grid),
            self._local_to_world(0.0, height, grid),
        )
        return (
            min(point[0] for point in corners),
            max(point[0] for point in corners),
            min(point[1] for point in corners),
            max(point[1] for point in corners),
        )

    @staticmethod
    def _bounds_expanded(before, after):
        if before is None or after is None:
            return False
        epsilon = 1e-6
        return (
            after[0] < before[0] - epsilon
            or after[1] > before[1] + epsilon
            or after[2] < before[2] - epsilon
            or after[3] > before[3] + epsilon
        )

    @staticmethod
    def _format_bounds(bounds):
        if bounds is None:
            return "unavailable"
        return "x=[%.2f, %.2f], y=[%.2f, %.2f]" % bounds

    @staticmethod
    def _goal_key(x, y):
        return (round(x, 3), round(y, 3))

    def _candidate(self, x, y, position, forward_distance, lateral_offset, is_final):
        key = self._goal_key(x, y)
        if key in self.sent_goal_keys:
            return None, "repeated"
        cell = self.world_to_grid(x, y)
        if cell is None:
            return None, "outside"
        if not self._has_boundary_margin(x, y):
            return None, "boundary"
        state = self._cell_state(*cell)
        if state in ("occupied", "intermediate"):
            return None, state
        if state == "unknown" and not self.allow_unknown_subgoal:
            return None, "unknown"
        if not self._clear_of_occupied(x, y):
            return None, "clearance"
        current_distance = math.hypot(self.goal_x - position[0], self.goal_y - position[1])
        candidate_distance = math.hypot(self.goal_x - x, self.goal_y - y)
        if not is_final and candidate_distance >= current_distance - 1e-6:
            return None, "no_progress"
        yaw = self.goal_yaw if is_final else math.atan2(self.goal_y - y, self.goal_x - x)
        return Subgoal(x, y, yaw, state, forward_distance, lateral_offset, is_final), None

    def _select_subgoal(self, position, distance_cap):
        final, reason = self._candidate(
            self.goal_x,
            self.goal_y,
            position,
            math.hypot(self.goal_x - position[0], self.goal_y - position[1]),
            0.0,
            True,
        )
        if final is not None:
            return final, {}

        rejected = {reason: 1}
        current_distance = math.hypot(self.goal_x - position[0], self.goal_y - position[1])
        direction_x = (self.goal_x - position[0]) / current_distance
        direction_y = (self.goal_y - position[1]) / current_distance
        free_candidates = []
        unknown_candidates = []
        distance = min(distance_cap, current_distance)
        while distance >= self.min_segment_length - 1e-9:
            for offset_index, lateral in enumerate(self._LATERAL_OFFSETS):
                x = position[0] + direction_x * distance - direction_y * lateral
                y = position[1] + direction_y * distance + direction_x * lateral
                candidate, reason = self._candidate(x, y, position, distance, lateral, False)
                if candidate is None:
                    rejected[reason] = rejected.get(reason, 0) + 1
                    continue
                ranking = (-distance, abs(lateral), offset_index, candidate)
                if candidate.state == "free":
                    free_candidates.append(ranking)
                else:
                    unknown_candidates.append(ranking)
            distance -= self.segment_length_step
        if free_candidates:
            return min(free_candidates)[3], rejected
        if self.allow_unknown_subgoal and unknown_candidates:
            return min(unknown_candidates)[3], rejected
        return None, rejected

    def _goal_message(self, subgoal):
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = subgoal.x
        pose.pose.position.y = subgoal.y
        pose.pose.orientation.z = math.sin(subgoal.yaw * 0.5)
        pose.pose.orientation.w = math.cos(subgoal.yaw * 0.5)
        goal.pose = pose
        return goal

    def _feedback_callback(self, feedback_message):
        now = time.monotonic()
        if now - self.last_feedback_log < 1.0:
            return
        self.last_feedback_log = now
        feedback = feedback_message.feedback
        eta = feedback.estimated_time_remaining
        eta_seconds = eta.sec + eta.nanosec / 1_000_000_000.0
        self.get_logger().info(
            "Navigation feedback: remaining=%.2f m, eta=%.1f s, recoveries=%d"
            % (feedback.distance_remaining, eta_seconds, feedback.number_of_recoveries)
        )

    def _goal_response_callback(self, future):
        try:
            self.goal_handle = future.result()
        except Exception as error:
            self.callback_error = error
        finally:
            self.goal_response_received = True
        if self.goal_handle is not None and self.goal_handle.accepted:
            self.get_logger().info("NavigateToPose goal accepted")
            self.goal_handle.get_result_async().add_done_callback(self._result_callback)

    def _result_callback(self, future):
        try:
            self.result_status = future.result().status
        except Exception as error:
            self.callback_error = error
        finally:
            self.result_received = True

    def _cancel_goal(self, reason):
        if self.goal_handle is None or not self.goal_handle.accepted:
            return
        self.get_logger().warning(f"Cancelling NavigateToPose goal: {reason}")
        cancel_future = self.goal_handle.cancel_goal_async()
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and not cancel_future.done() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)

    def _wait_for_server(self):
        deadline = time.monotonic() + self.server_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self.client.server_is_ready():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _wait_for_map(self):
        deadline = time.monotonic() + self.map_timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            if self._map_is_valid():
                return True
            rclpy.spin_once(self, timeout_sec=0.1)
        return False

    def _wait_for_map_update(self, previous_sequence):
        deadline = time.monotonic() + self.map_update_wait_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, deadline - time.monotonic()))
        return self.map_sequence > previous_sequence

    def _lookup_robot_position(self):
        deadline = time.monotonic() + self.tf_timeout_sec
        last_error = None
        while rclpy.ok() and time.monotonic() < deadline:
            try:
                if self.tf_buffer.can_transform(self.global_frame, self.robot_frame, Time()):
                    transform = self.tf_buffer.lookup_transform(
                        self.global_frame, self.robot_frame, Time()
                    )
                    translation = transform.transform.translation
                    return (translation.x, translation.y)
            except TransformException as error:
                last_error = error
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().error(
            "TF lookup %s -> %s timed out after %.1f s%s"
            % (
                self.global_frame,
                self.robot_frame,
                self.tf_timeout_sec,
                f": {last_error}" if last_error else "",
            )
        )
        return None

    def _navigate(self, subgoal):
        self.goal_handle = None
        self.goal_response_received = False
        self.result_status = None
        self.result_received = False
        self.callback_error = None
        self.last_feedback_log = 0.0
        try:
            send_future = self.client.send_goal_async(
                self._goal_message(subgoal), feedback_callback=self._feedback_callback
            )
            send_future.add_done_callback(self._goal_response_callback)
        except Exception as error:
            return False, f"send exception: {error}"

        while rclpy.ok() and not self.goal_response_received:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.callback_error is not None:
            return False, f"send failed: {self.callback_error}"
        if self.goal_handle is None or not self.goal_handle.accepted:
            return False, "goal rejected"

        deadline = time.monotonic() + self.navigation_timeout_sec
        while rclpy.ok() and not self.result_received:
            if time.monotonic() >= deadline:
                self._cancel_goal("single-segment navigation timeout")
                return False, "segment timeout"
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.callback_error is not None:
            return False, f"result failed: {self.callback_error}"
        if self.result_status == GoalStatus.STATUS_SUCCEEDED:
            return True, "succeeded"
        return False, f"finished with status {self.result_status}"

    def run(self):
        self.get_logger().info(
            "Waiting for map %s (%.1f s) and action server %s (%.1f s)"
            % (self.map_topic, self.map_timeout_sec, self.action_name, self.server_timeout_sec)
        )
        if not self._wait_for_map():
            self.get_logger().error("No valid OccupancyGrid map received before timeout")
            return 2
        if not self._wait_for_server():
            self.get_logger().error("NavigateToPose action server is unavailable")
            return 2

        failures = 0
        no_progress_segments = 0
        distance_cap = self.segment_length
        for segment_number in range(1, self.max_segments + 1):
            position = self._lookup_robot_position()
            if position is None:
                return 1
            remaining_distance = math.hypot(
                self.goal_x - position[0], self.goal_y - position[1]
            )
            if remaining_distance <= self.final_goal_tolerance:
                self.get_logger().info(
                    "Final goal reached within tolerance: remaining=%.3f m" % remaining_distance
                )
                return 0

            before_bounds = self._map_bounds()
            before_sequence = self.map_sequence
            subgoal, rejected = self._select_subgoal(position, distance_cap)
            if subgoal is None:
                self.get_logger().error(
                    "No legal subgoal: map=%s robot=(%.2f, %.2f) final=(%.2f, %.2f) rejected=%s"
                    % (
                        self._format_bounds(before_bounds),
                        position[0],
                        position[1],
                        self.goal_x,
                        self.goal_y,
                        rejected,
                    )
                )
                return 1

            self.get_logger().info(
                "Segment %d/%d: robot=(%.2f, %.2f) map=%s final=(%.2f, %.2f) "
                "remaining=%.2f subgoal=(%.2f, %.2f, %.2f) type=%s"
                % (
                    segment_number,
                    self.max_segments,
                    position[0],
                    position[1],
                    self._format_bounds(before_bounds),
                    self.goal_x,
                    self.goal_y,
                    remaining_distance,
                    subgoal.x,
                    subgoal.y,
                    subgoal.yaw,
                    "final" if subgoal.is_final else subgoal.state,
                )
            )
            self.sent_goal_keys.add(self._goal_key(subgoal.x, subgoal.y))
            succeeded, result = self._navigate(subgoal)
            self.get_logger().info("Segment %d result: %s" % (segment_number, result))
            if not succeeded:
                failures += 1
                distance_cap = max(
                    self.min_segment_length,
                    min(distance_cap, subgoal.forward_distance) - self.segment_length_step,
                )
                self.get_logger().warning(
                    "Segment %d failed (%d/%d); next distance cap=%.2f m"
                    % (
                        segment_number,
                        failures,
                        self.max_segment_failures,
                        distance_cap,
                    )
                )
                if failures >= self.max_segment_failures:
                    self.get_logger().error("Maximum consecutive segment failures reached")
                    return 1
                continue

            after_position = self._lookup_robot_position()
            if after_position is None:
                return 1
            after_remaining = math.hypot(
                self.goal_x - after_position[0], self.goal_y - after_position[1]
            )
            progress = remaining_distance - after_remaining
            if progress < self.minimum_progress:
                no_progress_segments += 1
            else:
                no_progress_segments = 0
            map_updated = self._wait_for_map_update(before_sequence)
            after_bounds = self._map_bounds()
            expanded = self._bounds_expanded(before_bounds, after_bounds)
            self.get_logger().info(
                "Segment %d progress=%.2f m remaining=%.2f m map_updated=%s map_expanded=%s"
                % (segment_number, progress, after_remaining, map_updated, expanded)
            )
            if no_progress_segments >= self.max_no_progress_segments:
                self.get_logger().error("Maximum consecutive no-progress segments reached")
                return 1
            failures = 0
            distance_cap = self.segment_length

        self.get_logger().error("Maximum number of navigation segments reached")
        return 1


def main():
    rclpy.init()
    node = None
    exit_code = 2
    try:
        node = StaticGoalNavigator()
        exit_code = node.run()
    except ValueError as error:
        print(f"Parameter error: {error}", file=sys.stderr)
        exit_code = 2
    except KeyboardInterrupt:
        if node is not None:
            node._cancel_goal("interrupted by user")
        exit_code = 1
    except Exception as error:
        print(f"static_goal_nav failed: {error}", file=sys.stderr)
        exit_code = 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
