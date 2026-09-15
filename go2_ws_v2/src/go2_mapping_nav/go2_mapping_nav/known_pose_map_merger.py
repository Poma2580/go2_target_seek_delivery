#!/usr/bin/env python3
"""Merge axis-aligned occupancy grids that already share world coordinates."""

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)


_RESOLUTION_TOLERANCE = 1.0e-6
_YAW_TOLERANCE = 1.0e-6


@dataclass(frozen=True)
class GridData:
    """ROS-independent occupancy-grid representation used by the merger."""

    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    origin_yaw: float
    data: Sequence[int]


@dataclass(frozen=True)
class MergedGrid:
    resolution: float
    width: int
    height: int
    origin_x: float
    origin_y: float
    data: tuple[int, ...]


def classify_cell(value: int, free_threshold: int, occupied_threshold: int) -> int:
    """Return -1 for unknown, 0 for free, or 100 for occupied."""
    if value >= occupied_threshold:
        return 100
    if 0 <= value <= free_threshold:
        return 0
    return -1


def validate_grid(grid: GridData, expected_resolution: float | None = None) -> None:
    """Raise ValueError when a grid cannot be merged without rotation."""
    numeric_values = (
        grid.resolution,
        grid.origin_x,
        grid.origin_y,
        grid.origin_yaw,
    )
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("grid resolution and origin must be finite")
    if grid.resolution <= 0.0:
        raise ValueError("grid resolution must be greater than zero")
    if grid.width <= 0 or grid.height <= 0:
        raise ValueError("grid width and height must be greater than zero")
    expected_cells = grid.width * grid.height
    if len(grid.data) != expected_cells:
        raise ValueError(
            f"grid data length is {len(grid.data)}, expected {expected_cells}"
        )
    if abs(grid.origin_yaw) > _YAW_TOLERANCE:
        raise ValueError(
            f"grid origin yaw must be zero, got {grid.origin_yaw:.9f} rad"
        )
    if (
        expected_resolution is not None
        and abs(grid.resolution - expected_resolution) > _RESOLUTION_TOLERANCE
    ):
        raise ValueError(
            "grid resolution "
            f"{grid.resolution:.9f} differs from expected "
            f"{expected_resolution:.9f}"
        )


def merge_grids(
    grids: Iterable[GridData],
    free_threshold: int = 25,
    occupied_threshold: int = 65,
) -> MergedGrid:
    """Merge grids using the agreed three-robot voting policy."""
    if not 0 <= free_threshold < occupied_threshold <= 100:
        raise ValueError(
            "thresholds must satisfy 0 <= free < occupied <= 100"
        )

    valid_grids = tuple(grids)
    if not valid_grids:
        raise ValueError("at least one grid is required")

    resolution = valid_grids[0].resolution
    for grid in valid_grids:
        validate_grid(grid, resolution)

    minimum_x = min(grid.origin_x for grid in valid_grids)
    minimum_y = min(grid.origin_y for grid in valid_grids)
    maximum_x = max(
        grid.origin_x + grid.width * resolution for grid in valid_grids
    )
    maximum_y = max(
        grid.origin_y + grid.height * resolution for grid in valid_grids
    )

    # Anchor the merged grid to world-resolution boundaries. Mapping source-cell
    # centers with floor() then remains stable if RTAB-Map expands its origin.
    origin_x = math.floor(minimum_x / resolution) * resolution
    origin_y = math.floor(minimum_y / resolution) * resolution
    width = max(1, math.ceil((maximum_x - origin_x) / resolution - 1.0e-9))
    height = max(1, math.ceil((maximum_y - origin_y) / resolution - 1.0e-9))
    occupied_votes = np.zeros((height, width), dtype=np.uint8)
    free_votes = np.zeros((height, width), dtype=np.uint8)

    for grid in valid_grids:
        source = np.asarray(grid.data, dtype=np.int16).reshape(
            grid.height, grid.width
        )
        destination_x = math.floor(
            (grid.origin_x + 0.5 * resolution - origin_x) / resolution
        )
        destination_y = math.floor(
            (grid.origin_y + 0.5 * resolution - origin_y) / resolution
        )
        destination = (
            slice(destination_y, destination_y + grid.height),
            slice(destination_x, destination_x + grid.width),
        )
        occupied_votes[destination] += source >= occupied_threshold
        free_votes[destination] += (
            (source >= 0) & (source <= free_threshold)
        )

    result = np.full((height, width), -1, dtype=np.int8)
    # Apply single occupied before single free; a 1/1/unknown conflict is
    # therefore occupied. Majority-free and majority-occupied override it.
    result[free_votes >= 1] = 0
    result[occupied_votes >= 1] = 100
    result[free_votes >= 2] = 0
    result[occupied_votes >= 2] = 100
    merged_data = tuple(int(value) for value in result.ravel())

    return MergedGrid(
        resolution=resolution,
        width=width,
        height=height,
        origin_x=origin_x,
        origin_y=origin_y,
        data=merged_data,
    )


def _yaw_from_quaternion(quaternion) -> float:
    return math.atan2(
        2.0
        * (
            quaternion.w * quaternion.z
            + quaternion.x * quaternion.y
        ),
        1.0
        - 2.0
        * (
            quaternion.y * quaternion.y
            + quaternion.z * quaternion.z
        ),
    )


def _grid_from_message(message: OccupancyGrid) -> GridData:
    origin = message.info.origin
    return GridData(
        resolution=float(message.info.resolution),
        width=int(message.info.width),
        height=int(message.info.height),
        origin_x=float(origin.position.x),
        origin_y=float(origin.position.y),
        origin_yaw=_yaw_from_quaternion(origin.orientation),
        data=message.data,
    )


class KnownPoseMapMerger(Node):
    """Fuse the latest cumulative RTAB-Map grids in their shared world frame."""

    def __init__(self):
        super().__init__("known_pose_map_merger")
        dynamic_parameter = ParameterDescriptor(dynamic_typing=True)
        self.declare_parameter(
            "map_topics",
            ["/go2_1/map", "/go2_2/map", "/go2_3/map"],
        )
        self.declare_parameter("output_topic", "/merged_map")
        self.declare_parameter("output_frame", "merged_map")
        self.declare_parameter("publish_rate", 1.0, dynamic_parameter)
        self.declare_parameter("occupied_threshold", 65, dynamic_parameter)
        self.declare_parameter("free_threshold", 25, dynamic_parameter)
        self.declare_parameter("stale_warning_sec", 5.0, dynamic_parameter)
        # Reserved for a future local-zero odometry mode. Non-zero transforms
        # are deliberately rejected in this ground-truth-only implementation.
        self.declare_parameter(
            "initial_se2_transforms",
            [0.0, 0.0, 0.0] * 3,
        )

        self.map_topics = self._string_list_parameter("map_topics")
        self.output_topic = self._string_parameter("output_topic")
        self.output_frame = self._string_parameter("output_frame")
        self.publish_rate = self._positive_float_parameter("publish_rate")
        self.stale_warning_sec = self._positive_float_parameter(
            "stale_warning_sec"
        )
        self.free_threshold = self._integer_parameter("free_threshold")
        self.occupied_threshold = self._integer_parameter(
            "occupied_threshold"
        )
        if not 0 <= self.free_threshold < self.occupied_threshold <= 100:
            raise ValueError(
                "thresholds must satisfy 0 <= free < occupied <= 100"
            )
        transforms = self.get_parameter("initial_se2_transforms").value
        if len(transforms) != len(self.map_topics) * 3:
            raise ValueError(
                "initial_se2_transforms must contain x, y, yaw per map topic"
            )
        if any(abs(float(value)) > 1.0e-12 for value in transforms):
            raise ValueError(
                "non-zero initial_se2_transforms are reserved for a future "
                "local-zero odometry mode and are not supported yet"
            )

        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.publisher = self.create_publisher(
            OccupancyGrid, self.output_topic, map_qos
        )
        self.latest_grids: dict[str, GridData] = {}
        self.last_received_ns: dict[str, int] = {}
        self.last_stale_warning_ns: dict[str, int] = {}
        self.invalid_reasons: dict[str, str] = {}
        self._map_subscriptions = [
            self.create_subscription(
                OccupancyGrid,
                topic,
                lambda message, source=topic: self._map_callback(
                    source, message
                ),
                map_qos,
            )
            for topic in self.map_topics
        ]
        self.create_timer(1.0 / self.publish_rate, self._publish_merged_map)
        self.get_logger().info(
            "Known-pose map merger ready: "
            f"{', '.join(self.map_topics)} -> {self.output_topic} "
            f"(frame={self.output_frame}, rate={self.publish_rate:g} Hz)"
        )

    def _string_parameter(self, name: str) -> str:
        value = self.get_parameter(name).value
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value.strip()

    def _string_list_parameter(self, name: str) -> tuple[str, ...]:
        value = self.get_parameter(name).value
        if not isinstance(value, Sequence) or isinstance(value, str):
            raise ValueError(f"{name} must be a string list")
        result = tuple(str(item).strip() for item in value)
        if not result or any(not item for item in result):
            raise ValueError(f"{name} must contain non-empty topic names")
        if len(set(result)) != len(result):
            raise ValueError(f"{name} must not contain duplicates")
        return result

    def _positive_float_parameter(self, name: str) -> float:
        value = float(self.get_parameter(name).value)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and greater than zero")
        return value

    def _integer_parameter(self, name: str) -> int:
        value = self.get_parameter(name).value
        if isinstance(value, bool) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        return int(value)

    def _map_callback(self, topic: str, message: OccupancyGrid) -> None:
        try:
            grid = _grid_from_message(message)
            expected_resolution = (
                next(iter(self.latest_grids.values())).resolution
                if self.latest_grids
                else None
            )
            validate_grid(grid, expected_resolution)
        except ValueError as error:
            reason = str(error)
            if self.invalid_reasons.get(topic) != reason:
                self.get_logger().error(f"Rejecting {topic}: {reason}")
                self.invalid_reasons[topic] = reason
            return

        self.invalid_reasons.pop(topic, None)
        self.latest_grids[topic] = grid
        self.last_received_ns[topic] = self.get_clock().now().nanoseconds

    def _publish_merged_map(self) -> None:
        if not self.latest_grids:
            return
        now = self.get_clock().now()
        now_ns = now.nanoseconds
        warning_interval_ns = int(self.stale_warning_sec * 1.0e9)
        for topic in self.map_topics:
            received_ns = self.last_received_ns.get(topic)
            if received_ns is None:
                continue
            age_ns = now_ns - received_ns
            last_warning_ns = self.last_stale_warning_ns.get(topic, 0)
            if (
                age_ns > warning_interval_ns
                and now_ns - last_warning_ns >= warning_interval_ns
            ):
                self.get_logger().warning(
                    f"{topic} has not updated for {age_ns / 1.0e9:.1f} s; "
                    "retaining its last cumulative map"
                )
                self.last_stale_warning_ns[topic] = now_ns

        try:
            merged = merge_grids(
                self.latest_grids.values(),
                self.free_threshold,
                self.occupied_threshold,
            )
        except ValueError as error:
            self.get_logger().error(f"Cannot publish merged map: {error}")
            return

        message = OccupancyGrid()
        message.header.stamp = now.to_msg()
        message.header.frame_id = self.output_frame
        message.info.map_load_time = now.to_msg()
        message.info.resolution = merged.resolution
        message.info.width = merged.width
        message.info.height = merged.height
        message.info.origin.position.x = merged.origin_x
        message.info.origin.position.y = merged.origin_y
        message.info.origin.position.z = 0.0
        message.info.origin.orientation.w = 1.0
        message.data = list(merged.data)
        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = KnownPoseMapMerger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            # A launch supervisor may deliver a second SIGINT while cleanup is
            # already in progress. The ROS context is exiting either way.
            pass


if __name__ == "__main__":
    main()
