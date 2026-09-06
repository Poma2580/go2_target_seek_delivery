#!/usr/bin/env python3
"""Measure cloud heights and scan rings in the level lidar frame once."""

import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud


def _percentiles(values):
    if not len(values):
        return "no points"
    levels = (0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100)
    result = np.percentile(values, levels)
    return ", ".join(
        f"p{level}={value:.3f}" for level, value in zip(levels, result)
    )


def _dominant_bin(values, width):
    if not len(values):
        return "none"
    lower = math.floor(float(np.min(values)) / width) * width
    upper = math.ceil(float(np.max(values)) / width) * width + width
    edges = np.arange(lower, upper + width * 0.5, width)
    counts, edges = np.histogram(values, bins=edges)
    index = int(np.argmax(counts))
    return f"[{edges[index]:.3f}, {edges[index + 1]:.3f}) count={counts[index]}"


def _xyz_array(cloud):
    """Read XYZ from clouds that also contain mixed-type fields such as ring."""
    records = point_cloud2.read_points(
        cloud, field_names=["x", "y", "z"], skip_nans=True
    )
    if records.dtype.names:
        return np.column_stack(
            [records["x"], records["y"], records["z"]]
        ).astype(np.float64, copy=False)
    return np.asarray(records, dtype=np.float64).reshape(-1, 3)


class LidarGroundDiagnostic(Node):
    def __init__(self):
        super().__init__("lidar_ground_diagnostic")
        self.declare_parameter("robot_name", "go2_2")
        self.declare_parameter("target_frame", "")
        self.declare_parameter("min_height", 0.10)
        self.declare_parameter("max_height", 0.50)
        robot = str(self.get_parameter("robot_name").value)
        self.min_height = float(self.get_parameter("min_height").value)
        self.max_height = float(self.get_parameter("max_height").value)
        configured_frame = str(self.get_parameter("target_frame").value)
        self.target_frame = configured_frame or f"{robot}/velodyne"
        self.cloud_topic = f"/{robot}/velodyne_points"
        self.scan_topic = f"/{robot}/scan"
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_report = None
        self.scan_report = None
        self.done = False
        self.create_subscription(
            PointCloud2, self.cloud_topic, self._cloud_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            LaserScan, self.scan_topic, self._scan_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(
            f"Waiting for {self.cloud_topic} and {self.scan_topic}; "
            f"target_frame={self.target_frame}"
        )

    def _cloud_callback(self, message):
        if self.cloud_report is not None:
            return
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                message.header.frame_id,
                Time.from_msg(message.header.stamp),
                timeout=Duration(seconds=0.5),
            )
        except TransformException as error:
            self.get_logger().warning(f"Waiting for point-cloud TF: {error}")
            return
        transformed = do_transform_cloud(message, transform)
        points = _xyz_array(transformed)
        if not len(points):
            self.get_logger().warning("Transformed cloud contains no finite XYZ points")
            return
        radius = np.hypot(points[:, 0], points[:, 1])
        useful = points[(radius >= 0.55) & (radius <= 5.0)]
        in_slice = useful[
            (useful[:, 2] >= self.min_height)
            & (useful[:, 2] <= self.max_height)
        ]
        self.cloud_report = (
            f"cloud frame={message.header.frame_id} -> {self.target_frame}\n"
            f"TF translation=({transform.transform.translation.x:.3f}, "
            f"{transform.transform.translation.y:.3f}, "
            f"{transform.transform.translation.z:.3f})\n"
            f"all z: {_percentiles(points[:, 2])}\n"
            f"0.55..5.0 m z: {_percentiles(useful[:, 2])}\n"
            f"dominant z bin (0.02 m): {_dominant_bin(useful[:, 2], 0.02)}\n"
            f"current slice [{self.min_height:.2f}, {self.max_height:.2f}] m: "
            f"{len(in_slice)}/{len(useful)} points"
        )
        self._report_if_ready()

    def _scan_callback(self, message):
        if self.scan_report is not None:
            return
        ranges = np.asarray(message.ranges, dtype=np.float64)
        finite = ranges[
            np.isfinite(ranges)
            & (ranges >= float(message.range_min))
            & (ranges <= float(message.range_max))
        ]
        self.scan_report = (
            f"scan frame={message.header.frame_id}, rays={len(ranges)}, "
            f"finite={len(finite)}\n"
            f"finite range: {_percentiles(finite)}\n"
            f"dominant range bin (0.10 m): {_dominant_bin(finite, 0.10)}"
        )
        self._report_if_ready()

    def _report_if_ready(self):
        if self.cloud_report is None or self.scan_report is None or self.done:
            return
        self.done = True
        self.get_logger().info(
            "\n===== LIDAR GROUND DIAGNOSTIC =====\n"
            f"{self.cloud_report}\n{self.scan_report}\n"
            "==================================="
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarGroundDiagnostic()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
