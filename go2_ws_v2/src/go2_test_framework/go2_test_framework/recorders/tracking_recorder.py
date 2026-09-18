#!/usr/bin/env python3
"""Record and evaluate one continuous target-tracking Case."""

import csv
import math
import time
from functools import partial
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401
import tf2_ros

from go2_test_framework.common.config import read_yaml
from go2_test_framework.evaluators.tracking import evaluate_tracking
from go2_test_framework.ground_truth.visibility import (
    CameraIntrinsics, project_camera_point, quaternion_conjugate_rotate,
)
from go2_test_framework.recorders.cache import TimeCache
from go2_test_framework.recorders.target_recorder import stamp_seconds
from go2_test_framework.reporting.results import write_yaml


ROBOTS = ("go2_1", "go2_2", "go2_3")
CSV_FIELDS = (
    "case_id", "eval_index", "eval_time", "perception_robot",
    "target_gt_x", "target_gt_y", "robot_gt_x", "robot_gt_y",
    "distance_m", "visible", "within_tracking_radius",
    "effective_tracking", "tracking_elapsed_sec", "infrastructure_valid",
)


class TrackingRecorder(Node):
    def __init__(self):
        super().__init__("tracking_test_recorder")
        self.declare_parameter("case_config", "")
        self.declare_parameter("output_dir", "")
        case_path = Path(str(self.get_parameter("case_config").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        if not case_path.is_file():
            raise ValueError(f"case_config does not exist: {case_path}")
        self.case = read_yaml(case_path)
        settings = self.case["settings"]
        metrics = self.case["metrics"]
        self.rate = float(settings["evaluation_rate_hz"])
        self.match_timeout = float(settings["match_timeout_sec"])
        self.startup_timeout = float(settings["startup_timeout_sec"])
        self.role_timeout = float(settings["role_timeout_sec"])
        self.data_ready_timeout = float(settings["data_ready_timeout_sec"])
        self.min_depth = float(settings["min_camera_depth_m"])
        self.max_depth = float(settings["max_camera_depth_m"])
        self.radius = float(metrics["tracking_radius_m"])
        self.minimum_duration = float(metrics["tracking_min_duration_sec"])
        self.acquisition_timeout = float(metrics["acquisition_timeout_sec"])
        self.role = None
        self.camera_info = None
        self.target_gt = TimeCache()
        self.robot_gt = TimeCache()
        self.rows = []
        self.t0 = None
        self.eval_index = 0
        self.started_wall = time.monotonic()
        self.infrastructure_valid = True
        self.infrastructure_errors = []
        self._done = False
        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        role_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.role_subscription = self.create_subscription(
            String, "/target_role/perception_robot", self._role_cb, role_qos,
            callback_group=self.callback_group,
        )
        self.target_subscription = self.create_subscription(
            Odometry, "/walking_target/odom", self._target_cb,
            qos_profile_sensor_data, callback_group=self.callback_group,
        )
        self._owned_subscriptions = []
        self.timer = self.create_timer(
            1.0 / self.rate, self._evaluate_tick,
            callback_group=self.callback_group,
        )

    def _role_cb(self, message):
        selected = message.data.strip("/")
        if selected not in ROBOTS:
            return
        if self.role is None:
            self.role = selected
            self._owned_subscriptions.extend((
                self.create_subscription(
                    Odometry, f"/{selected}/odom/ground_truth", self._robot_cb,
                    qos_profile_sensor_data, callback_group=self.callback_group,
                ),
                self.create_subscription(
                    CameraInfo, f"/{selected}/camera/depth/camera_info",
                    self._camera_cb, qos_profile_sensor_data,
                    callback_group=self.callback_group,
                ),
            ))
            self.get_logger().info(f"selected perception robot: {selected}")
        elif selected != self.role:
            self.get_logger().warning(
                f"ignoring conflicting role {selected}; locked={self.role}"
            )

    def _target_cb(self, message):
        self.target_gt.append(stamp_seconds(message.header.stamp), message)

    def _robot_cb(self, message):
        self.robot_gt.append(stamp_seconds(message.header.stamp), message)

    def _camera_cb(self, message):
        self.camera_info = message

    def _inputs_at(self, eval_time):
        if self.role is None:
            return None, None, "perception role unavailable"
        target = self.target_gt.nearest(eval_time, self.match_timeout)
        robot = self.robot_gt.nearest(eval_time, self.match_timeout)
        if target is None:
            return None, None, "walking target GT unavailable"
        if robot is None:
            return None, None, f"{self.role} GT unavailable"
        if self.camera_info is None:
            return None, None, "camera_info unavailable"
        return target[1], robot[1], None

    def _visibility(self, target, robot):
        info = self.camera_info
        if info.width <= 0 or info.height <= 0:
            raise ValueError("camera_info unavailable or invalid")
        target_position = target.pose.pose.position
        robot_position = robot.pose.pose.position
        q = robot.pose.pose.orientation
        delta_world = (
            target_position.x - robot_position.x,
            target_position.y - robot_position.y,
            target_position.z - robot_position.z,
        )
        delta_base = quaternion_conjugate_rotate(
            delta_world, (q.x, q.y, q.z, q.w)
        )
        point = PointStamped()
        point.header.frame_id = f"{self.role}/base_footprint"
        point.header.stamp = robot.header.stamp
        point.point.x, point.point.y, point.point.z = delta_base
        try:
            camera_point = self.tf_buffer.transform(
                point, info.header.frame_id,
                timeout=rclpy.duration.Duration(
                    seconds=min(0.1, self.match_timeout)
                ),
            )
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"camera TF unavailable: {error}") from error
        intrinsics = CameraIntrinsics(
            info.k[0], info.k[4], info.k[2], info.k[5], info.width, info.height
        )
        return project_camera_point(
            (camera_point.point.x, camera_point.point.y, camera_point.point.z),
            intrinsics, self.min_depth, self.max_depth,
        )[0]

    def _evaluate_tick(self):
        if self._done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        target, robot, reason = self._inputs_at(now)
        if self.t0 is None:
            if reason is None:
                try:
                    self._visibility(target, robot)
                except ValueError as error:
                    reason = str(error)
                else:
                    self.t0 = now
            if self.t0 is None:
                elapsed = time.monotonic() - self.started_wall
                timed_out = (
                    (self.role is None and elapsed >= self.role_timeout)
                    or (self.role is not None and elapsed >= self.data_ready_timeout)
                    or elapsed >= self.startup_timeout
                )
                if timed_out:
                    self._infrastructure_failure(reason or "tracking inputs unavailable")
                return
        eval_time = self.t0 + self.eval_index / self.rate
        target, robot, reason = self._inputs_at(eval_time)
        if reason is not None:
            self._infrastructure_failure(f"eval {self.eval_index}: {reason}")
            return
        try:
            visible = self._visibility(target, robot)
        except ValueError as error:
            self._infrastructure_failure(f"eval {self.eval_index}: {error}")
            return
        target_pos = target.pose.pose.position
        robot_pos = robot.pose.pose.position
        distance = math.hypot(
            target_pos.x - robot_pos.x, target_pos.y - robot_pos.y
        )
        within = distance <= self.radius
        effective = within and visible
        previous = evaluate_tracking(
            self.rows, self.minimum_duration, self.acquisition_timeout
        )
        acquired_at = (
            self.t0 + previous["acquisition_time_sec"]
            if previous["acquisition_time_sec"] is not None else None
        )
        if acquired_at is None and effective:
            acquired_at = eval_time
        elapsed = 0.0 if acquired_at is None else max(0.0, eval_time - acquired_at)
        self.rows.append({
            "case_id": self.case["case_id"],
            "eval_index": self.eval_index,
            "eval_time": f"{eval_time:.9f}",
            "perception_robot": self.role,
            "target_gt_x": target_pos.x,
            "target_gt_y": target_pos.y,
            "robot_gt_x": robot_pos.x,
            "robot_gt_y": robot_pos.y,
            "distance_m": distance,
            "visible": visible,
            "within_tracking_radius": within,
            "effective_tracking": effective,
            "tracking_elapsed_sec": elapsed,
            "infrastructure_valid": True,
        })
        self.eval_index += 1
        result = evaluate_tracking(
            self.rows, self.minimum_duration, self.acquisition_timeout
        )
        if result["complete"]:
            self._finish(result)

    def _infrastructure_failure(self, reason):
        self.infrastructure_valid = False
        self.infrastructure_errors.append(reason)
        self._finish()

    def _finish(self, result=None):
        if self._done:
            return
        self._done = True
        raw_path = self.output_dir / "raw/tracking_samples.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        result = result or evaluate_tracking(
            self.rows, self.minimum_duration, self.acquisition_timeout
        )
        provisional = bool(
            self.case.get("provisional", False) or not self.case.get("formal", False)
        )
        summary = {
            "infrastructure_valid": self.infrastructure_valid,
            "provisional": provisional,
            "tracking_success": bool(result["tracking_success"]),
            "acquisition_time_sec": result["acquisition_time_sec"],
            "continuous_tracking_time_sec": result["continuous_tracking_time_sec"],
            "failure_reason": result["failure_reason"],
            "pass": bool(
                self.infrastructure_valid and not provisional
                and result["tracking_success"]
            ),
        }
        if self.infrastructure_errors:
            summary["infrastructure_errors"] = self.infrastructure_errors
            summary["reason"] = self.infrastructure_errors[-1]
        elif provisional:
            summary["reason"] = "provisional configuration is not an official result"
        elif result["failure_reason"]:
            summary["reason"] = result["failure_reason"]
        write_yaml(self.output_dir / "case_summary.yaml", summary)
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TrackingRecorder()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node._done:
            node._infrastructure_failure("recorder interrupted")
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
