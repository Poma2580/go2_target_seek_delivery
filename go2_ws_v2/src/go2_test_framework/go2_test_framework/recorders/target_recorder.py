#!/usr/bin/env python3
"""Record one target-perception Case on a single 5 Hz evaluation clock."""

import csv
import json
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
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import tf2_ros
import tf2_geometry_msgs  # noqa: F401  (register PointStamped transforms)

from go2_test_framework.common.config import read_yaml
from go2_test_framework.ground_truth.visibility import CameraIntrinsics, project_camera_point, quaternion_conjugate_rotate
from go2_test_framework.recorders.cache import TimeCache
from go2_test_framework.reporting.results import evaluate_csv, write_yaml


ROBOTS = ("go2_1", "go2_2", "go2_3")
CSV_FIELDS = (
    "case_id", "eval_index", "eval_time", "perception_robot",
    "infrastructure_valid", "visible",
    "recognition_matched", "recognition_success",
    "localization_matched", "localization_success",
    "target_gt_x", "target_gt_y", "target_est_x", "target_est_y",
    "robot_gt_x", "robot_gt_y",
)


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def parse_status(message):
    try:
        value = json.loads(message)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid perception status JSON: {error}") from error
    required = (
        "schema_version", "stamp", "sample_id", "recognition_success",
        "confidence", "bbox", "localization_success",
    )
    missing = [key for key in required if key not in value]
    if missing:
        raise ValueError(f"perception status missing fields: {missing}")
    if value["schema_version"] != 1:
        raise ValueError("perception status schema_version must be 1")
    stamp = value["stamp"]
    if not isinstance(stamp, dict) or set(("sec", "nanosec")) - set(stamp):
        raise ValueError("perception status stamp must contain sec and nanosec")
    if not isinstance(value["sample_id"], int) or value["sample_id"] < 0:
        raise ValueError("perception status sample_id must be a non-negative integer")
    for key in ("recognition_success", "localization_success"):
        if not isinstance(value[key], bool):
            raise ValueError(f"perception status {key} must be boolean")
    if value["bbox"] is not None and (
        not isinstance(value["bbox"], list) or len(value["bbox"]) != 4
        or not all(isinstance(item, (int, float)) and math.isfinite(item) for item in value["bbox"])
    ):
        raise ValueError("perception status bbox must be null or four finite numbers")
    confidence = value["confidence"]
    if confidence is not None and (
        not isinstance(confidence, (int, float)) or not math.isfinite(confidence)
    ):
        raise ValueError("perception status confidence must be null or finite")
    value["timestamp"] = float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
    return value


class TargetTestRecorder(Node):
    def __init__(self):
        super().__init__("target_test_recorder")
        self.declare_parameter("case_config", "")
        self.declare_parameter("output_dir", "")
        case_path = Path(str(self.get_parameter("case_config").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        if not case_path.is_file():
            raise ValueError(f"case_config does not exist: {case_path}")
        if not str(self.output_dir):
            raise ValueError("output_dir must be non-empty")
        self.case = read_yaml(case_path)
        settings = self.case["settings"]
        self.rate = float(settings["evaluation_rate_hz"])
        self.duration = float(settings["evaluation_duration_sec"])
        self.match_timeout = float(settings["match_timeout_sec"])
        self.startup_timeout = float(settings["startup_timeout_sec"])
        self.role_timeout = float(settings["role_timeout_sec"])
        self.data_ready_timeout = float(settings["data_ready_timeout_sec"])
        self.min_depth = float(settings["min_camera_depth_m"])
        self.max_depth = float(settings["max_camera_depth_m"])
        self.sample_count = int(round(self.rate * self.duration))
        if self.sample_count <= 0:
            raise ValueError("evaluation window must contain at least one sample")

        self.role = None
        self.camera_info = {}
        self.target_gt = TimeCache()
        self.robot_gt = {name: TimeCache() for name in ROBOTS}
        self.status = {name: TimeCache(consumable=True) for name in ROBOTS}
        self.estimates = {name: TimeCache(consumable=True) for name in ROBOTS}
        self.rows = []
        self.t0 = None
        self.eval_index = 0
        self.infrastructure_valid = True
        self.failure_reasons = []
        self.started_wall = time.monotonic()
        self._done = False
        self.callback_group = ReentrantCallbackGroup()

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        role_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST, depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Do not reuse Node._subscriptions: it is rclpy's executor-owned registry.
        self.role_subscription = self.create_subscription(
            String, "/target_role/perception_robot", self._role_cb, role_qos,
            callback_group=self.callback_group)
        self.target_subscription = self.create_subscription(
            Odometry, "/walking_target/odom", self._target_cb,
            qos_profile_sensor_data, callback_group=self.callback_group)
        self._owned_subscriptions = []
        self.timer = self.create_timer(
            1.0 / self.rate, self._evaluate_tick,
            callback_group=self.callback_group)
        self.get_logger().info(f"waiting for first visible sample for {self.case['case_id']}")

    def _role_cb(self, message):
        selected = message.data.strip("/")
        if selected not in ROBOTS:
            self.get_logger().warning(f"ignoring unknown perception robot {selected!r}")
            return
        if self.role is None:
            self.role = selected
            self._subscribe_selected_robot(selected)
            self.get_logger().info(f"selected perception robot: {selected}")
        elif selected != self.role:
            self.get_logger().warning(f"ignoring conflicting role {selected}; locked={self.role}")

    def _subscribe_selected_robot(self, name):
        """Subscribe only after role lock to avoid three high-rate GT streams."""
        self._owned_subscriptions.extend((
            self.create_subscription(
                Odometry, f"/{name}/odom/ground_truth",
                partial(self._robot_cb, name), qos_profile_sensor_data,
                callback_group=self.callback_group),
            self.create_subscription(
                String, f"/{name}/target_perception/result_status",
                partial(self._status_cb, name), 10,
                callback_group=self.callback_group),
            self.create_subscription(
                Odometry, f"/{name}/target_estimated/odom",
                partial(self._estimate_cb, name), 10,
                callback_group=self.callback_group),
            self.create_subscription(
                CameraInfo, f"/{name}/camera/depth/camera_info",
                partial(self._camera_cb, name), qos_profile_sensor_data,
                callback_group=self.callback_group),
        ))

    def _target_cb(self, message):
        self.target_gt.append(stamp_seconds(message.header.stamp), message)

    def _robot_cb(self, name, message):
        self.robot_gt[name].append(stamp_seconds(message.header.stamp), message)

    def _status_cb(self, name, message):
        try:
            status = parse_status(message.data)
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=2.0)
            return
        self.status[name].append(status["timestamp"], status)

    def _estimate_cb(self, name, message):
        self.estimates[name].append(stamp_seconds(message.header.stamp), message)

    def _camera_cb(self, name, message):
        self.camera_info[name] = message

    def _visibility(self, target, robot, eval_time):
        info = self.camera_info.get(self.role)
        if info is None or info.width <= 0 or info.height <= 0:
            raise ValueError("camera_info unavailable or invalid")
        target_position = target.pose.pose.position
        robot_position = robot.pose.pose.position
        q = robot.pose.pose.orientation
        delta_world = (
            target_position.x - robot_position.x,
            target_position.y - robot_position.y,
            target_position.z - robot_position.z,
        )
        delta_base = quaternion_conjugate_rotate(delta_world, (q.x, q.y, q.z, q.w))
        point = PointStamped()
        point.header.frame_id = f"{self.role}/base_footprint"
        # The robot GT sample was already matched to eval_time within the suite
        # tolerance. Query TF at that actual sample stamp so a scheduled tick a
        # few milliseconds ahead of the latest transform is not misclassified
        # as an infrastructure failure.
        point.header.stamp = robot.header.stamp
        point.point.x, point.point.y, point.point.z = delta_base
        try:
            camera_point = self.tf_buffer.transform(
                point, info.header.frame_id,
                timeout=rclpy.duration.Duration(seconds=min(0.1, self.match_timeout)),
            )
        except Exception as error:  # noqa: BLE001
            raise ValueError(f"camera TF unavailable: {error}") from error
        intrinsics = CameraIntrinsics(info.k[0], info.k[4], info.k[2], info.k[5], info.width, info.height)
        visible, pixel = project_camera_point(
            (camera_point.point.x, camera_point.point.y, camera_point.point.z),
            intrinsics, self.min_depth, self.max_depth,
        )
        if not visible:
            self.get_logger().info(
                "target outside camera projection: optical=(%.2f, %.2f, %.2f) pixel=%s"
                % (camera_point.point.x, camera_point.point.y, camera_point.point.z, pixel),
                throttle_duration_sec=2.0,
            )
        return visible

    def _inputs_at(self, eval_time):
        if self.role is None:
            return None, None, "perception role unavailable"
        target = self.target_gt.nearest(eval_time, self.match_timeout)
        robot = self.robot_gt[self.role].nearest(eval_time, self.match_timeout)
        if target is None:
            return None, None, "walking target GT unavailable"
        if robot is None:
            return None, None, f"{self.role} GT unavailable"
        return target[1], robot[1], None

    def _evaluate_tick(self):
        if self._done:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            target, robot, reason = self._inputs_at(now)
            if reason is None:
                try:
                    visible = self._visibility(target, robot, now)
                except ValueError as error:
                    reason = str(error)
                    self.get_logger().warning(reason, throttle_duration_sec=2.0)
                    visible = False
                if visible:
                    self.t0 = now
                    self.get_logger().info(f"evaluation window started at {now:.3f}")
                    self._record(now, target, robot, True, True)
                    return
            elapsed = time.monotonic() - self.started_wall
            timed_out = (
                self.role is None and elapsed >= self.role_timeout
            ) or (
                self.role is not None and elapsed >= self.data_ready_timeout
            ) or elapsed >= self.startup_timeout
            if timed_out:
                self.infrastructure_valid = False
                self.failure_reasons.append(reason or "target never became visible")
                self._finish()
            return

        if self.eval_index >= self.sample_count:
            self._finish()
            return
        eval_time = self.t0 + self.eval_index / self.rate
        target, robot, reason = self._inputs_at(eval_time)
        if reason is not None:
            self.infrastructure_valid = False
            self.failure_reasons.append(f"eval {self.eval_index}: {reason}")
            self._record(eval_time, target, robot, "", False)
            return
        try:
            visible = self._visibility(target, robot, eval_time)
            self._record(eval_time, target, robot, visible, True)
        except ValueError as error:
            self.infrastructure_valid = False
            self.failure_reasons.append(f"eval {self.eval_index}: {error}")
            self._record(eval_time, target, robot, "", False)

    def _record(self, eval_time, target, robot, visible, infrastructure_valid):
        status_match = self.status[self.role].nearest(eval_time, self.match_timeout)
        status = status_match[1] if status_match else None
        recognition_matched = status is not None
        recognition_success = bool(status and status["recognition_success"])
        localization_success = bool(status and status["localization_success"])
        estimate = None
        localization_matched = False
        if localization_success:
            estimate_match = self.estimates[self.role].nearest(status["timestamp"], 1e-6)
            if estimate_match is not None:
                estimate = estimate_match[1]
                localization_matched = True
        target_pos = target.pose.pose.position if target is not None else None
        robot_pos = robot.pose.pose.position if robot is not None else None
        estimate_pos = estimate.pose.pose.position if estimate is not None else None
        self.rows.append({
            "case_id": self.case["case_id"],
            "eval_index": self.eval_index,
            "eval_time": f"{eval_time:.9f}",
            "perception_robot": self.role or "",
            "infrastructure_valid": infrastructure_valid,
            "visible": visible,
            "recognition_matched": recognition_matched,
            "recognition_success": recognition_success,
            "localization_matched": localization_matched,
            "localization_success": localization_success,
            "target_gt_x": "" if target_pos is None else target_pos.x,
            "target_gt_y": "" if target_pos is None else target_pos.y,
            "target_est_x": "" if estimate_pos is None else estimate_pos.x,
            "target_est_y": "" if estimate_pos is None else estimate_pos.y,
            "robot_gt_x": "" if robot_pos is None else robot_pos.x,
            "robot_gt_y": "" if robot_pos is None else robot_pos.y,
        })
        self.eval_index += 1
        if self.eval_index >= self.sample_count:
            self._finish()

    def _finish(self):
        if self._done:
            return
        self._done = True
        raw_path = self.output_dir / "raw/target_samples.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        summary = evaluate_csv(
            raw_path, self.output_dir / "metrics",
            infrastructure_valid=self.infrastructure_valid,
            provisional=bool(self.case.get("provisional", False) or not self.case.get("formal", False)),
            recognition_threshold=float(self.case.get("metrics", {}).get("recognition_pass_threshold_percent", 80.0)),
            localization_threshold=float(self.case.get("metrics", {}).get("localization_pass_threshold_percent", 15.0)),
        )
        if self.failure_reasons:
            summary["infrastructure_errors"] = self.failure_reasons
        write_yaml(self.output_dir / "case_summary.yaml", summary)
        self.get_logger().info(f"recorded {len(self.rows)} samples; pass={summary['pass']}")
        if rclpy.ok():
            rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TargetTestRecorder()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node._done:
            node.infrastructure_valid = False
            node.failure_reasons.append("recorder interrupted")
            node._finish()
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
