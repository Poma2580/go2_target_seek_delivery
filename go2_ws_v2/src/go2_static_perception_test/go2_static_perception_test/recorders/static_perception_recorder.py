#!/usr/bin/env python3
"""Record one static single-Go2 perception case."""

import csv
import json
import math
import time
from pathlib import Path

import rclpy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401
import tf2_ros

from go2_static_perception_test.common.config import read_yaml
from go2_static_perception_test.common.visibility import CameraIntrinsics, project_camera_point, quaternion_conjugate_rotate
from go2_static_perception_test.recorders.cache import TimeCache
from go2_static_perception_test.reporting.results import evaluate_csv


CSV_FIELDS = ("case_id", "eval_index", "eval_time", "target_key", "prompt",
              "infrastructure_valid", "visible", "recognition_matched",
              "recognition_success", "confidence", "bbox", "localization_matched",
              "localization_success", "target_gt_x", "target_gt_y", "target_est_x",
              "target_est_y", "robot_gt_x", "robot_gt_y")


def stamp_seconds(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def parse_status(raw, expected_prompt):
    try: value = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid perception status JSON: {error}") from error
    required = {"schema_version", "stamp", "sample_id", "prompt", "recognition_success",
                "confidence", "bbox", "localization_success"}
    if not isinstance(value, dict) or not required <= set(value) or value["schema_version"] != 1:
        raise ValueError("invalid perception status schema")
    if value["prompt"] != expected_prompt:
        raise ValueError("perception status prompt does not match case")
    stamp = value["stamp"]
    if not isinstance(stamp, dict) or not {"sec", "nanosec"} <= set(stamp):
        raise ValueError("invalid perception status stamp")
    if not isinstance(value["sample_id"], int) or value["sample_id"] < 0:
        raise ValueError("invalid perception sample_id")
    for key in ("recognition_success", "localization_success"):
        if not isinstance(value[key], bool): raise ValueError(f"{key} must be boolean")
    if value["localization_success"] and not value["recognition_success"]:
        raise ValueError("localization success requires recognition success")
    confidence, bbox = value["confidence"], value["bbox"]
    if confidence is not None and (isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence)):
        raise ValueError("invalid confidence")
    if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4 or
                             not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in bbox)):
        raise ValueError("invalid bbox")
    value["timestamp"] = float(stamp["sec"]) + float(stamp["nanosec"]) * 1e-9
    return value


class StaticPerceptionRecorder(Node):
    def __init__(self):
        super().__init__("static_perception_recorder")
        self.declare_parameter("case_config", "")
        self.declare_parameter("output_dir", "")
        case_path = Path(str(self.get_parameter("case_config").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        if not case_path.is_file() or not str(self.output_dir):
            raise ValueError("case_config must exist and output_dir must be non-empty")
        self.case = read_yaml(case_path)
        settings = self.case["settings"]
        self.rate = float(settings["evaluation_rate_hz"])
        self.duration = float(settings["evaluation_duration_sec"])
        self.match_timeout = float(settings["match_timeout_sec"])
        self.result_settle_delay = float(settings["result_settle_delay_sec"])
        self.startup_timeout = float(settings["startup_timeout_sec"])
        self.data_ready_timeout = float(settings["data_ready_timeout_sec"])
        self.min_depth = float(settings["min_camera_depth_m"])
        self.max_depth = float(settings["max_camera_depth_m"])
        self.sample_count = int(round(self.rate * self.duration))
        if self.sample_count <= 0: raise ValueError("evaluation window must contain samples")
        self.target_gt = TimeCache()
        self.robot_gt = TimeCache()
        self.status = TimeCache(consumable=True)
        self.estimates = TimeCache(consumable=True)
        self.camera_info = None
        self.rows, self.errors = [], []
        self.t0, self.eval_index = None, 0
        self.started_wall, self.done = time.monotonic(), False
        self.callback_group = ReentrantCallbackGroup()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        kwargs = {"callback_group": self.callback_group}
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models_cb, 10, **kwargs)
        self.create_subscription(Odometry, "/go2_1/odom/ground_truth", self._robot_cb, qos_profile_sensor_data, **kwargs)
        self.create_subscription(CameraInfo, "/go2_1/camera/depth/camera_info", self._camera_cb, qos_profile_sensor_data, **kwargs)
        self.create_subscription(String, "/go2_1/static_perception/result_status", self._status_cb, 10, **kwargs)
        self.create_subscription(PoseStamped, "/go2_1/static_perception/target_pose_estimated", self._estimate_cb, 10, **kwargs)
        self.create_timer(1.0 / self.rate, self._tick, **kwargs)

    def _models_cb(self, message):
        indices = [i for i, name in enumerate(message.name) if name == self.case["model_name"]]
        if len(indices) == 1:
            self.target_gt.append(self.get_clock().now().nanoseconds * 1e-9, message.pose[indices[0]])

    def _robot_cb(self, message): self.robot_gt.append(stamp_seconds(message.header.stamp), message)
    def _camera_cb(self, message): self.camera_info = message

    def _status_cb(self, message):
        try: status = parse_status(message.data, self.case["prompt"])
        except ValueError as error:
            self.get_logger().warning(str(error), throttle_duration_sec=2.0); return
        self.status.append(status["timestamp"], status)

    def _estimate_cb(self, message):
        if message.header.frame_id.strip("/") != "go2_1/odom":
            self.get_logger().warning("ignoring estimate outside go2_1/odom", throttle_duration_sec=2.0); return
        self.estimates.append(stamp_seconds(message.header.stamp), message)

    def _inputs(self, when):
        target = self.target_gt.nearest(when, self.match_timeout)
        robot = self.robot_gt.nearest(when, self.match_timeout)
        if target is None: return None, None, "target GT unavailable"
        if robot is None: return None, None, "go2_1 GT unavailable"
        return target[1], robot[1], None

    def _visible(self, target, robot):
        info = self.camera_info
        if info is None or info.width <= 0 or info.height <= 0:
            raise ValueError("CameraInfo unavailable or invalid")
        tp, rp, q = target.position, robot.pose.pose.position, robot.pose.pose.orientation
        delta = quaternion_conjugate_rotate((tp.x-rp.x, tp.y-rp.y, tp.z-rp.z), (q.x,q.y,q.z,q.w))
        point = PointStamped(); point.header.frame_id = "go2_1/base_footprint"
        point.header.stamp = robot.header.stamp
        point.point.x, point.point.y, point.point.z = delta
        try:
            camera = self.tf_buffer.transform(point, info.header.frame_id,
                timeout=rclpy.duration.Duration(seconds=min(0.1, self.match_timeout)))
        except Exception as error:
            raise ValueError(f"camera TF unavailable: {error}") from error
        intrinsics = CameraIntrinsics(info.k[0], info.k[4], info.k[2], info.k[5], info.width, info.height)
        return project_camera_point((camera.point.x, camera.point.y, camera.point.z),
                                    intrinsics, self.min_depth, self.max_depth)[0]

    def _tick(self):
        if self.done: return
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.t0 is None:
            target, robot, reason = self._inputs(now)
            if reason is None:
                try: visible = self._visible(target, robot)
                except ValueError as error: reason, visible = str(error), False
                if visible:
                    self.t0 = now
                    self.get_logger().info(
                        f"evaluation window anchored at {now:.3f}; "
                        f"settling perception results for {self.result_settle_delay:.2f}s"
                    )
                    return
            if time.monotonic() - self.started_wall >= min(self.startup_timeout, self.data_ready_timeout):
                self.errors.append(reason or "target never became visible"); self._finish(False)
            return
        when = self.t0 + self.eval_index / self.rate
        if now < when + self.result_settle_delay:
            return
        target, robot, reason = self._inputs(when)
        if reason:
            self.errors.append(f"eval {self.eval_index}: {reason}"); self._record(when, target, robot, "", False); return
        try: visible = self._visible(target, robot)
        except ValueError as error:
            self.errors.append(f"eval {self.eval_index}: {error}"); self._record(when, target, robot, "", False); return
        self._record(when, target, robot, visible, True)

    def _record(self, when, target, robot, visible, infrastructure_valid):
        matched = self.status.nearest(when, self.match_timeout)
        status = matched[1] if matched else None
        estimate = None
        if status and status["localization_success"]:
            found = self.estimates.nearest(status["timestamp"], 1e-6)
            estimate = found[1] if found else None
        tp = target.position if target else None
        rp = robot.pose.pose.position if robot else None
        ep = estimate.pose.position if estimate else None
        self.rows.append({"case_id": self.case["case_id"], "eval_index": self.eval_index,
            "eval_time": f"{when:.9f}", "target_key": self.case["target_key"], "prompt": self.case["prompt"],
            "infrastructure_valid": infrastructure_valid, "visible": visible,
            "recognition_matched": status is not None,
            "recognition_success": bool(status and status["recognition_success"]),
            "confidence": "" if not status or status["confidence"] is None else status["confidence"],
            "bbox": "" if not status or status["bbox"] is None else json.dumps(status["bbox"], separators=(",", ":")),
            "localization_matched": estimate is not None,
            "localization_success": bool(status and status["localization_success"]),
            "target_gt_x": "" if tp is None else tp.x, "target_gt_y": "" if tp is None else tp.y,
            "target_est_x": "" if ep is None else ep.x, "target_est_y": "" if ep is None else ep.y,
            "robot_gt_x": "" if rp is None else rp.x, "robot_gt_y": "" if rp is None else rp.y})
        self.eval_index += 1
        if self.eval_index >= self.sample_count: self._finish(not self.errors)

    def _finish(self, infrastructure_valid):
        if self.done: return
        self.done = True
        path = self.output_dir / "raw/static_samples.csv"; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS); writer.writeheader(); writer.writerows(self.rows)
        evaluate_csv(path, self.output_dir, self.case, infrastructure_valid, self.errors)
        self.get_logger().info(f"recorded {len(self.rows)} samples")
        if rclpy.ok(): rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args); node = StaticPerceptionRecorder(); executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try: executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException): pass
    finally:
        if not node.done:
            node.errors.append("recorder interrupted"); node._finish(False)
        executor.shutdown(); node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
