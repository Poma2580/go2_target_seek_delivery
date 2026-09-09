#!/usr/bin/env python3
"""Record latest-generation Nav2 endpoint performance for one T3 Case."""

import csv
import json
import math
import time
from functools import partial
from pathlib import Path

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import String
import tf2_geometry_msgs  # noqa: F401
import tf2_ros

from go2_test_framework.common.config import read_yaml
from go2_test_framework.evaluators.path_planning import PathPlanningEvaluator
from go2_test_framework.recorders.cache import TimeCache
from go2_test_framework.recorders.collisions import (
    CollisionDebouncer, parse_body_contacts,
)
from go2_test_framework.recorders.nav_goal_status import parse_nav_goal_status
from go2_test_framework.recorders.target_recorder import stamp_seconds
from go2_test_framework.reporting.results import write_yaml


ROBOTS = ("go2_1", "go2_2", "go2_3")
PATH_FIELDS = (
    "eval_time", "generation", "frame_id", "dog_1", "dog_1_gt_x",
    "dog_1_gt_y", "dog_1_goal_x", "dog_1_goal_y", "dog_1_endpoint_error_m",
    "dog_1_reached", "dog_2", "dog_2_gt_x", "dog_2_gt_y",
    "dog_2_goal_x", "dog_2_goal_y", "dog_2_endpoint_error_m",
    "dog_2_reached", "all_reached", "infrastructure_valid",
)
EVENT_FIELDS = (
    "stamp", "event", "generation", "frame_id", "navigation_dogs",
    "goals", "robot", "action_status",
)
COLLISION_FIELDS = (
    "stamp", "robot", "robot_collision", "object_model", "object_collision",
)


class PathPlanningRecorder(Node):
    def __init__(self):
        super().__init__("path_planning_recorder")
        self.declare_parameter("case_config", "")
        self.declare_parameter("output_dir", "")
        case_path = Path(str(self.get_parameter("case_config").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.case = read_yaml(case_path)
        metrics = self.case["metrics"]
        settings = self.case["settings"]
        self.rate = float(settings["evaluation_rate_hz"])
        self.match_timeout = float(settings["match_timeout_sec"])
        self.startup_timeout = float(settings["startup_timeout_sec"])
        self.data_ready_timeout = float(settings["data_ready_timeout_sec"])
        self.evaluator = PathPlanningEvaluator(
            metrics["endpoint_tolerance_m"], metrics["path_timeout_sec"]
        )
        self.robot_gt = {name: TimeCache() for name in ROBOTS}
        self.events = []
        self.rows = []
        self.collision_events = []
        self.collision_debouncer = CollisionDebouncer()
        self.contact_stream_seen = False
        self.started_wall = time.monotonic()
        self.dispatch_received_wall = None
        self.infrastructure_valid = True
        self.infrastructure_errors = []
        self._done = False
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_sub = self.create_subscription(
            String, "/dynamic_encircle/nav_goal_status", self._status_cb, qos
        )
        self.contact_sub = self.create_subscription(
            String, "/go2_test/body_contacts", self._contact_cb, 10
        )
        self.gt_subscriptions = [
            self.create_subscription(
                Odometry, f"/{name}/odom/ground_truth",
                partial(self._gt_cb, name), qos_profile_sensor_data,
            )
            for name in ROBOTS
        ]
        self.timer = self.create_timer(1.0 / self.rate, self._tick)

    def _gt_cb(self, name, message):
        self.robot_gt[name].append(stamp_seconds(message.header.stamp), message)

    def _status_cb(self, message):
        try:
            event = parse_nav_goal_status(message.data)
        except ValueError as error:
            self._infrastructure_failure(str(error))
            return
        self.events.append(event)
        previous = self.evaluator.latest_generation
        self.evaluator.observe_event(event)
        if (
            event["event"] == "DISPATCHED"
            and self.evaluator.latest_generation != previous
            and self.dispatch_received_wall is None
        ):
            self.dispatch_received_wall = time.monotonic()
        if self.evaluator.complete:
            self._maybe_finish()

    def _contact_cb(self, message):
        try:
            stamp, pairs = parse_body_contacts(message.data)
        except ValueError as error:
            self._infrastructure_failure(str(error))
            return
        self.contact_stream_seen = True
        stamp_seconds_value = stamp["sec"] + stamp["nanosec"] * 1e-9
        for event in self.collision_debouncer.update(pairs):
            self.collision_events.append({"stamp": stamp_seconds_value, **event})
        self._maybe_finish()

    def _point_in_goal_frame(self, message, frame_id):
        point = PointStamped()
        point.header = message.header
        point.point = message.pose.pose.position
        try:
            return self.tf_buffer.transform(
                point, frame_id,
                timeout=rclpy.duration.Duration(seconds=0.1),
            ).point
        except Exception as error:  # noqa: BLE001
            raise ValueError(
                f"cannot transform {message.header.frame_id!r} to {frame_id!r}: {error}"
            ) from error

    def _tick(self):
        if self._done:
            return
        if not self.contact_stream_seen:
            if time.monotonic() - self.started_wall >= self.data_ready_timeout:
                self._infrastructure_failure("body-contact stream unavailable")
            return
        if self.evaluator.latest_generation is None:
            if time.monotonic() - self.started_wall >= self.startup_timeout:
                self._infrastructure_failure("timed out waiting for DISPATCHED")
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        positions = {}
        for name in self.evaluator.navigation_dogs:
            match = self.robot_gt[name].nearest(now, self.match_timeout)
            if match is None:
                if time.monotonic() - self.dispatch_received_wall >= self.data_ready_timeout:
                    self._infrastructure_failure(f"{name} GT unavailable")
                return
            try:
                positions[name] = self._point_in_goal_frame(
                    match[1], self.evaluator.frame_id
                )
            except ValueError as error:
                if time.monotonic() - self.dispatch_received_wall >= self.data_ready_timeout:
                    self._infrastructure_failure(str(error))
                return
        errors = {}
        for name, position in positions.items():
            goal = self.evaluator.goals[name]
            errors[name] = math.hypot(
                position.x - goal["goal_x"], position.y - goal["goal_y"]
            )
        self.evaluator.observe_errors(now, errors)
        dogs = self.evaluator.navigation_dogs
        row = {
            "eval_time": f"{now:.9f}",
            "generation": self.evaluator.latest_generation,
            "frame_id": self.evaluator.frame_id,
            "infrastructure_valid": True,
            "all_reached": all(
                errors[name] <= self.evaluator.endpoint_tolerance_m for name in dogs
            ),
        }
        for index, name in enumerate(dogs, start=1):
            position = positions[name]
            goal = self.evaluator.goals[name]
            row.update({
                f"dog_{index}": name,
                f"dog_{index}_gt_x": position.x,
                f"dog_{index}_gt_y": position.y,
                f"dog_{index}_goal_x": goal["goal_x"],
                f"dog_{index}_goal_y": goal["goal_y"],
                f"dog_{index}_endpoint_error_m": errors[name],
                f"dog_{index}_reached": errors[name] <= self.evaluator.endpoint_tolerance_m,
            })
        self.rows.append(row)
        if self.evaluator.complete:
            self._maybe_finish()

    def _maybe_finish(self):
        if self.evaluator.complete and self.contact_stream_seen:
            self._finish()

    def _infrastructure_failure(self, reason):
        if self._done:
            return
        self.infrastructure_valid = False
        self.infrastructure_errors.append(reason)
        self._finish()

    def _write_raw(self):
        raw_dir = self.output_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "path_samples.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=PATH_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)
        with (raw_dir / "nav_goal_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=EVENT_FIELDS)
            writer.writeheader()
            for event in self.events:
                row = dict(event)
                row["stamp"] = json.dumps(row["stamp"], separators=(",", ":"))
                row["navigation_dogs"] = json.dumps(row["navigation_dogs"])
                row["goals"] = json.dumps(row["goals"], separators=(",", ":"))
                writer.writerow({key: row[key] for key in EVENT_FIELDS})
        with (raw_dir / "collision_events.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=COLLISION_FIELDS)
            writer.writeheader()
            writer.writerows(self.collision_events)

    def _finish(self):
        if self._done:
            return
        self._done = True
        self._write_raw()
        provisional = bool(
            self.case.get("provisional", False) or not self.case.get("formal", False)
        )
        collision_count = len(self.collision_events)
        path_success = bool(self.evaluator.path_success)
        failure_reason = self.evaluator.failure_reason
        if path_success and collision_count:
            failure_reason = "static_collision"
        summary = {
            "infrastructure_valid": self.infrastructure_valid,
            "provisional": provisional,
            "path_success": path_success,
            "all_reached": path_success,
            "collision_count": collision_count,
            "latest_generation": self.evaluator.latest_generation,
            "failure_reason": failure_reason,
            "pass": bool(
                self.infrastructure_valid and not provisional
                and path_success and collision_count == 0
            ),
        }
        if self.infrastructure_errors:
            summary["infrastructure_errors"] = self.infrastructure_errors
            summary["reason"] = self.infrastructure_errors[-1]
        elif provisional:
            summary["reason"] = "provisional configuration is not an official result"
        elif failure_reason:
            summary["reason"] = failure_reason
        write_yaml(self.output_dir / "case_summary.yaml", summary)


def main(args=None):
    rclpy.init(args=args)
    node = PathPlanningRecorder()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if not node._done:
            node._infrastructure_failure("path planning recorder interrupted")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
