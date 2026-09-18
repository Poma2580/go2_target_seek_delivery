#!/usr/bin/env python3
"""Phase-4 probe for the complete T3 startup chain."""

import csv
import time
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, String

from go2_test_framework.common.config import read_yaml
from go2_test_framework.recorders.nav_goal_status import parse_nav_goal_status
from go2_test_framework.reporting.results import write_yaml


class NavChainState:
    def __init__(self):
        self.role = None
        self.dispatched = []
        self.events = []
        self.maddpg_selected = False

    def observe_status(self, value):
        self.events.append(value)
        if value["event"] == "DISPATCHED":
            generation = value["generation"]
            if generation not in self.dispatched:
                self.dispatched.append(generation)

    @property
    def ready(self):
        return (
            self.role is not None
            and len(self.dispatched) >= 2
            and not self.maddpg_selected
        )


class NavChainProbe(Node):
    def __init__(self):
        super().__init__("nav_chain_probe")
        self.declare_parameter("case_config", "")
        self.declare_parameter("output_dir", "")
        case_path = Path(str(self.get_parameter("case_config").value))
        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.case = read_yaml(case_path)
        self.timeout = float(self.case["settings"]["startup_timeout_sec"])
        self.started = time.monotonic()
        self.state = NavChainState()
        self.errors = []
        self._done = False
        qos = QoSProfile(
            depth=100,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.role_sub = self.create_subscription(
            String, "/target_role/perception_robot", self._role_cb, qos
        )
        self.status_sub = self.create_subscription(
            String, "/dynamic_encircle/nav_goal_status", self._status_cb, qos
        )
        self.mux_sub = self.create_subscription(
            Bool, "/dynamic_encircle/use_maddpg", self._mux_cb, qos
        )
        self.timer = self.create_timer(0.2, self._tick)

    def _role_cb(self, message):
        role = message.data.strip("/")
        if role in ("go2_1", "go2_2", "go2_3") and self.state.role is None:
            self.state.role = role

    def _status_cb(self, message):
        try:
            self.state.observe_status(parse_nav_goal_status(message.data))
        except ValueError as error:
            self.errors.append(str(error))

    def _mux_cb(self, message):
        if message.data:
            self.state.maddpg_selected = True
            self.errors.append("mux switched away from Nav2")

    def _tick(self):
        if self._done:
            return
        if self.errors:
            self._finish(False, self.errors[-1])
        elif self.state.ready:
            self._finish(True, None)
        elif time.monotonic() - self.started >= self.timeout:
            self._finish(False, "timed out waiting for two DISPATCHED generations")

    def _finish(self, valid, reason):
        if self._done:
            return
        self._done = True
        raw_path = self.output_dir / "raw/nav_chain_events.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with raw_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("event", "generation", "robot", "action_status"),
            )
            writer.writeheader()
            for event in self.state.events:
                writer.writerow({key: event[key] for key in writer.fieldnames})
        dispatched = [
            event for event in self.state.events if event["event"] == "DISPATCHED"
        ]
        summary = {
            "infrastructure_valid": bool(valid),
            "provisional": True,
            "chain_ready": bool(valid),
            "perception_robot": self.state.role,
            "navigation_dogs": (
                dispatched[-1]["navigation_dogs"] if dispatched else []
            ),
            "first_dispatched_generation": (
                self.state.dispatched[0] if self.state.dispatched else None
            ),
            "latest_dispatched_generation": (
                self.state.dispatched[-1] if self.state.dispatched else None
            ),
            "mux_owner": "MADDPG" if self.state.maddpg_selected else "Nav2",
            "pass": False,
            "reason": reason or "Phase 4 chain probe is not an official result",
        }
        if reason:
            summary["infrastructure_errors"] = [reason]
        write_yaml(self.output_dir / "case_summary.yaml", summary)


def main(args=None):
    rclpy.init(args=args)
    node = NavChainProbe()
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if not node._done:
            node._finish(False, "nav chain probe interrupted")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
