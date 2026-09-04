"""Small ROS-facing wait primitives used by the process orchestrator."""

import time

from controller_manager_msgs.srv import ListControllers
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String


ROBOTS = ("go2_1", "go2_2", "go2_3")
REQUIRED_CONTROLLERS = (
    "joint_group_effort_controller",
    "joint_states_controller",
)


def active_controller_names(response):
    """Return the controller names reported in the active state."""
    return {
        controller.name
        for controller in response.controller
        if controller.state == "active"
    }


def wait_for_controllers_active(robot_name, timeout_sec, health_check=None):
    """Wait until both controllers required by a spawned Go2 are active."""
    service_name = f"/{robot_name}/controller_manager/list_controllers"
    rclpy.init()
    node = Node(f"target_test_{robot_name}_controller_waiter")
    client = node.create_client(ListControllers, service_name)
    deadline = time.monotonic() + timeout_sec
    try:
        while rclpy.ok():
            if health_check is not None:
                health_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError(
                    f"timed out waiting for active controllers for {robot_name}"
                )
            if not client.wait_for_service(timeout_sec=min(0.2, remaining)):
                continue
            future = client.call_async(ListControllers.Request())
            while rclpy.ok() and not future.done():
                if health_check is not None:
                    health_check()
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise RuntimeError(
                        f"timed out waiting for active controllers for {robot_name}"
                    )
                rclpy.spin_once(node, timeout_sec=min(0.2, remaining))
            if future.done() and future.exception() is None:
                active = active_controller_names(future.result())
                if set(REQUIRED_CONTROLLERS) <= active:
                    return
            time.sleep(0.2)
        raise RuntimeError(
            f"ROS shut down while waiting for controllers for {robot_name}"
        )
    finally:
        node.destroy_client(client)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def wait_for_perception_role(timeout_sec, health_check=None):
    """Return the latched perception role, using a wall-clock timeout."""
    selected = None
    rclpy.init()
    node = Node("target_test_role_waiter")
    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
    )

    def callback(message):
        nonlocal selected
        candidate = message.data.strip("/")
        if candidate in ROBOTS:
            selected = candidate

    subscription = node.create_subscription(
        String, "/target_role/perception_robot", callback, qos
    )
    deadline = time.monotonic() + timeout_sec
    try:
        while rclpy.ok() and selected is None:
            if health_check is not None:
                health_check()
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise RuntimeError("timed out waiting for perception role")
            rclpy.spin_once(node, timeout_sec=min(0.2, remaining))
        if selected is None:
            raise RuntimeError("ROS shut down while waiting for perception role")
        return selected
    finally:
        node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
