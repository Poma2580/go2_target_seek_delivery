"""Small ROS-facing wait primitives used by the process orchestrator."""

from dataclasses import asdict, dataclass
import math
import time

from controller_manager_msgs.srv import ListControllers
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from std_msgs.msg import String
from std_srvs.srv import Trigger


ROBOTS = ("go2_1", "go2_2", "go2_3")
REQUIRED_CONTROLLERS = (
    "joint_group_effort_controller",
    "joint_states_controller",
)


@dataclass(frozen=True)
class WalkingTargetStartObservation:
    movement_confirmed: bool
    displacement_m: float
    start_requests_sent: int

    def to_dict(self):
        return asdict(self)


class WalkingTargetStartError(RuntimeError):
    def __init__(self, message, observation):
        super().__init__(message)
        self.observation = observation


def start_walking_target(
    timeout_sec=10.0, min_displacement_m=0.05,
    retry_interval_sec=2.0, max_requests=3, health_check=None,
):
    """Request target motion and confirm it solely from observed displacement."""
    latest_position = None

    def odom_callback(message):
        nonlocal latest_position
        position = message.pose.pose.position
        latest_position = (position.x, position.y)

    rclpy.init()
    node = Node("target_test_walking_target_starter")
    client = node.create_client(Trigger, "/walking_target/start")
    subscription = node.create_subscription(
        Odometry, "/walking_target/odom", odom_callback,
        qos_profile_sensor_data,
    )
    deadline = time.monotonic() + timeout_sec
    baseline = None
    displacement = 0.0
    requests_sent = 0
    pending_requests = []
    next_request_at = None

    def observation(confirmed=False):
        return WalkingTargetStartObservation(
            movement_confirmed=confirmed,
            displacement_m=displacement,
            start_requests_sent=requests_sent,
        )

    try:
        while rclpy.ok() and time.monotonic() < deadline:
            if health_check is not None:
                health_check()
            remaining = deadline - time.monotonic()
            rclpy.spin_once(node, timeout_sec=min(0.1, max(0.0, remaining)))

            if baseline is None:
                if latest_position is None:
                    continue
                baseline = latest_position
                next_request_at = time.monotonic()

            if latest_position is not None:
                displacement = math.hypot(
                    latest_position[0] - baseline[0],
                    latest_position[1] - baseline[1],
                )
                if displacement >= min_displacement_m:
                    return observation(confirmed=True)

            now = time.monotonic()
            if requests_sent < max_requests and now >= next_request_at:
                if client.service_is_ready() or client.wait_for_service(
                    timeout_sec=min(0.1, max(0.0, deadline - now))
                ):
                    # Keep futures alive while observing motion, but deliberately
                    # do not inspect service responses: displacement is the only
                    # success criterion.
                    pending_requests.append(client.call_async(Trigger.Request()))
                    requests_sent += 1
                    next_request_at = now + retry_interval_sec

        if baseline is None:
            reason = "timed out waiting for initial /walking_target/odom"
        elif requests_sent == 0:
            reason = "timed out waiting for /walking_target/start service"
        else:
            reason = (
                "walking target did not move at least "
                f"{min_displacement_m:.3f} m within {timeout_sec:.1f} seconds"
            )
        raise WalkingTargetStartError(reason, observation())
    finally:
        node.destroy_subscription(subscription)
        node.destroy_client(client)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


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
