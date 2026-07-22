"""Bring up the shared map and namespaced Nav2 stacks in strict order."""

from collections.abc import Callable, Iterable
from typing import NamedTuple

from nav2_msgs.srv import ManageLifecycleNodes
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from multi_go2_nav2.scene_config import ROBOT_NAMES


class ManagedStack(NamedTuple):
    """Lifecycle manager endpoints belonging to one managed stack."""

    label: str
    manage_service: str
    active_service: str


def lifecycle_manager_sequence(robot_names=ROBOT_NAMES):
    """Return the only supported lifecycle startup order."""
    sequence = [ManagedStack(
        'shared map',
        '/lifecycle_manager_map/manage_nodes',
        '/lifecycle_manager_map/is_active',
    )]
    sequence.extend(
        ManagedStack(
            name,
            f'/{name}/lifecycle_manager_navigation/manage_nodes',
            f'/{name}/lifecycle_manager_navigation/is_active',
        )
        for name in robot_names
    )
    return tuple(sequence)


def run_bringup_sequence(
        sequence: Iterable[ManagedStack],
        start_stack: Callable[[ManagedStack], bool]):
    """Start stacks in order and stop immediately after the first failure."""
    for stack in sequence:
        if not start_stack(stack):
            return False
    return True


class Nav2BringupSequencer(Node):
    """Call Nav2 lifecycle managers sequentially to avoid startup overload."""

    def __init__(self):
        super().__init__('nav2_bringup_sequencer')
        self.declare_parameter('manager_timeout', 60.0)
        self.manager_timeout = float(
            self.get_parameter('manager_timeout').value)
        if self.manager_timeout <= 0.0:
            raise ValueError('manager_timeout must be positive')

    def bringup(self):
        """Start every managed stack and return whether all became active."""
        self.get_logger().info(
            'Starting lifecycle managers sequentially: map -> '
            + ' -> '.join(ROBOT_NAMES))
        success = run_bringup_sequence(
            lifecycle_manager_sequence(), self._start_stack)
        if success:
            self.get_logger().info(
                'Shared map and all three Nav2 stacks are active.')
        else:
            self.get_logger().error(
                'Sequential Nav2 lifecycle bringup failed; coordinator will '
                'not send navigation goals.')
        return success

    def _start_stack(self, stack):
        self.get_logger().info(f'Starting {stack.label} lifecycle stack...')
        manager = self.create_client(
            ManageLifecycleNodes, stack.manage_service)
        if not manager.wait_for_service(timeout_sec=self.manager_timeout):
            self.get_logger().error(
                f'Timed out waiting for {stack.manage_service}')
            return False

        request = ManageLifecycleNodes.Request()
        request.command = ManageLifecycleNodes.Request.STARTUP
        future = manager.call_async(request)
        rclpy.spin_until_future_complete(
            self, future, timeout_sec=self.manager_timeout)
        if not future.done():
            future.cancel()
            self.get_logger().error(
                f'Timed out starting {stack.label} after '
                f'{self.manager_timeout:.1f} seconds')
            return False
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(
                f'Lifecycle startup call for {stack.label} failed: {error}')
            return False
        if response is None or not response.success:
            self.get_logger().error(
                f'Lifecycle manager rejected startup for {stack.label}')
            return False

        active = self.create_client(Trigger, stack.active_service)
        if not active.wait_for_service(timeout_sec=self.manager_timeout):
            self.get_logger().error(
                f'Timed out waiting for {stack.active_service}')
            return False
        active_future = active.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            self, active_future, timeout_sec=self.manager_timeout)
        if not active_future.done():
            active_future.cancel()
            self.get_logger().error(
                f'Timed out confirming active state for {stack.label}')
            return False
        try:
            active_response = active_future.result()
        except Exception as error:
            self.get_logger().error(
                f'Active-state query for {stack.label} failed: {error}')
            return False
        if active_response is None or not active_response.success:
            message = (
                '' if active_response is None else active_response.message)
            self.get_logger().error(
                f'{stack.label} did not become active: {message}')
            return False

        self.get_logger().info(f'{stack.label} lifecycle stack is active.')
        return True


def main(args=None):
    """Run the one-shot lifecycle bringup process."""
    rclpy.init(args=args)
    node = None
    success = False
    try:
        node = Nav2BringupSequencer()
        success = node.bringup()
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
    if not success:
        raise SystemExit(1)
