"""NavigateToPose dispatch, throttling, cancellation, and callback ownership."""

import math
from functools import partial

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient

from .geometry import yaw_to_quaternion_components


def goal_status_label(status):
    """Return a readable action_msgs/GoalStatus label."""
    labels = {
        GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
        GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
        GoalStatus.STATUS_EXECUTING: "EXECUTING",
        GoalStatus.STATUS_CANCELING: "CANCELING",
        GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
        GoalStatus.STATUS_CANCELED: "CANCELED",
        GoalStatus.STATUS_ABORTED: "ABORTED",
    }
    return labels.get(status, f"UNKNOWN({status})")


class GoalUpdateState:
    """Throttle goal generations and reject stale asynchronous callbacks."""

    def __init__(self, period):
        """使用最小发送周期初始化 Nav2 目标状态。"""
        if not math.isfinite(period) or period <= 0.0:
            raise ValueError("goal update period must be finite and greater than zero")
        self.period = period
        self.last_dispatch = None
        self.generation = 0
        self.suspended = False
        self.completed = False

    def due(self, now):
        """Return whether another goal generation may be dispatched."""
        if self.suspended or self.completed:
            return False
        return self.last_dispatch is None or now - self.last_dispatch >= self.period - 1e-6

    def mark_dispatched(self, now):
        """Record dispatch time and return a new generation number."""
        if not self.due(now):
            raise RuntimeError("goal dispatch is not due")
        self.last_dispatch = now
        self.generation += 1
        return self.generation

    def make_due(self):
        """Allow an important plan change to bypass the periodic refresh wait."""
        if self.suspended or self.completed:
            return False
        self.last_dispatch = None
        return True

    def suspend(self):
        """Suspend dispatch and invalidate callbacks from the active generation."""
        if self.completed or self.suspended:
            return False
        self.suspended = True
        self.generation += 1
        return True

    def resume(self):
        """Resume dispatch and make the next generation immediately due."""
        if self.completed or not self.suspended:
            return False
        self.suspended = False
        self.last_dispatch = None
        return True

    def complete(self):
        """Latch completion and permanently prevent future dispatch."""
        if self.completed:
            return False
        self.completed = True
        self.suspended = False
        self.generation += 1
        return True

    def is_current(self, generation):
        """Return whether a callback belongs to the current live generation."""
        return (
            generation == self.generation
            and not self.suspended
            and not self.completed
        )


class NavGoalManager:
    """Own all NavigateToPose clients and their asynchronous lifecycle."""

    def __init__(self, node, robot_names, global_frame, update_period):
        """创建三套 action 客户端和目标更新状态。"""
        self.node = node
        self.robot_names = tuple(robot_names)
        self.global_frame = global_frame
        self.navigation_dogs = ()
        self.action_clients = {
            name: ActionClient(node, NavigateToPose, f"/{name}/navigate_to_pose")
            for name in self.robot_names
        }
        self.active_goal_handles = {name: {} for name in self.robot_names}
        self.pending_goal_sends = {name: set() for name in self.robot_names}
        self.cancel_inflight = {name: set() for name in self.robot_names}
        self._handoff_cancelling = False
        self._handoff_cancel_failed = False
        self.state = GoalUpdateState(update_period)
        self.plan = None

    @property
    def last_dispatch(self):
        """返回最近一次目标发送时刻。"""
        return self.state.last_dispatch

    @property
    def suspended(self):
        """返回目标更新是否暂时挂起。"""
        return self.state.suspended

    @property
    def completed(self):
        """返回围捕完成状态是否已永久锁存。"""
        return self.state.completed

    def set_navigation_dogs(self, names):
        """Select the two clients controlled by Nav2 after role election."""
        names = tuple(names)
        if len(names) != 2 or any(name not in self.robot_names for name in names):
            raise ValueError("navigation_dogs must contain two configured robots")
        self.navigation_dogs = names

    def set_plan(self, plan):
        """Store the latest formation plan for the next dispatch."""
        self.plan = plan

    def dispatch_if_due(self, now, force=False):
        """Send one goal generation when due and both servers are ready."""
        if force:
            self.state.make_due()
        if self.plan is None or not self.navigation_dogs or not self.state.due(now):
            return False
        if not all(
            self.action_clients[name].server_is_ready()
            for name in self.navigation_dogs
        ):
            self.node.get_logger().warning(
                "[nav_goal] Waiting for both NavigateToPose action servers",
                throttle_duration_sec=2.0,
            )
            return False

        generation = self.state.mark_dispatched(now)
        for name in self.navigation_dogs:
            self.pending_goal_sends[name].add(generation)
            point = self.plan.slots[name]
            future = self.action_clients[name].send_goal_async(
                self._goal_message(point)
            )
            future.add_done_callback(
                partial(self._goal_response_callback, name, generation, point)
            )
        self.node.get_logger().info(
            "[nav_goal] generation %d heading=%.1f deg: %s"
            % (
                generation,
                math.degrees(self.plan.route_heading),
                "; ".join(
                    "%s=(%.2f, %.2f)"
                    % (name, self.plan.slots[name][0], self.plan.slots[name][1])
                    for name in self.navigation_dogs
                ),
            )
        )
        return True

    def suspend(self, reason):
        """Suspend goal updates and cancel accepted active goals."""
        if self.state.suspend():
            self._cancel_active_goals(reason)
            return True
        return False

    def resume(self):
        """Resume updates after a temporary target loss."""
        return self.state.resume()

    def complete(self):
        """Permanently stop updates when the formation is complete."""
        if not self.state.complete():
            return False
        self._cancel_active_goals("encirclement formed")
        return True

    def begin_handoff_cancel(self):
        """Permanently stop dispatch and settle every pending/accepted goal."""
        if self._handoff_cancelling:
            return False
        self._handoff_cancelling = True
        self._handoff_cancel_failed = False
        self.state.complete()
        for name in self.navigation_dogs:
            for generation, goal_handle in tuple(
                self.active_goal_handles[name].items()
            ):
                self._request_handoff_cancel(name, generation, goal_handle)
        return True

    def handoff_cancel_complete(self):
        """Return true only after no send, cancel, or accepted goal remains."""
        return self._handoff_cancelling and not self._handoff_cancel_failed and all(
            not self.pending_goal_sends[name]
            and not self.cancel_inflight[name]
            and not self.active_goal_handles[name]
            for name in self.navigation_dogs
        )

    def handoff_cancel_failed(self):
        return self._handoff_cancel_failed

    def shutdown(self):
        """Invalidate callbacks and cancel goals before node destruction."""
        if not self.state.completed:
            self.state.complete()
        self._cancel_active_goals("node shutdown")

    def _goal_message(self, point):
        """Convert a formation slot into a NavigateToPose goal."""
        x, y, yaw = point
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = self.global_frame
        pose.header.stamp = self.node.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.z, pose.pose.orientation.w = (
            yaw_to_quaternion_components(yaw)
        )
        goal.pose = pose
        return goal

    def _goal_response_callback(self, name, generation, point, future):
        """Accept current handles and cancel accepted stale generations."""
        self.pending_goal_sends[name].discard(generation)
        try:
            goal_handle = future.result()
        except Exception as error:
            if self._handoff_cancelling:
                self._handoff_cancel_failed = True
            if self.state.is_current(generation):
                self.node.get_logger().error(
                    f"[nav_goal] {name} goal send failed: {error}"
                )
            return

        if self._handoff_cancelling:
            if goal_handle.accepted:
                self.active_goal_handles[name][generation] = goal_handle
                result_future = goal_handle.get_result_async()
                result_future.add_done_callback(
                    partial(self._goal_result_callback, name, generation)
                )
                self._request_handoff_cancel(name, generation, goal_handle)
            return

        if not self.state.is_current(generation):
            if goal_handle.accepted:
                goal_handle.cancel_goal_async()
            return
        if not goal_handle.accepted:
            self.node.get_logger().warning(
                "[nav_goal] %s rejected generation %d at (%.2f, %.2f)"
                % (name, generation, point[0], point[1])
            )
            return

        self.active_goal_handles[name][generation] = goal_handle
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(
            partial(self._goal_result_callback, name, generation)
        )

    def _goal_result_callback(self, name, generation, future):
        """Log a current result without treating it as formation completion."""
        self.active_goal_handles[name].pop(generation, None)
        self.cancel_inflight[name].discard(generation)
        if self._handoff_cancelling:
            try:
                future.result()
            except Exception as error:
                self._handoff_cancel_failed = True
                self.node.get_logger().error(
                    f"[nav_goal] {name} terminal result failed: {error}"
                )
            return
        if not self.state.is_current(generation):
            return
        try:
            status = future.result().status
        except Exception as error:
            self.node.get_logger().error(
                f"[nav_goal] {name} result failed: {error}"
            )
            return
        self.node.get_logger().info(
            f"[nav_goal] {name} generation {generation} finished: "
            f"{goal_status_label(status)}; overall success still uses "
            "simultaneous slot pose"
        )

    def _cancel_active_goals(self, reason):
        """Cancel and forget accepted handles for both navigation robots."""
        for name in self.navigation_dogs:
            for generation, goal_handle in tuple(
                self.active_goal_handles[name].items()
            ):
                self.node.get_logger().warning(
                    f"[nav_goal] Cancelling {name} goal: {reason}"
                )
                goal_handle.cancel_goal_async()
                self.active_goal_handles[name].pop(generation, None)

    def _request_handoff_cancel(self, name, generation, goal_handle):
        """Request cancellation once; terminal result confirms loss of control."""
        if generation in self.cancel_inflight[name]:
            return
        self.cancel_inflight[name].add(generation)
        self.node.get_logger().warning(
            f"[nav_goal] Cancelling {name} generation {generation} for handoff"
        )
        try:
            future = goal_handle.cancel_goal_async()
            future.add_done_callback(
                partial(self._cancel_response_callback, name, generation)
            )
        except Exception as error:
            self.cancel_inflight[name].discard(generation)
            self._handoff_cancel_failed = True
            self.node.get_logger().error(
                f"[nav_goal] {name} cancel request failed: {error}"
            )

    def _cancel_response_callback(self, name, generation, future):
        self.cancel_inflight[name].discard(generation)
        try:
            future.result()
        except Exception as error:
            self._handoff_cancel_failed = True
            self.node.get_logger().error(
                f"[nav_goal] {name} cancel response failed: {error}"
            )
