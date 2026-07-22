"""Assign three encirclement goals and delegate all motion to Nav2."""

from itertools import permutations
import math

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from multi_go2_nav2.scene_config import ROBOT_NAMES, load_scene_config


def yaw_to_quaternion(yaw):
    """Return the planar quaternion z/w components for a yaw angle."""
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def encircle_goals(target, radius, count=3, start_angle=math.pi):
    """Generate evenly spaced poses, with every robot facing the target."""
    goals = []
    first_angle = target.yaw + start_angle
    for index in range(count):
        angle = first_angle + 2.0 * math.pi * index / count
        x = target.x + radius * math.cos(angle)
        y = target.y + radius * math.sin(angle)
        yaw = math.atan2(target.y - y, target.x - x)
        goals.append((x, y, yaw))
    return goals


def path_length(path):
    """Compute planar length of a nav_msgs/Path-like object."""
    poses = path.poses
    return sum(
        math.hypot(
            second.pose.position.x - first.pose.position.x,
            second.pose.position.y - first.pose.position.y,
        )
        for first, second in zip(poses, poses[1:])
    )


def choose_assignment(robot_names, paths):
    """Choose the feasible 3! robot-goal permutation with least path length."""
    best = None
    for goal_indices in permutations(range(len(robot_names))):
        assignment = dict(zip(robot_names, goal_indices))
        selected = [paths.get((name, assignment[name])) for name in robot_names]
        if any(path is None or len(path.poses) < 2 for path in selected):
            continue
        total = sum(path_length(path) for path in selected)
        tie_break = tuple(goal_indices)
        candidate = (total, tie_break, assignment)
        if best is None or candidate[:2] < best[:2]:
            best = candidate
    return None if best is None else (best[2], best[0])


def planning_batches(robot_names, goal_count):
    """Return goal-major batches with one request per independent planner."""
    return [
        tuple((name, goal_index) for name in robot_names)
        for goal_index in range(goal_count)
    ]


def startup_prerequisites_ready(
        robot_names, odom_names, action_ready, lifecycle_active):
    """Require odometry, action discovery, and an active Nav2 stack per robot."""
    return all(
        name in odom_names
        and action_ready.get(name, False)
        and lifecycle_active.get(name, False)
        for name in robot_names
    )


def assignment_has_unique_goals(robot_names, assignment):
    """Return true only for a complete one-robot-to-one-goal assignment."""
    if set(assignment) != set(robot_names):
        return False
    indices = [assignment[name] for name in robot_names]
    return len(indices) == len(set(indices))


def action_status_name(status):
    """Make action result status useful in field logs."""
    names = {
        GoalStatus.STATUS_UNKNOWN: 'unknown',
        GoalStatus.STATUS_ACCEPTED: 'accepted',
        GoalStatus.STATUS_EXECUTING: 'executing',
        GoalStatus.STATUS_CANCELING: 'canceling',
        GoalStatus.STATUS_SUCCEEDED: 'succeeded',
        GoalStatus.STATUS_CANCELED: 'canceled',
        GoalStatus.STATUS_ABORTED: 'aborted',
    }
    return names.get(status, f'status_{status}')


class EncircleCoordinator(Node):
    """Use Nav2 path services for assignment and NavigateToPose for execution."""

    def __init__(self):
        super().__init__('encircle_coordinator')
        self.declare_parameter('scene_config', '')
        self.declare_parameter('autostart', True)
        config_file = self.get_parameter('scene_config').value
        if not config_file:
            raise ValueError('scene_config parameter is required')
        self.config = load_scene_config(config_file)
        self.robot_names = list(ROBOT_NAMES)

        self.plan_clients = {
            name: ActionClient(
                self, ComputePathToPose, f'/{name}/compute_path_to_pose')
            for name in self.robot_names
        }
        self.navigate_clients = {
            name: ActionClient(
                self, NavigateToPose, f'/{name}/navigate_to_pose')
            for name in self.robot_names
        }
        self.lifecycle_clients = {
            name: self.create_client(
                Trigger, f'/{name}/lifecycle_manager_navigation/is_active')
            for name in self.robot_names
        }
        self.lifecycle_active = {name: False for name in self.robot_names}
        self.lifecycle_queries_pending = set()
        self.latest_odom = {}
        for name in self.robot_names:
            self.create_subscription(
                Odometry,
                f'/{name}/odom',
                lambda message, robot=name: self._odom_callback(robot, message),
                10,
            )

        latched_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.path_publishers = {
            name: self.create_publisher(
                Path, f'/{name}/selected_plan', latched_qos)
            for name in self.robot_names
        }
        self.marker_publisher = self.create_publisher(
            MarkerArray, '/encircle_markers', latched_qos)

        radius = self.config.encircle.radius
        maximum = radius + self.config.encircle.max_radius_expansion
        step = self.config.encircle.radius_step
        self.radii = []
        while radius <= maximum + 1.0e-9:
            self.radii.append(round(radius, 9))
            radius += step
        self.state = 'waiting' if self.get_parameter('autostart').value else 'idle'
        self.radius_index = 0
        self.pending_plans = 0
        self.plan_results = {}
        self.plan_batches = []
        self.batch_index = 0
        self.batch_pending = set()
        self.current_goals = []
        self.navigation_handles = {}
        self.navigation_terminal = set()
        self.wait_count = 0
        self.timer = self.create_timer(1.0, self._startup_tick)
        self.get_logger().info(
            f'Loaded {self.config.scene} scene; pure Nav2 coordinator is {self.state}.')

    def _odom_callback(self, robot_name, message):
        self.latest_odom[robot_name] = message

    def _startup_tick(self):
        if self.state != 'waiting':
            return
        self._query_lifecycle_managers()
        action_ready = {
            name: self.plan_clients[name].server_is_ready()
            and self.navigate_clients[name].server_is_ready()
            for name in self.robot_names
        }
        ready = startup_prerequisites_ready(
            self.robot_names,
            self.latest_odom,
            action_ready,
            self.lifecycle_active,
        )
        if not ready:
            self.wait_count += 1
            if self.wait_count >= math.ceil(
                    self.config.planning.action_server_timeout):
                self.state = 'failed'
                self.timer.cancel()
                self.get_logger().error(
                    'Timed out waiting for odometry, actions, and active Nav2 stacks.')
                return
            if self.wait_count % 5 == 1:
                missing = [
                    name for name in self.robot_names
                    if name not in self.latest_odom
                    or not action_ready[name]
                    or not self.lifecycle_active[name]
                ]
                self.get_logger().info(
                    'Waiting for odometry, actions, and active Nav2 stacks: '
                    + ', '.join(missing))
            return
        self.timer.cancel()
        self.get_logger().info(
            'All three Nav2 lifecycle managers are active; starting planning.')
        self._plan_current_radius()

    def _query_lifecycle_managers(self):
        for name, client in self.lifecycle_clients.items():
            if self.lifecycle_active[name] or name in self.lifecycle_queries_pending:
                continue
            if not client.service_is_ready():
                continue
            self.lifecycle_queries_pending.add(name)
            future = client.call_async(Trigger.Request())
            future.add_done_callback(
                lambda result, robot=name:
                self._lifecycle_query_result(robot, result))

    def _lifecycle_query_result(self, name, future):
        self.lifecycle_queries_pending.discard(name)
        try:
            response = future.result()
            self.lifecycle_active[name] = bool(response.success)
        except Exception as error:
            self.lifecycle_active[name] = False
            self.get_logger().warn(
                f'Lifecycle readiness query for {name} failed: {error}')

    def _make_pose(self, values):
        x, y, yaw = values
        pose = PoseStamped()
        pose.header.frame_id = self.config.map.frame_id
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        z, w = yaw_to_quaternion(yaw)
        pose.pose.orientation.z = z
        pose.pose.orientation.w = w
        return pose

    def _plan_current_radius(self):
        if self.radius_index >= len(self.radii):
            self.state = 'failed'
            self.get_logger().error(
                'No radius produced three mutually feasible Nav2 paths; no robot moved.')
            return
        radius = self.radii[self.radius_index]
        self.current_goals = encircle_goals(
            self.config.target,
            radius,
            len(self.robot_names),
            self.config.encircle.candidate_start_angle,
        )
        self.plan_results = {}
        self.pending_plans = len(self.robot_names) * len(self.current_goals)
        self.plan_batches = planning_batches(
            self.robot_names, len(self.current_goals))
        self.batch_index = 0
        self.batch_pending = set()
        self.state = 'planning'
        self.get_logger().info(
            f'Asking Nav2 for assignment paths at encirclement radius {radius:.2f} m.')
        self._start_plan_batch()

    def _start_plan_batch(self):
        batch = self.plan_batches[self.batch_index]
        self.batch_pending = set(batch)
        goal_index = batch[0][1]
        x, y, _yaw = self.current_goals[goal_index]
        self.get_logger().info(
            f'Planning candidate {goal_index} at ({x:.2f}, {y:.2f}) '
            'with three independent planners in parallel.')
        for name, goal_index in batch:
            goal = ComputePathToPose.Goal()
            goal.goal = self._make_pose(self.current_goals[goal_index])
            goal.planner_id = self.config.planning.planner_id
            goal.use_start = False
            try:
                future = self.plan_clients[name].send_goal_async(goal)
                future.add_done_callback(
                    lambda result, key=(name, goal_index):
                    self._plan_goal_response(key, result))
            except Exception as error:
                self.get_logger().error(f'Path request {(name, goal_index)} failed: {error}')
                self._store_plan((name, goal_index), None, 'request_error')

    def _plan_goal_response(self, key, future):
        try:
            handle = future.result()
        except Exception as error:  # ROS action transport failure
            self.get_logger().error(f'Path request {key} failed: {error}')
            self._store_plan(key, None, 'request_error')
            return
        if not handle.accepted:
            self._store_plan(key, None, 'rejected')
            return
        result_future = handle.get_result_async()
        result_future.add_done_callback(
            lambda result, plan_key=key: self._plan_result(plan_key, result))

    def _plan_result(self, key, future):
        path = None
        status_name = 'result_error'
        try:
            wrapped = future.result()
            status_name = action_status_name(wrapped.status)
            if wrapped.status == GoalStatus.STATUS_SUCCEEDED:
                path = wrapped.result.path
        except Exception as error:  # ROS action transport failure
            self.get_logger().error(f'Path result {key} failed: {error}')
        if path is not None and len(path.poses) < 2:
            path = None
            status_name = 'invalid_path'
        self._store_plan(key, path, status_name)

    def _store_plan(self, key, path, status_name):
        if key in self.plan_results:
            return
        self.plan_results[key] = path
        self.pending_plans -= 1
        self.batch_pending.discard(key)
        name, goal_index = key
        x, y, _yaw = self.current_goals[goal_index]
        length_text = 'unavailable'
        if path is not None:
            length_text = f'{path_length(path):.2f} m'
        self.get_logger().info(
            f'Path {name} -> goal {goal_index} ({x:.2f}, {y:.2f}): '
            f'{status_name}, length={length_text}.')
        if self.batch_pending:
            return
        if self.batch_index + 1 < len(self.plan_batches):
            self.batch_index += 1
            self._start_plan_batch()
        elif self.pending_plans == 0:
            self._finish_assignment()
        else:
            self.state = 'failed'
            self.get_logger().error(
                'Internal planning batch count mismatch; no robot moved.')

    def _finish_assignment(self):
        selected = choose_assignment(self.robot_names, self.plan_results)
        if selected is None:
            self.radius_index += 1
            self._plan_current_radius()
            return
        assignment, total_length = selected
        if not assignment_has_unique_goals(self.robot_names, assignment):
            self.state = 'failed'
            self.get_logger().error(
                f'Planner produced a non-unique assignment {assignment}; no robot moved.')
            return
        selected_coordinates = [
            self.current_goals[assignment[name]][:2]
            for name in self.robot_names
        ]
        if len(set(selected_coordinates)) != len(self.robot_names):
            self.state = 'failed'
            self.get_logger().error(
                'Assigned goal coordinates are not unique; no robot moved.')
            return
        radius = self.radii[self.radius_index]
        self.get_logger().info(
            f'Nav2 assignment selected at radius {radius:.2f} m; '
            f'total planned length {total_length:.2f} m: {assignment}')
        for name in self.robot_names:
            goal_index = assignment[name]
            path = self.plan_results[(name, goal_index)]
            x, y, _yaw = self.current_goals[goal_index]
            self.get_logger().info(
                f'FINAL {name} -> goal {goal_index} ({x:.2f}, {y:.2f}), '
                f'path length={path_length(path):.2f} m.')
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_publishers[name].publish(path)
        self._publish_markers(assignment)
        self._navigate(assignment)

    def _publish_markers(self, assignment):
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()
        target = Marker()
        target.header.frame_id = self.config.map.frame_id
        target.header.stamp = now
        target.ns = 'encircle_target'
        target.id = 0
        target.type = Marker.CYLINDER
        target.action = Marker.ADD
        target.pose.position.x = self.config.target.x
        target.pose.position.y = self.config.target.y
        target.pose.orientation.w = 1.0
        target.scale.x = target.scale.y = 0.70
        target.scale.z = 0.15
        target.color.r = 1.0
        target.color.g = 0.65
        target.color.a = 0.95
        markers.markers.append(target)
        for index, name in enumerate(self.robot_names, start=1):
            goal_index = assignment[name]
            x, y, yaw = self.current_goals[goal_index]
            color = self.config.robots[name].color
            marker = Marker()
            marker.header.frame_id = self.config.map.frame_id
            marker.header.stamp = now
            marker.ns = 'encircle_goals'
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose = self._make_pose((x, y, yaw)).pose
            marker.scale.x = 0.80
            marker.scale.y = 0.16
            marker.scale.z = 0.16
            marker.color.r, marker.color.g, marker.color.b = color
            marker.color.a = 1.0
            markers.markers.append(marker)

            label = Marker()
            label.header.frame_id = self.config.map.frame_id
            label.header.stamp = now
            label.ns = 'encircle_labels'
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x
            label.pose.position.y = y
            label.pose.position.z = 0.8
            label.pose.orientation.w = 1.0
            label.scale.z = 0.45
            label.color.r, label.color.g, label.color.b = color
            label.color.a = 1.0
            label.text = name
            markers.markers.append(label)
        self.marker_publisher.publish(markers)

    def _navigate(self, assignment):
        self.state = 'navigating'
        self.navigation_handles = {}
        self.navigation_terminal = set()
        for name in self.robot_names:
            goal = NavigateToPose.Goal()
            goal.pose = self._make_pose(self.current_goals[assignment[name]])
            future = self.navigate_clients[name].send_goal_async(goal)
            future.add_done_callback(
                lambda result, robot=name: self._navigation_goal_response(robot, result))

    def _navigation_goal_response(self, name, future):
        try:
            handle = future.result()
        except Exception as error:
            self.get_logger().error(f'{name} NavigateToPose request failed: {error}')
            self._navigation_failed(name)
            return
        if not handle.accepted:
            self.get_logger().error(f'{name} NavigateToPose goal was rejected.')
            self._navigation_failed(name)
            return
        if self.state != 'navigating':
            handle.cancel_goal_async()
            self.get_logger().warn(
                f'Canceling late accepted goal for {name} after group failure.')
            return
        self.navigation_handles[name] = handle
        result = handle.get_result_async()
        result.add_done_callback(
            lambda completed, robot=name: self._navigation_result(robot, completed))

    def _navigation_result(self, name, future):
        try:
            status = future.result().status
        except Exception as error:
            self.get_logger().error(f'{name} navigation result failed: {error}')
            self._navigation_failed(name)
            return
        self.navigation_terminal.add(name)
        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().error(
                f'{name} navigation ended with action status {status}.')
            self._cancel_remaining(name)
            self.state = 'failed'
            return
        self.get_logger().info(f'{name} reached its assigned encirclement pose.')
        if len(self.navigation_terminal) == len(self.robot_names):
            self.state = 'succeeded'
            self.get_logger().info(
                'All three robots reached their poses using pure Nav2 navigation.')

    def _navigation_failed(self, failed_name):
        self.navigation_terminal.add(failed_name)
        self._cancel_remaining(failed_name)
        self.state = 'failed'

    def _cancel_remaining(self, failed_name):
        for name, handle in self.navigation_handles.items():
            if name != failed_name and name not in self.navigation_terminal:
                handle.cancel_goal_async()


def main(args=None):
    rclpy.init(args=args)
    node = EncircleCoordinator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()
