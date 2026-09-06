"""Deploy discrete MADDPG as a one-hertz Nav2 goal selector."""

import math
import os
import sys
from dataclasses import dataclass, fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import rclpy
import torch
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32MultiArray, Int32MultiArray, String
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

import tf2_geometry_msgs  # noqa: F401

from .dynamic_encircle.geometry import quaternion_to_yaw, yaw_to_quaternion_components
from .dynamic_encircle.nav_goal_manager import NavGoalManager


DEFAULT_ACTION = 2
OBSERVATION_SLICES = {
    "lidar_sectors": (0, 36),
    "candidate_features": (36, 61),
    "default_relative": (61, 63),
    "own_velocity": (63, 65),
    "leader_relative": (65, 67),
    "leader_velocity": (67, 69),
    "teammate_relative": (69, 71),
    "teammate_velocity": (71, 73),
    "role": (73, 75),
    "previous_action": (75, 80),
    "goal_relative": (80, 82),
    "last_progress": (82, 83),
}


def _repo_root():
    configured = os.environ.get("DELIVERY_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "waypoint_maddpg_v0").is_dir():
            return parent
    raise RuntimeError("Set DELIVERY_ROOT so waypoint_maddpg_v0 can be imported")


REPO_ROOT = _repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from waypoint_maddpg_v0.config import EnvConfig  # noqa: E402
from waypoint_maddpg_v0.discrete_maddpg import DiscreteMADDPG  # noqa: E402
from waypoint_maddpg_v0.geometry import rotate, world_to_body  # noqa: E402
from waypoint_maddpg_v0.lidar import candidate_features  # noqa: E402


DEFAULT_MODEL = (
    REPO_ROOT
    / "waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt"
)


@dataclass
class RobotSample:
    odom: Odometry = None
    odom_received_at: object = None
    scan: LaserScan = None
    scan_received_at: object = None


def environment_from_checkpoint(payload):
    values = payload.get("metadata", {}).get("environment", {})
    valid = {field.name for field in fields(EnvConfig)}
    values = {key: value for key, value in values.items() if key in valid}
    for name in ("candidate_offsets", "obstacle_spawn_x", "obstacle_abs_y_range"):
        if name in values:
            values[name] = tuple(values[name])
    return EnvConfig(**values)


def candidate_points(leader_xy, leader_yaw, config):
    forward = np.asarray([math.cos(leader_yaw), math.sin(leader_yaw)], dtype=np.float32)
    left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
    bases = np.stack(
        [
            leader_xy + config.formation_forward * forward + config.formation_side * left,
            leader_xy + config.formation_forward * forward - config.formation_side * left,
        ]
    )
    offsets = np.asarray(config.candidate_offsets, dtype=np.float32)
    return bases[:, None, :] + offsets[None, :, None] * left[None, None, :]


def scan_features(scan, follower_xy, follower_yaw, leader_yaw, config):
    """Convert Gazebo LaserScan data to training-compatible sectors and hits."""
    ranges = np.asarray(scan.ranges, dtype=np.float32)
    angles = scan.angle_min + np.arange(len(ranges), dtype=np.float32) * scan.angle_increment
    if len(ranges) < config.lidar_sim_rays:
        raise ValueError(
            f"LaserScan has {len(ranges)} rays; checkpoint requires at least "
            f"{config.lidar_sim_rays}"
        )
    # The VLP-16/pointcloud converter currently supplies 220 rays.  Select the
    # nearest physical ray for each of the checkpoint's 108 leader-frame ray
    # directions before 3:1 minimum pooling.  This keeps both the sector input
    # and candidate-clearance point density aligned with training.
    target_relative = np.linspace(
        -math.pi,
        math.pi,
        config.lidar_sim_rays,
        endpoint=False,
        dtype=np.float32,
    )
    target_scan_angles = target_relative + leader_yaw - follower_yaw
    scan_span = float(scan.angle_increment) * len(ranges)
    target_scan_angles = (
        np.mod(target_scan_angles - scan.angle_min, scan_span) + scan.angle_min
    )
    selected_indices = np.rint(
        (target_scan_angles - scan.angle_min) / scan.angle_increment
    ).astype(int) % len(ranges)
    ranges = ranges[selected_indices]
    angles = angles[selected_indices]
    valid = (
        np.isfinite(ranges)
        & (ranges >= config.lidar_min_range)
        & (ranges <= config.lidar_policy_max_range)
    )
    ranges = ranges[valid]
    angles = angles[valid]
    if not len(ranges):
        return (
            np.zeros(config.lidar_observation_size, dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    # /scan is expressed in the physical lidar frame, whose origin is
    # config.lidar_sensor_x ahead of the robot centre.  Add that translation
    # exactly once to recover hit points in the robot-centred convention.
    points_robot = np.stack(
        [
            config.lidar_sensor_x + ranges * np.cos(angles),
            ranges * np.sin(angles),
        ],
        axis=1,
    )
    c, s = math.cos(follower_yaw), math.sin(follower_yaw)
    robot_rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    points_world = follower_xy + points_robot @ robot_rotation.T
    c, s = math.cos(-leader_yaw), math.sin(-leader_yaw)
    leader_rotation = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    points_leader = (points_world - follower_xy) @ leader_rotation.T

    # Sector identity follows the emitted ray direction, exactly as in the
    # training simulator.  Do not derive it from the hit point direction:
    # the 0.20 m forward sensor offset would shift sector boundaries.
    sector_count = config.lidar_observation_size
    rays_per_sector = config.lidar_sim_rays // sector_count
    # Invalid/far returns are clear sectors, equivalent to the policy max.
    ordered_ranges = np.full(
        config.lidar_sim_rays, config.lidar_policy_max_range, dtype=np.float32
    )
    ordered_ranges[valid] = ranges
    pooled = ordered_ranges.reshape(sector_count, rays_per_sector).min(axis=1)
    sectors = (config.lidar_policy_max_range - pooled) / (
        config.lidar_policy_max_range - config.lidar_min_range
    )
    return sectors.astype(np.float32), points_leader.astype(np.float32)


class MaddpgWaypointSelector(Node):
    """Output only candidate goals; Nav2 exclusively owns motion and cmd_vel."""

    def __init__(self):
        super().__init__("maddpg_waypoint_selector")
        self.declare_parameter("model_path", str(DEFAULT_MODEL))
        self.declare_parameter("leader_name", "go2_1")
        self.declare_parameter("follower_1", "go2_2")
        self.declare_parameter("follower_2", "go2_3")
        self.declare_parameter("robot_names", ["go2_1", "go2_2", "go2_3"])
        self.declare_parameter("perception_robot_topic", "")
        self.declare_parameter("wait_for_enable", False)
        self.declare_parameter("enable_topic", "/dynamic_encircle/maddpg_enable")
        self.declare_parameter("controller_ready_topic", "/maddpg_waypoint/controller_ready")
        self.declare_parameter("controller_active_topic", "/maddpg_waypoint/controller_active")
        self.declare_parameter("global_frame", "merged_map")
        self.declare_parameter("decision_period", 1.0)
        self.declare_parameter("nav_goal_update_period", 3.0)
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("scan_timeout", 1.0)
        self.declare_parameter("tf_timeout", 0.2)
        self.declare_parameter("max_input_skew", 0.2)
        self.declare_parameter("speed_tolerance", 0.03)
        self.declare_parameter("leader_speed_tolerance", 0.03)
        self.declare_parameter(
            "initial_formation_tolerance", 0.75
        )
        self.declare_parameter("require_initial_formation", True)
        self.declare_parameter("dry_run", True)
        self.declare_parameter("enabled", True)

        self.model_path = Path(self.get_parameter("model_path").value).expanduser().resolve()
        self.leader_name = str(self.get_parameter("leader_name").value)
        self.follower_names = (
            str(self.get_parameter("follower_1").value),
            str(self.get_parameter("follower_2").value),
        )
        self.robot_names = tuple(self.get_parameter("robot_names").value)
        self.perception_robot_topic = str(
            self.get_parameter("perception_robot_topic").value
        )
        self.wait_for_enable = bool(self.get_parameter("wait_for_enable").value)
        self.enable_topic = str(self.get_parameter("enable_topic").value)
        self.controller_ready_topic = str(
            self.get_parameter("controller_ready_topic").value
        )
        self.controller_active_topic = str(
            self.get_parameter("controller_active_topic").value
        )
        self.global_frame = str(self.get_parameter("global_frame").value)
        self.decision_period = float(self.get_parameter("decision_period").value)
        self.nav_goal_update_period = float(
            self.get_parameter("nav_goal_update_period").value
        )
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.scan_timeout = float(self.get_parameter("scan_timeout").value)
        self.tf_timeout = float(self.get_parameter("tf_timeout").value)
        self.max_input_skew = float(self.get_parameter("max_input_skew").value)
        self.speed_tolerance = float(self.get_parameter("speed_tolerance").value)
        self.leader_speed_tolerance = float(
            self.get_parameter("leader_speed_tolerance").value
        )
        self.initial_formation_tolerance = float(
            self.get_parameter("initial_formation_tolerance").value
        )
        self.require_initial_formation = bool(
            self.get_parameter("require_initial_formation").value
        )
        self.dry_run = bool(self.get_parameter("dry_run").value)
        self.enabled = bool(self.get_parameter("enabled").value) and not self.wait_for_enable
        self.dynamic_role = bool(self.perception_robot_topic)
        self.role_received = not self.dynamic_role
        self.controller_active = False
        if len(self.follower_names) != 2 or self.leader_name in self.follower_names:
            raise ValueError("leader_name and two unique follower_names are required")
        if len(self.robot_names) != 3 or len(set(self.robot_names)) != 3:
            raise ValueError("robot_names must contain exactly three unique names")
        if self.dynamic_role and any(
            not name or not isinstance(name, str) for name in self.robot_names
        ):
            raise ValueError("dynamic robot names must be non-empty strings")
        if self.decision_period < 1.0:
            raise ValueError("decision_period must be at least 1.0 second")
        if self.nav_goal_update_period <= 0.0:
            raise ValueError("nav_goal_update_period must be positive")
        if (
            self.max_input_skew <= 0.0
            or self.speed_tolerance < 0.0
            or self.leader_speed_tolerance < 0.0
        ):
            raise ValueError("input skew must be positive and speed tolerances nonnegative")
        if self.initial_formation_tolerance <= 0.0:
            raise ValueError("initial_formation_tolerance must be positive")

        payload = torch.load(self.model_path, map_location="cpu", weights_only=False)
        self.config = environment_from_checkpoint(payload)
        self.policy = DiscreteMADDPG(
            2,
            83,
            self.config.num_actions,
            hidden_dim=int(payload.get("hidden_dim", 256)),
            device=torch.device("cpu"),
            shared_actor=bool(payload.get("shared_actor", False)),
        )
        self.policy.load(self.model_path)
        expected_obs_dim = max(stop for _, stop in OBSERVATION_SLICES.values())
        if expected_obs_dim != 83 or self.config.lidar_observation_size != 36:
            raise ValueError("checkpoint/runtime observation layout must be 83-D with 36 lidar sectors")
        if self.config.num_actions != 5:
            raise ValueError("checkpoint must use exactly five candidate actions")
        if abs(self.decision_period - self.config.marl_dt) > 1e-6:
            raise ValueError(
                "decision_period must match checkpoint marl_dt "
                f"({self.config.marl_dt:.3f} s)"
            )

        names = self.robot_names if self.dynamic_role else (
            self.leader_name,
            *self.follower_names,
        )
        self.samples = {name: RobotSample() for name in names}
        for name in names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda message, robot=name: self._odom_callback(robot, message),
                20,
            )
        scan_names = names if self.dynamic_role else self.follower_names
        for name in scan_names:
            self.create_subscription(
                LaserScan,
                f"/{name}/scan",
                lambda message, robot=name: self._scan_callback(robot, message),
                qos_profile_sensor_data,
            )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav_goals = NavGoalManager(
            self, names, self.global_frame, self.nav_goal_update_period
        )
        self.nav_goals.set_navigation_dogs(self.follower_names)
        if not self.enabled:
            self.nav_goals.suspend("waiting for handoff enable")

        self.action_publisher = self.create_publisher(
            Int32MultiArray, "/maddpg_waypoint/actions", 10
        )
        self.goal_publisher = self.create_publisher(
            PoseArray, "/maddpg_waypoint/goals", 10
        )
        self.ready_publisher = self.create_publisher(
            Bool, "/maddpg_waypoint/ready", 10
        )
        status_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.controller_ready_publisher = self.create_publisher(
            Bool, self.controller_ready_topic, status_qos
        )
        self.controller_active_publisher = self.create_publisher(
            Bool, self.controller_active_topic, status_qos
        )
        self.observation_publisher = self.create_publisher(
            Float32MultiArray, "/maddpg_waypoint/observations", 10
        )
        self.error_publisher = self.create_publisher(
            Float32MultiArray, "/maddpg_waypoint/errors", 10
        )
        self.marker_publisher = self.create_publisher(
            MarkerArray, "/maddpg_waypoint/markers", 10
        )
        self.create_service(SetBool, "/maddpg_waypoint/set_enabled", self._set_enabled)
        if self.dynamic_role:
            self.create_subscription(
                String,
                self.perception_robot_topic,
                self._role_callback,
                status_qos,
            )
        if self.wait_for_enable:
            self.create_subscription(
                Bool,
                self.enable_topic,
                self._enable_callback,
                status_qos,
            )

        self.previous_actions = np.full(2, DEFAULT_ACTION, dtype=np.int64)
        self.current_goals = np.zeros((2, 2), dtype=np.float32)
        self.last_positions = None
        self.last_progress = np.zeros(2, dtype=np.float32)
        self.blocked_latched = np.zeros(2, dtype=bool)
        self.clear_counts = np.zeros(2, dtype=np.int64)
        self.formation_initialized = not self.require_initial_formation
        self._publish_controller_status(False)
        self.timer = self.create_timer(self.decision_period, self._decision_callback)
        self.get_logger().info(
            "Loaded %s; leader=%s followers=%s decision=%.1fs nav_goal=%.1fs "
            "dry_run=%s dynamic_role=%s wait_for_enable=%s initial_alignment=%s. "
            "MADDPG publishes goals only; Nav2 owns cmd_vel."
            % (
                self.model_path,
                self.leader_name,
                ",".join(self.follower_names),
                self.decision_period,
                self.nav_goal_update_period,
                self.dry_run,
                self.dynamic_role,
                self.wait_for_enable,
                self.require_initial_formation,
            )
        )

    def _role_callback(self, message):
        selected = message.data.strip("/")
        if selected not in self.robot_names:
            self.get_logger().warning(f"Ignoring unknown perception robot: {selected}")
            return
        if self.role_received:
            if selected != self.leader_name:
                self.get_logger().warning(
                    f"Ignoring role change to {selected}; role is locked to {self.leader_name}"
                )
            return
        self.leader_name = selected
        self.follower_names = tuple(
            name for name in self.robot_names if name != selected
        )
        self.nav_goals.set_navigation_dogs(self.follower_names)
        self.role_received = True
        self._reset_policy_state()
        self.get_logger().info(
            "Dynamic role locked: leader=%s followers=%s"
            % (self.leader_name, ",".join(self.follower_names))
        )

    def _reset_policy_state(self):
        self.previous_actions = np.full(2, DEFAULT_ACTION, dtype=np.int64)
        self.current_goals = np.zeros((2, 2), dtype=np.float32)
        self.last_positions = None
        self.last_progress = np.zeros(2, dtype=np.float32)
        self.blocked_latched = np.zeros(2, dtype=bool)
        self.clear_counts = np.zeros(2, dtype=np.int64)
        self.formation_initialized = not self.require_initial_formation

    def _inputs_ready(self):
        if not self.role_received:
            return False
        required = (self.leader_name, *self.follower_names)
        return all(
            self.samples[name].odom is not None
            and self._fresh(self.samples[name].odom_received_at, self.odom_timeout)
            for name in required
        ) and all(
            self.samples[name].scan is not None
            and self._fresh(self.samples[name].scan_received_at, self.scan_timeout)
            for name in self.follower_names
        )

    def _publish_controller_status(self, ready):
        self.controller_ready_publisher.publish(Bool(data=bool(ready)))
        self.controller_active_publisher.publish(Bool(data=self.controller_active))

    def _set_enabled_state(self, enabled):
        self.enabled = bool(enabled)
        self.controller_active = False
        self._reset_policy_state()
        if self.enabled:
            self.nav_goals.resume()
        else:
            self.nav_goals.suspend("selector disabled")
        self._publish_controller_status(self._inputs_ready())

    def _enable_callback(self, message):
        requested = bool(message.data)
        if requested != self.enabled:
            self._set_enabled_state(requested)
            self.get_logger().info(
                "Handoff command: waypoint selector %s"
                % ("enabled" if requested else "disabled")
            )

    def _odom_callback(self, name, message):
        sample = self.samples[name]
        sample.odom = message
        sample.odom_received_at = self.get_clock().now()

    def _scan_callback(self, name, message):
        sample = self.samples[name]
        sample.scan = message
        sample.scan_received_at = self.get_clock().now()

    def _fresh(self, received_at, timeout):
        return received_at is not None and (
            self.get_clock().now() - received_at
        ).nanoseconds * 1e-9 <= timeout

    def _global_state(self, name):
        message = self.samples[name].odom
        pose = PoseStamped()
        pose.header = message.header
        pose.header.stamp = Time().to_msg()
        pose.pose = message.pose.pose
        if pose.header.frame_id != self.global_frame:
            pose = self.tf_buffer.transform(
                pose,
                self.global_frame,
                timeout=Duration(seconds=self.tf_timeout),
            )
        yaw = quaternion_to_yaw(pose.pose.orientation)
        body_vx = float(message.twist.twist.linear.x)
        body_vy = float(message.twist.twist.linear.y)
        c, s = math.cos(yaw), math.sin(yaw)
        return SimpleNamespace(
            xy=np.asarray([pose.pose.position.x, pose.pose.position.y], dtype=np.float32),
            yaw=yaw,
            velocity=np.asarray(
                [c * body_vx - s * body_vy, s * body_vx + c * body_vy],
                dtype=np.float32,
            ),
        )

    def _observations(self, states, candidates):
        leader = states[self.leader_name]
        followers = [states[name] for name in self.follower_names]
        defaults = candidates[:, DEFAULT_ACTION]
        current_positions = np.stack([state.xy for state in followers])
        if self.last_positions is None:
            self.last_progress[:] = 0.0
        else:
            forward = np.asarray([math.cos(leader.yaw), math.sin(leader.yaw)])
            self.last_progress = ((current_positions - self.last_positions) @ forward).astype(np.float32)

        observations, metrics_by_agent = [], []
        for index, follower in enumerate(followers):
            sectors, lidar_points = scan_features(
                self.samples[self.follower_names[index]].scan,
                follower.xy,
                follower.yaw,
                leader.yaw,
                self.config,
            )
            candidate_obs, metrics = candidate_features(
                candidates[index], follower.xy, leader.yaw, lidar_points, self.config
            )
            other = followers[1 - index]
            own_velocity = rotate(follower.velocity, -leader.yaw)
            leader_rel = world_to_body(leader.xy, follower.xy, leader.yaw)
            leader_velocity = rotate(leader.velocity, -leader.yaw)
            teammate_rel = world_to_body(other.xy, follower.xy, leader.yaw)
            teammate_velocity = rotate(other.velocity - follower.velocity, -leader.yaw)
            default_rel = world_to_body(defaults[index], follower.xy, leader.yaw)
            goal = self.current_goals[index] if np.any(self.current_goals[index]) else defaults[index]
            goal_rel = world_to_body(goal, follower.xy, leader.yaw)
            role = np.asarray([1.0, 0.0] if index == 0 else [0.0, 1.0], dtype=np.float32)
            previous = np.eye(self.config.num_actions, dtype=np.float32)[self.previous_actions[index]]
            observation = np.concatenate(
                [
                    sectors,
                    candidate_obs,
                    np.clip(default_rel / 6.0, -1.0, 1.0),
                    np.clip(own_velocity / self.config.follower_max_speed, -1.0, 1.0),
                    np.clip(leader_rel / 6.0, -1.0, 1.0),
                    np.clip(leader_velocity / self.config.follower_max_speed, -1.0, 1.0),
                    np.clip(teammate_rel / 8.0, -1.0, 1.0),
                    np.clip(teammate_velocity / self.config.follower_max_speed, -1.0, 1.0),
                    role,
                    previous,
                    np.clip(goal_rel / 6.0, -1.0, 1.0),
                    np.asarray(
                        [np.clip(self.last_progress[index] / self.config.follower_max_speed, -1.0, 1.0)],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)
            if observation.shape != (83,):
                raise RuntimeError(f"unexpected observation shape: {observation.shape}")
            observations.append(observation)
            metrics_by_agent.append(metrics)
        self.last_positions = current_positions
        return np.stack(observations), metrics_by_agent

    def _update_blocked_state(self, metrics_by_agent):
        for index, metrics in enumerate(metrics_by_agent):
            if metrics[DEFAULT_ACTION]["blocked"]:
                self.blocked_latched[index] = True
                self.clear_counts[index] = 0
            elif self.blocked_latched[index]:
                self.clear_counts[index] += 1
                if self.clear_counts[index] >= self.config.default_clear_release_steps:
                    self.blocked_latched[index] = False
                    self.clear_counts[index] = 0

    def _action_masks(self, metrics_by_agent):
        indices = np.arange(self.config.num_actions, dtype=np.int64)
        masks = (
            np.abs(indices[None, :] - self.previous_actions[:, None])
            <= self.config.max_action_index_change
        )
        for index in range(2):
            if self.blocked_latched[index]:
                safe = np.asarray([not item["blocked"] for item in metrics_by_agent[index]])
                safe_adjacent = masks[index] & safe
                if np.any(safe_adjacent):
                    masks[index] = safe_adjacent
                continue
            masks[index] = False
            previous = int(self.previous_actions[index])
            if previous == DEFAULT_ACTION:
                masks[index, DEFAULT_ACTION] = True
            else:
                step = 1 if previous < DEFAULT_ACTION else -1
                masks[index, previous + step] = True
        return masks

    @staticmethod
    def _marker_color(marker, rgba):
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = rgba

    def _publish_debug(self, actions, goals, yaw, current_positions):
        self.action_publisher.publish(Int32MultiArray(data=actions.tolist()))
        message = PoseArray()
        message.header.frame_id = self.global_frame
        message.header.stamp = self.get_clock().now().to_msg()
        qz, qw = yaw_to_quaternion_components(yaw)
        for point in goals:
            pose = Pose()
            pose.position.x = float(point[0])
            pose.position.y = float(point[1])
            pose.orientation.z = qz
            pose.orientation.w = qw
            message.poses.append(pose)
        self.goal_publisher.publish(message)

        errors = np.linalg.norm(goals - current_positions, axis=1)
        self.error_publisher.publish(
            Float32MultiArray(data=errors.astype(np.float32).tolist())
        )

        markers = MarkerArray()
        colors = ((0.22, 0.55, 0.95, 1.0), (1.0, 0.55, 0.10, 1.0))
        stamp = message.header.stamp
        for index, name in enumerate(self.follower_names):
            current = Marker()
            current.header.frame_id = self.global_frame
            current.header.stamp = stamp
            current.ns = "maddpg_current"
            current.id = index
            current.type = Marker.SPHERE
            current.action = Marker.ADD
            current.pose.position.x = float(current_positions[index, 0])
            current.pose.position.y = float(current_positions[index, 1])
            current.pose.position.z = 0.20
            current.pose.orientation.w = 1.0
            current.scale.x = current.scale.y = current.scale.z = 0.38
            self._marker_color(current, colors[index])
            markers.markers.append(current)

            goal = Marker()
            goal.header.frame_id = self.global_frame
            goal.header.stamp = stamp
            goal.ns = "maddpg_goal"
            goal.id = index
            goal.type = Marker.CUBE
            goal.action = Marker.ADD
            goal.pose.position.x = float(goals[index, 0])
            goal.pose.position.y = float(goals[index, 1])
            goal.pose.position.z = 0.22
            goal.pose.orientation.z = qz
            goal.pose.orientation.w = qw
            goal.scale.x = goal.scale.y = goal.scale.z = 0.48
            self._marker_color(goal, colors[index])
            markers.markers.append(goal)

            line = Marker()
            line.header.frame_id = self.global_frame
            line.header.stamp = stamp
            line.ns = "maddpg_error_line"
            line.id = index
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.07
            self._marker_color(line, colors[index])
            line.points = [
                Point(
                    x=float(current_positions[index, 0]),
                    y=float(current_positions[index, 1]),
                    z=0.24,
                ),
                Point(
                    x=float(goals[index, 0]),
                    y=float(goals[index, 1]),
                    z=0.24,
                ),
            ]
            markers.markers.append(line)

            label = Marker()
            label.header.frame_id = self.global_frame
            label.header.stamp = stamp
            label.ns = "maddpg_error_text"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = float(
                0.5 * (current_positions[index, 0] + goals[index, 0])
            )
            label.pose.position.y = float(
                0.5 * (current_positions[index, 1] + goals[index, 1])
            )
            label.pose.position.z = 0.75 + 0.25 * index
            label.pose.orientation.w = 1.0
            label.scale.z = 0.38
            self._marker_color(label, colors[index])
            label.text = (
                f"{name}  action={int(actions[index])}  "
                f"error={float(errors[index]):.2f} m"
            )
            markers.markers.append(label)
        self.marker_publisher.publish(markers)

    def _dispatch_goals(self, goals, yaw, force=False):
        self.nav_goals.set_plan(
            SimpleNamespace(
                slots={
                    name: (float(goals[index, 0]), float(goals[index, 1]), yaw)
                    for index, name in enumerate(self.follower_names)
                },
                route_heading=yaw,
            )
        )
        self.nav_goals.dispatch_if_due(
            self.get_clock().now().nanoseconds * 1e-9, force=force
        )

    def _decision_callback(self):
        required = (self.leader_name, *self.follower_names)
        ready = self._inputs_ready()
        self._publish_controller_status(ready)
        if not ready or not self.enabled:
            self.ready_publisher.publish(Bool(data=False))
            return
        if not self.controller_active:
            self.controller_active = True
            self._publish_controller_status(True)
            self.get_logger().info(
                "Waypoint selector active; Nav2 remains cmd_vel owner"
            )
        try:
            receive_times = [
                self.samples[name].odom_received_at.nanoseconds * 1e-9
                for name in required
            ] + [
                self.samples[name].scan_received_at.nanoseconds * 1e-9
                for name in self.follower_names
            ]
            if max(receive_times) - min(receive_times) > self.max_input_skew:
                raise RuntimeError(
                    "odom/scan input skew exceeds "
                    f"{self.max_input_skew:.2f} s"
                )
            states = {name: self._global_state(name) for name in required}
            leader = states[self.leader_name]
            leader_speed = float(np.linalg.norm(leader.velocity))
            follower_speeds = [
                float(np.linalg.norm(states[name].velocity))
                for name in self.follower_names
            ]
            if leader_speed > self.config.leader_speed + self.leader_speed_tolerance:
                raise RuntimeError(
                    f"leader speed {leader_speed:.3f} m/s exceeds training range"
                )
            if any(
                speed > self.config.follower_max_speed + self.speed_tolerance
                for speed in follower_speeds
            ):
                raise RuntimeError(
                    "follower speed exceeds training range: "
                    f"{[round(speed, 3) for speed in follower_speeds]}"
                )
            candidates = candidate_points(leader.xy, leader.yaw, self.config)
            follower_positions = np.stack(
                [states[name].xy for name in self.follower_names]
            )
            if not self.formation_initialized:
                defaults = candidates[:, DEFAULT_ACTION]
                errors = np.linalg.norm(follower_positions - defaults, axis=1)
                self.previous_actions[:] = DEFAULT_ACTION
                self.current_goals = defaults.copy()
                self._publish_debug(
                    self.previous_actions, defaults, leader.yaw, follower_positions
                )
                if not self.dry_run:
                    self._dispatch_goals(defaults, leader.yaw)
                if np.all(errors <= self.initial_formation_tolerance):
                    self.formation_initialized = True
                    self.last_positions = follower_positions.copy()
                    self.get_logger().info(
                        "Initial formation aligned; MADDPG inference enabled"
                    )
                else:
                    self.get_logger().info(
                        "Waiting for Nav2 initial formation: errors=%s%s"
                        % (
                            [round(float(error), 2) for error in errors],
                            " [DRY RUN]" if self.dry_run else "",
                        )
                    )
                self.ready_publisher.publish(
                    Bool(data=self.formation_initialized)
                )
                return
            self.ready_publisher.publish(Bool(data=True))
            observations, metrics = self._observations(states, candidates)
            if observations.shape != (2, 83) or not np.isfinite(observations).all():
                raise RuntimeError("policy observations must be finite with shape (2, 83)")
            self.observation_publisher.publish(
                Float32MultiArray(data=observations.reshape(-1).tolist())
            )
            self._update_blocked_state(metrics)
            masks = self._action_masks(metrics)
            actions = self.policy.act(observations, action_masks=masks, deterministic=True)
            goals = np.stack([candidates[index, actions[index]] for index in range(2)])
            action_changed = not np.array_equal(actions, self.previous_actions)
            self.previous_actions = actions.copy()
            self.current_goals = goals.copy()
            self._publish_debug(actions, goals, leader.yaw, follower_positions)
            if not self.dry_run:
                self._dispatch_goals(goals, leader.yaw, force=action_changed)
            self.get_logger().info(
                "actions=%s blocked=%s goals=%s%s"
                % (
                    actions.tolist(),
                    self.blocked_latched.tolist(),
                    [tuple(np.round(goal, 2)) for goal in goals],
                    " [DRY RUN]" if self.dry_run else "",
                )
            )
        except Exception as error:
            self.get_logger().warning(
                f"waypoint decision skipped: {error}", throttle_duration_sec=2.0
            )

    def _set_enabled(self, request, response):
        self._set_enabled_state(bool(request.data))
        response.success = True
        response.message = "enabled" if self.enabled else "disabled"
        return response

    def stop(self):
        self.controller_active = False
        self._publish_controller_status(False)
        self.nav_goals.shutdown()
