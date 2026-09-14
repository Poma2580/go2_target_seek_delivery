"""Two-stage online MADDPG fine-tuning with Gazebo observations and Nav2.

The node intentionally reuses :class:`MaddpgWaypointSelector` for observation
construction and Nav2 goal dispatch.  Stage 1 contains no spawned obstacle so
the actor learns to tolerate Gazebo/Nav2 tracking error while holding the
default waypoint.  Stage 2 spawns one randomized static box per episode and
continues training the same policy.  Stage-1 replay is retained as rehearsal
data so obstacle fine-tuning does not immediately forget clear-lane behavior.
"""

import csv
import json
import math
import random
import re
from collections import deque
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import rclpy
import torch
from gazebo_msgs.msg import EntityState, ModelStates
from gazebo_msgs.srv import DeleteEntity, SetEntityState, SpawnEntity
from geometry_msgs.msg import Pose, Twist
from std_msgs.msg import Bool, Float32MultiArray

from .maddpg_waypoint_selector import (
    DEFAULT_ACTION,
    MaddpgWaypointSelector,
    candidate_points,
)

# Importing the selector first establishes DELIVERY_ROOT on sys.path for an
# installed ROS package, so the standalone learner modules resolve reliably.
from waypoint_maddpg_v0.geometry import segment_segment_distance  # noqa: E402
from waypoint_maddpg_v0.replay_buffer import ReplayBuffer  # noqa: E402


def obstacle_sdf(name, length, width, height):
    """Return a lidar-visible, static Gazebo box with collision geometry."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError("obstacle_name must contain only letters, digits and underscores")
    values = (float(length), float(width), float(height))
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("obstacle dimensions must be finite and positive")
    size = " ".join(f"{value:.6f}" for value in values)
    return f"""<?xml version='1.0'?>
<sdf version='1.6'>
  <model name='{name}'>
    <static>true</static>
    <link name='link'>
      <collision name='collision'>
        <geometry><box><size>{size}</size></box></geometry>
      </collision>
      <visual name='visual'>
        <geometry><box><size>{size}</size></box></geometry>
        <material>
          <ambient>0.85 0.12 0.12 1</ambient>
          <diffuse>0.90 0.18 0.18 1</diffuse>
        </material>
      </visual>
    </link>
  </model>
</sdf>"""


def oriented_box_clearance(point, center, yaw, length, width, radius):
    """Signed circle-to-oriented-box clearance in the horizontal plane."""
    delta = np.asarray(point, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.asarray([c * delta[0] + s * delta[1], -s * delta[0] + c * delta[1]])
    half = np.asarray([0.5 * length, 0.5 * width], dtype=np.float32)
    outside = np.maximum(np.abs(local) - half, 0.0)
    outside_distance = float(np.linalg.norm(outside))
    if np.all(np.abs(local) <= half):
        boundary = float(np.min(half - np.abs(local)))
        return -boundary - float(radius)
    return outside_distance - float(radius)


class GazeboMaddpgFinetuner(MaddpgWaypointSelector):
    """Waypoint selector with clear-to-one-obstacle curriculum learning."""

    def __init__(self):
        super().__init__()
        self.declare_parameter("finetune_seed", 2709)
        self.declare_parameter("run_dir", "")
        self.declare_parameter("obstacle_name", "maddpg_finetune_box")
        self.declare_parameter("obstacle_min_size", 0.8)
        self.declare_parameter("obstacle_max_size", 1.2)
        self.declare_parameter("obstacle_height", 1.0)
        self.declare_parameter("follower_max_speed", 0.20)
        self.declare_parameter("spawn_min_distance", 6.0)
        self.declare_parameter("spawn_max_distance", 9.0)
        self.declare_parameter("lane_jitter", 0.40)
        self.declare_parameter("pass_distance", 2.0)
        self.declare_parameter("clear_stage_steps", 10000)
        self.declare_parameter("one_obstacle_stage_steps", 30000)
        self.declare_parameter("clear_success_distance", 10.0)
        self.declare_parameter("episode_max_steps", 130)
        self.declare_parameter("replay_size", 50000)
        self.declare_parameter("batch_size", 128)
        self.declare_parameter("warmup_steps", 256)
        self.declare_parameter("updates_per_step", 1)
        self.declare_parameter("actor_lr", 1.0e-5)
        self.declare_parameter("critic_lr", 5.0e-5)
        self.declare_parameter("initial_epsilon", 0.10)
        self.declare_parameter("final_epsilon", 0.03)
        self.declare_parameter("epsilon_decay_steps", 30000)
        self.declare_parameter("temperature", 0.25)
        self.declare_parameter("checkpoint_interval", 5000)
        self.declare_parameter("enable_tensorboard", True)
        self.declare_parameter("leader_forward_speed", 0.15)
        self.declare_parameter("leader_command_rate", 20.0)
        self.declare_parameter("leader_heading_kp", 1.5)
        self.declare_parameter("leader_max_yaw_rate", 0.60)
        self.declare_parameter("reset_robots_each_episode", True)
        self.declare_parameter("reset_settle_seconds", 2.0)
        self.declare_parameter(
            "set_entity_state_service", "/gazebo/set_entity_state"
        )
        # All robots face +x. Keep the followers one metre farther back than
        # the original x=-13 reset positions so they start at x=-14.
        self.declare_parameter("reset_x", [-15.0, -14.0, -14.0])
        self.declare_parameter("reset_y", [4.0, 6.0, 2.0])
        self.declare_parameter("reset_z", [0.30, 0.30, 0.30])
        self.declare_parameter("reset_yaw", [0.0, 0.0, 0.0])

        self.seed = int(self.get_parameter("finetune_seed").value)
        self.rng = np.random.default_rng(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.obstacle_name = str(self.get_parameter("obstacle_name").value)
        obstacle_sdf(self.obstacle_name, 1.0, 1.0, 1.0)
        self.min_size = float(self.get_parameter("obstacle_min_size").value)
        self.max_size = float(self.get_parameter("obstacle_max_size").value)
        self.obstacle_height = float(self.get_parameter("obstacle_height").value)
        follower_max_speed = float(self.get_parameter("follower_max_speed").value)
        # Preserve every reward coefficient from the seed-27 checkpoint.  Only
        # the Gazebo motion scale and this fine-tune stage's obstacle count
        # differ from the source environment.
        self.config = replace(
            self.config,
            follower_max_speed=follower_max_speed,
            obstacle_count=1,
        )
        self.spawn_min = float(self.get_parameter("spawn_min_distance").value)
        self.spawn_max = float(self.get_parameter("spawn_max_distance").value)
        self.lane_jitter = float(self.get_parameter("lane_jitter").value)
        self.pass_distance = float(self.get_parameter("pass_distance").value)
        self.clear_stage_steps = int(
            self.get_parameter("clear_stage_steps").value
        )
        self.one_obstacle_stage_steps = int(
            self.get_parameter("one_obstacle_stage_steps").value
        )
        self.clear_success_distance = float(
            self.get_parameter("clear_success_distance").value
        )
        self.total_training_steps = (
            self.clear_stage_steps + self.one_obstacle_stage_steps
        )
        self.episode_max_steps = int(self.get_parameter("episode_max_steps").value)
        self.batch_size = int(self.get_parameter("batch_size").value)
        self.warmup_steps = int(self.get_parameter("warmup_steps").value)
        self.updates_per_step = int(self.get_parameter("updates_per_step").value)
        self.initial_epsilon = float(self.get_parameter("initial_epsilon").value)
        self.final_epsilon = float(self.get_parameter("final_epsilon").value)
        self.epsilon_decay_steps = int(self.get_parameter("epsilon_decay_steps").value)
        self.temperature = float(self.get_parameter("temperature").value)
        self.checkpoint_interval = int(self.get_parameter("checkpoint_interval").value)
        self.enable_tensorboard = bool(
            self.get_parameter("enable_tensorboard").value
        )
        self.leader_forward_speed = float(
            self.get_parameter("leader_forward_speed").value
        )
        leader_command_rate = float(self.get_parameter("leader_command_rate").value)
        self.leader_heading_kp = float(
            self.get_parameter("leader_heading_kp").value
        )
        self.leader_max_yaw_rate = float(
            self.get_parameter("leader_max_yaw_rate").value
        )
        self.reset_robots_each_episode = bool(
            self.get_parameter("reset_robots_each_episode").value
        )
        self.reset_settle_seconds = float(
            self.get_parameter("reset_settle_seconds").value
        )
        self.set_entity_state_service = str(
            self.get_parameter("set_entity_state_service").value
        )
        reset_values = {
            key: tuple(float(value) for value in self.get_parameter(key).value)
            for key in ("reset_x", "reset_y", "reset_z", "reset_yaw")
        }
        self.reset_poses = tuple(
            zip(
                reset_values["reset_x"], reset_values["reset_y"],
                reset_values["reset_z"], reset_values["reset_yaw"],
            )
        )
        self._validate_parameters()
        if leader_command_rate <= 0.0:
            raise ValueError("leader_command_rate must be positive")

        actor_lr = float(self.get_parameter("actor_lr").value)
        critic_lr = float(self.get_parameter("critic_lr").value)
        for optimizer in self.policy.actor_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = actor_lr
        for optimizer in self.policy.critic_optimizers:
            for group in optimizer.param_groups:
                group["lr"] = critic_lr
        self.replay = ReplayBuffer(
            int(self.get_parameter("replay_size").value),
            2,
            83,
            self.config.num_actions,
            self.policy.device,
        )

        requested_dir = str(self.get_parameter("run_dir").value).strip()
        if requested_dir:
            self.run_dir = Path(requested_dir).expanduser().resolve()
        else:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_dir = (
                self.model_path.parents[3] / f"gazebo_finetune_seed27_{stamp}"
                if len(self.model_path.parents) >= 4
                else Path.cwd() / f"gazebo_finetune_{stamp}"
            )
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(
                f"fine-tune run_dir must be new or empty: {self.run_dir}"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_writer = None
        if self.enable_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as error:
                raise RuntimeError(
                    "TensorBoard logging is enabled but tensorboard is not installed; "
                    "run 'python3 -m pip install tensorboard'"
                ) from error
            self.tensorboard_writer = SummaryWriter(
                log_dir=str(self.run_dir / "tensorboard")
            )
        self.recent_returns = deque(maxlen=50)
        self.recent_successes = deque(maxlen=50)
        self.recent_collisions = deque(maxlen=50)
        self.recent_pair_collisions = deque(maxlen=50)
        self.csv_path = self.run_dir / "episodes.csv"
        with self.csv_path.open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    "episode", "stage", "global_step", "stage_step", "length",
                    "return", "success", "collision", "pair_collision", "epsilon",
                    "box_length", "box_width", "blocked_lane", "actor_loss",
                    "critic_loss",
                ]
            )

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.set_state_client = self.create_client(
            SetEntityState, self.set_entity_state_service
        )
        self.leader_command_publisher = self.create_publisher(
            Twist, f"/{self.leader_name}/cmd_vel", 10
        )
        self.leader_command_timer = self.create_timer(
            1.0 / leader_command_rate, self._publish_leader_command
        )
        self.create_subscription(ModelStates, "/gazebo/model_states", self._models_callback, 10)
        self.models = {}
        self.obstacle = None
        self.entity_request_pending = False
        self.episode_finishing = False
        self.pending_transition = None
        self.global_step = 0
        self.episode = 0
        self.episode_step = 0
        self.episode_return = 0.0
        self.recovery_count = 0
        self.best_success_return = {
            "clear": -math.inf,
            "one_obstacle": -math.inf,
        }
        self.last_actor_loss = math.nan
        self.last_critic_loss = math.nan
        self.route_directions = np.zeros(2, dtype=np.int64)
        self.previous_previous_actions = np.full(2, DEFAULT_ACTION, dtype=np.int64)
        self.episode_reset_required = self.reset_robots_each_episode
        self.reset_requests_sent = False
        self.reset_requests_pending = 0
        self.reset_requests_succeeded = True
        self.reset_complete_at = None
        self.training_complete = False
        self._write_manifest(actor_lr, critic_lr)
        self.get_logger().info(
            f"Gazebo curriculum ready: output={self.run_dir}, "
            f"stage1=clear/{self.clear_stage_steps} steps, "
            f"stage2=one_box/{self.one_obstacle_stage_steps} steps, "
            f"box={self.min_size:.2f}-{self.max_size:.2f} m, "
            f"epsilon={self.initial_epsilon:.2f}->{self.final_epsilon:.2f}"
        )

    def _validate_parameters(self):
        if not (0.0 < self.min_size <= self.max_size):
            raise ValueError("obstacle sizes must satisfy 0 < min <= max")
        if self.obstacle_height <= 0.0:
            raise ValueError("obstacle_height must be positive")
        if self.config.follower_max_speed <= 0.0:
            raise ValueError("follower_max_speed must be positive")
        if not (0.0 < self.spawn_min <= self.spawn_max):
            raise ValueError("spawn distances must satisfy 0 < min <= max")
        if self.lane_jitter < 0.0 or self.pass_distance < 0.0:
            raise ValueError("lane_jitter and pass_distance must be nonnegative")
        if self.clear_stage_steps < 1 or self.one_obstacle_stage_steps < 1:
            raise ValueError("both curriculum stage step counts must be positive")
        if not math.isfinite(self.clear_success_distance) or self.clear_success_distance <= 0.0:
            raise ValueError("clear_success_distance must be finite and positive")
        if min(self.episode_max_steps, self.batch_size, self.warmup_steps, self.updates_per_step) < 1:
            raise ValueError("step, batch, warmup and update counts must be positive")
        if not (0.0 <= self.final_epsilon <= self.initial_epsilon <= 1.0):
            raise ValueError("epsilon must satisfy 0 <= final <= initial <= 1")
        if self.epsilon_decay_steps < 1 or self.checkpoint_interval < 1:
            raise ValueError("decay and checkpoint intervals must be positive")
        if not math.isfinite(self.leader_forward_speed) or self.leader_forward_speed <= 0.0:
            raise ValueError("leader_forward_speed must be finite and positive")
        if not math.isfinite(self.leader_heading_kp) or self.leader_heading_kp <= 0.0:
            raise ValueError("leader_heading_kp must be finite and positive")
        if not math.isfinite(self.leader_max_yaw_rate) or self.leader_max_yaw_rate <= 0.0:
            raise ValueError("leader_max_yaw_rate must be finite and positive")
        if self.reset_settle_seconds < 0.0:
            raise ValueError("reset_settle_seconds must be nonnegative")
        if len(self.reset_poses) != 3 or any(len(values) != 4 for values in self.reset_poses):
            raise ValueError("reset pose arrays must each contain exactly three values")
        if not all(math.isfinite(value) for pose in self.reset_poses for value in pose):
            raise ValueError("reset poses must be finite")

    def _write_manifest(self, actor_lr, critic_lr):
        data = {
            "initialized_from": str(self.model_path),
            "seed": self.seed,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "follower_max_speed": self.config.follower_max_speed,
            "leader_forward_speed": self.leader_forward_speed,
            "leader_heading_kp": self.leader_heading_kp,
            "leader_max_yaw_rate": self.leader_max_yaw_rate,
            "reset_robots_each_episode": self.reset_robots_each_episode,
            "tensorboard": self.enable_tensorboard,
            "reset_poses": {
                name: list(pose)
                for name, pose in zip(
                    (self.leader_name, *self.follower_names), self.reset_poses
                )
            },
            "reward_source": "seed27 checkpoint environment metadata",
            "curriculum": {
                "stage1": {
                    "name": "clear",
                    "steps": self.clear_stage_steps,
                    "success_distance": self.clear_success_distance,
                },
                "stage2": {
                    "name": "one_obstacle",
                    "steps": self.one_obstacle_stage_steps,
                },
                "total_steps": self.total_training_steps,
                "retain_stage1_replay": True,
            },
            "obstacle": {
                "shape": "box",
                "length_range": [self.min_size, self.max_size],
                "width_range": [self.min_size, self.max_size],
                "height": self.obstacle_height,
                "spawn_distance_range": [self.spawn_min, self.spawn_max],
                "lane_jitter": self.lane_jitter,
            },
        }
        (self.run_dir / "config.json").write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _models_callback(self, message):
        self.models = {
            name: (pose, twist)
            for name, pose, twist in zip(message.name, message.pose, message.twist)
        }

    def _publish_leader_command(self):
        command = Twist()
        if (
            self.enabled
            and not self.episode_reset_required
            and self.obstacle is not None
            and self.obstacle.get("confirmed", False)
        ):
            leader_model = self.models.get(self.leader_name)
            if leader_model is not None:
                pose, _ = leader_model
                target_yaw = self.reset_poses[0][3]
                yaw_error = math.atan2(
                    math.sin(target_yaw - self._yaw(pose)),
                    math.cos(target_yaw - self._yaw(pose)),
                )
                command.angular.z = float(np.clip(
                    self.leader_heading_kp * yaw_error,
                    -self.leader_max_yaw_rate,
                    self.leader_max_yaw_rate,
                ))
                # Reduce forward motion while correcting a large disturbance;
                # at the reset heading this remains exactly the configured speed.
                command.linear.x = self.leader_forward_speed * max(
                    0.0, math.cos(yaw_error)
                )
        self.leader_command_publisher.publish(command)

    def _begin_robot_reset(self):
        if not self.reset_robots_each_episode:
            return
        self.episode_reset_required = True
        self.reset_requests_sent = False
        self.reset_requests_pending = 0
        self.reset_requests_succeeded = True
        self.reset_complete_at = None
        self.nav_goals.suspend("Gazebo episode reset")
        self.leader_command_publisher.publish(Twist())

    def _advance_robot_reset(self):
        if not self.episode_reset_required:
            return True
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.reset_complete_at is not None:
            if now < self.reset_complete_at:
                return False
            self.episode_reset_required = False
            self.reset_complete_at = None
            self.nav_goals.resume()
            self.get_logger().info("Robot reset settled; starting the next episode")
            return True
        if self.reset_requests_sent:
            return False
        if not self.set_state_client.service_is_ready():
            self.get_logger().warning(
                f"Waiting for Gazebo {self.set_entity_state_service}",
                throttle_duration_sec=5.0,
            )
            return False
        self.reset_requests_sent = True
        self.reset_requests_pending = 3
        self.reset_requests_succeeded = True
        for name, (x, y, z, yaw) in zip(
            (self.leader_name, *self.follower_names), self.reset_poses
        ):
            request = SetEntityState.Request()
            request.state = EntityState()
            request.state.name = name
            request.state.reference_frame = "world"
            request.state.pose.position.x = x
            request.state.pose.position.y = y
            request.state.pose.position.z = z
            request.state.pose.orientation.z = math.sin(0.5 * yaw)
            request.state.pose.orientation.w = math.cos(0.5 * yaw)
            request.state.twist = Twist()
            future = self.set_state_client.call_async(request)
            future.add_done_callback(
                lambda completed, robot=name: self._robot_reset_done(robot, completed)
            )
        return False

    def _robot_reset_done(self, name, future):
        succeeded = False
        try:
            response = future.result()
            succeeded = bool(response.success)
            if not succeeded:
                self.get_logger().error(f"Gazebo rejected reset for {name}")
        except Exception as error:
            self.get_logger().error(f"Failed to reset {name}: {error}")
        self.reset_requests_succeeded &= succeeded
        self.reset_requests_pending -= 1
        if self.reset_requests_pending:
            return
        if not self.reset_requests_succeeded:
            self.reset_requests_sent = False
            return
        self._reset_policy_state()
        self.reset_complete_at = (
            self.get_clock().now().nanoseconds * 1e-9 + self.reset_settle_seconds
        )

    @staticmethod
    def _yaw(pose):
        q = pose.orientation
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def _curriculum_stage(self):
        if self.global_step < self.clear_stage_steps:
            return "clear"
        if self.global_step < self.total_training_steps:
            return "one_obstacle"
        return "complete"

    def _stage_start_step(self, stage=None):
        stage = stage or self._curriculum_stage()
        return 0 if stage == "clear" else self.clear_stage_steps

    def _stage_budget(self, stage=None):
        stage = stage or self._curriculum_stage()
        if stage == "clear":
            return self.clear_stage_steps
        if stage == "one_obstacle":
            return self.one_obstacle_stage_steps
        return 0

    def _stage_step(self, stage=None):
        stage = stage or self._curriculum_stage()
        return max(0, self.global_step - self._stage_start_step(stage))

    def _epsilon(self, stage=None):
        stage = stage or self._curriculum_stage()
        decay_steps = min(self.epsilon_decay_steps, self._stage_budget(stage))
        fraction = min(1.0, self._stage_step(stage) / max(decay_steps, 1))
        return self.initial_epsilon + fraction * (self.final_epsilon - self.initial_epsilon)

    def _start_clear_episode(self, leader_pose):
        """Start an episode without creating any Gazebo entity."""
        yaw = self._yaw(leader_pose)
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        center = np.asarray(
            [leader_pose.position.x, leader_pose.position.y], dtype=np.float32
        )
        self.obstacle = {
            "clear": True,
            "confirmed": True,
            "center": center,
            "forward": forward,
            "yaw": yaw,
            "length": 0.0,
            "width": 0.0,
            "blocked_agent": -1,
        }
        self.episode += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.recovery_count = 0
        self.pending_transition = None
        self.route_directions[:] = 0
        self.previous_previous_actions[:] = DEFAULT_ACTION
        self._reset_policy_state()
        self.get_logger().info(
            "Episode %d stage=clear start=(%.2f, %.2f), target_distance=%.2f m"
            % (
                self.episode,
                center[0],
                center[1],
                self.clear_success_distance,
            )
        )

    def _request_spawn(self):
        if self.entity_request_pending or self.obstacle is not None or not self.role_received:
            return
        leader_model = self.models.get(self.leader_name)
        if leader_model is None or not self.spawn_client.service_is_ready():
            self.get_logger().warning(
                "Waiting for Gazebo model state and /spawn_entity",
                throttle_duration_sec=5.0,
            )
            return
        pose, _ = leader_model
        yaw = self._yaw(pose)
        forward = np.asarray([math.cos(yaw), math.sin(yaw)], dtype=np.float32)
        left = np.asarray([-forward[1], forward[0]], dtype=np.float32)
        distance = float(self.rng.uniform(self.spawn_min, self.spawn_max))
        blocked_agent = int(self.rng.integers(0, 2))
        lane = self.config.formation_side if blocked_agent == 0 else -self.config.formation_side
        lateral = lane + float(self.rng.uniform(-self.lane_jitter, self.lane_jitter))
        center = np.asarray([pose.position.x, pose.position.y], dtype=np.float32)
        center = center + distance * forward + lateral * left
        length = float(self.rng.uniform(self.min_size, self.max_size))
        width = float(self.rng.uniform(self.min_size, self.max_size))
        request = SpawnEntity.Request()
        request.name = self.obstacle_name
        request.xml = obstacle_sdf(
            self.obstacle_name, length, width, self.obstacle_height
        )
        request.reference_frame = "world"
        request.initial_pose = Pose()
        request.initial_pose.position.x = float(center[0])
        request.initial_pose.position.y = float(center[1])
        request.initial_pose.position.z = 0.5 * self.obstacle_height
        request.initial_pose.orientation.z = math.sin(0.5 * yaw)
        request.initial_pose.orientation.w = math.cos(0.5 * yaw)
        self.obstacle = {
            "clear": False,
            "center": center,
            "yaw": yaw,
            "forward": forward,
            "length": length,
            "width": width,
            "blocked_agent": blocked_agent,
            "confirmed": False,
        }
        self.entity_request_pending = True
        future = self.spawn_client.call_async(request)
        future.add_done_callback(self._spawn_done)

    def _spawn_done(self, future):
        self.entity_request_pending = False
        try:
            response = future.result()
        except Exception as error:
            self.get_logger().error(f"Obstacle spawn service failed: {error}")
            self.obstacle = None
            return
        if not response.success:
            self.get_logger().warning(
                f"Obstacle spawn rejected: {response.status_message}; cleaning stale entity"
            )
            self.obstacle = None
            self._request_delete()
            return
        self.obstacle["confirmed"] = True
        self.episode += 1
        self.episode_step = 0
        self.episode_return = 0.0
        self.recovery_count = 0
        self.pending_transition = None
        self.route_directions[:] = 0
        self.previous_previous_actions[:] = DEFAULT_ACTION
        self._reset_policy_state()
        self.get_logger().info(
            "Episode %d stage=one_obstacle box %.2fx%.2f m at (%.2f, %.2f), blocking %s"
            % (
                self.episode,
                self.obstacle["length"],
                self.obstacle["width"],
                self.obstacle["center"][0],
                self.obstacle["center"][1],
                self.follower_names[self.obstacle["blocked_agent"]],
            )
        )

    def _request_delete(self):
        if self.obstacle is not None and self.obstacle.get("clear", False):
            self.obstacle = None
            self.episode_finishing = False
            if not self.training_complete:
                self._begin_robot_reset()
            return
        if self.entity_request_pending or not self.delete_client.service_is_ready():
            return
        request = DeleteEntity.Request()
        request.name = self.obstacle_name
        self.entity_request_pending = True
        future = self.delete_client.call_async(request)
        future.add_done_callback(self._delete_done)

    def _delete_done(self, future):
        self.entity_request_pending = False
        try:
            response = future.result()
            if not response.success:
                self.get_logger().warning(f"Obstacle delete: {response.status_message}")
        except Exception as error:
            self.get_logger().warning(f"Obstacle delete service failed: {error}")
        self.obstacle = None
        self.episode_finishing = False
        if not self.training_complete:
            self._begin_robot_reset()

    @staticmethod
    def _danger(clearance, safe_distance):
        ratio = np.clip((safe_distance - clearance) / max(safe_distance, 1e-6), 0.0, 1.0)
        return float(ratio * ratio)

    def _formation_term(self, actions, action_before, metrics, blocked_latched):
        offsets = np.asarray(self.config.candidate_offsets, dtype=np.float32)
        weights = np.where(
            blocked_latched,
            self.config.blocked_default_offset_weight,
            self.config.clear_default_offset_weight,
        )
        values = -weights * np.abs(offsets[actions])
        switched = actions != action_before
        values -= self.config.formation_switch_weight * switched.astype(np.float32)
        oscillated = switched & (actions == self.previous_previous_actions)
        values -= self.config.formation_oscillation_weight * oscillated.astype(np.float32)
        for index in range(2):
            direction = int(np.sign(offsets[actions[index]]))
            if metrics[index][DEFAULT_ACTION]["blocked"] and direction:
                if self.route_directions[index] and direction != self.route_directions[index]:
                    values[index] -= self.config.formation_route_reversal_weight
                self.route_directions[index] = direction
            elif actions[index] == DEFAULT_ACTION:
                self.route_directions[index] = 0
        return float(np.mean(values))

    def _gazebo_positions(self):
        result = []
        for name in self.follower_names:
            model = self.models.get(name)
            if model is None:
                return None
            pose, _ = model
            result.append([pose.position.x, pose.position.y])
        return np.asarray(result, dtype=np.float32)

    def _transition_reward(self, current_states, current_candidates):
        transition = self.pending_transition
        gazebo_positions = self._gazebo_positions()
        if gazebo_positions is None:
            gazebo_positions = np.stack(
                [current_states[name].xy for name in self.follower_names]
            )
        clear_episode = bool(self.obstacle.get("clear", False))
        if clear_episode:
            obstacle_clearances = np.full(2, np.inf, dtype=np.float32)
        else:
            obstacle_clearances = np.asarray(
                [
                    oriented_box_clearance(
                        point,
                        self.obstacle["center"],
                        self.obstacle["yaw"],
                        self.obstacle["length"],
                        self.obstacle["width"],
                        self.config.robot_radius,
                    )
                    for point in gazebo_positions
                ],
                dtype=np.float32,
            )
        obstacle_collision = bool(np.any(obstacle_clearances <= 0.0))
        pair_distance = float(np.linalg.norm(gazebo_positions[0] - gazebo_positions[1]))
        pair_collision = pair_distance <= self.config.pair_collision_distance
        obstacle_values = np.zeros(2, dtype=np.float32)
        for index in range(2):
            obstacle_values[index] -= self.config.obstacle_clearance_weight * self._danger(
                float(obstacle_clearances[index]), self.config.obstacle_safe_clearance
            )
            selected = transition["metrics"][index][transition["actions"][index]]
            obstacle_values[index] -= self.config.selected_path_weight * self._danger(
                selected["path_clearance"], self.config.selected_path_safe_clearance
            )
            if selected["blocked"]:
                obstacle_values[index] -= self.config.blocked_path_penalty
        obstacle_reward = float(np.mean(obstacle_values))
        if obstacle_collision:
            obstacle_reward -= self.config.collision_penalty
        path_pair_distance = segment_segment_distance(
            transition["positions"][0], transition["goals"][0],
            transition["positions"][1], transition["goals"][1],
        )
        pair_reward = -self.config.pair_clearance_weight * self._danger(
            pair_distance, self.config.pair_safe_distance
        )
        pair_reward -= self.config.pair_path_weight * self._danger(
            path_pair_distance, self.config.pair_path_safe_distance
        )
        if pair_collision and not obstacle_collision:
            pair_reward -= self.config.collision_penalty
        current_positions = np.stack(
            [current_states[name].xy for name in self.follower_names]
        )
        forward = transition["leader_forward"]
        progress = (current_positions - transition["positions"]) @ forward
        progress_reward = self.config.progress_weight * float(
            np.mean(np.clip(progress / self.config.follower_max_speed, -1.0, 1.0))
        )
        default_errors = np.linalg.norm(current_positions - current_candidates[:, DEFAULT_ACTION], axis=1)
        leader_model = self.models.get(self.leader_name)
        passed = False
        if leader_model is not None:
            leader_pose, _ = leader_model
            leader_xy = np.asarray([leader_pose.position.x, leader_pose.position.y])
            if clear_episode:
                progress = float(
                    np.dot(
                        leader_xy - self.obstacle["center"],
                        self.obstacle["forward"],
                    )
                )
                passed = progress >= self.clear_success_distance
            else:
                relative = float(
                    np.dot(
                        self.obstacle["center"] - leader_xy,
                        self.obstacle["forward"],
                    )
                )
                passed = relative <= -self.pass_distance
        holding_default = bool(
            np.all(transition["actions"] == DEFAULT_ACTION)
        )
        if (
            passed
            and holding_default
            and np.all(default_errors < self.config.recovery_error_tolerance)
        ):
            self.recovery_count += 1
        else:
            self.recovery_count = 0
        success = self.recovery_count >= self.config.recovery_hold_steps
        timeout = self.episode_step >= self.episode_max_steps
        stage_boundary = (
            self.global_step + 1
            >= self._stage_start_step(transition["stage"])
            + self._stage_budget(transition["stage"])
        )
        done = bool(
            success
            or obstacle_collision
            or pair_collision
            or timeout
            or stage_boundary
        )
        reward = (
            -self.config.time_penalty
            + obstacle_reward
            + pair_reward
            + transition["formation_reward"]
            + progress_reward
        )
        if success:
            reward += self.config.success_bonus
        return float(reward), done, success, obstacle_collision, pair_collision

    def _store_and_update(self, next_observations, next_masks, reward, done):
        transition = self.pending_transition
        one_hot = np.eye(self.config.num_actions, dtype=np.float32)[transition["actions"]]
        rewards = np.full(2, reward, dtype=np.float32)
        self.replay.add(
            transition["observations"], one_hot, transition["masks"], rewards,
            next_observations, next_masks, done,
        )
        self.global_step += 1
        self.episode_return += reward
        if len(self.replay) >= max(self.batch_size, self.warmup_steps):
            for _ in range(self.updates_per_step):
                losses = self.policy.update(self.replay.sample(self.batch_size), self.temperature)
                self.last_actor_loss = losses["actor_loss"]
                self.last_critic_loss = losses["critic_loss"]
                if self.tensorboard_writer is not None:
                    self.tensorboard_writer.add_scalar(
                        "loss/actor", self.last_actor_loss, self.global_step
                    )
                    self.tensorboard_writer.add_scalar(
                        "loss/critic", self.last_critic_loss, self.global_step
                    )
        if self.global_step % self.checkpoint_interval == 0:
            self._save_checkpoint("latest_model.pt")

    def _save_checkpoint(self, name, stage=None):
        stage = stage or self._curriculum_stage()
        self.policy.save(
            self.run_dir / name,
            metadata={
                "global_step": self.global_step,
                "environment": self.config.to_dict(),
                "initialized_from": str(self.model_path),
                "gazebo_finetune": True,
                "episode": self.episode,
                "curriculum_stage": stage,
                "clear_stage_steps": self.clear_stage_steps,
                "one_obstacle_stage_steps": self.one_obstacle_stage_steps,
            },
        )

    def _finish_episode(self, success, collision, pair_collision):
        obstacle = self.obstacle
        stage = "clear" if obstacle.get("clear", False) else "one_obstacle"
        epsilon = self._epsilon(stage)
        blocked_lane = (
            "none"
            if stage == "clear"
            else self.follower_names[obstacle["blocked_agent"]]
        )
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(
                [
                    self.episode, stage, self.global_step,
                    self._stage_step(stage), self.episode_step,
                    f"{self.episode_return:.6f}", int(success), int(collision),
                    int(pair_collision), f"{epsilon:.6f}",
                    f"{obstacle['length']:.4f}", f"{obstacle['width']:.4f}",
                    blocked_lane,
                    f"{self.last_actor_loss:.6f}", f"{self.last_critic_loss:.6f}",
                ]
            )
        self.recent_returns.append(self.episode_return)
        self.recent_successes.append(float(success))
        self.recent_collisions.append(float(collision))
        self.recent_pair_collisions.append(float(pair_collision))
        if self.tensorboard_writer is not None:
            episode_metrics = {
                "train/episode_reward": self.episode_return,
                "train/episode_length": self.episode_step,
                "train/reward_moving_average_50": float(np.mean(self.recent_returns)),
                "train/success": float(success),
                "train/success_rate_50": float(np.mean(self.recent_successes)),
                "train/collision": float(collision),
                "train/collision_rate_50": float(np.mean(self.recent_collisions)),
                "train/pair_collision": float(pair_collision),
                "train/pair_collision_rate_50": float(
                    np.mean(self.recent_pair_collisions)
                ),
                "exploration/epsilon": epsilon,
                "exploration/temperature": self.temperature,
                "curriculum/stage": 0.0 if stage == "clear" else 1.0,
                "curriculum/stage_step": float(self._stage_step(stage)),
                "obstacle/length": obstacle["length"],
                "obstacle/width": obstacle["width"],
                "replay/size": len(self.replay),
            }
            for tag, value in episode_metrics.items():
                self.tensorboard_writer.add_scalar(
                    f"{stage}/{tag}", value, self.global_step
                )
            self.tensorboard_writer.flush()
        self._save_checkpoint("latest_model.pt", stage=stage)
        if success and self.episode_return > self.best_success_return[stage]:
            self.best_success_return[stage] = self.episode_return
            self._save_checkpoint(f"{stage}_best_model.pt", stage=stage)
            if stage == "one_obstacle":
                self._save_checkpoint("best_model.pt", stage=stage)
        stage_finished = (
            self.global_step
            >= self._stage_start_step(stage) + self._stage_budget(stage)
        )
        if stage_finished and stage == "clear":
            self._save_checkpoint("stage1_clear_model.pt", stage=stage)
            self.get_logger().info(
                "Stage 1 complete at %d steps; keeping replay and switching to one obstacle"
                % self.global_step
            )
        elif stage_finished and stage == "one_obstacle":
            self._save_checkpoint(
                "stage2_one_obstacle_final_model.pt", stage=stage
            )
            self.training_complete = True
            self.nav_goals.suspend("Gazebo curriculum complete")
            self.leader_command_publisher.publish(Twist())
            self.get_logger().info(
                "Gazebo curriculum complete at %d steps; final checkpoint=%s"
                % (
                    self.global_step,
                    self.run_dir / "stage2_one_obstacle_final_model.pt",
                )
            )
        self.get_logger().info(
            "Episode %d stage=%s done: success=%s collision=%s pair_collision=%s "
            "steps=%d return=%.2f replay=%d"
            % (
                self.episode, stage, success, collision, pair_collision,
                self.episode_step, self.episode_return, len(self.replay),
            )
        )
        self.pending_transition = None
        self._reset_policy_state()
        self.episode_finishing = True
        self._request_delete()

    def _decision_callback(self):
        required = (self.leader_name, *self.follower_names)
        ready = self._inputs_ready()
        self._publish_controller_status(ready)
        if not ready or not self.enabled:
            self.ready_publisher.publish(Bool(data=False))
            return
        try:
            if self.episode_finishing:
                self._request_delete()
                return
            if self.training_complete:
                self.ready_publisher.publish(Bool(data=False))
                return
            receive_times = [
                self.samples[name].odom_received_at.nanoseconds * 1e-9
                for name in required
            ] + [
                self.samples[name].scan_received_at.nanoseconds * 1e-9
                for name in self.follower_names
            ]
            if max(receive_times) - min(receive_times) > self.max_input_skew:
                raise RuntimeError(
                    f"odom/scan input skew exceeds {self.max_input_skew:.2f} s"
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
            follower_positions = np.stack([states[name].xy for name in self.follower_names])
            if self.obstacle is None or not self.obstacle.get("confirmed", False):
                if not self._advance_robot_reset():
                    return
                stage = self._curriculum_stage()
                if stage == "complete":
                    self.training_complete = True
                    self.nav_goals.suspend("Gazebo curriculum complete")
                    self.leader_command_publisher.publish(Twist())
                    return
                if stage == "clear":
                    leader_model = self.models.get(self.leader_name)
                    if leader_model is None:
                        return
                    self._start_clear_episode(leader_model[0])
                else:
                    self._request_spawn()
                defaults = candidates[:, DEFAULT_ACTION]
                self.previous_actions[:] = DEFAULT_ACTION
                self.current_goals = defaults.copy()
                self._publish_debug(self.previous_actions, defaults, leader.yaw, follower_positions)
                if not self.dry_run:
                    self._dispatch_goals(defaults, leader.yaw)
                return
            if not self.controller_active:
                self.controller_active = True
                self._publish_controller_status(True)
            observations, metrics = self._observations(states, candidates)
            if observations.shape != (2, 83) or not np.isfinite(observations).all():
                raise RuntimeError("policy observations must be finite with shape (2,83)")
            self.observation_publisher.publish(
                Float32MultiArray(data=observations.reshape(-1).tolist())
            )
            self._update_blocked_state(metrics)
            masks = self._action_masks(metrics)
            if self.pending_transition is not None:
                reward, done, success, collision, pair_collision = self._transition_reward(
                    states, candidates
                )
                self._store_and_update(observations, masks, reward, done)
                if done:
                    self._finish_episode(success, collision, pair_collision)
                    return
            epsilon = self._epsilon()
            actions = self.policy.act(
                observations, action_masks=masks, epsilon=epsilon, deterministic=False
            )
            goals = np.stack([candidates[index, actions[index]] for index in range(2)])
            action_before = self.previous_actions.copy()
            formation_reward = self._formation_term(
                actions, action_before, metrics, self.blocked_latched.copy()
            )
            self._publish_decision_diagnostics(
                observations, metrics, masks, actions, action_before
            )
            self.previous_previous_actions = action_before.copy()
            self.previous_actions = actions.copy()
            self.current_goals = goals.copy()
            self.episode_step += 1
            self.pending_transition = {
                "stage": self._curriculum_stage(),
                "observations": observations.copy(),
                "masks": masks.copy(),
                "actions": actions.copy(),
                "metrics": metrics,
                "positions": follower_positions.copy(),
                "goals": goals.copy(),
                "leader_forward": np.asarray(
                    [math.cos(leader.yaw), math.sin(leader.yaw)], dtype=np.float32
                ),
                "formation_reward": formation_reward,
            }
            self._publish_debug(actions, goals, leader.yaw, follower_positions)
            if not self.dry_run:
                self._dispatch_goals(goals, leader.yaw)
            self.get_logger().info(
                "finetune stage=%s episode=%d step=%d total=%d actions=%s epsilon=%.3f"
                % (
                    self.pending_transition["stage"], self.episode,
                    self.episode_step, self.global_step, actions.tolist(), epsilon,
                )
            )
        except Exception as error:
            self.get_logger().warning(
                f"Gazebo fine-tune decision skipped: {error}", throttle_duration_sec=2.0
            )

    def stop(self):
        self.leader_command_publisher.publish(Twist())
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
        if self.global_step:
            self._save_checkpoint("latest_model.pt")
        if not rclpy.ok():
            return
        if (
            self.obstacle is not None
            and self.delete_client.service_is_ready()
        ):
            try:
                request = DeleteEntity.Request()
                request.name = self.obstacle_name
                self.delete_client.call_async(request)
            except Exception as error:
                self.get_logger().warning(f"Final obstacle cleanup failed: {error}")
        super().stop()
