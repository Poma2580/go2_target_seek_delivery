#!/usr/bin/env python3
"""MADDPG controller for three real Go2 robots with one virtual agent.

The trained MADDPG policy expects four agents. This node maps agent_0, agent_1
and agent_3 to go2_1, go2_2 and go2_3, while agent_2 is kept as a virtual
follower slot. Only the three real robots receive cmd_vel commands.
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


REPO_ROOT = Path(os.environ.get("DELIVERY_ROOT", Path(__file__).resolve().parents[4])).resolve()
DEFAULT_MADDPG_ROOT = REPO_ROOT / "三角形MADDPG"
DEFAULT_MODEL_PATH = (
    DEFAULT_MADDPG_ROOT
    / "runs"
    / "stage4_b512_usteps20_g0.99_t0.005_alr5e-05_clr0.0005_n0.14_minn0.02_h128,128_20260430_132728"
    / "best_model.pt"
)

FOLLOWER_OFFSETS = {
    "agent_1": np.array([-0.60, 0.65], dtype=np.float32),
    "agent_2": np.array([-0.60, 0], dtype=np.float32),
    "agent_3": np.array([-0.60, -0.65], dtype=np.float32),
}

REAL_AGENT_MAP = {
    0: "go2_1",
    1: "go2_2",
    3: "go2_3",
}


@dataclass
class EntityState:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    received: bool = False
    receive_time = None


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def normalize_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


class MarlThreeRealController(Node):
    def __init__(self):
        super().__init__("marl_three_real_controller")

        self.declare_parameter("maddpg_root", str(DEFAULT_MADDPG_ROOT))
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("target_odom_topic", "/walking_target/odom")
        self.declare_parameter("control_rate", 5.0)
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("target_timeout", 2.0)
        self.declare_parameter("max_linear", 0.35)
        self.declare_parameter("max_angular", 0.6)
        self.declare_parameter("heading_deadband", 0.15)
        self.declare_parameter("position_scale", 15.0)
        self.declare_parameter("formation_scale", 1.0)
        self.declare_parameter("leader_visual_servo", True)
        self.declare_parameter("leader_linear", 0.20)
        self.declare_parameter("leader_follow_distance", 2.0)
        self.declare_parameter("leader_distance_deadband", 0.25)
        self.declare_parameter("leader_k_linear", 0.8)
        self.declare_parameter("leader_k_angular", 0.8)
        self.declare_parameter("slot_attraction_gain", 0.25)
        self.declare_parameter("slot_attraction_max", 0.8)
        self.declare_parameter("slot_deadband", 0.25)

        self.maddpg_root = Path(self.get_parameter("maddpg_root").value).expanduser().resolve()
        self.model_path = Path(self.get_parameter("model_path").value).expanduser().resolve()
        self.target_odom_topic = self.get_parameter("target_odom_topic").value
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.heading_deadband = float(self.get_parameter("heading_deadband").value)
        self.position_scale = max(float(self.get_parameter("position_scale").value), 1e-6)
        self.formation_scale = max(float(self.get_parameter("formation_scale").value), 1e-6)
        self.leader_visual_servo = bool(self.get_parameter("leader_visual_servo").value)
        self.leader_linear = float(self.get_parameter("leader_linear").value)
        self.leader_follow_distance = float(self.get_parameter("leader_follow_distance").value)
        self.leader_distance_deadband = float(self.get_parameter("leader_distance_deadband").value)
        self.leader_k_linear = float(self.get_parameter("leader_k_linear").value)
        self.leader_k_angular = float(self.get_parameter("leader_k_angular").value)
        self.slot_attraction_gain = float(self.get_parameter("slot_attraction_gain").value)
        self.slot_attraction_max = max(float(self.get_parameter("slot_attraction_max").value), 0.0)
        self.slot_deadband = max(float(self.get_parameter("slot_deadband").value), 0.0)

        self.real_agent_names = ("go2_1", "go2_2", "go2_3")
        self.states = {name: EntityState() for name in self.real_agent_names}
        self.target = EntityState()
        self.cmd_pubs = {
            name: self.create_publisher(Twist, f"/{name}/cmd_vel", 10)
            for name in self.real_agent_names
        }

        for name in self.real_agent_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot_name=name: self._odom_cb(robot_name, msg),
                10,
            )
        self.create_subscription(Odometry, self.target_odom_topic, self._target_cb, 10)

        self.maddpg = self._load_model()
        rate = float(self.get_parameter("control_rate").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-6), self._timer_cb)
        self.get_logger().info(
            "marl_three_real_controller started: "
            f"target={self.target_odom_topic}, max_linear={self.max_linear:.2f}, "
            f"max_angular={self.max_angular:.2f}, position_scale={self.position_scale:.2f}, "
            f"formation_scale={self.formation_scale:.2f}, "
            f"leader_visual_servo={self.leader_visual_servo}, "
            f"slot_attraction_gain={self.slot_attraction_gain:.2f}, "
            f"slot_attraction_max={self.slot_attraction_max:.2f}"
        )

    def _load_model(self):
        if not self.maddpg_root.exists():
            raise FileNotFoundError(f"MADDPG root not found: {self.maddpg_root}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"MADDPG model not found: {self.model_path}")

        sys.path.insert(0, str(self.maddpg_root))
        from maddpg import MADDPG

        maddpg = MADDPG(
            state_sizes=[23, 23, 23, 23],
            action_sizes=[2, 2, 2, 2],
            hidden_sizes=(128, 128),
            action_low=-1.0,
            action_high=1.0,
        )
        maddpg.load(str(self.model_path))
        return maddpg

    def _now_age(self, receive_time):
        if receive_time is None:
            return float("inf")
        return (self.get_clock().now() - receive_time).nanoseconds * 1e-9

    def _odom_cb(self, name, msg):
        state = self.states[name]
        state.x = msg.pose.pose.position.x
        state.y = msg.pose.pose.position.y
        state.yaw = quaternion_to_yaw(msg.pose.pose.orientation)
        state.vx = msg.twist.twist.linear.x
        state.vy = msg.twist.twist.linear.y
        state.received = True
        state.receive_time = self.get_clock().now()

    def _target_cb(self, msg):
        self.target.x = msg.pose.pose.position.x
        self.target.y = msg.pose.pose.position.y
        self.target.vx = msg.twist.twist.linear.x
        self.target.vy = msg.twist.twist.linear.y
        self.target.received = True
        self.target.receive_time = self.get_clock().now()

    def _data_ready(self):
        missing = []
        for name in self.real_agent_names:
            state = self.states[name]
            if not state.received or self._now_age(state.receive_time) > self.odom_timeout:
                missing.append(name)

        if not self.target.received or self._now_age(self.target.receive_time) > self.target_timeout:
            missing.append("target")

        if missing:
            self._publish_all_zero()
            self.get_logger().info(f"waiting for fresh odom: {', '.join(missing)}")
            return False
        return True

    def _agent_arrays(self):
        leader = self.states["go2_1"]
        virtual_pos = (
            np.array([leader.x, leader.y], dtype=np.float32)
            + FOLLOWER_OFFSETS["agent_2"] * self.formation_scale
        )
        virtual_vel = np.array([leader.vx, leader.vy], dtype=np.float32)

        positions = np.array(
            [
                [self.states["go2_1"].x, self.states["go2_1"].y],
                [self.states["go2_2"].x, self.states["go2_2"].y],
                virtual_pos,
                [self.states["go2_3"].x, self.states["go2_3"].y],
            ],
            dtype=np.float32,
        )
        velocities = np.array(
            [
                [self.states["go2_1"].vx, self.states["go2_1"].vy],
                [self.states["go2_2"].vx, self.states["go2_2"].vy],
                virtual_vel,
                [self.states["go2_3"].vx, self.states["go2_3"].vy],
            ],
            dtype=np.float32,
        )
        return positions, velocities

    def _build_observations(self):
        positions, velocities = self._agent_arrays()
        leader_pos = positions[0]
        target_pos = np.array([self.target.x, self.target.y], dtype=np.float32)

        observations = []
        for i in range(4):
            own_pos = positions[i]
            own_vel = velocities[i]

            if i == 0:
                own_slot_rel = np.zeros(2, dtype=np.float32)
                role_flag = np.array([1.0], dtype=np.float32)
            else:
                own_slot_rel = leader_pos + FOLLOWER_OFFSETS[f"agent_{i}"] * self.formation_scale - own_pos
                role_flag = np.array([0.0], dtype=np.float32)

            other_positions = np.delete(positions, i, axis=0)
            # The policy was trained in a small [-3, 3] world. Scale Gazebo city
            # coordinates before inference so observations stay near that range.
            scale = self.position_scale
            obs = np.concatenate(
                [
                    own_pos / scale,
                    own_vel / scale,
                    own_slot_rel / scale,
                    (leader_pos - own_pos) / scale,
                    (target_pos - own_pos) / scale,
                    ((other_positions - own_pos) / scale).reshape(-1),
                    role_flag,
                    np.zeros(6, dtype=np.float32),
                ]
            ).astype(np.float32)
            observations.append(obs)
        return observations

    def _action_to_twist(self, robot_name, action):
        state = self.states[robot_name]
        ax = float(clamp(action[0], -1.0, 1.0))
        ay = float(clamp(action[1], -1.0, 1.0))
        speed = min(math.hypot(ax, ay), 1.0) * self.max_linear

        if speed < 1e-4:
            return Twist()

        desired_yaw = math.atan2(ay, ax)
        yaw_error = normalize_angle(desired_yaw - state.yaw)

        cmd = Twist()
        if abs(yaw_error) < self.heading_deadband:
            cmd.linear.x = speed
        else:
            cmd.linear.x = speed * max(0.0, math.cos(yaw_error))
        cmd.angular.z = clamp(yaw_error, -self.max_angular, self.max_angular)
        return cmd

    def _leader_visual_servo_twist(self, action):
        state = self.states["go2_1"]
        dx = self.target.x - state.x
        dy = self.target.y - state.y
        target_dist = math.hypot(dx, dy)
        bearing = math.atan2(dy, dx)
        yaw_error = normalize_angle(bearing - state.yaw)

        cmd = Twist()
        cmd.angular.z = clamp(self.leader_k_angular * yaw_error, -self.max_angular, self.max_angular)

        distance_error = target_dist - self.leader_follow_distance
        if abs(distance_error) < self.leader_distance_deadband:
            distance_error = 0.0

        target_feedforward = self.target.vx * math.cos(bearing) + self.target.vy * math.sin(bearing)
        gate = max(math.cos(yaw_error), 0.25)
        cmd.linear.x = clamp(
            (self.leader_k_linear * distance_error + target_feedforward) * gate,
            -0.5 * self.leader_linear,
            self.leader_linear,
        )
        return cmd

    def _slot_error(self, agent_idx, robot_name):
        leader = self.states["go2_1"]
        state = self.states[robot_name]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        actual = np.array([state.x, state.y], dtype=np.float32)
        slot = leader_pos + FOLLOWER_OFFSETS[f"agent_{agent_idx}"] * self.formation_scale
        error = slot - actual
        return slot, actual, error, float(np.linalg.norm(error))

    def _follower_action_with_slot_attraction(self, agent_idx, robot_name, action):
        _, _, error, error_norm = self._slot_error(agent_idx, robot_name)
        corrected = np.array(action, dtype=np.float32)

        if error_norm <= self.slot_deadband or self.slot_attraction_max <= 0.0:
            return corrected

        usable_error = error_norm - self.slot_deadband
        correction = error / max(error_norm, 1e-6) * usable_error * self.slot_attraction_gain
        correction_norm = float(np.linalg.norm(correction))
        if correction_norm > self.slot_attraction_max:
            correction = correction / correction_norm * self.slot_attraction_max

        corrected = corrected + correction.astype(np.float32)
        return np.clip(corrected, -1.0, 1.0)

    def _publish_all_zero(self):
        for pub in self.cmd_pubs.values():
            pub.publish(Twist())

    def _slot_error_text(self):
        leader = self.states["go2_1"]
        leader_pos = np.array([leader.x, leader.y], dtype=np.float32)
        parts = []

        for agent_idx, robot_name in ((1, "go2_2"), (3, "go2_3")):
            slot, actual, error, error_norm = self._slot_error(agent_idx, robot_name)
            parts.append(
                f"{robot_name}_slot=({slot[0]:+.2f},{slot[1]:+.2f}) "
                f"actual=({actual[0]:+.2f},{actual[1]:+.2f}) "
                f"err=({error[0]:+.2f},{error[1]:+.2f}) |e|={error_norm:.2f}m"
            )

        return "; ".join(parts)

    def _timer_cb(self):
        if not self._data_ready():
            return

        observations = self._build_observations()
        actions = self.maddpg.act(observations, add_noise=False)
        actions = [np.asarray(action, dtype=np.float32) for action in actions]

        if not all(np.all(np.isfinite(action)) for action in actions):
            self.get_logger().error("MADDPG produced NaN or Inf action. Publishing zero velocity.")
            self._publish_all_zero()
            return

        command_actions = list(actions)
        for agent_idx, robot_name in REAL_AGENT_MAP.items():
            if robot_name == "go2_1" and self.leader_visual_servo:
                cmd = self._leader_visual_servo_twist(actions[agent_idx])
            else:
                command_actions[agent_idx] = self._follower_action_with_slot_attraction(
                    agent_idx, robot_name, actions[agent_idx]
                )
                cmd = self._action_to_twist(robot_name, command_actions[agent_idx])
            self.cmd_pubs[robot_name].publish(cmd)

        action_text = ", ".join(
            f"{REAL_AGENT_MAP[idx]}<=agent_{idx}"
            f"[raw={actions[idx][0]:+.3f},{actions[idx][1]:+.3f}; "
            f"cmd={command_actions[idx][0]:+.3f},{command_actions[idx][1]:+.3f}]"
            for idx in sorted(REAL_AGENT_MAP)
        )
        self.get_logger().info(f"{action_text}; {self._slot_error_text()}")


def main(args=None):
    rclpy.init(args=args)
    node = MarlThreeRealController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_all_zero()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
