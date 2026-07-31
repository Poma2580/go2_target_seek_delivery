#!/usr/bin/env python3
"""Read-only MADDPG observation and inference checker.

This node subscribes to three real Go2 odometry topics plus one target odometry
topic, creates a virtual fourth agent at its ideal follower slot, builds the
four 23-D observations expected by the trained MADDPG model, runs inference,
and prints the four action vectors. It never publishes cmd_vel.
"""

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
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
    "agent_1": np.array([-0.60, -0.65], dtype=np.float32),
    "agent_2": np.array([0.0, -0.65], dtype=np.float32),
    "agent_3": np.array([0.60, -0.65], dtype=np.float32),
}


@dataclass
class EntityState:
    x: float = 0.0
    y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    received: bool = False
    receive_time = None


class MarlReadonlyObserver(Node):
    def __init__(self):
        super().__init__("marl_readonly_observer")

        self.declare_parameter("maddpg_root", str(DEFAULT_MADDPG_ROOT))
        self.declare_parameter("model_path", str(DEFAULT_MODEL_PATH))
        self.declare_parameter("target_odom_topic", "/walking_target/odom")
        self.declare_parameter("timer_rate", 2.0)
        self.declare_parameter("odom_timeout", 1.0)
        self.declare_parameter("target_timeout", 2.0)
        self.declare_parameter("marl_switch_radius", 5.0)

        self.maddpg_root = Path(self.get_parameter("maddpg_root").value).expanduser().resolve()
        self.model_path = Path(self.get_parameter("model_path").value).expanduser().resolve()
        self.target_odom_topic = self.get_parameter("target_odom_topic").value
        self.odom_timeout = float(self.get_parameter("odom_timeout").value)
        self.target_timeout = float(self.get_parameter("target_timeout").value)
        self.marl_switch_radius = float(self.get_parameter("marl_switch_radius").value)

        self.real_agent_names = ("go2_1", "go2_2", "go2_3")
        self.states = {name: EntityState() for name in self.real_agent_names}
        self.target = EntityState()

        for name in self.real_agent_names:
            self.create_subscription(
                Odometry,
                f"/{name}/odom",
                lambda msg, robot_name=name: self._odom_cb(robot_name, msg),
                10,
            )
        self.create_subscription(Odometry, self.target_odom_topic, self._target_cb, 10)

        self.maddpg = self._load_model()
        rate = float(self.get_parameter("timer_rate").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-6), self._timer_cb)
        self.get_logger().info(
            "marl_readonly_observer started: "
            f"target={self.target_odom_topic}, switch_radius={self.marl_switch_radius:.2f}m"
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
            self.get_logger().info(f"waiting for fresh odom: {', '.join(missing)}")
            return False
        return True

    def _agent_arrays(self):
        leader = self.states["go2_1"]
        virtual_pos = np.array([leader.x, leader.y], dtype=np.float32) + FOLLOWER_OFFSETS["agent_2"]
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
                own_slot_rel = leader_pos + FOLLOWER_OFFSETS[f"agent_{i}"] - own_pos
                role_flag = np.array([0.0], dtype=np.float32)

            other_positions = np.delete(positions, i, axis=0)
            obs = np.concatenate(
                [
                    own_pos,
                    own_vel,
                    own_slot_rel,
                    leader_pos - own_pos,
                    target_pos - own_pos,
                    (other_positions - own_pos).reshape(-1),
                    role_flag,
                    np.zeros(6, dtype=np.float32),
                ]
            ).astype(np.float32)
            observations.append(obs)
        return observations, positions

    def _all_real_dogs_near_target(self):
        dists = {}
        for name in self.real_agent_names:
            state = self.states[name]
            dists[name] = math.hypot(self.target.x - state.x, self.target.y - state.y)

        ready = all(dist < self.marl_switch_radius for dist in dists.values())
        return ready, dists

    def _timer_cb(self):
        if not self._data_ready():
            return

        observations, positions = self._build_observations()
        actions = self.maddpg.act(observations, add_noise=False)
        actions = [np.asarray(action, dtype=np.float32) for action in actions]

        if not all(np.all(np.isfinite(action)) for action in actions):
            self.get_logger().error("MADDPG produced NaN or Inf action.")
            return

        ready, dists = self._all_real_dogs_near_target()
        virtual_pos = positions[2]
        action_text = ", ".join(
            f"agent_{idx}=[{action[0]:+.3f},{action[1]:+.3f}]"
            for idx, action in enumerate(actions)
        )
        dist_text = ", ".join(f"{name}={dist:.2f}m" for name, dist in dists.items())
        self.get_logger().info(
            f"switch_ready={ready} ({dist_text}); "
            f"virtual_agent_2=({virtual_pos[0]:.2f},{virtual_pos[1]:.2f}); "
            f"{action_text}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = MarlReadonlyObserver()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
