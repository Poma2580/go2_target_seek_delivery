"""Contract tests between waypoint training features and Gazebo preprocessing."""

import math
from dataclasses import replace

import numpy as np
import torch
from sensor_msgs.msg import LaserScan

from go2_mapping_nav.maddpg_waypoint_selector import (
    DEFAULT_MODEL,
    MaddpgWaypointSelector,
    OBSERVATION_SLICES,
    candidate_points,
    environment_from_checkpoint,
    scan_features,
)
from waypoint_maddpg_v0.environment import WaypointSelectionEnv
from waypoint_maddpg_v0.geometry import ray_aabb_distance, ray_circle_distance, rotate
from waypoint_maddpg_v0.lidar import PlanarLidar


def test_selector_goal_dispatch_never_bypasses_fixed_period():
    class Clock:
        @staticmethod
        def now():
            return type("Now", (), {"nanoseconds": 12_500_000_000})()

    class GoalManager:
        def __init__(self):
            self.plan = None
            self.dispatch_times = []

        def set_plan(self, plan):
            self.plan = plan

        def dispatch_if_due(self, now):
            self.dispatch_times.append(now)

    node = type("SelectorStub", (), {})()
    node.follower_names = ("go2_2", "go2_3")
    node.nav_goals = GoalManager()
    node.get_clock = lambda: Clock()
    goals = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    MaddpgWaypointSelector._dispatch_goals(node, goals, yaw=0.25)

    assert node.nav_goals.dispatch_times == [12.5]
    assert node.nav_goals.plan.slots == {
        "go2_2": (1.0, 2.0, 0.25),
        "go2_3": (3.0, 4.0, 0.25),
    }


def _exact_scan(config, robot_pos, yaw, obstacles, other_robots):
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = 2.0 * math.pi / config.lidar_sim_rays
    origin = robot_pos + rotate([config.lidar_sensor_x, 0.0], yaw)
    ranges = []
    for index in range(config.lidar_sim_rays):
        relative = scan.angle_min + index * scan.angle_increment
        direction = np.asarray(
            [math.cos(yaw + relative), math.sin(yaw + relative)], dtype=np.float32
        )
        distance = config.lidar_physical_max_range
        for obstacle in obstacles:
            if obstacle["shape"] == "square":
                hit = ray_aabb_distance(
                    origin,
                    direction,
                    obstacle["lower"],
                    obstacle["upper"],
                    config.lidar_physical_max_range,
                )
            else:
                hit = ray_circle_distance(
                    origin,
                    direction,
                    obstacle["center"],
                    obstacle["radius"],
                    config.lidar_physical_max_range,
                )
            distance = min(distance, hit)
        for center, radius in other_robots:
            distance = min(
                distance,
                ray_circle_distance(
                    origin,
                    direction,
                    center,
                    radius,
                    config.lidar_physical_max_range,
                ),
            )
        if distance < config.lidar_min_range or distance > config.lidar_policy_max_range:
            ranges.append(float("inf"))
        else:
            ranges.append(float(distance))
    scan.ranges = ranges
    return scan


def test_checkpoint_and_observation_layout_are_exact():
    payload = torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=False)
    config = environment_from_checkpoint(payload)
    assert config.lidar_observation_size == 36
    assert config.lidar_sim_rays == 108
    assert config.marl_dt == 1.0
    assert config.follower_max_speed == 0.15
    assert config.leader_speed == 0.10
    assert config.candidate_offsets == (2.0, 1.0, 0.0, -1.0, -2.0)
    assert payload["actors"][0]["net.0.weight"].shape[1] == 83
    assert payload["actors"][0]["net.4.weight"].shape[0] == 5

    cursor = 0
    for start, stop in OBSERVATION_SLICES.values():
        assert start == cursor
        assert stop > start
        cursor = stop
    assert cursor == 83


def test_candidate_coordinates_match_training_environment():
    config = replace(environment_from_checkpoint(
        torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=False)
    ), lidar_noise_std=0.0)
    env = WaypointSelectionEnv(config, seed=3, lidar_noise=False)
    env.leader_pos = np.asarray([4.2, -1.7], dtype=np.float32)
    env.leader_yaw = 0.63
    expected = env._candidate_points()
    actual = candidate_points(env.leader_pos, env.leader_yaw, config)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_laserscan_preprocessing_matches_training_lidar():
    config = replace(environment_from_checkpoint(
        torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=False)
    ), lidar_noise_std=0.0)
    robot = np.asarray([2.0, 2.0], dtype=np.float32)
    yaw = 0.37
    square_center = np.asarray([7.8, 2.2], dtype=np.float32)
    half = config.obstacle_size / 2.0
    obstacles = [
        {
            "shape": "square",
            "center": square_center,
            "lower": square_center - half,
            "upper": square_center + half,
        },
        {
            "shape": "circle",
            "center": np.asarray([8.5, -2.1], dtype=np.float32),
            "radius": config.obstacle_circle_radius,
        },
    ]
    other_robots = [
        (np.asarray([3.0, 0.0], dtype=np.float32), config.robot_radius)
    ]
    training_lidar = PlanarLidar(config, np.random.default_rng(5))
    expected_sectors, expected_hits = training_lidar.scan(
        robot, yaw, obstacles, other_robots
    )
    scan = _exact_scan(config, robot, yaw, obstacles, other_robots)
    actual_sectors, actual_hits = scan_features(scan, robot, yaw, yaw, config)
    np.testing.assert_allclose(actual_sectors, expected_sectors, atol=2e-6)
    np.testing.assert_allclose(actual_hits, expected_hits, atol=2e-5)


def test_gazebo_220_rays_are_resampled_to_checkpoint_density():
    config = environment_from_checkpoint(
        torch.load(DEFAULT_MODEL, map_location="cpu", weights_only=False)
    )
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = 2.0 * math.pi / 220.0
    scan.ranges = [5.0] * 220
    sectors, hits = scan_features(
        scan,
        np.zeros(2, dtype=np.float32),
        follower_yaw=0.4,
        leader_yaw=-0.2,
        config=config,
    )
    assert sectors.shape == (36,)
    assert hits.shape == (108, 2)
    expected = (config.lidar_policy_max_range - 5.0) / (
        config.lidar_policy_max_range - config.lidar_min_range
    )
    np.testing.assert_allclose(sectors, expected, atol=1e-6)
