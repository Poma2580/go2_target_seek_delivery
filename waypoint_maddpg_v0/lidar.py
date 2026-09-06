"""Gazebo-aligned planar lidar proxy and fixed-size policy features."""

import math

import numpy as np

from .geometry import (
    point_segment_distance,
    ray_aabb_distance,
    ray_circle_distance,
    rotate,
    world_to_body,
)


class PlanarLidar:
    """Simulate the effective 2-D observation produced from the VLP-16 cloud.

    The first curriculum uses 36 rays directly.  ``sim_rays`` may later be set
    to 108; the result is still minimum-pooled to the same 36 policy sectors.
    """

    def __init__(self, config, rng):
        self.cfg = config
        self.rng = rng
        if config.lidar_sim_rays % config.lidar_observation_size != 0:
            raise ValueError("lidar_sim_rays must be divisible by lidar_observation_size")
        self.rays_per_sector = config.lidar_sim_rays // config.lidar_observation_size
        self.relative_angles = np.linspace(
            -math.pi,
            math.pi,
            config.lidar_sim_rays,
            endpoint=False,
            dtype=np.float32,
        )

    def scan(self, robot_pos, robot_yaw, obstacles, other_robots):
        sensor_origin = robot_pos + rotate([self.cfg.lidar_sensor_x, 0.0], robot_yaw)
        raw = np.full(
            self.cfg.lidar_sim_rays,
            self.cfg.lidar_physical_max_range,
            dtype=np.float32,
        )
        hit_points_world = []

        for index, relative_angle in enumerate(self.relative_angles):
            angle = robot_yaw + float(relative_angle)
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
            distance = self.cfg.lidar_physical_max_range
            for obstacle in obstacles:
                if obstacle["shape"] == "square":
                    obstacle_distance = ray_aabb_distance(
                        sensor_origin,
                        direction,
                        obstacle["lower"],
                        obstacle["upper"],
                        self.cfg.lidar_physical_max_range,
                    )
                else:
                    obstacle_distance = ray_circle_distance(
                        sensor_origin,
                        direction,
                        obstacle["center"],
                        obstacle["radius"],
                        self.cfg.lidar_physical_max_range,
                    )
                distance = min(distance, obstacle_distance)
            for other_pos, other_radius in other_robots:
                distance = min(
                    distance,
                    ray_circle_distance(
                        sensor_origin,
                        direction,
                        other_pos,
                        other_radius,
                        self.cfg.lidar_physical_max_range,
                    ),
                )

            if distance < self.cfg.lidar_min_range:
                # gazebo_ros_velodyne_laser omits returns inside min_range.
                distance = self.cfg.lidar_physical_max_range
            elif distance < self.cfg.lidar_physical_max_range:
                distance += float(self.rng.normal(0.0, self.cfg.lidar_noise_std))
                distance = float(np.clip(
                    distance,
                    self.cfg.lidar_min_range,
                    self.cfg.lidar_physical_max_range,
                ))

            raw[index] = distance
            if distance <= self.cfg.lidar_policy_max_range:
                hit_points_world.append(sensor_origin + distance * direction)

        pooled = raw.reshape(self.cfg.lidar_observation_size, self.rays_per_sector).min(axis=1)
        clipped = np.clip(
            pooled,
            self.cfg.lidar_min_range,
            self.cfg.lidar_policy_max_range,
        )
        # Zero means clear/far; one means a return at the minimum useful range.
        sectors = (
            self.cfg.lidar_policy_max_range - clipped
        ) / (
            self.cfg.lidar_policy_max_range - self.cfg.lidar_min_range
        )
        sectors[pooled > self.cfg.lidar_policy_max_range] = 0.0

        if hit_points_world:
            points_body = np.stack(
                [world_to_body(point, robot_pos, robot_yaw) for point in hit_points_world],
                axis=0,
            ).astype(np.float32)
        else:
            points_body = np.empty((0, 2), dtype=np.float32)
        return sectors.astype(np.float32), points_body


def candidate_features(candidate_points_world, robot_pos, robot_yaw, lidar_points_body, config):
    """Return normalized 5-D features and raw metrics for each candidate."""
    features, metrics = [], []
    start = np.zeros(2, dtype=np.float32)
    cap = float(config.candidate_clearance_cap)

    for point_world in candidate_points_world:
        relative = world_to_body(point_world, robot_pos, robot_yaw)
        lookahead_end = relative + np.array(
            [config.candidate_forward_lookahead, 0.0], dtype=np.float32
        )
        if len(lidar_points_body):
            endpoint_clearance = float(
                np.min(np.linalg.norm(lidar_points_body - relative, axis=1))
            )
            path_clearance = min(
                min(
                    point_segment_distance(point, start, relative),
                    point_segment_distance(point, relative, lookahead_end),
                )
                for point in lidar_points_body
            )
        else:
            endpoint_clearance = cap
            path_clearance = cap

        endpoint_clearance = min(endpoint_clearance, cap)
        path_clearance = min(path_clearance, cap)
        blocked = (
            endpoint_clearance < config.endpoint_blocked_clearance
            or path_clearance < config.path_blocked_clearance
        )
        features.extend(
            [
                float(np.clip(relative[0] / 6.0, -1.0, 1.0)),
                float(np.clip(relative[1] / 6.0, -1.0, 1.0)),
                endpoint_clearance / cap,
                path_clearance / cap,
                float(blocked),
            ]
        )
        metrics.append(
            {
                "endpoint_clearance": endpoint_clearance,
                "path_clearance": path_clearance,
                "blocked": bool(blocked),
            }
        )
    return np.asarray(features, dtype=np.float32), metrics
