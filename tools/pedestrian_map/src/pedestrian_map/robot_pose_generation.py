#!/usr/bin/env python3
"""Generate deterministic, scene-local Go2 spawn pose groups around frozen reference robots."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Mapping, Optional, Sequence

import cv2
import numpy as np
import yaml

from .grid_geometry import snap_grid_coordinate, supercover_cells
from .map_io import read_grayscale_image
from .paths import MAPS_ROOT, ROBOT_POSE_GROUPS, ROBOT_POSE_REPORT_ROOT, TARGET_ROUTES


SCENES = ("city", "forest", "airport")
ROUTES = ("straight", "rectangle", "v_shape")
ROBOTS = ("go2_1", "go2_2", "go2_3")
REFERENCE_ROBOTS = dict(zip(SCENES, ROBOTS))
DEFAULT_REFERENCE_POSES = ROBOT_POSE_REPORT_ROOT / "reference_robot_poses.yaml"
DEFAULT_ROUTES = TARGET_ROUTES
DEFAULT_MAPS_ROOT = MAPS_ROOT
DEFAULT_OUTPUT = ROBOT_POSE_GROUPS
DEFAULT_REPORT_DIR = ROBOT_POSE_REPORT_ROOT
OUTPUT_DECIMALS = 2


class TwoDecimalSafeDumper(yaml.SafeDumper):
    """Emit pose floating-point values with a stable two-decimal YAML spelling."""


def _represent_two_decimal_float(
    dumper: yaml.SafeDumper, value: float
) -> yaml.nodes.ScalarNode:
    return dumper.represent_scalar("tag:yaml.org,2002:float", f"{value:.2f}")


TwoDecimalSafeDumper.add_representer(float, _represent_two_decimal_float)


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


@dataclass(frozen=True)
class OccupancyMap:
    scene: str
    yaml_path: Path
    pixels: np.ndarray
    blocked: np.ndarray
    clearance: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    def world_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        grid_x = snap_grid_coordinate((x - self.origin_x) / self.resolution)
        grid_y = snap_grid_coordinate((y - self.origin_y) / self.resolution)
        gx, gy = math.floor(grid_x), math.floor(grid_y)
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return self.height - 1 - gy, gx

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        return (
            self.origin_x + (col + 0.5) * self.resolution,
            self.origin_y + (self.height - 1 - row + 0.5) * self.resolution,
        )


@dataclass(frozen=True)
class GenerationParameters:
    seed: int = 20260901
    group_count: int = 11
    neighbor_radius_min: float = 3.0
    neighbor_radius_max: float = 8.0
    spawn_z: Optional[float] = None
    spawn_clearance: float = 0.8
    min_robot_separation: float = 2.0
    max_attempts_per_robot: int = 1_000_000

    def validate(self) -> None:
        for name in ("neighbor_radius_min", "neighbor_radius_max",
                     "spawn_clearance", "min_robot_separation"):
            if _finite(getattr(self, name), name) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.spawn_z is not None and _finite(self.spawn_z, "spawn_z") <= 0.0:
            raise ValueError("spawn_z must be finite and positive")
        if self.spawn_z is not None and round(self.spawn_z, OUTPUT_DECIMALS) <= 0.0:
            raise ValueError("spawn_z must remain positive after rounding")
        if self.neighbor_radius_min >= self.neighbor_radius_max:
            raise ValueError("neighbor_radius_min must be less than neighbor_radius_max")
        for name in ("group_count", "max_attempts_per_robot"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")


def _group_names(group_count: int) -> tuple[str, ...]:
    return tuple(f"group_{index:02d}" for index in range(1, group_count + 1))


def _parse_pose(raw: object, label: str) -> dict[str, float]:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be a mapping")
    try:
        pose = {field: _finite(raw[field], f"{label}.{field}")
                for field in ("x", "y", "z", "yaw")}
    except KeyError as error:
        raise ValueError(f"{label}.{error.args[0]} is required") from error
    if pose["z"] <= 0.0:
        raise ValueError(f"{label}.z must be positive")
    return pose


def load_reference_poses(path: Path, group_count: int = 11) -> dict:
    """Load the frozen pre-migration perception poses, never the generated output."""
    try:
        document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 1:
            raise ValueError("reference poses schema_version must be 1")
        scenes = document["scenes"]
        result = {}
        for scene, robot in REFERENCE_ROBOTS.items():
            entry = scenes[scene]
            if entry["reference_robot"] != robot:
                raise ValueError(f"{scene}.reference_robot must be {robot}")
            groups = entry["poses"]
            if not isinstance(groups, dict) or tuple(groups) != _group_names(group_count):
                raise ValueError(f"{scene}.poses must contain ordered reference groups")
            result[scene] = {name: _parse_pose(groups[name], f"{scene}.{name}.{robot}")
                             for name in _group_names(group_count)}
        return result
    except (OSError, yaml.YAMLError, KeyError, TypeError) as error:
        raise ValueError(f"invalid reference poses {path}: {error}") from error


def load_occupancy_map(scene: str, yaml_path: Path) -> OccupancyMap:
    yaml_path = Path(yaml_path).resolve()
    try:
        metadata = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        image_path = Path(str(metadata["image"])).expanduser()
        if not image_path.is_absolute():
            image_path = yaml_path.parent / image_path
        resolution = _finite(metadata["resolution"], "resolution")
        origin_x = _finite(metadata["origin"][0], "origin[0]")
        origin_y = _finite(metadata["origin"][1], "origin[1]")
        occupied_thresh = _finite(
            metadata.get("occupied_thresh", 0.65), "occupied_thresh"
        )
        free_thresh = _finite(metadata.get("free_thresh", 0.196), "free_thresh")
        negate = bool(int(metadata.get("negate", 0)))
    except (OSError, KeyError, TypeError, IndexError, yaml.YAMLError) as error:
        raise ValueError(f"invalid map YAML {yaml_path}: {error}") from error
    if resolution <= 0.0:
        raise ValueError(f"{yaml_path}: resolution must be positive")
    if not 0.0 <= free_thresh < occupied_thresh <= 1.0:
        raise ValueError(f"{yaml_path}: invalid occupancy thresholds")

    pixels = read_grayscale_image(image_path.resolve())
    probability = pixels.astype(np.float32) / 255.0
    if not negate:
        probability = 1.0 - probability
    occupied = probability > occupied_thresh
    free = probability < free_thresh
    blocked = np.logical_or(occupied, np.logical_not(np.logical_or(occupied, free)))

    # float32 OpenCV distance transforms are much smaller than retaining scipy's
    # nearest-cell index arrays for all three large maps.
    distance = cv2.distanceTransform(
        np.logical_not(blocked).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    )
    clearance = np.maximum(
        0.0, distance * resolution - resolution / math.sqrt(2.0)
    )
    return OccupancyMap(
        scene=scene,
        yaml_path=yaml_path,
        pixels=pixels,
        blocked=blocked,
        clearance=clearance,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
    )


def load_maps(maps_root: Path) -> dict[str, OccupancyMap]:
    root = Path(maps_root)
    return {
        scene: load_occupancy_map(scene, root / scene / f"{scene}.yaml")
        for scene in SCENES
    }


def load_route_p1s(routes_path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    try:
        root = yaml.safe_load(Path(routes_path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to read routes {routes_path}: {error}") from error
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("routes YAML schema_version must be 1")
    result: dict[str, dict[str, tuple[float, float]]] = {}
    try:
        for scene in SCENES:
            result[scene] = {}
            for route in ROUTES:
                point = root["scenes"][scene]["routes"][route]["points"][0]
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"{scene}.{route}.P1 must be [x, y]")
                result[scene][route] = (
                    _finite(point[0], f"{scene}.{route}.P1.x"),
                    _finite(point[1], f"{scene}.{route}.P1.y"),
                )
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError(f"routes YAML is missing required P1 data: {error}") from error
    return result


def compute_anchors(
    p1s: Mapping[str, Mapping[str, tuple[float, float]]]
) -> dict[str, tuple[float, float]]:
    anchors = {}
    for scene in SCENES:
        points = [p1s[scene][route] for route in ROUTES]
        anchors[scene] = (
            sum(point[0] for point in points) / len(points),
            sum(point[1] for point in points) / len(points),
        )
    return anchors


def normalize_angle(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def sample_annulus(
    rng: random.Random,
    anchor: tuple[float, float],
    radius_min: float,
    radius_max: float,
) -> tuple[float, float]:
    radius = math.sqrt(rng.uniform(radius_min**2, radius_max**2))
    theta = rng.uniform(-math.pi, math.pi)
    return anchor[0] + radius * math.cos(theta), anchor[1] + radius * math.sin(theta)


def yaw_toward(point: tuple[float, float], target: tuple[float, float]) -> float:
    return normalize_angle(math.atan2(target[1] - point[1], target[0] - point[0]))


def check_map_position(
    map_data: OccupancyMap, point: tuple[float, float], clearance: float
) -> dict[str, object]:
    cell = map_data.world_to_cell(*point)
    if cell is None:
        return {"pass": False, "reason": "outside_map", "clearance_m": None}
    if map_data.blocked[cell]:
        return {
            "pass": False,
            "reason": "occupied_or_unknown",
            "clearance_m": 0.0,
        }
    actual = max(0.0, float(map_data.clearance[cell]))
    return {
        "pass": actual + 1e-12 >= clearance,
        "reason": None if actual + 1e-12 >= clearance else "below_clearance",
        "clearance_m": actual,
    }


def line_of_sight(
    map_data: OccupancyMap,
    start: tuple[float, float],
    end: tuple[float, float],
) -> bool:
    if map_data.world_to_cell(*start) is None or map_data.world_to_cell(*end) is None:
        return False
    return all(
        not bool(map_data.blocked[cell])
        for cell in supercover_cells(map_data, start, end)
    )


def check_visibility(
    map_data: OccupancyMap,
    pose: Mapping[str, float],
    targets: Mapping[str, tuple[float, float]],
    max_distance: float,
    hfov_deg: float,
) -> list[dict[str, object]]:
    start = float(pose["x"]), float(pose["y"])
    yaw = float(pose["yaw"])
    half_fov = math.radians(hfov_deg) / 2.0
    results = []
    for route in ROUTES:
        target = targets[route]
        distance = math.dist(start, target)
        bearing = math.atan2(target[1] - start[1], target[0] - start[0])
        offset = abs(normalize_angle(bearing - yaw))
        los = line_of_sight(map_data, start, target)
        distance_pass = distance <= max_distance + 1e-12
        fov_pass = offset <= half_fov + 1e-12
        results.append(
            {
                "route": route,
                "target": [target[0], target[1]],
                "distance_m": distance,
                "distance_pass": distance_pass,
                "angle_offset_deg": math.degrees(offset),
                "fov_pass": fov_pass,
                "los_pass": los,
                "pass": distance_pass and fov_pass and los,
            }
        )
    return results


def _rounded_pose(
    point: tuple[float, float], anchor: tuple[float, float], spawn_z: float
) -> dict[str, float]:
    x, y = round(point[0], OUTPUT_DECIMALS), round(point[1], OUTPUT_DECIMALS)
    x = 0.0 if x == 0.0 else x
    y = 0.0 if y == 0.0 else y
    yaw = round(yaw_toward((x, y), anchor), OUTPUT_DECIMALS)
    yaw = 0.0 if yaw == 0.0 else yaw
    return {
        "x": x,
        "y": y,
        "z": round(spawn_z, OUTPUT_DECIMALS),
        "yaw": yaw,
    }


def generate_poses(
    maps: Mapping[str, OccupancyMap],
    p1s: Mapping[str, Mapping[str, tuple[float, float]]],
    parameters: GenerationParameters,
    reference_poses: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> tuple[dict, dict]:
    parameters.validate()
    anchors = compute_anchors(p1s)
    generated, statistics = {}, {}
    for scene in SCENES:
        generated[scene], statistics[scene] = {}, {}
        reference_robot = REFERENCE_ROBOTS[scene]
        for name in _group_names(parameters.group_count):
            reference = _parse_pose(reference_poses[scene][name], f"{scene}/{name}/{reference_robot}")
            center = reference["x"], reference["y"]
            reference_check = check_map_position(maps[scene], center, parameters.spawn_clearance)
            if not reference_check["pass"]:
                raise ValueError(f"{scene}/{name}/{reference_robot}: fixed reference failed: {reference_check}")
            # String seeding is stable across Python processes; each group is independent.
            rng = random.Random(f"{parameters.seed}:{scene}:{name}")
            robots, accepted = {}, []
            statistics[scene][name] = {}
            for robot in ROBOTS:
                if robot == reference_robot:
                    robots[robot] = dict(reference)
                    continue
                rejected = {"annulus": 0, "map": 0, "separation": 0}
                for attempt in range(1, parameters.max_attempts_per_robot + 1):
                    pose = _rounded_pose(
                        sample_annulus(rng, center, parameters.neighbor_radius_min,
                                       parameters.neighbor_radius_max),
                        anchors[scene],
                        reference["z"] if parameters.spawn_z is None else parameters.spawn_z,
                    )
                    # All acceptance checks use the actual serialized XY coordinates.
                    point = pose["x"], pose["y"]
                    radius = math.dist(point, center)
                    if not parameters.neighbor_radius_min <= radius <= parameters.neighbor_radius_max:
                        rejected["annulus"] += 1
                        continue
                    if not check_map_position(maps[scene], point, parameters.spawn_clearance)["pass"]:
                        rejected["map"] += 1
                        continue
                    if any(math.dist(point, old) < parameters.min_robot_separation for old in accepted):
                        rejected["separation"] += 1
                        continue
                    robots[robot] = pose
                    accepted.append(point)
                    statistics[scene][name][robot] = {
                        "attempts": attempt, "accepted": 1, "rejected": rejected,
                    }
                    break
                else:
                    raise RuntimeError(
                        f"{scene}/{name}/{robot}: accepted 0/1 after "
                        f"{parameters.max_attempts_per_robot} attempts; rejected={rejected}"
                    )
            generated[scene][name] = {"resolved": True, "robots": robots}
    return generated, {"anchors": anchors, "sampling": statistics}


def validate_generated_poses(
    poses: Mapping,
    maps: Mapping[str, OccupancyMap],
    p1s: Mapping[str, Mapping[str, tuple[float, float]]],
    parameters: GenerationParameters,
    reference_poses: Mapping,
) -> list[dict[str, object]]:
    parameters.validate()
    # Validate shape and numeric fields independently of the generation routine.
    poses = poses_from_document(pose_groups_document(poses, parameters.group_count),
                                parameters.group_count)
    anchors, details = compute_anchors(p1s), []
    for scene in SCENES:
        reference_robot = REFERENCE_ROBOTS[scene]
        for name in _group_names(parameters.group_count):
            robots = poses[scene][name]["robots"]
            reference = _parse_pose(reference_poses[scene][name], f"{scene}/{name}/reference")
            center = reference["x"], reference["y"]
            navigation = [robot for robot in ROBOTS if robot != reference_robot]
            navigation_separation = math.dist(
                (robots[navigation[0]]["x"], robots[navigation[0]]["y"]),
                (robots[navigation[1]]["x"], robots[navigation[1]]["y"]),
            )
            for robot in ROBOTS:
                pose = robots[robot]
                point = pose["x"], pose["y"]
                radius = math.dist(point, center)
                map_check = check_map_position(maps[scene], point, parameters.spawn_clearance)
                is_reference = robot == reference_robot
                yaw_error = abs(normalize_angle(pose["yaw"] - yaw_toward(point, anchors[scene])))
                reference_match = pose == reference if is_reference else None
                radius_pass = None if is_reference else parameters.neighbor_radius_min <= radius <= parameters.neighbor_radius_max
                separation_pass = None if is_reference else navigation_separation >= parameters.min_robot_separation
                expected_z = reference["z"] if parameters.spawn_z is None else round(parameters.spawn_z, OUTPUT_DECIMALS)
                passed = map_check["pass"] and (
                    reference_match if is_reference else
                    radius_pass and separation_pass
                    and yaw_error <= 0.5 * 10**-OUTPUT_DECIMALS + 1e-12
                    and pose["z"] == expected_z
                )
                details.append({
                    "scene": scene, "group": name, "robot": robot,
                    "reference_robot": reference_robot, "is_reference": is_reference,
                    "pose": pose, "distance_to_reference_robot": radius,
                    "clearance": map_check["clearance_m"], "map_check": map_check,
                    "robot_separation": {
                        other: math.dist(point, (other_pose["x"], other_pose["y"]))
                        for other, other_pose in robots.items() if other != robot
                    },
                    "navigation_robot_separation": navigation_separation,
                    "reference_match": reference_match, "radius_pass": radius_pass,
                    "separation_pass": separation_pass,
                    "yaw_error_rad": None if is_reference else yaw_error,
                    "pass": bool(passed),
                })
    return details


def pose_groups_document(poses: Mapping, group_count: int) -> dict:
    for scene in SCENES:
        if scene not in poses or tuple(poses[scene]) != _group_names(group_count):
            raise ValueError(f"{scene} must contain {group_count} ordered pose groups")
    return {
        "schema_version": 2,
        "coordinate_mode": "scene_absolute",
        "scenes": {scene: {"pose_groups": poses[scene]} for scene in SCENES},
    }


def dump_pose_groups_yaml(document: Mapping[str, object]) -> str:
    return yaml.dump(
        document,
        Dumper=TwoDecimalSafeDumper,
        sort_keys=False,
        allow_unicode=True,
    )


def poses_from_document(document: Mapping, group_count: int = 11) -> dict:
    if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document["schema_version"] != 2:
        raise ValueError("pose groups schema_version must be 2")
    if document.get("coordinate_mode") != "scene_absolute":
        raise ValueError("coordinate_mode must be scene_absolute")
    try:
        result = {}
        for scene in SCENES:
            groups = document["scenes"][scene]["pose_groups"]
            if not isinstance(groups, dict) or tuple(groups) != _group_names(group_count):
                raise ValueError(f"{scene} must contain {group_count} ordered pose groups")
            result[scene] = {}
            for name in _group_names(group_count):
                group = groups[name]
                if group["resolved"] is not True:
                    raise ValueError(f"{scene}/{name} is not resolved")
                robots = {robot: _parse_pose(group["robots"][robot], f"{scene}/{name}/{robot}")
                          for robot in ROBOTS}
                result[scene][name] = {"resolved": True, "robots": robots}
        return result
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid robot pose groups document: {error}") from error


def build_report(
    parameters: GenerationParameters,
    generation: Mapping[str, object],
    validation: Sequence[Mapping[str, object]],
    reference_path: Path,
) -> dict[str, object]:
    import hashlib

    return {
        "schema_version": 2,
        "status": "PASS" if all(item["pass"] for item in validation) else "FAIL",
        "units": {"distance": "m", "yaw": "rad"},
        "generation": asdict(parameters),
        "reference_source": {
            "filename": Path(reference_path).name,
            "sha256": hashlib.sha256(Path(reference_path).read_bytes()).hexdigest(),
        },
        "reference_robots": REFERENCE_ROBOTS,
        "anchors": {scene: list(point) for scene, point in generation["anchors"].items()},
        "sampling": generation["sampling"],
        "robots": list(validation),
    }


def _world_to_pixel(map_data: OccupancyMap, point: tuple[float, float]) -> tuple[int, int]:
    cell = map_data.world_to_cell(*point)
    if cell is None:
        raise ValueError(f"point {point} lies outside {map_data.scene} map")
    return cell[1], cell[0]


def render_scene(
    scene: str,
    map_data: OccupancyMap,
    poses: Mapping,
    p1s: Mapping[str, tuple[float, float]],
    anchor: tuple[float, float],
) -> bytes:
    """Render an overview and a readable local view of this scene's 33 robots."""
    groups = poses[scene]
    colors = {"go2_1": (40, 70, 230), "go2_2": (50, 160, 60), "go2_3": (220, 90, 40)}
    pixels = [_world_to_pixel(map_data, (pose["x"], pose["y"]))
              for group in groups.values() for pose in group["robots"].values()]
    pixels += [_world_to_pixel(map_data, point) for point in (*p1s.values(), anchor)]
    margin = int(math.ceil(3.0 / map_data.resolution))
    x0 = max(0, min(p[0] for p in pixels) - margin)
    y0 = max(0, min(p[1] for p in pixels) - margin)
    x1 = min(map_data.width, max(p[0] for p in pixels) + margin + 1)
    y1 = min(map_data.height, max(p[1] for p in pixels) + margin + 1)

    def panel(bounds, width, height, detailed):
        left, top, right, bottom = bounds
        crop = map_data.pixels[top:bottom, left:right]
        scale = min(width / crop.shape[1], height / crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                    max(1, round(crop.shape[0] * scale))),
                             interpolation=cv2.INTER_NEAREST)
        image = np.full((height, width, 3), 245, dtype=np.uint8)
        image[:resized.shape[0], :resized.shape[1]] = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

        def pixel(point):
            x, y = _world_to_pixel(map_data, point)
            return round((x - left) * scale), round((y - top) * scale)

        for name, group in groups.items():
            reference = group["robots"][REFERENCE_ROBOTS[scene]]
            start = pixel((reference["x"], reference["y"]))
            for robot, pose in group["robots"].items():
                position = pixel((pose["x"], pose["y"]))
                if detailed and robot != REFERENCE_ROBOTS[scene]:
                    cv2.line(image, start, position, (175, 175, 175), 1, cv2.LINE_AA)
        for name, group in groups.items():
            for robot, pose in group["robots"].items():
                position = pixel((pose["x"], pose["y"]))
                cv2.circle(image, position, 5 if detailed else 3, colors[robot], -1, cv2.LINE_AA)
                if detailed:
                    length = max(10, round(0.9 * scale / map_data.resolution))
                    end = (round(position[0] + length * math.cos(pose["yaw"])),
                           round(position[1] - length * math.sin(pose["yaw"])))
                    cv2.arrowedLine(image, position, end, colors[robot], 1, cv2.LINE_AA, tipLength=0.3)
                    if robot == REFERENCE_ROBOTS[scene]:
                        cv2.circle(image, position, 8, colors[robot], 1, cv2.LINE_AA)
                    cv2.putText(image, name[-2:], (position[0] + 7, position[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[robot], 1, cv2.LINE_AA)
        for route, point in p1s.items():
            position = pixel(point)
            cv2.drawMarker(image, position, (200, 30, 200), cv2.MARKER_DIAMOND, 12, 2)
            if detailed:
                cv2.putText(image, route + " P1", (position[0] + 8, position[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 30, 200), 1, cv2.LINE_AA)
        cv2.drawMarker(image, pixel(anchor), (0, 160, 255), cv2.MARKER_STAR, 16, 2)
        if not detailed:
            cv2.rectangle(image, (round(x0 * scale), round(y0 * scale)),
                          (round(x1 * scale), round(y1 * scale)), (0, 130, 255), 2)
        return image

    overview = panel((0, 0, map_data.width, map_data.height), 480, 860, False)
    local = panel((x0, y0, x1, y1), 1000, 860, True)
    image = np.full((950, 1480, 3), 255, dtype=np.uint8)
    image[90:, :480] = overview
    image[90:, 480:] = local
    cv2.putText(image, f"{scene}: 11 groups / 33 robots; reference={REFERENCE_ROBOTS[scene]} (ring)",
                (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(image, "Overview (orange = detail area)       Local view: group numbers, yaw arrows, group links; star = P1 mean anchor",
                (15, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
    for index, (robot, color) in enumerate(colors.items()):
        cv2.putText(image, robot, (15 + index * 150, 77), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode {scene} visualization")
    return encoded.tobytes()


def _stage_bytes(destination: Path, data: bytes) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temp_name, 0o644)
    except Exception:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return Path(temp_name)


def write_outputs(
    output_path: Path,
    report_dir: Path,
    document: Mapping[str, object],
    report: Mapping[str, object],
    images: Mapping[str, bytes],
) -> None:
    outputs = {
        Path(output_path): dump_pose_groups_yaml(document).encode("utf-8"),
        Path(report_dir) / "validation_report.json": (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }
    outputs.update(
        {Path(report_dir) / f"{scene}_pose_groups.png": data for scene, data in images.items()}
    )
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, data in outputs.items():
            staged.append((_stage_bytes(destination, data), destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def _print_summary(report: Mapping[str, object]) -> None:
    print(f"Validation: {report['status']} ({len(report['robots'])} robot poses)")
    for scene in SCENES:
        records = [item for item in report["robots"] if item["scene"] == scene]
        print(f"  {scene}: {sum(item['pass'] for item in records)}/{len(records)} PASS; "
              f"reference={REFERENCE_ROBOTS[scene]}")


def run(
    routes_path: Path,
    maps_root: Path,
    output_path: Path,
    report_dir: Path,
    parameters: GenerationParameters,
    check: bool = False,
    reference_path: Path = DEFAULT_REFERENCE_POSES,
) -> dict[str, object]:
    parameters.validate()
    reference_poses = load_reference_poses(reference_path, parameters.group_count)
    p1s = load_route_p1s(routes_path)
    maps = load_maps(maps_root)
    poses, generation = generate_poses(maps, p1s, parameters, reference_poses)
    document = pose_groups_document(poses, parameters.group_count)
    if check:
        try:
            existing = yaml.safe_load(Path(output_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"failed to read frozen poses {output_path}: {error}") from error
        poses = poses_from_document(existing, parameters.group_count)
        if existing != document:
            raise ValueError(f"{output_path} does not match deterministic generation")
    validation = validate_generated_poses(poses, maps, p1s, parameters, reference_poses)
    report = build_report(parameters, generation, validation, reference_path)
    if report["status"] != "PASS":
        failed = [f"{item['scene']}/{item['group']}/{item['robot']}"
                  for item in validation if not item["pass"]]
        raise ValueError(f"pose validation failed: {', '.join(failed)}")
    if not check:
        anchors = compute_anchors(p1s)
        images = {
            scene: render_scene(scene, maps[scene], poses, p1s[scene], anchors[scene])
            for scene in SCENES
        }
        write_outputs(output_path, report_dir, document, report, images)
    _print_summary(report)
    return report


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--routes", type=Path, default=DEFAULT_ROUTES)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--reference-poses", type=Path, default=DEFAULT_REFERENCE_POSES)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--neighbor-radius-min", type=float, default=3.0)
    parser.add_argument("--neighbor-radius-max", type=float, default=8.0)
    parser.add_argument("--spawn-z", type=float, default=None,
                        help="New robot height; default: inherit the group's reference height")
    parser.add_argument("--spawn-clearance", type=float, default=0.8)
    parser.add_argument("--min-robot-separation", type=float, default=2.0)
    parser.add_argument("--max-attempts-per-robot", type=int, default=1_000_000)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    parameters = GenerationParameters(
        seed=args.seed,
        neighbor_radius_min=args.neighbor_radius_min,
        neighbor_radius_max=args.neighbor_radius_max,
        spawn_z=args.spawn_z,
        spawn_clearance=args.spawn_clearance,
        min_robot_separation=args.min_robot_separation,
        max_attempts_per_robot=args.max_attempts_per_robot,
    )
    try:
        run(args.routes, args.maps_root, args.output, args.report_dir, parameters,
            args.check, args.reference_poses)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
