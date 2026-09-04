#!/usr/bin/env python3
"""Generate deterministic, cross-scene-valid Go2 spawn pose groups."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
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
RESPONSIBILITY = dict(zip(ROBOTS, SCENES))
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
    radius_min: float = 6.0
    radius_max: float = 10.0
    spawn_z: float = 0.4
    spawn_clearance: float = 0.8
    max_target_distance: float = 25.0
    camera_hfov_deg: float = 60.0
    min_pose_separation: float = 2.0
    go2_2_camera_hfov_deg: float = 90.0
    go2_2_min_pose_separation: float = 0.5
    max_attempts_per_robot: int = 1_000_000

    def validate(self) -> None:
        numeric_positive = {
            "radius_min": self.radius_min,
            "radius_max": self.radius_max,
            "spawn_z": self.spawn_z,
            "spawn_clearance": self.spawn_clearance,
            "max_target_distance": self.max_target_distance,
            "camera_hfov_deg": self.camera_hfov_deg,
            "min_pose_separation": self.min_pose_separation,
            "go2_2_camera_hfov_deg": self.go2_2_camera_hfov_deg,
            "go2_2_min_pose_separation": self.go2_2_min_pose_separation,
        }
        for name, value in numeric_positive.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.radius_min >= self.radius_max:
            raise ValueError("radius_min must be less than radius_max")
        for name, value in (
            ("camera_hfov_deg", self.camera_hfov_deg),
            ("go2_2_camera_hfov_deg", self.go2_2_camera_hfov_deg),
        ):
            if value > 360.0:
                raise ValueError(f"{name} must not exceed 360 degrees")
        if self.group_count <= 0:
            raise ValueError("group_count must be positive")
        if self.max_attempts_per_robot <= 0:
            raise ValueError("max_attempts_per_robot must be positive")

    def hfov_deg(self, robot: str) -> float:
        return self.go2_2_camera_hfov_deg if robot == "go2_2" else self.camera_hfov_deg

    def separation(self, robot: str) -> float:
        return (
            self.go2_2_min_pose_separation
            if robot == "go2_2"
            else self.min_pose_separation
        )


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


def _first_visibility_rejection(items: Sequence[Mapping[str, object]]) -> str:
    if any(not item["distance_pass"] for item in items):
        return "distance"
    if any(not item["fov_pass"] for item in items):
        return "fov"
    return "los"


def generate_poses(
    maps: Mapping[str, OccupancyMap],
    p1s: Mapping[str, Mapping[str, tuple[float, float]]],
    parameters: GenerationParameters,
) -> tuple[dict[str, list[dict[str, float]]], dict[str, object]]:
    parameters.validate()
    anchors = compute_anchors(p1s)
    rng = random.Random(parameters.seed)
    generated: dict[str, list[dict[str, float]]] = {}
    statistics: dict[str, object] = {}

    for robot in ROBOTS:
        scene = RESPONSIBILITY[robot]
        accepted: list[dict[str, float]] = []
        attempts = 0
        rejected = {
            "annulus": 0,
            "map": 0,
            "distance": 0,
            "fov": 0,
            "los": 0,
            "separation": 0,
        }
        while len(accepted) < parameters.group_count:
            if attempts >= parameters.max_attempts_per_robot:
                raise RuntimeError(
                    f"{robot}/{scene}: accepted {len(accepted)}/{parameters.group_count} "
                    f"after {attempts} attempts; rejected={rejected}"
                )
            attempts += 1
            pose = _rounded_pose(
                sample_annulus(
                    rng, anchors[scene], parameters.radius_min, parameters.radius_max
                ),
                anchors[scene],
                parameters.spawn_z,
            )
            point = pose["x"], pose["y"]
            radius = math.dist(point, anchors[scene])
            if not parameters.radius_min <= radius <= parameters.radius_max:
                rejected["annulus"] += 1
                continue
            map_checks = {
                name: check_map_position(map_data, point, parameters.spawn_clearance)
                for name, map_data in maps.items()
            }
            if not all(check["pass"] for check in map_checks.values()):
                rejected["map"] += 1
                continue
            visibility = check_visibility(
                maps[scene],
                pose,
                p1s[scene],
                parameters.max_target_distance,
                parameters.hfov_deg(robot),
            )
            if not all(item["pass"] for item in visibility):
                rejected[_first_visibility_rejection(visibility)] += 1
                continue
            if any(
                math.dist(point, (old["x"], old["y"]))
                < parameters.separation(robot) - 1e-12
                for old in accepted
            ):
                rejected["separation"] += 1
                continue
            accepted.append(pose)
        generated[robot] = accepted
        statistics[robot] = {
            "scene": scene,
            "attempts": attempts,
            "accepted": len(accepted),
            "rejected": rejected,
        }
    return generated, {"anchors": anchors, "sampling": statistics}


def validate_generated_poses(
    poses: Mapping[str, Sequence[Mapping[str, float]]],
    maps: Mapping[str, OccupancyMap],
    p1s: Mapping[str, Mapping[str, tuple[float, float]]],
    parameters: GenerationParameters,
) -> list[dict[str, object]]:
    anchors = compute_anchors(p1s)
    for robot in ROBOTS:
        if len(poses.get(robot, ())) != parameters.group_count:
            raise ValueError(f"{robot} must contain {parameters.group_count} poses")
    details = []
    for index in range(parameters.group_count):
        group = {"id": index + 1, "robots": {}}
        for robot in ROBOTS:
            scene = RESPONSIBILITY[robot]
            pose = dict(poses[robot][index])
            point = float(pose["x"]), float(pose["y"])
            expected_yaw = yaw_toward(point, anchors[scene])
            yaw_error = abs(normalize_angle(float(pose["yaw"]) - expected_yaw))
            radius = math.dist(point, anchors[scene])
            map_checks = {
                name: check_map_position(map_data, point, parameters.spawn_clearance)
                for name, map_data in maps.items()
            }
            visibility = check_visibility(
                maps[scene],
                pose,
                p1s[scene],
                parameters.max_target_distance,
                parameters.hfov_deg(robot),
            )
            prior = poses[robot][:index]
            separation_pass = all(
                math.dist(point, (float(old["x"]), float(old["y"])))
                >= parameters.separation(robot) - 1e-12
                for old in prior
            )
            passed = (
                parameters.radius_min - 1e-8 <= radius <= parameters.radius_max + 1e-8
                and yaw_error <= 0.5 * 10**-OUTPUT_DECIMALS + 1e-12
                and float(pose["z"]) == round(parameters.spawn_z, OUTPUT_DECIMALS)
                and all(item["pass"] for item in map_checks.values())
                and all(item["pass"] for item in visibility)
                and separation_pass
            )
            group["robots"][robot] = {
                "pass": passed,
                "pose": pose,
                "responsibility": scene,
                "radius_m": radius,
                "yaw_error_rad": yaw_error,
                "map_checks": map_checks,
                "visibility": visibility,
                "separation_pass": separation_pass,
            }
            if not passed:
                raise ValueError(f"group_{index + 1:02d}.{robot} failed validation")
        details.append(group)
    return details


def pose_groups_document(
    poses: Mapping[str, Sequence[Mapping[str, float]]], group_count: int
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "coordinate_mode": "shared_absolute",
        "pose_groups": {
            f"group_{index + 1:02d}": {
                "resolved": True,
                "robots": {robot: dict(poses[robot][index]) for robot in ROBOTS},
            }
            for index in range(group_count)
        },
    }


def dump_pose_groups_yaml(document: Mapping[str, object]) -> str:
    return yaml.dump(
        document,
        Dumper=TwoDecimalSafeDumper,
        sort_keys=False,
        allow_unicode=True,
    )


def poses_from_document(
    document: Mapping[str, object], group_count: int = 11
) -> dict[str, list[dict[str, float]]]:
    try:
        groups = document["pose_groups"]
        result = {robot: [] for robot in ROBOTS}
        for index in range(1, group_count + 1):
            group = groups[f"group_{index:02d}"]
            if group["resolved"] is not True:
                raise ValueError(f"group_{index:02d} is not resolved")
            for robot in ROBOTS:
                result[robot].append(
                    {key: float(group["robots"][robot][key]) for key in ("x", "y", "z", "yaw")}
                )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid robot pose groups document: {error}") from error
    return result


def _parameters_dict(parameters: GenerationParameters) -> dict[str, object]:
    return {
        "seed": parameters.seed,
        "group_count": parameters.group_count,
        "radius_min": parameters.radius_min,
        "radius_max": parameters.radius_max,
        "spawn_z": parameters.spawn_z,
        "spawn_clearance": parameters.spawn_clearance,
        "max_target_distance": parameters.max_target_distance,
        "camera_hfov_deg": parameters.camera_hfov_deg,
        "min_pose_separation": parameters.min_pose_separation,
        "role_overrides": {
            "go2_2": {
                "camera_hfov_deg": parameters.go2_2_camera_hfov_deg,
                "min_pose_separation": parameters.go2_2_min_pose_separation,
            }
        },
        "max_attempts_per_robot": parameters.max_attempts_per_robot,
    }


def build_report(
    parameters: GenerationParameters,
    generation: Mapping[str, object],
    validation: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    anchors = {
        scene: [point[0], point[1]]
        for scene, point in generation["anchors"].items()
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "generation": _parameters_dict(parameters),
        "responsibility": RESPONSIBILITY,
        "anchors": anchors,
        "sampling": generation["sampling"],
        "groups": list(validation),
        "notes": [
            "Forest/go2_2 uses 90 deg HFOV and 0.5 m inter-group separation because the original 60 deg/2.0 m constraints have no solution in the 6-10 m annulus."
        ],
    }


def _world_to_pixel(map_data: OccupancyMap, point: tuple[float, float]) -> tuple[int, int]:
    cell = map_data.world_to_cell(*point)
    if cell is None:
        raise ValueError(f"point {point} lies outside {map_data.scene} map")
    return cell[1], cell[0]


def render_scene(
    scene: str,
    map_data: OccupancyMap,
    poses: Mapping[str, Sequence[Mapping[str, float]]],
    p1s: Mapping[str, tuple[float, float]],
    anchor: tuple[float, float],
) -> bytes:
    image = cv2.cvtColor(map_data.pixels, cv2.COLOR_GRAY2BGR)
    colors = {"go2_1": (40, 70, 230), "go2_2": (50, 180, 60), "go2_3": (220, 90, 40)}
    responsible = ROBOTS[SCENES.index(scene)]
    for pose in poses[responsible]:
        start = _world_to_pixel(map_data, (pose["x"], pose["y"]))
        for target in p1s.values():
            cv2.line(image, start, _world_to_pixel(map_data, target), (170, 170, 170), 1, cv2.LINE_AA)
    for robot in ROBOTS:
        for index, pose in enumerate(poses[robot], start=1):
            pixel = _world_to_pixel(map_data, (pose["x"], pose["y"]))
            cv2.circle(image, pixel, 5, colors[robot], -1, cv2.LINE_AA)
            cv2.putText(image, str(index), (pixel[0] + 5, pixel[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[robot], 1, cv2.LINE_AA)
    for route, target in p1s.items():
        pixel = _world_to_pixel(map_data, target)
        cv2.drawMarker(image, pixel, (200, 30, 200), cv2.MARKER_DIAMOND, 14, 2, cv2.LINE_AA)
        cv2.putText(image, route, (pixel[0] + 7, pixel[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 30, 200), 1, cv2.LINE_AA)
    cv2.drawMarker(image, _world_to_pixel(map_data, anchor), (0, 180, 255), cv2.MARKER_STAR, 20, 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (min(image.shape[1] - 1, 720), 32), (255, 255, 255), -1)
    cv2.putText(image, f"{scene}: all poses; LOS={responsible}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2, cv2.LINE_AA)
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
    print("Target anchors:")
    for scene, point in report["anchors"].items():
        print(f"  {scene:7s} ({point[0]:.6f}, {point[1]:.6f})")
    for group in report["groups"]:
        print(f"Group {group['id']:02d}")
        for robot in ROBOTS:
            item = group["robots"][robot]
            visible = sum(target["pass"] for target in item["visibility"])
            print(
                f"  {robot} PASS  maps=3/3  "
                f"{item['responsibility']} P1 visibility={visible}/3"
            )


def run(
    routes_path: Path,
    maps_root: Path,
    output_path: Path,
    report_dir: Path,
    parameters: GenerationParameters,
    check: bool = False,
) -> dict[str, object]:
    p1s = load_route_p1s(routes_path)
    maps = load_maps(maps_root)
    poses, generation = generate_poses(maps, p1s, parameters)
    document = pose_groups_document(poses, parameters.group_count)
    if check:
        try:
            existing = yaml.safe_load(Path(output_path).read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as error:
            raise ValueError(f"failed to read frozen poses {output_path}: {error}") from error
        if existing != document:
            raise ValueError(f"{output_path} does not match deterministic generation")
        poses = poses_from_document(existing, parameters.group_count)
    validation = validate_generated_poses(poses, maps, p1s, parameters)
    report = build_report(parameters, generation, validation)
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
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--radius-min", type=float, default=6.0)
    parser.add_argument("--radius-max", type=float, default=10.0)
    parser.add_argument("--spawn-z", type=float, default=0.4)
    parser.add_argument("--spawn-clearance", type=float, default=0.8)
    parser.add_argument("--max-target-distance", type=float, default=25.0)
    parser.add_argument("--camera-hfov-deg", type=float, default=60.0)
    parser.add_argument("--min-pose-separation", type=float, default=2.0)
    parser.add_argument("--go2-2-camera-hfov-deg", type=float, default=90.0)
    parser.add_argument("--go2-2-min-pose-separation", type=float, default=0.5)
    parser.add_argument("--max-attempts-per-robot", type=int, default=1_000_000)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    parameters = GenerationParameters(
        seed=args.seed,
        group_count=11,
        radius_min=args.radius_min,
        radius_max=args.radius_max,
        spawn_z=args.spawn_z,
        spawn_clearance=args.spawn_clearance,
        max_target_distance=args.max_target_distance,
        camera_hfov_deg=args.camera_hfov_deg,
        min_pose_separation=args.min_pose_separation,
        go2_2_camera_hfov_deg=args.go2_2_camera_hfov_deg,
        go2_2_min_pose_separation=args.go2_2_min_pose_separation,
        max_attempts_per_robot=args.max_attempts_per_robot,
    )
    try:
        run(args.routes, args.maps_root, args.output, args.report_dir, parameters, args.check)
    except (ValueError, RuntimeError) as error:
        print(f"ERROR: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
