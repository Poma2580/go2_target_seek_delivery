#!/usr/bin/env python3
"""Validate pedestrian waypoint routes against a Nav2 occupancy map."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable, Optional, Sequence

import numpy as np
from scipy import ndimage
import yaml

from .grid_geometry import snap_grid_coordinate, supercover_cells
from .map_io import read_grayscale_image
from .paths import MAPS_ROOT, ROUTE_VALIDATION_ROOT


DEFAULT_SAFETY_DISTANCE = 0.4
SHAPE_POINT_COUNTS = {"straight": 2, "rectangle": 4, "v": 3}


@dataclass(frozen=True)
class MapData:
    yaml_path: Path
    image_path: Path
    pixels: np.ndarray
    blocked: np.ndarray
    resolution: float
    origin_x: float
    origin_y: float
    negate: bool
    occupied_thresh: float
    free_thresh: float
    clearance: np.ndarray
    nearest_rows: Optional[np.ndarray]
    nearest_cols: Optional[np.ndarray]

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def max_x(self) -> float:
        return self.origin_x + self.width * self.resolution

    @property
    def max_y(self) -> float:
        return self.origin_y + self.height * self.resolution

    def world_to_cell(self, x: float, y: float) -> Optional[tuple[int, int]]:
        """Return (row, column), with the PGM's top-down row convention."""
        grid_x = snap_grid_coordinate((x - self.origin_x) / self.resolution)
        grid_y = snap_grid_coordinate((y - self.origin_y) / self.resolution)
        gx = math.floor(grid_x)
        gy = math.floor(grid_y)
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return self.height - 1 - gy, gx

    def cell_center(self, row: int, col: int) -> tuple[float, float]:
        gx = col
        gy = self.height - 1 - row
        return (
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def clearance_at_cell(self, row: int, col: int) -> float:
        if self.blocked[row, col]:
            return 0.0
        return max(0.0, float(self.clearance[row, col]))

    def nearest_obstacle(self, row: int, col: int) -> Optional[tuple[float, float]]:
        if self.nearest_rows is None or self.nearest_cols is None:
            return None
        return self.cell_center(
            int(self.nearest_rows[row, col]), int(self.nearest_cols[row, col])
        )


@dataclass(frozen=True)
class CheckResult:
    label: str
    passed: bool
    clearance: Optional[float]
    reason: Optional[str] = None
    obstacle: Optional[tuple[float, float]] = None


@dataclass(frozen=True)
class RouteResult:
    scene: str
    shape: str
    safety_distance: float
    points: tuple[tuple[float, float], ...]
    point_results: tuple[CheckResult, ...]
    segment_results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return all(item.passed for item in self.point_results + self.segment_results)

    @property
    def minimum_clearance(self) -> Optional[float]:
        values = [
            item.clearance
            for item in self.point_results + self.segment_results
            if item.clearance is not None
        ]
        return min(values) if values else None


def load_map(yaml_path: Path) -> MapData:
    yaml_path = yaml_path.expanduser().resolve()
    if not yaml_path.is_file():
        raise ValueError(f"map YAML does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    if not isinstance(metadata, dict):
        raise ValueError(f"invalid map YAML: {yaml_path}")

    try:
        image_value = str(metadata["image"])
        resolution = float(metadata["resolution"])
        origin = metadata["origin"]
        origin_x, origin_y = float(origin[0]), float(origin[1])
        negate = bool(int(metadata.get("negate", 0)))
        occupied_thresh = float(metadata.get("occupied_thresh", 0.65))
        free_thresh = float(metadata.get("free_thresh", 0.196))
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"invalid map metadata in {yaml_path}: {error}") from error
    if resolution <= 0.0:
        raise ValueError("map resolution must be positive")
    if not 0.0 <= free_thresh < occupied_thresh <= 1.0:
        raise ValueError("map thresholds must satisfy 0 <= free < occupied <= 1")

    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = yaml_path.parent / image_path
    image_path = image_path.resolve()
    if not image_path.is_file():
        raise ValueError(f"map image does not exist: {image_path}")
    pixels = read_grayscale_image(image_path)
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError(f"map image is not a non-empty grayscale image: {image_path}")

    probability = pixels.astype(np.float64) / 255.0
    if not negate:
        probability = 1.0 - probability
    occupied = probability > occupied_thresh
    free = probability < free_thresh
    blocked = np.logical_or(occupied, np.logical_not(np.logical_or(occupied, free)))

    if np.any(blocked):
        distance_pixels, indices = ndimage.distance_transform_edt(
            np.logical_not(blocked), return_indices=True
        )
        # Convert centre-to-centre distance to a conservative distance to the
        # occupied square.  Half a pixel diagonal bounds every cell edge.
        clearance = np.maximum(
            0.0, distance_pixels * resolution - resolution / math.sqrt(2.0)
        )
        nearest_rows, nearest_cols = indices[0], indices[1]
    else:
        clearance = np.full(pixels.shape, np.inf, dtype=np.float64)
        nearest_rows = nearest_cols = None

    return MapData(
        yaml_path=yaml_path,
        image_path=image_path,
        pixels=pixels,
        blocked=blocked,
        resolution=resolution,
        origin_x=origin_x,
        origin_y=origin_y,
        negate=negate,
        occupied_thresh=occupied_thresh,
        free_thresh=free_thresh,
        clearance=clearance,
        nearest_rows=nearest_rows,
        nearest_cols=nearest_cols,
    )


def route_segments(shape: str, point_count: int) -> tuple[tuple[int, int], ...]:
    expected = SHAPE_POINT_COUNTS.get(shape)
    if expected is None:
        raise ValueError(f"unsupported shape: {shape}")
    if point_count != expected:
        raise ValueError(f"shape '{shape}' requires {expected} points, got {point_count}")
    if shape == "straight":
        return ((0, 1),)
    if shape == "rectangle":
        return ((0, 1), (1, 2), (2, 3), (3, 0))
    return ((0, 1), (1, 2))


def _minimum_sampled_clearance(
    map_data: MapData, start: tuple[float, float], end: tuple[float, float]
) -> tuple[float, Optional[tuple[float, float]]]:
    length = math.dist(start, end)
    count = max(1, int(math.ceil(length / (map_data.resolution / 4.0))))
    best = math.inf
    obstacle = None
    for index in range(count + 1):
        fraction = index / count
        x = start[0] + (end[0] - start[0]) * fraction
        y = start[1] + (end[1] - start[1]) * fraction
        cell = map_data.world_to_cell(x, y)
        if cell is None:
            return 0.0, None
        clearance = map_data.clearance_at_cell(*cell)
        if clearance < best:
            best = clearance
            obstacle = map_data.nearest_obstacle(*cell)
    return best, obstacle


def validate_route(
    scene: str,
    shape: str,
    points: Sequence[tuple[float, float]],
    safety_distance: float,
    map_data: MapData,
) -> RouteResult:
    if safety_distance < 0.0 or not math.isfinite(safety_distance):
        raise ValueError("safety distance must be a finite non-negative number")
    segments = route_segments(shape, len(points))
    point_results = []
    for index, point in enumerate(points):
        label = f"P{index + 1}"
        cell = map_data.world_to_cell(*point)
        if cell is None:
            point_results.append(CheckResult(label, False, None, "outside map"))
            continue
        if map_data.blocked[cell]:
            point_results.append(
                CheckResult(
                    label,
                    False,
                    0.0,
                    "inside obstacle or unknown cell",
                    map_data.cell_center(*cell),
                )
            )
            continue
        clearance = map_data.clearance_at_cell(*cell)
        obstacle = map_data.nearest_obstacle(*cell)
        if clearance < safety_distance:
            point_results.append(
                CheckResult(label, False, clearance, "below safety distance", obstacle)
            )
        else:
            point_results.append(CheckResult(label, True, clearance, obstacle=obstacle))

    segment_results = []
    for start_index, end_index in segments:
        label = f"P{start_index + 1} -> P{end_index + 1}"
        start, end = points[start_index], points[end_index]
        if map_data.world_to_cell(*start) is None or map_data.world_to_cell(*end) is None:
            segment_results.append(CheckResult(label, False, None, "endpoint outside map"))
            continue
        cells = supercover_cells(map_data, start, end)
        collision = next((cell for cell in cells if map_data.blocked[cell]), None)
        clearance, obstacle = _minimum_sampled_clearance(map_data, start, end)
        if collision is not None:
            segment_results.append(
                CheckResult(
                    label,
                    False,
                    0.0,
                    "segment crosses obstacle or unknown cell",
                    map_data.cell_center(*collision),
                )
            )
        elif clearance < safety_distance:
            segment_results.append(
                CheckResult(label, False, clearance, "below safety distance", obstacle)
            )
        else:
            segment_results.append(CheckResult(label, True, clearance, obstacle=obstacle))

    return RouteResult(
        scene=scene,
        shape=shape,
        safety_distance=safety_distance,
        points=tuple(points),
        point_results=tuple(point_results),
        segment_results=tuple(segment_results),
    )


def _format_result(item: CheckResult) -> str:
    text = f"{item.label}: {'PASS' if item.passed else 'FAIL'}"
    details = []
    if item.reason:
        details.append(item.reason)
    if item.clearance is not None:
        details.append(f"clearance={item.clearance:.2f} m")
    if item.obstacle is not None:
        details.append(f"nearest obstacle=({item.obstacle[0]:.2f}, {item.obstacle[1]:.2f})")
    return text + (" - " + "; ".join(details) if details else "")


def print_result(result: RouteResult) -> None:
    print(f"Scene: {result.scene}")
    print(f"Shape: {result.shape}")
    print(f"Safety distance: {result.safety_distance:.2f} m\n")
    for item in result.point_results:
        print(_format_result(item))
    print()
    for item in result.segment_results:
        print(_format_result(item))
    minimum = result.minimum_clearance
    print(f"\nMinimum clearance: {'N/A' if minimum is None else f'{minimum:.2f} m'}")
    print(f"\nRESULT: {'PASS' if result.passed else 'FAIL'}")


def plot_result(map_data: MapData, result: RouteResult, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 8), constrained_layout=True)
    axis.imshow(
        map_data.pixels,
        cmap="gray",
        origin="upper",
        extent=(map_data.origin_x, map_data.max_x, map_data.origin_y, map_data.max_y),
        vmin=0,
        vmax=255,
    )
    segments = route_segments(result.shape, len(result.points))
    for (start_index, end_index), check in zip(segments, result.segment_results):
        start, end = result.points[start_index], result.points[end_index]
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color="tab:green" if check.passed else "tab:red",
            linewidth=2.0,
        )
    for index, (point, check) in enumerate(zip(result.points, result.point_results), 1):
        color = "tab:green" if check.passed else "tab:red"
        axis.scatter([point[0]], [point[1]], color=color, s=45, zorder=3)
        axis.annotate(f"P{index}", point, xytext=(5, 5), textcoords="offset points", color=color)
    axis.set_title(
        f"{result.scene} {result.shape}: {'PASS' if result.passed else 'FAIL'} "
        f"(safety {result.safety_distance:.2f} m)"
    )
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_aspect("equal")
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _parse_interactive_points(text: str) -> list[tuple[float, float]]:
    points = []
    for item in text.split(";"):
        values = item.replace(",", " ").split()
        if len(values) != 2:
            raise ValueError("points must use 'x,y; x,y; ...' format")
        points.append((float(values[0]), float(values[1])))
    return points


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("city", "forest", "airport"))
    parser.add_argument("--shape", choices=tuple(SHAPE_POINT_COUNTS))
    parser.add_argument("--point", nargs=2, type=float, action="append", metavar=("X", "Y"))
    parser.add_argument("--safety-distance", type=float)
    parser.add_argument(
        "--map-root",
        type=Path,
        default=MAPS_ROOT,
    )
    parser.add_argument(
        "--plot-root",
        type=Path,
        default=ROUTE_VALIDATION_ROOT,
        help="directory used for default route overlay paths",
    )
    parser.add_argument("--plot", action="store_true", help="save a route overlay PNG")
    parser.add_argument("--plot-output", type=Path, help="override route overlay path")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        scene = args.scene or input("Scene [city/forest/airport]: ").strip().lower()
        shape = args.shape or input("Shape [straight/rectangle/v]: ").strip().lower()
        if scene not in ("city", "forest", "airport"):
            raise ValueError(f"unsupported scene: {scene}")
        if shape not in SHAPE_POINT_COUNTS:
            raise ValueError(f"unsupported shape: {shape}")
        points = [tuple(point) for point in args.point] if args.point else []
        if not points:
            prompt = f"Enter {SHAPE_POINT_COUNTS[shape]} points (x,y; x,y; ...): "
            points = _parse_interactive_points(input(prompt).strip())
        safety_distance = args.safety_distance
        if safety_distance is None:
            if args.scene is None or args.shape is None or args.point is None:
                text = input(f"Safety distance [{DEFAULT_SAFETY_DISTANCE}]: ").strip()
                safety_distance = float(text) if text else DEFAULT_SAFETY_DISTANCE
            else:
                safety_distance = DEFAULT_SAFETY_DISTANCE

        map_data = load_map(args.map_root / scene / f"{scene}.yaml")
        result = validate_route(scene, shape, points, safety_distance, map_data)
        print_result(result)
        if args.plot:
            output = args.plot_output or (
                args.plot_root / scene / f"{shape}_validation.png"
            )
            plot_result(map_data, result, output)
            print(f"Plot: {output.resolve()}")
        return 0 if result.passed else 1
    except (ValueError, OSError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
