import math
from pathlib import Path
import random

import numpy as np
import pytest
import yaml

from pedestrian_map.robot_pose_generation import (
    GenerationParameters,
    OccupancyMap,
    check_map_position,
    check_visibility,
    compute_anchors,
    dump_pose_groups_yaml,
    generate_poses,
    line_of_sight,
    load_occupancy_map,
    normalize_angle,
    parse_args,
    pose_groups_document,
    poses_from_document,
    run,
    sample_annulus,
    validate_generated_poses,
    yaw_toward,
)
from pedestrian_map.paths import MAPS_ROOT, ROBOT_POSE_GROUPS, ROBOT_POSE_REPORT_ROOT


def p1s(center=(0.0, 0.0)):
    x, y = center
    return {
        scene: {
            "straight": (x, y),
            "rectangle": (x, y),
            "v_shape": (x, y),
        }
        for scene in ("city", "forest", "airport")
    }


def memory_map(scene="city", *, size=100, resolution=1.0, origin=(-50.0, -50.0)):
    pixels = np.full((size, size), 255, dtype=np.uint8)
    blocked = np.zeros_like(pixels, dtype=bool)
    clearance = np.full_like(pixels, 1000.0, dtype=np.float32)
    return OccupancyMap(
        scene=scene,
        yaml_path=Path(f"{scene}.yaml"),
        pixels=pixels,
        blocked=blocked,
        clearance=clearance,
        resolution=resolution,
        origin_x=origin[0],
        origin_y=origin[1],
    )


def write_map(root: Path, scene: str, pixels: np.ndarray) -> Path:
    directory = root / scene
    directory.mkdir(parents=True)
    image = directory / f"{scene}.pgm"
    height, width = pixels.shape
    image.write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels.tobytes())
    metadata = {
        "image": image.name,
        "resolution": 1.0,
        "origin": [-10.0, -10.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.20,
    }
    path = directory / f"{scene}.yaml"
    path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return path


def write_routes(path: Path, center=(0.0, 0.0)) -> None:
    scenes = {}
    for scene in ("city", "forest", "airport"):
        scenes[scene] = {
            "routes": {
                route: {"points": [list(center), [center[0] + 1.0, center[1]]]}
                for route in ("straight", "rectangle", "v_shape")
            }
        }
    path.write_text(yaml.safe_dump({"schema_version": 1, "scenes": scenes}), encoding="utf-8")


def test_compute_anchors_uses_all_three_p1_values():
    values = p1s()
    values["city"] = {
        "straight": (-10.0, 0.0),
        "rectangle": (-13.0, 4.0),
        "v_shape": (-15.0, 8.0),
    }
    assert compute_anchors(values)["city"] == pytest.approx((-38.0 / 3.0, 4.0))


def test_area_uniform_annulus_and_yaw_toward_anchor():
    rng = random.Random(123)
    squared_radii = []
    for _ in range(5000):
        point = sample_annulus(rng, (2.0, -3.0), 6.0, 10.0)
        radius = math.dist(point, (2.0, -3.0))
        assert 6.0 <= radius <= 10.0
        squared_radii.append(radius**2)
    assert np.mean(squared_radii) == pytest.approx((36.0 + 100.0) / 2.0, abs=1.0)
    assert yaw_toward((0.0, 0.0), (0.0, 2.0)) == pytest.approx(math.pi / 2.0)


def test_map_bounds_unknown_obstacle_and_clearance(tmp_path):
    pixels = np.full((20, 20), 255, dtype=np.uint8)
    pixels[10, 10] = 0
    pixels[11, 11] = 127
    loaded = load_occupancy_map("city", write_map(tmp_path, "city", pixels))
    assert check_map_position(loaded, (-11.0, 0.0), 0.0)["reason"] == "outside_map"
    obstacle = loaded.cell_center(10, 10)
    unknown = loaded.cell_center(11, 11)
    assert check_map_position(loaded, obstacle, 0.0)["reason"] == "occupied_or_unknown"
    assert check_map_position(loaded, unknown, 0.0)["reason"] == "occupied_or_unknown"
    near = loaded.cell_center(10, 11)
    assert check_map_position(loaded, near, 0.8)["reason"] == "below_clearance"
    far = loaded.cell_center(0, 0)
    assert check_map_position(loaded, far, 0.8)["pass"]


def test_los_and_fov_include_boundaries_and_wrap_angles():
    grid = memory_map(size=30, origin=(-15.0, -15.0))
    pose = {"x": 0.0, "y": 0.0, "z": 0.4, "yaw": math.radians(179.0)}
    angle = math.radians(-151.0)
    target = (10.0 * math.cos(angle), 10.0 * math.sin(angle))
    targets = {route: target for route in ("straight", "rectangle", "v_shape")}
    results = check_visibility(grid, pose, targets, 10.0, 60.0)
    assert all(item["distance_pass"] and item["fov_pass"] for item in results)
    assert abs(normalize_angle(math.radians(-151.0 - 179.0))) == pytest.approx(math.radians(30.0))
    assert line_of_sight(grid, (0.0, 0.0), target)
    blocked_cell = grid.world_to_cell(*(target[0] / 2.0, target[1] / 2.0))
    grid.blocked[blocked_cell] = True
    assert not line_of_sight(grid, (0.0, 0.0), target)


def test_three_map_filter_seed_reproducibility_overrides_and_serialization():
    maps = {scene: memory_map(scene) for scene in ("city", "forest", "airport")}
    parameters = GenerationParameters(
        seed=9,
        group_count=3,
        radius_min=6.0,
        radius_max=10.0,
        spawn_clearance=0.1,
        max_target_distance=25.0,
        min_pose_separation=0.1,
        go2_2_min_pose_separation=0.05,
    )
    first, _ = generate_poses(maps, p1s(), parameters)
    second, _ = generate_poses(maps, p1s(), parameters)
    different, _ = generate_poses(
        maps, p1s(), GenerationParameters(**{**parameters.__dict__, "seed": 10})
    )
    assert first == second
    assert first != different
    assert all(
        value == round(value, 2)
        for robot_poses in first.values()
        for pose in robot_poses
        for value in pose.values()
    )
    assert parameters.hfov_deg("go2_1") == 60.0
    assert parameters.hfov_deg("go2_2") == 90.0
    assert parameters.separation("go2_1") == 0.1
    assert parameters.separation("go2_2") == 0.05
    document = pose_groups_document(first, 3)
    rendered = dump_pose_groups_yaml(document)
    assert "z: 0.40" in rendered
    assert yaml.safe_load(rendered) == document
    restored = poses_from_document(document, 3)
    assert restored == first
    details = validate_generated_poses(restored, maps, p1s(), parameters)
    assert len(details) == 3
    assert all(item["pass"] for group in details for item in group["robots"].values())


def test_blocked_in_any_map_rejects_candidate():
    maps = {scene: memory_map(scene) for scene in ("city", "forest", "airport")}
    point = (7.0, 0.0)
    cell = maps["airport"].world_to_cell(*point)
    maps["airport"].blocked[cell] = True
    checks = {
        scene: check_map_position(item, point, 0.1) for scene, item in maps.items()
    }
    assert checks["city"]["pass"] and checks["forest"]["pass"]
    assert not checks["airport"]["pass"]


def test_generation_failure_does_not_overwrite_output(tmp_path):
    maps_root = tmp_path / "maps"
    pixels = np.zeros((20, 20), dtype=np.uint8)
    for scene in ("city", "forest", "airport"):
        write_map(maps_root, scene, pixels)
    routes = tmp_path / "routes.yaml"
    write_routes(routes)
    output = tmp_path / "robot_pose_groups.yaml"
    output.write_text("keep-me\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="accepted 0/11"):
        run(
            routes,
            maps_root,
            output,
            tmp_path / "reports",
            GenerationParameters(max_attempts_per_robot=2),
        )
    assert output.read_text(encoding="utf-8") == "keep-me\n"
    assert not (tmp_path / "reports").exists()


def test_default_and_overridden_paths(tmp_path):
    defaults = parse_args([])
    assert defaults.maps_root == MAPS_ROOT
    assert defaults.output == ROBOT_POSE_GROUPS
    assert defaults.report_dir == ROBOT_POSE_REPORT_ROOT

    custom = parse_args([
        "--maps-root", str(tmp_path / "maps"),
        "--output", str(tmp_path / "poses.yaml"),
        "--report-dir", str(tmp_path / "reports"),
    ])
    assert custom.maps_root == tmp_path / "maps"
    assert custom.output == tmp_path / "poses.yaml"
    assert custom.report_dir == tmp_path / "reports"
