import copy
import math
from pathlib import Path
import random

import numpy as np
import pytest
import yaml

from pedestrian_map.robot_pose_generation import (
    DEFAULT_REFERENCE_POSES,
    REFERENCE_ROBOTS,
    SCENES,
    ROBOTS,
    GenerationParameters,
    OccupancyMap,
    check_map_position,
    check_visibility,
    compute_anchors,
    dump_pose_groups_yaml,
    generate_poses,
    line_of_sight,
    load_occupancy_map,
    load_maps,
    load_reference_poses,
    load_route_p1s,
    render_scene,
    normalize_angle,
    parse_args,
    pose_groups_document,
    poses_from_document,
    run,
    sample_annulus,
    validate_generated_poses,
    yaw_toward,
)
from pedestrian_map.paths import MAPS_ROOT, ROBOT_POSE_GROUPS, ROBOT_POSE_REPORT_ROOT, TARGET_ROUTES


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


def references(group_count=11):
    return {
        scene: {f"group_{index:02d}": {"x": 0.0, "y": 0.0, "z": 0.6, "yaw": 0.13}
                for index in range(1, group_count + 1)}
        for scene in SCENES
    }


def write_references(path, values):
    path.write_text(yaml.safe_dump({
        "schema_version": 1,
        "scenes": {scene: {"reference_robot": REFERENCE_ROBOTS[scene], "poses": groups}
                   for scene, groups in values.items()},
    }, sort_keys=False))


def test_scene_generation_preserves_references_constraints_and_serialization():
    maps = {scene: memory_map(scene) for scene in SCENES}
    parameters = GenerationParameters(seed=9)
    frozen = references()
    original = copy.deepcopy(frozen)
    first, _ = generate_poses(maps, p1s(), parameters, frozen)
    second, _ = generate_poses(maps, p1s(), parameters, frozen)
    different, _ = generate_poses(maps, p1s(), GenerationParameters(seed=10), frozen)
    assert first == second and first != different
    assert frozen == original
    for scene in SCENES:
        assert len(first[scene]) == 11
        for name, group in first[scene].items():
            assert set(group["robots"]) == set(ROBOTS)
            reference = frozen[scene][name]
            assert group["robots"][REFERENCE_ROBOTS[scene]] == reference
            navigation = [pose for robot, pose in group["robots"].items()
                          if robot != REFERENCE_ROBOTS[scene]]
            for pose in navigation:
                assert 3 <= math.hypot(pose["x"], pose["y"]) <= 8
                assert pose["z"] == reference["z"]
                assert pose["yaw"] == round(yaw_toward((pose["x"], pose["y"]), (0, 0)), 2)
                assert check_map_position(maps[scene], (pose["x"], pose["y"]), .8)["pass"]
            assert math.dist((navigation[0]["x"], navigation[0]["y"]),
                             (navigation[1]["x"], navigation[1]["y"])) >= 2
    document = pose_groups_document(first, 11)
    rendered = dump_pose_groups_yaml(document)
    assert "z: 0.60" in rendered
    assert document["schema_version"] == 2
    assert document["coordinate_mode"] == "scene_absolute"
    assert yaml.safe_load(rendered) == document
    restored = poses_from_document(document)
    assert restored == first
    details = validate_generated_poses(restored, maps, p1s(), parameters, frozen)
    assert len(details) == 99
    assert all(item["pass"] for item in details)


def test_current_scene_only_and_no_visibility_constraint(monkeypatch):
    from pedestrian_map import robot_pose_generation as generation

    maps = {scene: memory_map(scene) for scene in SCENES}
    frozen = references(1)
    parameters = GenerationParameters(group_count=1)
    first, _ = generate_poses(maps, p1s(), parameters, frozen)
    city_pose = first["city"]["group_01"]["robots"]["go2_2"]
    cell = maps["airport"].world_to_cell(city_pose["x"], city_pose["y"])
    maps["airport"].blocked[cell] = True
    assert not check_map_position(maps["airport"], (city_pose["x"], city_pose["y"]), .8)["pass"]
    # All P1s are behind an obstacle, but neither navigation robot needs visibility.
    targets = p1s((15.0, 15.0))
    for grid in maps.values():
        grid.blocked[grid.world_to_cell(15, 15)] = True
    monkeypatch.setattr(generation, "check_visibility", lambda *a: pytest.fail("visibility must not be used"))
    second, _ = generate_poses(maps, targets, parameters, frozen)
    assert second["city"]["group_01"]["robots"]["go2_2"]["x"] == city_pose["x"]
    assert second["city"]["group_01"]["robots"]["go2_2"]["y"] == city_pose["y"]
    assert all(item["pass"] for item in validate_generated_poses(second, maps, targets, parameters, frozen))


def test_scene_random_streams_are_independent():
    maps = {scene: memory_map(scene) for scene in SCENES}
    frozen, parameters = references(2), GenerationParameters(group_count=2)
    first, _ = generate_poses(maps, p1s(), parameters, frozen)
    candidate = first["city"]["group_01"]["robots"]["go2_2"]
    maps["city"].blocked[maps["city"].world_to_cell(candidate["x"], candidate["y"])] = True
    second, _ = generate_poses(maps, p1s(), parameters, frozen)
    assert second["city"]["group_01"] != first["city"]["group_01"]
    assert second["forest"] == first["forest"]
    assert second["airport"] == first["airport"]


def test_rounded_positions_are_checked_at_radius_and_separation_boundaries(monkeypatch):
    from pedestrian_map import robot_pose_generation as generation

    maps = {scene: memory_map(scene) for scene in SCENES}
    frozen = references(1)
    # Radius rounds outside the maximum, then to the minimum; second dog first
    # rounds to insufficient separation, then exactly to the allowed maximum.
    samples = iter([(8.006, 0), (2.999, 0), (4.994, 0), (7.999, 0)] * 3)
    monkeypatch.setattr(generation, "sample_annulus", lambda *a: next(samples))
    parameters = GenerationParameters(group_count=1)
    poses, statistics = generate_poses(maps, p1s(), parameters, frozen)
    for scene in SCENES:
        navigation = [p for robot, p in poses[scene]["group_01"]["robots"].items()
                      if robot != REFERENCE_ROBOTS[scene]]
        assert [p["x"] for p in navigation] == [3.0, 8.0]
        counts = list(statistics["sampling"][scene]["group_01"].values())
        assert counts[0]["rejected"]["annulus"] == 1
        assert counts[1]["rejected"]["separation"] == 1
    assert all(item["pass"] for item in validate_generated_poses(poses, maps, p1s(), parameters, frozen))


@pytest.mark.parametrize("change", ["reference", "radius", "separation", "yaw", "z", "clearance"])
def test_validation_detects_corruption(change):
    maps = {scene: memory_map(scene) for scene in SCENES}
    frozen, parameters = references(1), GenerationParameters(group_count=1)
    poses, _ = generate_poses(maps, p1s(), parameters, frozen)
    robots = poses["city"]["group_01"]["robots"]
    if change == "reference":
        robots["go2_1"]["yaw"] += .1
    elif change == "radius":
        robots["go2_2"]["x"], robots["go2_2"]["y"] = 20.0, 0.0
    elif change == "separation":
        robots["go2_3"] = dict(robots["go2_2"])
    elif change in ("yaw", "z"):
        robots["go2_2"][change] += .1
    else:
        pose = robots["go2_2"]
        maps["city"].clearance[maps["city"].world_to_cell(pose["x"], pose["y"])] = .79
    details = validate_generated_poses(poses, maps, p1s(), parameters, frozen)
    assert any(not item["pass"] for item in details if item["scene"] == "city")
    assert all(item["pass"] for item in details if item["scene"] != "city")


def test_spawn_height_override_only_changes_new_robots():
    maps = {scene: memory_map(scene) for scene in SCENES}
    frozen, parameters = references(1), GenerationParameters(group_count=1, spawn_z=.8)
    poses, _ = generate_poses(maps, p1s(), parameters, frozen)
    for scene in SCENES:
        for robot, pose in poses[scene]["group_01"]["robots"].items():
            assert pose["z"] == (.6 if robot == REFERENCE_ROBOTS[scene] else .8)
    assert all(item["pass"] for item in validate_generated_poses(poses, maps, p1s(), parameters, frozen))


def test_generation_failure_does_not_overwrite_output(tmp_path):
    maps_root = tmp_path / "maps"
    pixels = np.zeros((20, 20), dtype=np.uint8)
    # Fixed robot is valid, but there are no valid navigation positions.
    pixels[9, 10] = 255
    for scene in SCENES:
        write_map(maps_root, scene, pixels)
    routes, reference_path = tmp_path / "routes.yaml", tmp_path / "references.yaml"
    write_routes(routes)
    write_references(reference_path, references())
    output = tmp_path / "robot_pose_groups.yaml"
    output.write_text("keep-me\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="city/group_01/go2_2: accepted 0/1"):
        run(routes, maps_root, output, tmp_path / "reports",
            GenerationParameters(max_attempts_per_robot=2, spawn_clearance=.1),
            reference_path=reference_path)
    assert output.read_text(encoding="utf-8") == "keep-me\n"
    assert not (tmp_path / "reports").exists()


def test_run_outputs_are_reproducible_and_check_is_read_only(tmp_path):
    maps_root = tmp_path / "maps"
    for scene in SCENES:
        write_map(maps_root, scene, np.full((20, 20), 255, dtype=np.uint8))
    routes, reference_path = tmp_path / "routes.yaml", tmp_path / "references.yaml"
    write_routes(routes)
    write_references(reference_path, references())
    output, report_dir = tmp_path / "poses.yaml", tmp_path / "reports"
    args = (routes, maps_root, output, report_dir, GenerationParameters())
    report = run(*args, reference_path=reference_path)
    paths = [output, *sorted(report_dir.iterdir())]
    assert len(paths) == 5
    before = {path: path.read_bytes() for path in paths}
    second = run(*args, reference_path=reference_path)
    assert report == second
    assert before == {path: path.read_bytes() for path in paths}
    timestamps = {path: path.stat().st_mtime_ns for path in paths}
    assert run(*args, check=True, reference_path=reference_path) == report
    assert timestamps == {path: path.stat().st_mtime_ns for path in paths}
    assert before == {path: path.read_bytes() for path in paths}
    document = yaml.safe_load(output.read_text())
    document["scenes"]["city"]["pose_groups"]["group_01"]["robots"]["go2_2"]["x"] += .1
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    corrupted = output.read_bytes()
    with pytest.raises(ValueError, match="does not match deterministic generation"):
        run(*args, check=True, reference_path=reference_path)
    assert output.read_bytes() == corrupted
    assert all(path.read_bytes() == before[path] for path in paths if path != output)


def test_render_only_uses_current_scene():
    maps = {scene: memory_map(scene) for scene in SCENES}
    poses, _ = generate_poses(maps, p1s(), GenerationParameters(group_count=1), references(1))
    anchors = compute_anchors(p1s())
    first = render_scene("city", maps["city"], poses, p1s()["city"], anchors["city"])
    del poses["forest"]
    del poses["airport"]
    assert render_scene("city", maps["city"], poses, p1s()["city"], anchors["city"]) == first


@pytest.mark.parametrize("override", [
    {"neighbor_radius_min": 0}, {"neighbor_radius_max": float("nan")},
    {"neighbor_radius_min": 8, "neighbor_radius_max": 3},
    {"spawn_clearance": -1}, {"min_robot_separation": True},
    {"spawn_z": 0}, {"spawn_z": .001}, {"spawn_z": float("inf")},
    {"group_count": 0}, {"group_count": 1.5}, {"max_attempts_per_robot": 0}, {"seed": True},
])
def test_invalid_parameters(override):
    with pytest.raises(ValueError):
        GenerationParameters(**override).validate()


@pytest.mark.parametrize("mutation", ["schema", "role", "missing", "nan", "z"])
def test_reference_loader_rejects_invalid_baselines(tmp_path, mutation):
    path = tmp_path / "references.yaml"
    write_references(path, references())
    document = yaml.safe_load(path.read_text())
    entry = document["scenes"]["city"]
    if mutation == "schema":
        document["schema_version"] = True
    elif mutation == "role":
        entry["reference_robot"] = "go2_2"
    elif mutation == "missing":
        del entry["poses"]["group_11"]
    else:
        entry["poses"]["group_01"]["x" if mutation == "nan" else "z"] = float("nan") if mutation == "nan" else 0
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError):
        load_reference_poses(path)


def test_committed_scene_poses_match_frozen_baseline_and_real_maps():
    frozen = load_reference_poses(DEFAULT_REFERENCE_POSES)
    poses = poses_from_document(yaml.safe_load(ROBOT_POSE_GROUPS.read_text()))
    maps, targets = load_maps(MAPS_ROOT), load_route_p1s(TARGET_ROUTES)
    generated, _ = generate_poses(maps, targets, GenerationParameters(), frozen)
    assert poses == generated
    details = validate_generated_poses(poses, maps, targets, GenerationParameters(), frozen)
    assert len(details) == 99 and all(item["pass"] for item in details)
    assert sum(item["reference_match"] is True for item in details) == 33


def test_default_and_overridden_paths(tmp_path):
    defaults = parse_args([])
    assert defaults.maps_root == MAPS_ROOT
    assert defaults.output == ROBOT_POSE_GROUPS
    assert defaults.report_dir == ROBOT_POSE_REPORT_ROOT
    assert defaults.reference_poses == DEFAULT_REFERENCE_POSES
    assert defaults.spawn_z is None
    assert defaults.neighbor_radius_min == 3.0
    assert defaults.neighbor_radius_max == 8.0
    assert defaults.spawn_clearance == .8
    assert defaults.min_robot_separation == 2.0
    custom = parse_args([
        "--maps-root", str(tmp_path / "maps"), "--output", str(tmp_path / "poses.yaml"),
        "--report-dir", str(tmp_path / "reports"),
        "--reference-poses", str(tmp_path / "references.yaml"),
    ])
    assert custom.maps_root == tmp_path / "maps"
    assert custom.output == tmp_path / "poses.yaml"
    assert custom.report_dir == tmp_path / "reports"
    assert custom.reference_poses == tmp_path / "references.yaml"
