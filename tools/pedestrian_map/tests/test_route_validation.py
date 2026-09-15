from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml


from pedestrian_map import route_validation as validator
from pedestrian_map.grid_geometry import supercover_cells
from pedestrian_map.paths import MAPS_ROOT, REPO_ROOT, ROUTE_VALIDATION_ROOT


def write_map(root: Path, pixels: np.ndarray, *, ascii_pgm: bool = False) -> Path:
    scene_dir = root / "city"
    scene_dir.mkdir(parents=True)
    image_path = scene_dir / "city.pgm"
    if ascii_pgm:
        rows = [" ".join(str(int(value)) for value in row) for row in pixels]
        image_path.write_text(
            f"P2\n{pixels.shape[1]} {pixels.shape[0]}\n255\n" + "\n".join(rows) + "\n",
            encoding="ascii",
        )
    else:
        Image.fromarray(pixels.astype(np.uint8), mode="L").save(image_path)
    metadata = {
        "image": "city.pgm",
        "mode": "trinary",
        "resolution": 1.0,
        "origin": [10.0, 20.0, 0.0],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.20,
    }
    yaml_path = scene_dir / "city.yaml"
    yaml_path.write_text(yaml.safe_dump(metadata), encoding="utf-8")
    return yaml_path


@pytest.mark.parametrize("ascii_pgm", [False, True])
def test_load_p5_and_p2_with_world_y_flip_and_exclusive_max(tmp_path, ascii_pgm):
    pixels = np.full((3, 4), 255, dtype=np.uint8)
    pixels[0, 1] = 0
    map_data = validator.load_map(write_map(tmp_path, pixels, ascii_pgm=ascii_pgm))

    assert map_data.world_to_cell(11.5, 22.5) == (0, 1)
    assert map_data.blocked[map_data.world_to_cell(11.5, 22.5)]
    assert map_data.world_to_cell(10.0, 20.0) == (2, 0)
    assert map_data.world_to_cell(14.0, 20.0) is None
    assert map_data.world_to_cell(10.0, 23.0) is None


def test_unknown_cells_are_blocked(tmp_path):
    pixels = np.full((3, 3), 255, dtype=np.uint8)
    pixels[1, 1] = 128
    map_data = validator.load_map(write_map(tmp_path, pixels))
    assert map_data.blocked[1, 1]


def test_shape_connections_and_point_counts():
    assert validator.route_segments("straight", 2) == ((0, 1),)
    assert validator.route_segments("rectangle", 4) == (
        (0, 1), (1, 2), (2, 3), (3, 0)
    )
    assert validator.route_segments("v", 3) == ((0, 1), (1, 2))
    with pytest.raises(ValueError, match="requires 3 points"):
        validator.route_segments("v", 2)


def route_map(tmp_path: Path):
    pixels = np.full((10, 10), 255, dtype=np.uint8)
    # row 4 is world grid y=5, so this obstacle centre is (15.5, 25.5).
    pixels[4, 5] = 0
    return validator.load_map(write_map(tmp_path, pixels))


def test_pass_collision_clearance_and_nearest_obstacle(tmp_path):
    map_data = route_map(tmp_path)
    safe = validator.validate_route(
        "city", "straight", [(11.5, 21.5), (18.5, 21.5)], 0.4, map_data
    )
    assert safe.passed
    assert safe.minimum_clearance > 2.0

    collision = validator.validate_route(
        "city", "straight", [(11.5, 25.5), (18.5, 25.5)], 0.4, map_data
    )
    assert not collision.passed
    assert collision.segment_results[0].reason == "segment crosses obstacle or unknown cell"
    assert collision.segment_results[0].obstacle == pytest.approx((15.5, 25.5))

    too_close = validator.validate_route(
        "city", "straight", [(11.5, 24.5), (18.5, 24.5)], 0.4, map_data
    )
    assert not too_close.passed
    assert too_close.segment_results[0].reason == "below safety distance"
    assert too_close.segment_results[0].clearance == pytest.approx(1 - 1 / np.sqrt(2))


def test_waypoint_outside_and_inside_obstacle(tmp_path):
    map_data = route_map(tmp_path)
    result = validator.validate_route(
        "city", "straight", [(15.5, 25.5), (20.0, 21.0)], 0.0, map_data
    )
    assert result.point_results[0].reason == "inside obstacle or unknown cell"
    assert result.point_results[1].reason == "outside map"
    assert result.segment_results[0].reason == "endpoint outside map"


def test_supercover_detects_corner_touch(tmp_path):
    pixels = np.full((4, 4), 255, dtype=np.uint8)
    # The diagonal touches this side cell at the (1,1) grid corner.
    pixels[2, 1] = 0
    map_data = validator.load_map(write_map(tmp_path, pixels))
    result = validator.validate_route(
        "city", "straight", [(10.5, 20.5), (12.5, 22.5)], 0.0, map_data
    )
    assert not result.segment_results[0].passed
    assert result.segment_results[0].reason == "segment crosses obstacle or unknown cell"


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ((11.0, 22.0), (16.0, 22.0)),  # horizontal grid edge
        ((13.0, 21.0), (13.0, 26.0)),  # vertical grid edge
        ((11.0, 21.0), (16.0, 26.0)),  # diagonal through grid corners
        ((11.0, 25.0), (16.0, 21.0)),  # oblique line in the reverse Y direction
    ],
)
def test_supercover_grid_boundaries_are_direction_invariant(tmp_path, start, end):
    pixels = np.full((8, 8), 255, dtype=np.uint8)
    map_data = validator.load_map(write_map(tmp_path, pixels))
    forward = set(supercover_cells(map_data, start, end))
    reverse = set(supercover_cells(map_data, end, start))
    assert forward
    assert forward == reverse


def test_city_v_route_integer_coordinates_do_not_exceed_traversal_bound():
    map_data = validator.load_map(MAPS_ROOT / "city/city.yaml")
    points = [(-15.0, 8.0), (-2.0, -30.0), (-18.0, -15.0)]
    for start_index, end_index in validator.route_segments("v", len(points)):
        forward = set(
            supercover_cells(
                map_data, points[start_index], points[end_index]
            )
        )
        reverse = set(
            supercover_cells(
                map_data, points[end_index], points[start_index]
            )
        )
        assert forward == reverse


def test_plot_and_cli_exit_codes(tmp_path):
    map_data = route_map(tmp_path)
    result = validator.validate_route(
        "city", "v", [(11.5, 21.5), (12.5, 22.5), (13.5, 21.5)], 0.0, map_data
    )
    output = tmp_path / "plot.png"
    validator.plot_result(map_data, result, output)
    assert output.is_file() and output.stat().st_size > 0

    assert validator.main([
        "--scene", "city", "--shape", "straight",
        "--point", "11.5", "21.5", "--point", "18.5", "21.5",
        "--safety-distance", "0.4", "--map-root", str(tmp_path),
    ]) == 0
    assert validator.main([
        "--scene", "city", "--shape", "straight",
        "--point", "11.5", "25.5", "--point", "18.5", "25.5",
        "--safety-distance", "0.4", "--map-root", str(tmp_path),
    ]) == 1
    assert validator.main([
        "--scene", "city", "--shape", "v",
        "--point", "11.5", "21.5", "--point", "12.5", "21.5",
        "--safety-distance", "0.4", "--map-root", str(tmp_path),
    ]) == 2


def test_default_and_overridden_artifact_paths(tmp_path):
    defaults = validator.build_parser().parse_args([])
    assert defaults.map_root == MAPS_ROOT
    assert MAPS_ROOT == REPO_ROOT / "tools/gazebo_map_creator/artifacts/maps"
    assert defaults.plot_root == ROUTE_VALIDATION_ROOT

    pixels = np.full((10, 10), 255, dtype=np.uint8)
    write_map(tmp_path / "maps", pixels)
    plot_root = tmp_path / "route_plots"
    common = [
        "--scene", "city", "--shape", "straight",
        "--point", "11.5", "21.5", "--point", "18.5", "21.5",
        "--safety-distance", "0.4", "--map-root", str(tmp_path / "maps"),
        "--plot",
    ]
    assert validator.main([*common, "--plot-root", str(plot_root)]) == 0
    assert (plot_root / "city/straight_validation.png").is_file()

    explicit_plot = tmp_path / "custom.png"
    assert validator.main([*common, "--plot-output", str(explicit_plot)]) == 0
    assert explicit_plot.is_file()
