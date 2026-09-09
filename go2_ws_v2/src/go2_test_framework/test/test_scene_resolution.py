"""Resolved dynamic-encircle scene configuration tests."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import yaml

from go2_scenario_config.scene_config import load_scene_config
from go2_test_framework.common.scene_resolution import (
    resolved_route_points,
    write_resolved_scene_config,
)


PACKAGE_ROOT = Path(__file__).parents[1]
SCENE_ROOT = PACKAGE_ROOT.parent / "go2_scenario_config/config/scenes"


def route(points, traversal):
    return {
        "points": tuple(points),
        "traversal": traversal,
        "speed": 0.25,
        "turn_duration": 1.5,
        "provisional": False,
    }


def test_closed_route_is_unchanged():
    value = route(((0, 0), (2, 0), (2, 2)), "closed")
    assert resolved_route_points(value) == ((0, 0), (2, 0), (2, 2))


def test_multi_point_ping_pong_reverses_interior_points():
    value = route(((0, 0), (1, 0), (2, 0), (3, 0)), "ping_pong")
    assert resolved_route_points(value) == (
        (0, 0), (1, 0), (2, 0), (3, 0), (2, 0), (1, 0)
    )


def test_two_point_ping_pong_inserts_midpoint():
    value = route(((0, 2), (4, 6)), "ping_pong")
    assert resolved_route_points(value) == ((0, 2), (2.0, 4.0), (4, 6))


def test_resolved_scene_only_overrides_dynamic_route_fields(tmp_path):
    source_path = SCENE_ROOT / "city.yaml"
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    case = SimpleNamespace(
        scene="city",
        route_config=route(((1, 2), (5, 6)), "ping_pong"),
    )
    output = tmp_path / "resolved_scene_config.yaml"

    resolved = write_resolved_scene_config(case, source_path, output)

    expected = deepcopy(source)
    expected["dynamic_target"].update({
        "speed": 0.25,
        "turn_duration": 1.5,
        "route": [
            {"x": 1.0, "y": 2.0},
            {"x": 3.0, "y": 4.0},
            {"x": 5.0, "y": 6.0},
        ],
    })
    assert resolved == expected
    loaded = load_scene_config("city", output)
    assert loaded.dynamic_target.route == ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
