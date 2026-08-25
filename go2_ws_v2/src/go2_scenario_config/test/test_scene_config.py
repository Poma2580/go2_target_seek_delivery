"""Scene-route validation and Gazebo actor consistency tests."""

import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml

from go2_scenario_config.scene_config import (
    load_dynamic_target_config,
    load_scene_config,
)


PACKAGE_ROOT = Path(__file__).parents[1]
DELIVERY_ROOT = PACKAGE_ROOT.parents[2]
SCENE_ROOT = PACKAGE_ROOT / "config/scenes"
WORLD_PATHS = {
    "city": DELIVERY_ROOT / "QY_MODEL/target_seek",
    "forest": DELIVERY_ROOT / "KD_MODEL/world/forestV3_dynamic.world",
    "airport": DELIVERY_ROOT / "KD_MODEL/world/airport_dynamic.world",
}


def scene_config(name):
    return load_dynamic_target_config(name, SCENE_ROOT / f"{name}.yaml")


def test_complete_scene_config_exposes_spawn_and_dynamic_target_only():
    scene = load_scene_config("city", SCENE_ROOT / "city.yaml")
    assert scene.name == "city"
    assert scene.world_path == "QY_MODEL/target_seek"
    assert scene.robots["go2_1"].spawn == (-30.0, -10.0, 0.30, -1.54)
    raw = yaml.safe_load((SCENE_ROOT / "city.yaml").read_text(encoding="utf-8"))
    assert not ({"map", "target", "encircle", "controller"} & raw.keys())


@pytest.mark.parametrize("scene", ("city", "forest", "airport"))
def test_default_dynamic_target_routes_are_valid(scene):
    target = scene_config(scene)
    assert target.loop is True
    assert target.speed == pytest.approx(0.15)
    assert target.turn_duration == pytest.approx(2.0)
    assert len(target.route) == 4
    assert len(set(target.route)) == 4


def test_forest_route_and_initial_detection_spawn_are_configured():
    target = scene_config("forest")
    assert target.route == (
        (-50.0, 0.0),
        (-10.0, 0.0),
        (-6.0, -43.0),
        (-23.0, -75.0),
    )
    root = yaml.safe_load((SCENE_ROOT / "forest.yaml").read_text(encoding="utf-8"))
    assert root["robots"]["go2_2"]["spawn"] == {
        "x": -42.0,
        "y": 8.0,
        "z": 0.80,
        "yaw": -2.35619449,
    }


def test_dynamic_target_config_rejects_invalid_fields(tmp_path):
    base = yaml.safe_load((SCENE_ROOT / "city.yaml").read_text(encoding="utf-8"))
    invalid_values = (
        lambda data: data.pop("dynamic_target"),
        lambda data: data["dynamic_target"].update(loop=False),
        lambda data: data["dynamic_target"].update(speed=0.0),
        lambda data: data["dynamic_target"].update(turn_duration=-1.0),
        lambda data: data["dynamic_target"].update(
            route=[{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 0.0}]
        ),
        lambda data: data["dynamic_target"].update(
            route=[{"x": 0.0, "y": 0.0}, {"x": 0.0, "y": 0.0},
                   {"x": 1.0, "y": 1.0}]
        ),
        lambda data: data["dynamic_target"]["route"][0].update(x=float("nan")),
    )
    for index, mutate in enumerate(invalid_values):
        data = yaml.safe_load(yaml.safe_dump(base))
        mutate(data)
        path = tmp_path / f"invalid_{index}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        with pytest.raises(ValueError):
            load_dynamic_target_config("city", path)
    with pytest.raises(ValueError, match="scene must be one of"):
        load_dynamic_target_config("unknown", SCENE_ROOT / "city.yaml")


@pytest.mark.parametrize("scene", ("city", "forest", "airport"))
def test_world_actor_trajectory_matches_scene_yaml(scene):
    target = scene_config(scene)
    root = ET.parse(WORLD_PATHS[scene]).getroot()
    actors = root.findall(".//actor[@name='walking_target']")
    assert len(actors) == 1
    actor = actors[0]
    plugins = actor.findall("./plugin[@name='walking_target_controller']")
    assert len(plugins) == 1
    assert plugins[0].get("filename") == "libwalking_target_controller.so"
    assert plugins[0].findtext("service_prefix") == target.service_prefix
    assert actor.findtext("./script/loop") == "true"

    elements = actor.findall("./script/trajectory/waypoint")
    assert len(elements) == 2 * len(target.route) + 1
    waypoints = []
    for element in elements:
        pose = [float(value) for value in element.findtext("pose").split()]
        waypoints.append((float(element.findtext("time")), pose[0], pose[1]))

    collapsed = []
    for _, x, y in waypoints:
        point = (x, y)
        if not collapsed or point != collapsed[-1]:
            collapsed.append(point)
    assert collapsed == [*target.route, target.route[0]]

    for index, start in enumerate(target.route):
        end = target.route[(index + 1) % len(target.route)]
        departure = waypoints[2 * index]
        arrival = waypoints[2 * index + 1]
        turned = waypoints[2 * index + 2]
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        assert arrival[0] - departure[0] == pytest.approx(
            distance / target.speed, abs=0.002
        )
        assert turned[0] - arrival[0] == pytest.approx(
            target.turn_duration, abs=0.001
        )
