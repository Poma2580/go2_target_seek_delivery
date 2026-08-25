"""Load and validate shared simulation scene configuration."""

import math
from dataclasses import dataclass
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


VALID_SCENES = ("city", "forest", "airport")


@dataclass(frozen=True)
class DynamicTargetConfig:
    """Validated dynamic-target data shared by Gazebo and encirclement."""

    model_name: str
    service_prefix: str
    loop: bool
    speed: float
    turn_duration: float
    route: tuple


@dataclass(frozen=True)
class RobotConfig:
    """Spawn pose and default sensors for one robot."""

    spawn: tuple
    lidar: bool
    camera: bool


@dataclass(frozen=True)
class SceneConfig:
    """Validated configuration shared by spawn and mission nodes."""

    name: str
    world_path: str
    robots: dict
    dynamic_target: DynamicTargetConfig


def _mapping(value, field):
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a YAML mapping")
    return value


def _finite_number(value, field):
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _config_path(scene, scene_config=""):
    scene = str(scene).strip().lower()
    if scene not in VALID_SCENES:
        raise ValueError(
            f"scene must be one of {', '.join(VALID_SCENES)}, got {scene!r}"
        )
    configured_path = str(scene_config).strip()
    return scene, (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else Path(get_package_share_directory("go2_scenario_config"))
        / "config" / "scenes" / f"{scene}.yaml"
    )


def load_scene_config(scene, scene_config=""):
    """Load a complete scene used by robot spawn and target tracking."""
    scene, config_path = _config_path(scene, scene_config)
    try:
        root = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(
            f"failed to read scene config {config_path}: {error}"
        ) from error
    root = _mapping(root, "root")
    if root.get("schema_version") != 1:
        raise ValueError("scene config schema_version must be 1")
    if root.get("scene") != scene:
        raise ValueError(
            f"scene config declares {root.get('scene')!r}, expected {scene!r}"
        )
    world = _mapping(root.get("world"), "world")
    world_path = world.get("path")
    if not isinstance(world_path, str) or not world_path.strip():
        raise ValueError("world.path must be a non-empty string")

    robots_value = _mapping(root.get("robots"), "robots")
    robots = {}
    for robot_name in ("go2_1", "go2_2", "go2_3"):
        robot = _mapping(robots_value.get(robot_name), f"robots.{robot_name}")
        spawn = _mapping(robot.get("spawn"), f"robots.{robot_name}.spawn")
        sensors = _mapping(robot.get("sensors"), f"robots.{robot_name}.sensors")
        pose = tuple(
            _finite_number(spawn.get(key), f"robots.{robot_name}.spawn.{key}")
            for key in ("x", "y", "z", "yaw")
        )
        if pose[2] <= 0.0:
            raise ValueError(f"robots.{robot_name}.spawn.z must be positive")
        for key in ("lidar", "camera"):
            if not isinstance(sensors.get(key), bool):
                raise ValueError(f"robots.{robot_name}.sensors.{key} must be boolean")
        robots[robot_name] = RobotConfig(
            spawn=pose,
            lidar=sensors["lidar"],
            camera=sensors["camera"],
        )

    target = _mapping(root.get("dynamic_target"), "dynamic_target")
    model_name = target.get("model_name")
    service_prefix = target.get("service_prefix")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("dynamic_target.model_name must be a non-empty string")
    if not isinstance(service_prefix, str) or not service_prefix.strip():
        raise ValueError(
            "dynamic_target.service_prefix must be a non-empty string"
        )
    if target.get("loop") is not True:
        raise ValueError("dynamic_target.loop must be true")
    speed = _finite_number(target.get("speed"), "dynamic_target.speed")
    turn_duration = _finite_number(
        target.get("turn_duration"), "dynamic_target.turn_duration"
    )
    if speed <= 0.0:
        raise ValueError("dynamic_target.speed must be greater than zero")
    if turn_duration < 0.0:
        raise ValueError("dynamic_target.turn_duration must be non-negative")
    route_value = target.get("route")
    if not isinstance(route_value, list) or len(route_value) < 3:
        raise ValueError("dynamic_target.route must contain at least three points")
    route = []
    for index, raw_point in enumerate(route_value):
        point = _mapping(raw_point, f"dynamic_target.route[{index}]")
        route.append((
            _finite_number(point.get("x"), f"dynamic_target.route[{index}].x"),
            _finite_number(point.get("y"), f"dynamic_target.route[{index}].y"),
        ))
    if len(set(route)) < 3:
        raise ValueError("dynamic_target.route must contain three distinct points")
    closed_route = route + [route[0]]
    for index, (start, end) in enumerate(zip(closed_route, closed_route[1:])):
        if start == end:
            raise ValueError(
                f"dynamic_target.route segment {index} has identical endpoints"
            )
    dynamic_target = DynamicTargetConfig(
        model_name=model_name.strip(),
        service_prefix=service_prefix.strip(),
        loop=True,
        speed=speed,
        turn_duration=turn_duration,
        route=tuple(route),
    )
    return SceneConfig(
        name=scene,
        world_path=world_path.strip(),
        robots=robots,
        dynamic_target=dynamic_target,
    )


def load_dynamic_target_config(scene, scene_config=""):
    """Load one scene's validated closed dynamic-target route."""
    return load_scene_config(scene, scene_config).dynamic_target
