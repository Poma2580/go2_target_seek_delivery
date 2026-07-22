"""Load and validate the shared scene configuration used by Gazebo and Nav2."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml


ROBOT_NAMES = ('go2_1', 'go2_2', 'go2_3')


@dataclass(frozen=True)
class SpawnPose:
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class RobotConfig:
    name: str
    spawn: SpawnPose
    color: Tuple[float, float, float]


@dataclass(frozen=True)
class TargetConfig:
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class EncircleConfig:
    radius: float
    radius_step: float
    max_radius_expansion: float
    candidate_start_angle: float


@dataclass(frozen=True)
class PlanningConfig:
    planner_id: str
    action_server_timeout: float


@dataclass(frozen=True)
class MapConfig:
    package: str
    yaml: str
    frame_id: str


@dataclass(frozen=True)
class SceneConfig:
    scene: str
    world_path: str
    map: Optional[MapConfig]
    target: TargetConfig
    encircle: EncircleConfig
    planning: PlanningConfig
    robots: Dict[str, RobotConfig]
    source: Path

    def resolve_world(self, repository_root: Path) -> Path:
        """Resolve a world path relative to this repository."""
        raw = Path(self.world_path)
        return raw if raw.is_absolute() else repository_root / raw

    def resolve_map_yaml(self) -> Path:
        """Resolve the installed map YAML through the ament resource index."""
        if self.map is None:
            raise ValueError(
                f'scene {self.scene!r} does not define a static map yet')
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(self.map.package)) / self.map.yaml


def _mapping(value, field):
    if not isinstance(value, dict):
        raise ValueError(f'{field} must be a YAML mapping')
    return value


def _number(mapping, key, field):
    if key not in mapping:
        raise ValueError(f'{field}.{key} is required')
    try:
        value = float(mapping[key])
    except (TypeError, ValueError) as error:
        raise ValueError(f'{field}.{key} must be numeric') from error
    if not math.isfinite(value):
        raise ValueError(f'{field}.{key} must be finite')
    return value


def _positive(mapping, key, field, allow_zero=False):
    value = _number(mapping, key, field)
    invalid = value < 0.0 if allow_zero else value <= 0.0
    if invalid:
        relation = 'non-negative' if allow_zero else 'positive'
        raise ValueError(f'{field}.{key} must be {relation}')
    return value


def load_scene_config(filename) -> SceneConfig:
    """Load a scene YAML and fail early on unsafe or ambiguous values."""
    source = Path(filename).expanduser().resolve()
    with source.open('r', encoding='utf-8') as stream:
        data = yaml.safe_load(stream)
    root = _mapping(data, 'root')
    if root.get('schema_version') != 1:
        raise ValueError('schema_version must be 1')

    scene = root.get('scene')
    if not isinstance(scene, str) or not scene.strip():
        raise ValueError('scene must be a non-empty string')
    world = _mapping(root.get('world'), 'world')
    world_path = world.get('path')
    if not isinstance(world_path, str) or not world_path.strip():
        raise ValueError('world.path must be a non-empty string')

    map_value = root.get('map')
    map_config = None
    if map_value is not None:
        map_data = _mapping(map_value, 'map')
        for key in ('package', 'yaml', 'frame_id'):
            if not isinstance(map_data.get(key), str) or not map_data[key]:
                raise ValueError(f'map.{key} must be a non-empty string')
        map_config = MapConfig(
            package=map_data['package'],
            yaml=map_data['yaml'],
            frame_id=map_data['frame_id'],
        )

    target_data = _mapping(root.get('target'), 'target')
    target = TargetConfig(
        x=_number(target_data, 'x', 'target'),
        y=_number(target_data, 'y', 'target'),
        yaw=_number(target_data, 'yaw', 'target'),
    )

    encircle_data = _mapping(root.get('encircle'), 'encircle')
    encircle = EncircleConfig(
        radius=_positive(encircle_data, 'radius', 'encircle'),
        radius_step=_positive(encircle_data, 'radius_step', 'encircle'),
        max_radius_expansion=_positive(
            encircle_data, 'max_radius_expansion', 'encircle', allow_zero=True),
        candidate_start_angle=_number(
            encircle_data, 'candidate_start_angle', 'encircle'),
    )

    planning_data = _mapping(root.get('planning'), 'planning')
    planner_id = planning_data.get('planner_id')
    if not isinstance(planner_id, str) or not planner_id:
        raise ValueError('planning.planner_id must be a non-empty string')
    planning = PlanningConfig(
        planner_id=planner_id,
        action_server_timeout=_positive(
            planning_data, 'action_server_timeout', 'planning'),
    )

    robots_data = _mapping(root.get('robots'), 'robots')
    if tuple(sorted(robots_data)) != tuple(sorted(ROBOT_NAMES)):
        raise ValueError(
            'robots must contain exactly go2_1, go2_2 and go2_3')
    robots = {}
    spawn_xy = set()
    for name in ROBOT_NAMES:
        robot_data = _mapping(robots_data[name], f'robots.{name}')
        spawn_data = _mapping(robot_data.get('spawn'), f'robots.{name}.spawn')
        spawn = SpawnPose(
            x=_number(spawn_data, 'x', f'robots.{name}.spawn'),
            y=_number(spawn_data, 'y', f'robots.{name}.spawn'),
            z=_number(spawn_data, 'z', f'robots.{name}.spawn'),
            yaw=_number(spawn_data, 'yaw', f'robots.{name}.spawn'),
        )
        if spawn.z <= 0.0:
            raise ValueError(f'robots.{name}.spawn.z must be positive')
        xy = (spawn.x, spawn.y)
        if xy in spawn_xy:
            raise ValueError('robot spawn x/y positions must be distinct')
        spawn_xy.add(xy)

        color_value = robot_data.get('color')
        if not isinstance(color_value, list) or len(color_value) != 3:
            raise ValueError(f'robots.{name}.color must have three values')
        color = tuple(float(component) for component in color_value)
        if any(not math.isfinite(component) or not 0.0 <= component <= 1.0
               for component in color):
            raise ValueError(f'robots.{name}.color values must be in [0, 1]')
        robots[name] = RobotConfig(name=name, spawn=spawn, color=color)

    return SceneConfig(
        scene=scene,
        world_path=world_path,
        map=map_config,
        target=target,
        encircle=encircle,
        planning=planning,
        robots=robots,
        source=source,
    )
