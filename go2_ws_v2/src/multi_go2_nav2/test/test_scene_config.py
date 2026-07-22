from pathlib import Path

import pytest
import yaml

from multi_go2_nav2.scene_config import ROBOT_NAMES, load_scene_config


PACKAGE_ROOT = Path(__file__).parents[1]


def test_airport_configuration_is_complete():
    config = load_scene_config(
        PACKAGE_ROOT / 'config' / 'scenes' / 'airport.yaml')

    assert config.scene == 'airport'
    assert tuple(config.robots) == ROBOT_NAMES
    assert config.map.yaml == 'maps/airport.yaml'
    assert config.target.x == 80.0
    assert config.encircle.radius > 0.0
    assert len({
        (robot.spawn.x, robot.spawn.y)
        for robot in config.robots.values()
    }) == len(ROBOT_NAMES)


def test_world_path_is_resolved_from_repository_root(tmp_path):
    config = load_scene_config(
        PACKAGE_ROOT / 'config' / 'scenes' / 'airport.yaml')

    assert config.resolve_world(tmp_path) == tmp_path / 'KD_MODEL/world/airport'


def test_duplicate_spawn_positions_are_rejected(tmp_path):
    source = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'scenes' / 'airport.yaml').read_text())
    source['robots']['go2_2']['spawn'] = dict(
        source['robots']['go2_1']['spawn'])
    filename = tmp_path / 'duplicate.yaml'
    filename.write_text(yaml.safe_dump(source))

    with pytest.raises(ValueError, match='must be distinct'):
        load_scene_config(filename)


def test_missing_map_is_allowed_for_gazebo_only_scene():
    config = load_scene_config(
        PACKAGE_ROOT / 'config' / 'scenes' / 'forest.yaml')

    assert config.map is None
    with pytest.raises(ValueError, match='does not define a static map'):
        config.resolve_map_yaml()
