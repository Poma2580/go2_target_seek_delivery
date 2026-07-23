import math
from pathlib import Path

from multi_go2_waypoint.waypoint_encircle import (
    load_astar_config,
    load_controller_config,
    load_manual_config,
    load_scene_config,
    load_visualization_config,
)


CONFIG_ROOT = Path(__file__).parents[1] / 'config'
ROBOT_NAMES = {'go2_1', 'go2_2', 'go2_3'}


def test_default_scene_configs_are_valid():
    scenes = {
        name: load_scene_config(CONFIG_ROOT / 'scenes' / f'{name}.yaml')
        for name in ('airport', 'city', 'forest')
    }

    for name, scene in scenes.items():
        assert scene.name == name
        assert all(math.isfinite(value) for value in (
            scene.target_x, scene.target_y, scene.target_yaw))
        assert scene.encircle_radius > 0.0
        assert set(scene.robots) == ROBOT_NAMES
        assert all(
            all(math.isfinite(value) for value in position)
            for position in scene.robots.values())

    airport = scenes['airport']
    assert airport.map_package == 'multi_go2_waypoint'
    assert airport.map_yaml == 'maps/airport.yaml'
    assert scenes['city'].map_yaml is None
    assert scenes['forest'].map_yaml is None


def test_manual_configs_cover_every_scene_and_robot():
    manual = load_manual_config(CONFIG_ROOT / 'planner' / 'manual.yaml')

    assert set(manual) == {'airport', 'city', 'forest'}
    for paths in manual.values():
        assert set(paths) == ROBOT_NAMES
        assert all(
            all(
                len(waypoint) == 2 and
                all(math.isfinite(value) for value in waypoint)
                for waypoint in waypoints)
            for waypoints in paths.values())


def test_default_tunable_configs_have_valid_values():
    astar = load_astar_config(CONFIG_ROOT / 'planner' / 'astar.yaml')
    controller = load_controller_config(
        CONFIG_ROOT / 'controller' / 'p_controller.yaml')
    visualization = load_visualization_config(
        CONFIG_ROOT / 'visualization' / 'astar_rviz.yaml')

    assert astar['inflation_radius'] >= 0.0
    assert astar['max_waypoint_spacing'] > 0.0
    assert astar['max_goal_radius_expansion'] >= 0.0
    assert controller['reach_threshold'] > 0.0
    assert controller['yaw_threshold'] > 0.0
    assert controller['turn_in_place_thresh'] >= 0.0
    assert controller['max_linear'] >= 0.0
    assert controller['max_angular'] >= 0.0
    assert controller['k_linear'] >= 0.0
    assert controller['k_angular'] >= 0.0
    assert controller['control_period'] > 0.0
    assert visualization.frame_id
    assert visualization.trajectory_sample_distance > 0.0
    assert visualization.max_trajectory_points >= 2
    assert visualization.trajectory_publish_rate > 0.0
    assert isinstance(visualization.publish_world_to_odom_tf, bool)
