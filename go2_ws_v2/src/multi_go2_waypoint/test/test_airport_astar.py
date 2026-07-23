import math
from pathlib import Path

import pytest
import rclpy
from nav_msgs.msg import Odometry

from multi_go2_waypoint.grid_astar import (
    GridMap,
    astar,
    grid_path_to_waypoints,
    line_is_free,
    simplify_grid_path,
)
from multi_go2_waypoint.waypoint_encircle import (
    WaypointEncircle,
    assign_encircle_points,
    load_astar_config,
    load_scene_config,
    solve_encircle_points,
)


MAP_YAML = Path(__file__).parents[1] / 'maps' / 'airport.yaml'
CONFIG_ROOT = Path(__file__).parents[1] / 'config'
SCENE_CONFIG = CONFIG_ROOT / 'scenes' / 'airport.yaml'
PLANNER_CONFIG = CONFIG_ROOT / 'planner' / 'astar.yaml'
CONTROLLER_CONFIG = CONFIG_ROOT / 'controller' / 'p_controller.yaml'


@pytest.mark.skipif(
    not MAP_YAML.is_file(), reason='airport map is not present')
def test_three_airport_paths_exist_on_inflated_map():
    scene = load_scene_config(SCENE_CONFIG)
    planner = load_astar_config(PLANNER_CONFIG)
    grid = GridMap.from_yaml(MAP_YAML).inflated(
        planner['inflation_radius'])
    starts = [
        (name, *position)
        for name, position in scene.robots.items()
    ]

    requested_radius = scene.encircle_radius
    actual_radius = None
    points = None
    max_steps = int(math.floor(
        planner['max_goal_radius_expansion'] / grid.resolution + 1e-9))
    for step in range(max_steps + 1):
        candidate_radius = requested_radius + step * grid.resolution
        candidate_points = solve_encircle_points(
            scene.target_x, scene.target_y, candidate_radius,
            len(starts), scene.target_yaw)
        if all(grid.is_free_world(x, y) for x, y, _ in candidate_points):
            actual_radius = candidate_radius
            points = candidate_points
            break
    assert actual_radius is not None
    assert actual_radius <= (
        requested_radius + planner['max_goal_radius_expansion'] + 1e-9)

    assigned = assign_encircle_points(starts, points)
    for name, start_x, start_y in starts:
        goal_x, goal_y, goal_yaw = assigned[name]
        result = astar(
            grid, grid.world_to_grid(start_x, start_y),
            grid.world_to_grid(goal_x, goal_y))
        assert result.cells == astar(
            grid, grid.world_to_grid(start_x, start_y),
            grid.world_to_grid(goal_x, goal_y)).cells
        simplified = simplify_grid_path(grid, result.cells)
        assert all(
            line_is_free(grid, simplified[index], simplified[index + 1])
            for index in range(len(simplified) - 1))
        waypoints = grid_path_to_waypoints(
            grid, simplified, (start_x, start_y), (goal_x, goal_y),
            goal_yaw, max_spacing=planner['max_waypoint_spacing'])
        assert waypoints
        assert waypoints[-1] == pytest.approx((goal_x, goal_y, goal_yaw))
        assert all(yaw is None for _, _, yaw in waypoints[:-1])


@pytest.mark.skipif(
    not MAP_YAML.is_file(), reason='airport map is not present')
def test_waypoint_node_waits_for_odom_then_installs_all_three_paths():
    scene = load_scene_config(SCENE_CONFIG)
    planner = load_astar_config(PLANNER_CONFIG)
    args = [
        '--ros-args',
        '-p', 'scene:=airport',
        '-p', 'planner_mode:=astar',
        '-p', f'map_yaml:={MAP_YAML}',
        '-p', f'scene_config:={SCENE_CONFIG}',
        '-p', f'planner_config:={PLANNER_CONFIG}',
        '-p', f'controller_config:={CONTROLLER_CONFIG}',
    ]
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointEncircle()
        assert not node.planning_complete
        for dog in node.dogs:
            x, y = scene.robots[dog.name]
            odom = Odometry()
            odom.pose.pose.position.x = x
            odom.pose.pose.position.y = y
            odom.pose.pose.orientation.w = 1.0
            dog.odom_callback(odom)

        node.control_loop()
        assert node.planning_complete
        assert not node.planning_failed
        assert all(dog.waypoints for dog in node.dogs)
        assert all(dog.waypoints[-1][2] is not None for dog in node.dogs)
        radii = [
            ((dog.waypoints[-1][0] - scene.target_x) ** 2 +
             (dog.waypoints[-1][1] - scene.target_y) ** 2) ** 0.5
            for dog in node.dogs
        ]
        assert radii == pytest.approx([radii[0]] * len(radii))
        assert scene.encircle_radius <= radii[0]
        assert radii[0] <= (
            scene.encircle_radius + planner['max_goal_radius_expansion'])
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


@pytest.mark.skipif(
    not MAP_YAML.is_file(), reason='airport map is not present')
def test_command_line_parameters_override_yaml_defaults():
    planner = load_astar_config(PLANNER_CONFIG)
    args = [
        '--ros-args',
        '-p', 'scene:=airport',
        '-p', 'planner_mode:=astar',
        '-p', f'map_yaml:={MAP_YAML}',
        '-p', f'scene_config:={SCENE_CONFIG}',
        '-p', f'planner_config:={PLANNER_CONFIG}',
        '-p', f'controller_config:={CONTROLLER_CONFIG}',
        '-p', 'inflation_radius:=0.40',
        '-p', 'max_linear:=0.42',
    ]
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointEncircle()
        assert node.inflation_radius == pytest.approx(0.40)
        assert node.max_linear == pytest.approx(0.42)
        assert node.max_waypoint_spacing == pytest.approx(
            planner['max_waypoint_spacing'])
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
