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
    solve_encircle_points,
)


MAP_YAML = Path(__file__).parents[1] / 'maps' / 'airport.yaml'


@pytest.mark.skipif(not MAP_YAML.is_file(), reason='airport map is not present')
def test_three_airport_paths_exist_on_inflated_map():
    grid = GridMap.from_yaml(MAP_YAML).inflated(0.55)
    starts = [
        ('go2_1', 0.0, -4.0),
        ('go2_2', 2.0, -4.0),
        ('go2_3', 0.0, -6.0),
    ]

    requested_radius = 3.0
    actual_radius = None
    points = None
    for step in range(16):
        candidate_radius = requested_radius + step * grid.resolution
        candidate_points = solve_encircle_points(80.0, -25.0, candidate_radius, 3, -1.0)
        if all(grid.is_free_world(x, y) for x, y, _ in candidate_points):
            actual_radius = candidate_radius
            points = candidate_points
            break
    assert actual_radius == pytest.approx(4.0)

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
            goal_yaw, max_spacing=1.5)
        assert waypoints
        assert waypoints[-1] == pytest.approx((goal_x, goal_y, goal_yaw))
        assert all(yaw is None for _, _, yaw in waypoints[:-1])


@pytest.mark.skipif(not MAP_YAML.is_file(), reason='airport map is not present')
def test_waypoint_node_waits_for_odom_then_installs_all_three_paths():
    args = [
        '--ros-args',
        '-p', 'scene:=airport',
        '-p', 'planner_mode:=astar',
        '-p', f'map_yaml:={MAP_YAML}',
    ]
    rclpy.init(args=args)
    node = None
    try:
        node = WaypointEncircle()
        assert not node.planning_complete
        starts = ((0.0, -4.0), (2.0, -4.0), (0.0, -6.0))
        for dog, (x, y) in zip(node.dogs, starts):
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
            ((dog.waypoints[-1][0] - 80.0) ** 2 +
             (dog.waypoints[-1][1] + 25.0) ** 2) ** 0.5
            for dog in node.dogs
        ]
        assert radii == pytest.approx([4.0, 4.0, 4.0])
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
