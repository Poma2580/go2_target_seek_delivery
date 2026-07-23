from pathlib import Path

from nav_msgs.msg import Odometry
import yaml

from multi_go2_waypoint.astar_visualizer import (
    TrajectoryBuffer,
    occupancy_grid_from_map,
    world_to_odom_transforms,
)
from multi_go2_waypoint.grid_astar import GridMap
from multi_go2_waypoint.waypoint_encircle import execution_path_from_waypoints


RVIZ_FILE = Path(__file__).parents[1] / 'rviz' / 'astar_waypoint.rviz'


def test_execution_path_includes_current_odom_and_all_waypoints():
    path = execution_path_from_waypoints(
        (1.0, -2.0, 0.5),
        [(2.0, -2.0, None), (3.0, -1.0, 1.2)],
        'world', Odometry().header.stamp)

    assert path.header.frame_id == 'world'
    assert len(path.poses) == 3
    assert path.poses[0].pose.position.x == 1.0
    assert path.poses[-1].pose.position.y == -1.0
    assert path.poses[-1].pose.orientation.w != 1.0


def test_actual_trajectory_sampling_and_truncation():
    buffer = TrajectoryBuffer('world', sample_distance=0.5, max_points=2)
    first = Odometry()
    first.pose.pose.position.x = 1.0
    second = Odometry()
    second.pose.pose.position.x = 1.2
    third = Odometry()
    third.pose.pose.position.x = 2.0
    fourth = Odometry()
    fourth.pose.pose.position.x = 3.0

    assert buffer.append_odometry(first)
    assert not buffer.append_odometry(second)
    assert buffer.append_odometry(third)
    assert buffer.append_odometry(fourth)
    assert len(buffer.path.poses) == 2
    assert buffer.path.poses[0].pose.position.x == 2.0
    assert buffer.message(Odometry().header.stamp).header.frame_id == 'world'


def test_map_conversion_and_world_to_odom_transforms():
    grid = GridMap(2, 2, 0.2, -1.0, -2.0, bytearray([0, 1, 0, 0]))
    map_message = occupancy_grid_from_map(
        grid, 'world', Odometry().header.stamp)
    transforms = world_to_odom_transforms(
        'world', ('go2_1', 'go2_2', 'go2_3'), Odometry().header.stamp)

    assert map_message.header.frame_id == 'world'
    assert map_message.info.width == 2
    assert map_message.info.origin.position.y == -2.0
    assert list(map_message.data) == [0, 100, 0, 0]
    assert [transform.child_frame_id for transform in transforms] == [
        'go2_1/odom', 'go2_2/odom', 'go2_3/odom']
    assert all(
        transform.transform.rotation.w == 1.0 for transform in transforms)


def test_rviz_config_enables_all_astar_visualization_topics():
    config = yaml.safe_load(RVIZ_FILE.read_text(encoding='utf-8'))
    displays = config['Visualization Manager']['Displays']
    groups = {
        display.get('Name'): display
        for display in displays
        if display.get('Class') == 'rviz_common/Group'
    }

    assert (
        config['Visualization Manager']['Global Options']['Fixed Frame']
        == 'world')
    assert any(
        display.get('Topic', {}).get('Value') == '/astar_map'
        for display in displays)
    assert groups['AStar Planned Paths']['Enabled'] is True
    assert groups['AStar Actual Paths']['Enabled'] is True
    assert {
        display['Topic']['Value']
        for display in groups['AStar Planned Paths']['Displays']
    } == {
        '/go2_1/astar_plan',
        '/go2_2/astar_plan',
        '/go2_3/astar_plan',
    }
    assert {
        display['Topic']['Value']
        for display in groups['AStar Actual Paths']['Displays']
    } == {
        '/go2_1/astar_actual_path',
        '/go2_2/astar_actual_path',
        '/go2_3/astar_actual_path',
    }
