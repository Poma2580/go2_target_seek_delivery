from types import SimpleNamespace

import pytest

from multi_go2_nav2.encircle_coordinator import (
    action_status_name,
    assignment_has_unique_goals,
    choose_assignment,
    encircle_goals,
    path_length,
    planning_batches,
    startup_prerequisites_ready,
)
from action_msgs.msg import GoalStatus
from multi_go2_nav2.scene_config import TargetConfig


def make_path(points):
    poses = []
    for x, y in points:
        position = SimpleNamespace(x=x, y=y)
        poses.append(SimpleNamespace(pose=SimpleNamespace(position=position)))
    return SimpleNamespace(poses=poses)


def test_encircle_goals_are_even_and_face_target():
    target = TargetConfig(x=10.0, y=-5.0, yaw=0.2)
    goals = encircle_goals(target, 3.0)

    assert len(goals) == 3
    for x, y, yaw in goals:
        assert ((x - target.x) ** 2 + (y - target.y) ** 2) ** 0.5 \
            == pytest.approx(3.0)
        assert yaw == pytest.approx(
            __import__('math').atan2(target.y - y, target.x - x))


def test_path_length_uses_all_segments():
    assert path_length(make_path([(0, 0), (3, 0), (3, 4)])) == 7.0


def test_assignment_uses_nav2_path_lengths_and_rejects_missing_paths():
    robots = ['go2_1', 'go2_2', 'go2_3']
    paths = {}
    for robot_index, robot in enumerate(robots):
        for goal_index in range(3):
            length = 1.0 if robot_index == goal_index else 10.0
            paths[(robot, goal_index)] = make_path([(0, 0), (length, 0)])
    paths[('go2_2', 1)] = None

    assignment, total = choose_assignment(robots, paths)

    assert assignment != {'go2_1': 0, 'go2_2': 1, 'go2_3': 2}
    assert assignment_has_unique_goals(robots, assignment)
    assert total == pytest.approx(21.0)


def test_planning_batches_never_overlap_requests_for_one_robot():
    robots = ['go2_1', 'go2_2', 'go2_3']
    batches = planning_batches(robots, 3)
    active = {name: 0 for name in robots}
    maximum = {name: 0 for name in robots}

    assert len(batches) == 3
    for goal_index, batch in enumerate(batches):
        assert batch == tuple((name, goal_index) for name in robots)
        for name, _goal in batch:
            active[name] += 1
            maximum[name] = max(maximum[name], active[name])
        assert sum(active.values()) == 3
        for name, _goal in batch:
            active[name] -= 1

    assert maximum == {name: 1 for name in robots}


def test_startup_waits_for_delayed_third_lifecycle_manager():
    robots = ['go2_1', 'go2_2', 'go2_3']
    odom_names = set(robots)
    actions = {name: True for name in robots}
    lifecycle = {'go2_1': True, 'go2_2': True, 'go2_3': False}

    assert not startup_prerequisites_ready(
        robots, odom_names, actions, lifecycle)
    lifecycle['go2_3'] = True
    assert startup_prerequisites_ready(
        robots, odom_names, actions, lifecycle)


def test_assignment_rejects_duplicate_or_incomplete_goal_indices():
    robots = ['go2_1', 'go2_2', 'go2_3']

    assert assignment_has_unique_goals(
        robots, {'go2_1': 1, 'go2_2': 2, 'go2_3': 0})
    assert not assignment_has_unique_goals(
        robots, {'go2_1': 2, 'go2_2': 2, 'go2_3': 2})
    assert not assignment_has_unique_goals(
        robots, {'go2_1': 0, 'go2_2': 1})


def test_failed_action_statuses_are_reported_explicitly():
    assert action_status_name(GoalStatus.STATUS_ABORTED) == 'aborted'
    assert action_status_name(GoalStatus.STATUS_CANCELED) == 'canceled'
    assert action_status_name(GoalStatus.STATUS_SUCCEEDED) == 'succeeded'


def test_timed_out_robot_prevents_partial_group_assignment():
    robots = ['go2_1', 'go2_2', 'go2_3']
    paths = {
        (robot, goal): make_path([(0, 0), (goal + 1, 0)])
        for robot in robots
        for goal in range(3)
    }
    for goal in range(3):
        paths[('go2_3', goal)] = None

    assert choose_assignment(robots, paths) is None
