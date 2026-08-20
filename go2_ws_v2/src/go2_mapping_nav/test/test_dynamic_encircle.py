"""Unit and static integration tests for Nav2 dynamic encirclement."""

import importlib.util
import math
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
DELIVERY_ROOT = PACKAGE_ROOT.parents[2]
SCRIPT = PACKAGE_ROOT / "scripts" / "dynamic_encircle.py"
SPEC = importlib.util.spec_from_file_location("nav2_dynamic_encircle_script", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_encircle_points_are_uniform_and_anchored_to_go2_1_direction():
    points = MODULE.solve_encircle_points(4.0, -2.0, 3.0, 3, math.pi / 2.0)

    assert points[0][0] == pytest.approx(4.0)
    assert points[0][1] == pytest.approx(1.0)
    for x, y, _ in points:
        assert math.hypot(x - 4.0, y + 2.0) == pytest.approx(3.0)

    angles = [math.atan2(y + 2.0, x - 4.0) for x, y, _ in points]
    assert MODULE.normalize_angle(angles[1] - angles[0]) == pytest.approx(
        2.0 * math.pi / 3.0
    )
    assert MODULE.normalize_angle(angles[2] - angles[1]) == pytest.approx(
        2.0 * math.pi / 3.0
    )


def test_initial_assignment_minimizes_distance_and_remains_sticky():
    initial_points = (
        (0.0, 0.0, 0.0),
        (10.0, 0.0, 0.0),
        (-10.0, 0.0, 0.0),
    )
    assignment = MODULE.assign_remaining_slots(
        {"go2_2": (-9.0, 0.0), "go2_3": (9.0, 0.0)}, initial_points
    )
    assert assignment == {"go2_2": 2, "go2_3": 1}

    moved_points = (
        (1.0, 1.0, 0.0),
        (2.0, 3.0, 0.0),
        (4.0, 5.0, 0.0),
    )
    updated = MODULE.update_assigned_slots(assignment, moved_points)
    assert updated == {"go2_2": moved_points[2], "go2_3": moved_points[1]}


def test_success_requires_both_navigation_dogs_in_tolerance_simultaneously():
    goals = {"go2_2": (1.0, 1.0, 0.0), "go2_3": (5.0, 5.0, 0.0)}
    assert MODULE.encircle_reached(
        {"go2_2": (2.0, 1.0), "go2_3": (3.5, 5.0)}, goals, 2.0
    )
    assert not MODULE.encircle_reached(
        {"go2_2": (2.0, 1.0), "go2_3": (2.9, 5.0)}, goals, 2.0
    )


@pytest.mark.parametrize(
    "call",
    [
        lambda: MODULE.solve_encircle_points(0.0, 0.0, 0.0, 3, 0.0),
        lambda: MODULE.solve_encircle_points(0.0, 0.0, 1.0, 0, 0.0),
        lambda: MODULE.assign_remaining_slots(
            {"go2_2": (0.0, 0.0)}, ((0.0, 0.0, 0.0),) * 3
        ),
        lambda: MODULE.encircle_reached(
            {"go2_2": (0.0, 0.0)}, {"go2_2": (0.0, 0.0, 0.0)}, 0.0
        ),
    ],
)
def test_geometry_rejects_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_goal_update_state_sends_immediately_then_every_five_seconds():
    state = MODULE.GoalUpdateState(period=5.0)
    assert state.due(10.0)
    first_generation = state.mark_dispatched(10.0)
    assert first_generation == 1
    assert not state.due(14.999)
    assert state.due(15.0)

    second_generation = state.mark_dispatched(15.0)
    assert second_generation == 2
    assert not state.is_current(first_generation)
    assert state.is_current(second_generation)


def test_goal_update_state_handles_loss_recovery_and_latched_completion():
    state = MODULE.GoalUpdateState(period=5.0)
    generation = state.mark_dispatched(0.0)
    assert state.suspend()
    assert not state.is_current(generation)
    assert not state.due(20.0)

    assert state.resume()
    assert state.due(20.0)
    recovered_generation = state.mark_dispatched(20.0)
    assert state.is_current(recovered_generation)

    assert state.complete()
    assert not state.due(30.0)
    assert not state.resume()
    assert not state.is_current(recovered_generation)


def test_build_and_startup_integration_registers_new_node_after_perception():
    cmake = (PACKAGE_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    assert "scripts/dynamic_encircle.py" in cmake
    assert "scripts/follower_cmd_vel_mux.py" in cmake
    assert "test/test_dynamic_encircle.py" in cmake

    start_script = (
        DELIVERY_ROOT / "Scripts" / "start_three_go2_dynamic_tracking.sh"
    ).read_text(encoding="utf-8")
    perception = start_script.index("ros2 run multi_go2_waypoint target_perception")
    dynamic = start_script.index("ros2 run go2_mapping_nav dynamic_encircle.py")
    rqt = start_script.index("ros2 run rqt_image_view rqt_image_view")
    assert perception < dynamic < rqt
    assert "ros2 run multi_go2_waypoint dynamic_encircle" not in start_script
    assert "ros2 run go2_mapping_nav follower_cmd_vel_mux.py" in start_script
    assert "ros2 run multi_go2_waypoint gazebo_leader_slot_controller" in start_script
    assert 'if [ "$robot_index" -ge 2 ]' in start_script
    assert 'nav_cmd_vel_arg="cmd_vel_topic:=/${robot_name}/nav_cmd_vel"' in start_script
