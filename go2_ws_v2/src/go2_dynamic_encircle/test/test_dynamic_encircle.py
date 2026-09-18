"""Geometry, role, and static integration tests for modular encirclement."""

import math
from pathlib import Path

import pytest

from go2_dynamic_encircle.geometry import (
    assign_remaining_slots,
    encircle_reached,
    navigation_dog_names,
    solve_encircle_points,
)
from go2_dynamic_encircle.nav_goal_manager import GoalUpdateState


PACKAGE_ROOT = Path(__file__).parents[1]
DELIVERY_ROOT = PACKAGE_ROOT.parents[2]


def test_geometry_requires_simultaneous_position_and_yaw():
    goals = {"go2_2": (1.0, 1.0, 0.0), "go2_3": (5.0, 5.0, 0.0)}
    assert encircle_reached(
        {"go2_2": (2.0, 1.0, 0.2), "go2_3": (3.5, 5.0, -0.2)},
        goals, 2.0, math.radians(60.0),
    )
    assert not encircle_reached(
        {"go2_2": (2.0, 1.0, 1.2), "go2_3": (3.5, 5.0, 0.0)},
        goals, 2.0, math.radians(60.0),
    )


@pytest.mark.parametrize(
    "perception, expected",
    [
        ("go2_1", ("go2_2", "go2_3")),
        ("go2_2", ("go2_1", "go2_3")),
        ("go2_3", ("go2_1", "go2_2")),
    ],
)
def test_all_dynamic_role_combinations(perception, expected):
    assert navigation_dog_names(("go2_1", "go2_2", "go2_3"), perception) == expected


def test_slot_assignment_and_goal_generation_are_sticky():
    points = solve_encircle_points(0.0, 0.0, 3.0, 3, 0.0)
    assignment = assign_remaining_slots(
        {"go2_2": points[2], "go2_3": points[1]}, points
    )
    assert assignment == {"go2_2": 2, "go2_3": 1}
    state = GoalUpdateState(5.0)
    generation = state.mark_dispatched(0.0)
    assert state.is_current(generation)
    assert state.complete()
    assert not state.due(10.0)


def test_startup_routes_all_nav2_commands_through_mux():
    script = (DELIVERY_ROOT / "Scripts/start_three_go2_dynamic_tracking.sh").read_text()
    assert "go2_dynamic_encircle follower_cmd_vel_mux" in script
    assert "gazebo_leader_slot_controller" in script
    assert 'nav_cmd_vel_arg="cmd_vel_topic:=/${robot_name}/nav_cmd_vel"' in script
    assert 'if [ "$robot_index" -ge 2 ]' not in script


def test_startup_restarts_when_nav2_lifecycle_times_out():
    script = (DELIVERY_ROOT / "Scripts/start_three_go2_dynamic_tracking.sh").read_text()
    assert "NAV2_LIFECYCLE_TIMEOUT=${NAV2_LIFECYCLE_TIMEOUT:-20}" in script
    assert (
        '[[ "$NAV2_LIFECYCLE_TIMEOUT" =~ ^[1-9][0-9]*$ ]]' in script
    )
    assert 'local description=$3\n    local timeout_seconds=$4' in script
    assert "for robot_index in 1 2 3; do" in script
    assert '"${robot_name} Nav2 lifecycle" \\\n        "$NAV2_LIFECYCLE_TIMEOUT"' in script
    assert "trigger_auto_restart()" in script
    assert script.count("trigger_auto_restart \\") == 2
    assert 'GO2_RESTART_COUNT="$next_restart_count"' in script
    assert 'MAX_GO2_RESTARTS="$MAX_GO2_RESTARTS"' in script

    mapping_launch = script.index('launch_terminal "mapping_nav_${robot_name}"')
    lifecycle_wait = script.index("if ! wait_for_log_message", mapping_launch)
    merged_map_wait = script.index('if [ "$merged_map_ready" = false ]', mapping_launch)
    assert mapping_launch < lifecycle_wait < merged_map_wait
