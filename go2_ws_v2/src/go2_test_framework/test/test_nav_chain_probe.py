import json

import pytest

from go2_test_framework.recorders.nav_chain_probe import NavChainState
from go2_test_framework.recorders.nav_goal_status import parse_nav_goal_status


def event(generation=1, name="DISPATCHED"):
    return {
        "schema_version": 1,
        "stamp": {"sec": 1, "nanosec": 2},
        "event": name,
        "generation": generation,
        "frame_id": "merged_map",
        "navigation_dogs": ["go2_2", "go2_3"],
        "goals": {
            "go2_2": {"goal_x": 1.0, "goal_y": 2.0, "goal_yaw": 0.0},
            "go2_3": {"goal_x": 3.0, "goal_y": 4.0, "goal_yaw": 0.1},
        },
        "robot": None,
        "action_status": None,
    }


def test_nav_goal_json_schema_and_two_generations():
    first = parse_nav_goal_status(json.dumps(event(1)))
    second = parse_nav_goal_status(json.dumps(event(2)))
    state = NavChainState()
    state.role = "go2_1"
    state.observe_status(first)
    assert not state.ready
    state.observe_status(second)
    assert state.ready
    assert state.dispatched == [1, 2]


def test_nav_goal_schema_rejects_mismatched_goals():
    value = event()
    value["goals"].pop("go2_3")
    with pytest.raises(ValueError, match="exactly match"):
        parse_nav_goal_status(json.dumps(value))


def test_mux_handoff_prevents_chain_success():
    state = NavChainState()
    state.role = "go2_1"
    state.observe_status(event(1))
    state.observe_status(event(2))
    state.maddpg_selected = True
    assert not state.ready
