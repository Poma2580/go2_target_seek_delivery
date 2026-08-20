"""Pure role-assignment tests for the runtime MADDPG controller."""

import numpy as np

from multi_go2_waypoint.gazebo_leader_slot_controller import (
    EntityState,
    GazeboLeaderSlotController,
    choose_follower_slots,
)


def test_enable_assignment_chooses_shorter_roles_and_latches_robot_names():
    slots = np.array([[2.7, 1.8], [2.7, -1.8]], dtype=np.float32)
    mapping, normal_cost, swapped_cost = choose_follower_slots(
        ("go2_1", "go2_3"),
        (np.array([2.5, -1.7], dtype=np.float32),
         np.array([2.6, 1.7], dtype=np.float32)),
        slots,
    )
    assert swapped_cost < normal_cost
    assert mapping == {"go2_1": 1, "go2_3": 0}


def test_disabled_auto_assignment_keeps_documented_default_roles():
    slots = np.array([[2.7, 1.8], [2.7, -1.8]], dtype=np.float32)
    mapping, _, _ = choose_follower_slots(
        ("go2_1", "go2_2"),
        (np.array([2.5, -1.7], dtype=np.float32),
         np.array([2.6, 1.7], dtype=np.float32)),
        slots,
        automatic=False,
    )
    assert mapping == {"go2_1": 0, "go2_2": 1}


def test_dynamic_leader_keeps_25d_left_right_observation_semantics():
    controller = GazeboLeaderSlotController.__new__(GazeboLeaderSlotController)
    controller.leader_name = "go2_2"
    controller.follower_names = ("go2_3", "go2_1")
    controller.states = {
        "go2_1": EntityState(x=2.0, y=-1.0, received=True),
        "go2_2": EntityState(x=0.0, y=0.0, received=True),
        "go2_3": EntityState(x=2.0, y=1.0, received=True),
    }
    controller.side_dist = 1.8
    controller.leader_follow_dist = 2.7
    controller.dt = 0.1
    controller.last_slots = None
    controller.follower_max_angular = 0.45
    controller.follower_max_linear = 0.45
    controller.follower_prev_actions = {
        name: [0.0, 0.0] for name in controller.states
    }
    observations, diagnostics = controller._build_observations()
    assert [observation.shape for observation in observations] == [(25,), (25,)]
    assert np.array_equal(observations[0][14:16], [1.0, 0.0])
    assert np.array_equal(observations[1][14:16], [0.0, 1.0])
    assert [item["slot_side"] for item in diagnostics] == ["left", "right"]
