"""Pure role-assignment tests for the runtime MADDPG controller."""

import numpy as np

from multi_go2_waypoint.gazebo_leader_slot_controller import choose_follower_slots


def test_enable_assignment_chooses_shorter_roles_and_latches_robot_names():
    slots = np.array([[2.7, 1.8], [2.7, -1.8]], dtype=np.float32)
    mapping, normal_cost, swapped_cost = choose_follower_slots(
        np.array([2.5, -1.7], dtype=np.float32),
        np.array([2.6, 1.7], dtype=np.float32),
        slots,
    )
    assert swapped_cost < normal_cost
    assert mapping == {"go2_2": 1, "go2_3": 0}


def test_disabled_auto_assignment_keeps_documented_default_roles():
    slots = np.array([[2.7, 1.8], [2.7, -1.8]], dtype=np.float32)
    mapping, _, _ = choose_follower_slots(
        np.array([2.5, -1.7], dtype=np.float32),
        np.array([2.6, 1.7], dtype=np.float32),
        slots,
        automatic=False,
    )
    assert mapping == {"go2_2": 0, "go2_3": 1}
