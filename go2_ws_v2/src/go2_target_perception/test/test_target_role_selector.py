"""Pure tests for first-detector role selection."""

from go2_target_perception.target_role_selector import RoleElectionState


def test_first_robot_with_enough_recent_samples_is_latched():
    election = RoleElectionState(("go2_1", "go2_2", "go2_3"), 3, 1.0)
    assert election.observe("go2_1", 0.0) is None
    assert election.observe("go2_2", 0.1) is None
    assert election.observe("go2_2", 0.2) is None
    assert election.observe("go2_2", 0.3) == "go2_2"
    assert election.observe("go2_1", 0.4) == "go2_2"


def test_reject_clears_only_one_robots_confirmation_history():
    election = RoleElectionState(("go2_1", "go2_2"), 2, 1.0)
    election.observe("go2_1", 0.0)
    election.observe("go2_2", 0.1)
    election.reject("go2_1")
    assert election.observe("go2_2", 0.2) == "go2_2"
