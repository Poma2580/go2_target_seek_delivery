from go2_test_framework.evaluators.path_planning import PathPlanningEvaluator


def dispatched(generation, stamp=0.0):
    sec = int(stamp)
    return {
        "event": "DISPATCHED", "generation": generation,
        "stamp": {"sec": sec, "nanosec": int((stamp - sec) * 1e9)},
        "frame_id": "merged_map",
        "navigation_dogs": ["go2_1", "go2_3"],
        "goals": {
            "go2_1": {"goal_x": 1.0, "goal_y": 2.0, "goal_yaw": 3.0},
            "go2_3": {"goal_x": 4.0, "goal_y": 5.0, "goal_yaw": -2.0},
        },
    }


def action(event, generation):
    value = dispatched(generation)
    value["event"] = event
    return value


def test_new_dispatch_replaces_latest_generation():
    evaluator = PathPlanningEvaluator()
    evaluator.observe_event(dispatched(1))
    evaluator.observe_event(dispatched(2, 5.0))
    assert evaluator.latest_generation == 2
    assert evaluator.first_dispatch_time == 0.0


def test_stale_terminal_generation_is_ignored():
    evaluator = PathPlanningEvaluator()
    evaluator.observe_event(dispatched(1))
    evaluator.observe_event(dispatched(2))
    evaluator.observe_event(action("ABORTED", 1))
    assert not evaluator.complete


def test_both_dogs_must_reach_same_latest_generation_and_yaw_is_ignored():
    evaluator = PathPlanningEvaluator(endpoint_tolerance_m=1.0)
    evaluator.observe_event(dispatched(3))
    evaluator.observe_errors(1.0, {"go2_1": 0.5, "go2_3": 1.1})
    assert not evaluator.complete
    evaluator.observe_errors(1.5, {"go2_1": 0.9, "go2_3": 1.0})
    assert evaluator.complete and evaluator.path_success


def test_current_aborted_does_not_finish_evaluation():
    evaluator = PathPlanningEvaluator()
    evaluator.observe_event(dispatched(4))
    evaluator.observe_event(action("ABORTED", 4))
    assert not evaluator.complete
    assert not evaluator.path_success
    assert evaluator.failure_reason is None


def test_current_aborted_can_still_finish_by_reaching_endpoint():
    evaluator = PathPlanningEvaluator(endpoint_tolerance_m=1.0)
    evaluator.observe_event(dispatched(4))
    evaluator.observe_event(action("ABORTED", 4))
    evaluator.observe_errors(1.0, {"go2_1": 0.9, "go2_3": 1.0})
    assert evaluator.complete and evaluator.path_success
    assert evaluator.failure_reason is None


def test_current_aborted_can_still_finish_by_path_timeout():
    evaluator = PathPlanningEvaluator(path_timeout_sec=300.0)
    evaluator.observe_event(dispatched(4, 10.0))
    evaluator.observe_event(action("ABORTED", 4))
    evaluator.observe_errors(310.0, {"go2_1": 2.0, "go2_3": 2.0})
    assert evaluator.complete and not evaluator.path_success
    assert evaluator.failure_reason == "path_timeout"


def test_current_rejected_is_a_valid_failure():
    evaluator = PathPlanningEvaluator()
    evaluator.observe_event(dispatched(4))
    evaluator.observe_event(action("REJECTED", 4))
    assert evaluator.complete and not evaluator.path_success
    assert evaluator.failure_reason == "nav_goal_rejected"


def test_path_timeout_does_not_reset_on_new_generation():
    evaluator = PathPlanningEvaluator(path_timeout_sec=300.0)
    evaluator.observe_event(dispatched(1, 10.0))
    evaluator.observe_event(dispatched(2, 250.0))
    evaluator.observe_errors(310.0, {"go2_1": 2.0, "go2_3": 2.0})
    assert evaluator.complete
    assert evaluator.failure_reason == "path_timeout"
