"""T2 continuous tracking evaluator tests."""

from go2_test_framework.evaluators.tracking import evaluate_tracking


def sample(time_value, effective, visible=True):
    return {
        "eval_time": time_value,
        "effective_tracking": effective,
        "visible": visible,
    }


def test_continuous_five_seconds_succeeds():
    rows = [sample(0.0, False)] + [
        sample(float(index), True) for index in range(1, 7)
    ]
    result = evaluate_tracking(rows)
    assert result["complete"]
    assert result["tracking_success"]
    assert result["acquisition_time_sec"] == 1.0
    assert result["continuous_tracking_time_sec"] == 5.0


def test_disjoint_tracking_time_is_not_accumulated():
    rows = [sample(0.0, True), sample(3.0, True), sample(3.5, False)]
    result = evaluate_tracking(rows)
    assert result["complete"]
    assert not result["tracking_success"]
    assert result["continuous_tracking_time_sec"] == 3.5
    assert result["failure_reason"] == "tracking_radius_exceeded"


def test_visibility_loss_has_stable_failure_reason():
    result = evaluate_tracking([
        sample(0.0, True), sample(0.5, False, visible=False)
    ])
    assert result["failure_reason"] == "visibility_lost"


def test_acquisition_timeout_is_valid_failure():
    result = evaluate_tracking([
        sample(0.0, False), sample(29.5, False), sample(30.0, False)
    ])
    assert result["complete"]
    assert not result["tracking_success"]
    assert result["acquisition_time_sec"] is None
    assert result["failure_reason"] == "acquisition_timeout"


def test_pending_window_is_not_failed_early():
    result = evaluate_tracking([sample(0.0, False), sample(29.5, False)])
    assert not result["complete"]
    assert result["failure_reason"] is None
