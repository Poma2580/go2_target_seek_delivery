import json

import pytest
from builtin_interfaces.msg import Time

from go2_target_perception.target_perception import TargetPerception, result_status_json


def test_result_status_three_outcomes_are_strict_and_versioned():
    stamp = Time(sec=12, nanosec=34)
    failed = json.loads(result_status_json(stamp, 0))
    recognized = json.loads(result_status_json(
        stamp, 1, True, 0.8, (1, 2, 3, 4), False))
    localized = json.loads(result_status_json(
        stamp, 2, True, 0.9, (5, 6, 7, 8), True))
    assert failed["recognition_success"] is False
    assert failed["confidence"] is None and failed["bbox"] is None
    assert recognized["recognition_success"] is True
    assert recognized["localization_success"] is False
    assert localized["localization_success"] is True
    assert localized["stamp"] == {"sec": 12, "nanosec": 34}


def test_result_status_rejects_non_finite_json_values():
    with pytest.raises(ValueError, match="有限"):
        result_status_json(Time(), 1, confidence=float("nan"))


@pytest.mark.parametrize(
    "outcome",
    [
        (False, None, None, False),
        (True, 0.8, (1.0, 2.0, 3.0, 4.0), False),
        (True, 0.9, (5.0, 6.0, 7.0, 8.0), True),
    ],
)
def test_rgbd_wrapper_publishes_exactly_one_status_for_each_outcome(outcome):
    node = TargetPerception.__new__(TargetPerception)
    node._next_sample_id = 10
    node._process_rgbd_pair_impl = lambda rgb, depth: outcome
    published = []
    node._publish_result_status = lambda *args, **kwargs: published.append((args, kwargs))
    depth = type("Depth", (), {"header": type("Header", (), {"stamp": Time()})()})()
    node._process_rgbd_pair(object(), depth)
    assert len(published) == 1
    assert published[0][0][1] == 10
    assert published[0][1]["recognition_success"] is outcome[0]
    assert published[0][1]["localization_success"] is outcome[3]


def test_rgbd_wrapper_converts_unexpected_exception_to_one_failure_status():
    node = TargetPerception.__new__(TargetPerception)
    node._next_sample_id = 3
    node._process_rgbd_pair_impl = lambda rgb, depth: (_ for _ in ()).throw(RuntimeError("boom"))
    node._warn = lambda text: None
    published = []
    node._publish_result_status = lambda *args, **kwargs: published.append((args, kwargs))
    depth = type("Depth", (), {"header": type("Header", (), {"stamp": Time()})()})()
    node._process_rgbd_pair(object(), depth)
    assert len(published) == 1
    assert published[0][1]["recognition_success"] is False
    assert published[0][1]["localization_success"] is False
