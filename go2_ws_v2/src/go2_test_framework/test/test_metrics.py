import pytest

from go2_test_framework.evaluators.localization import evaluate_localization
from go2_test_framework.evaluators.recognition import evaluate_recognition


def test_recognition_counts_unmatched_visible_frame_as_incorrect():
    result = evaluate_recognition([
        {"visible": True, "recognition_matched": True, "recognition_success": True},
        {"visible": True, "recognition_matched": False, "recognition_success": False},
        {"visible": False, "recognition_matched": True, "recognition_success": True},
    ], threshold_percent=50.0)
    assert result == {"valid_frames": 2, "correct_frames": 1, "accuracy": 50.0, "pass": True}


def test_recognition_zero_visible_frames_fails_without_division():
    result = evaluate_recognition([])
    assert result["accuracy"] is None
    assert result["pass"] is False


def test_localization_computes_mean_relative_percent_only_from_valid_samples():
    row = {
        "visible": True, "localization_matched": True, "localization_success": True,
        "target_gt_x": 3.0, "target_gt_y": 4.0,
        "target_est_x": 3.3, "target_est_y": 4.4,
        "robot_gt_x": 0.0, "robot_gt_y": 0.0,
    }
    result = evaluate_localization([row], threshold_percent=15.0)
    assert result["valid_samples"] == 1
    assert result["mean_relative_error"] == pytest.approx(10.0)
    assert result["pass"] is True


def test_localization_zero_reference_and_zero_samples_fail():
    zero = {
        "visible": True, "localization_matched": True, "localization_success": True,
        "target_gt_x": 1.0, "target_gt_y": 2.0,
        "target_est_x": 1.0, "target_est_y": 2.0,
        "robot_gt_x": 1.0, "robot_gt_y": 2.0,
    }
    assert evaluate_localization([zero])["pass"] is False
    assert evaluate_localization([])["mean_relative_error"] is None
