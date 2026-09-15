"""Mean relative localization error from recorder CSV facts."""

import math

from go2_test_framework.evaluators.recognition import _is_true


def _number(row, key):
    value = row.get(key)
    if value in (None, ""):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_localization(rows, threshold_percent=15.0, distance_epsilon=1e-6):
    relative_errors = []
    for row in rows:
        if not (
            _is_true(row.get("visible"))
            and _is_true(row.get("localization_matched"))
            and _is_true(row.get("localization_success"))
        ):
            continue
        values = [_number(row, key) for key in (
            "target_gt_x", "target_gt_y", "target_est_x", "target_est_y",
            "robot_gt_x", "robot_gt_y",
        )]
        if any(value is None for value in values):
            continue
        target_x, target_y, estimate_x, estimate_y, robot_x, robot_y = values
        reference = math.hypot(target_x - robot_x, target_y - robot_y)
        if reference <= distance_epsilon:
            return {
                "valid_samples": len(relative_errors),
                "mean_relative_error": None,
                "pass": False,
                "reason": "target-to-robot reference distance is zero",
            }
        absolute = math.hypot(estimate_x - target_x, estimate_y - target_y)
        relative_errors.append(absolute / reference)
    if not relative_errors:
        return {
            "valid_samples": 0,
            "mean_relative_error": None,
            "pass": False,
            "reason": "no valid localization samples",
        }
    mean_percent = sum(relative_errors) / len(relative_errors) * 100.0
    return {
        "valid_samples": len(relative_errors),
        "mean_relative_error": mean_percent,
        "pass": mean_percent <= float(threshold_percent),
    }
