"""T1-compatible recognition and two-dimensional localization metrics."""

import math


def is_true(value):
    return value is True or str(value).strip().lower() == "true"


def number(row, key):
    try:
        value = float(row.get(key))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def recognition(rows, threshold=80.0):
    visible = [row for row in rows if is_true(row.get("visible"))]
    correct = sum(is_true(row.get("recognition_matched")) and
                  is_true(row.get("recognition_success")) for row in visible)
    accuracy = 100.0 * correct / len(visible) if visible else None
    result = {"valid_frames": len(visible), "correct_frames": correct,
              "accuracy": accuracy, "pass": accuracy is not None and accuracy >= threshold}
    if accuracy is None:
        result["reason"] = "no visible evaluation frames"
    return result


def relative_errors(rows, distance_epsilon=1e-6):
    values = []
    for row in rows:
        if not (is_true(row.get("visible")) and is_true(row.get("localization_matched"))
                and is_true(row.get("localization_success"))):
            continue
        nums = [number(row, key) for key in ("target_gt_x", "target_gt_y", "target_est_x",
                                             "target_est_y", "robot_gt_x", "robot_gt_y")]
        if any(value is None for value in nums):
            continue
        tx, ty, ex, ey, rx, ry = nums
        reference = math.hypot(tx - rx, ty - ry)
        if reference <= distance_epsilon:
            raise ZeroDivisionError("target-to-robot reference distance is zero")
        values.append(math.hypot(ex - tx, ey - ty) / reference)
    return values


def localization(rows, threshold=15.0):
    try:
        errors = relative_errors(rows)
    except ZeroDivisionError as error:
        return {"valid_samples": 0, "mean_relative_error": None,
                "pass": False, "reason": str(error)}
    mean = 100.0 * math.fsum(errors) / len(errors) if errors else None
    result = {"valid_samples": len(errors), "mean_relative_error": mean,
              "pass": mean is not None and mean <= threshold}
    if mean is None:
        result["reason"] = "no valid localization samples"
    return result
