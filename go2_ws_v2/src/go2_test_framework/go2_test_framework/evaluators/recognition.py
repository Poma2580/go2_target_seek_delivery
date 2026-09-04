"""Recognition metric from recorder CSV facts."""


def _is_true(value):
    return value is True or str(value).strip().lower() == "true"


def evaluate_recognition(rows, threshold_percent=80.0):
    visible = [row for row in rows if _is_true(row.get("visible"))]
    valid_frames = len(visible)
    correct_frames = sum(
        _is_true(row.get("recognition_matched"))
        and _is_true(row.get("recognition_success"))
        for row in visible
    )
    if valid_frames == 0:
        return {
            "valid_frames": 0,
            "correct_frames": 0,
            "accuracy": None,
            "pass": False,
            "reason": "no visible evaluation frames",
        }
    accuracy = correct_frames / valid_frames * 100.0
    return {
        "valid_frames": valid_frames,
        "correct_frames": correct_frames,
        "accuracy": accuracy,
        "pass": accuracy >= float(threshold_percent),
    }
