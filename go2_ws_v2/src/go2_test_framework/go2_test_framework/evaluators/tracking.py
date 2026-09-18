"""Continuous target-tracking metric evaluation."""


def evaluate_tracking(
    rows,
    tracking_min_duration_sec=5.0,
    acquisition_timeout_sec=30.0,
):
    """Evaluate ordered samples without accumulating disjoint tracking spans."""
    result = {
        "complete": False,
        "tracking_success": False,
        "acquisition_time_sec": None,
        "continuous_tracking_time_sec": 0.0,
        "failure_reason": None,
    }
    if not rows:
        return result
    started = float(rows[0]["eval_time"])
    acquired = None
    for row in rows:
        now = float(row["eval_time"])
        effective = bool(row["effective_tracking"])
        if acquired is None:
            if effective:
                acquired = now
                result["acquisition_time_sec"] = now - started
            elif now - started >= acquisition_timeout_sec:
                result.update({
                    "complete": True,
                    "failure_reason": "acquisition_timeout",
                })
                return result
            continue

        elapsed = now - acquired
        result["continuous_tracking_time_sec"] = max(0.0, elapsed)
        if not effective:
            result.update({
                "complete": True,
                "failure_reason": (
                    "visibility_lost"
                    if not bool(row["visible"])
                    else "tracking_radius_exceeded"
                ),
            })
            return result
        if elapsed >= tracking_min_duration_sec:
            result.update({"complete": True, "tracking_success": True})
            return result
    return result
