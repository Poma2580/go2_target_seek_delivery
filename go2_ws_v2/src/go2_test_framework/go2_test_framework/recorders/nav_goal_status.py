"""Validation shared by T3 NavGoal event consumers."""

import json
import math


EVENTS = {
    "DISPATCHED", "ACCEPTED", "REJECTED",
    "SUCCEEDED", "CANCELED", "ABORTED",
}


def parse_nav_goal_status(data):
    try:
        value = json.loads(data)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid nav goal status JSON: {error}") from error
    required = {
        "schema_version", "stamp", "event", "generation", "frame_id",
        "navigation_dogs", "goals", "robot", "action_status",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"nav goal status missing fields: {sorted(missing)}")
    if value["schema_version"] != 1 or value["event"] not in EVENTS:
        raise ValueError("unsupported nav goal status schema or event")
    if not isinstance(value["generation"], int) or value["generation"] <= 0:
        raise ValueError("nav goal generation must be a positive integer")
    dogs = value["navigation_dogs"]
    if not isinstance(dogs, list) or len(dogs) != 2 or len(set(dogs)) != 2:
        raise ValueError("navigation_dogs must contain two unique names")
    if set(value["goals"]) != set(dogs):
        raise ValueError("goals must exactly match navigation_dogs")
    for name in dogs:
        goal = value["goals"][name]
        if set(goal) != {"goal_x", "goal_y", "goal_yaw"} or not all(
            isinstance(item, (int, float)) and not isinstance(item, bool)
            and math.isfinite(item) for item in goal.values()
        ):
            raise ValueError(f"invalid goal for {name}")
    if not isinstance(value["frame_id"], str) or not value["frame_id"]:
        raise ValueError("frame_id must be non-empty")
    return value
