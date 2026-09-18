"""Strict parsing and active-set debouncing for Go2 trunk contacts."""

import json


IGNORED_MODELS = {"ground_plane", "walking_target"}


def _scoped_parts(value):
    return value.split("::")


def _trunk_robot(value):
    parts = _scoped_parts(value)
    if len(parts) >= 3 and parts[0].startswith("go2_") and (
        parts[1] == "trunk"
        or "fixed_joint_lump__trunk_collision" in parts[2]
    ):
        return parts[0]
    return None


def parse_body_contacts(payload):
    try:
        value = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid body-contact JSON: {error}") from error
    if set(value) != {"schema_version", "stamp", "contacts"}:
        raise ValueError("body-contact message has unexpected fields")
    if value["schema_version"] != 1:
        raise ValueError("unsupported body-contact schema_version")
    stamp = value["stamp"]
    if (
        not isinstance(stamp, dict) or set(stamp) != {"sec", "nanosec"}
        or not all(isinstance(stamp[key], int) for key in stamp)
    ):
        raise ValueError("body-contact stamp must contain integer sec/nanosec")
    contacts = value["contacts"]
    if not isinstance(contacts, list):
        raise ValueError("body-contact contacts must be a list")
    parsed = []
    for item in contacts:
        if (
            not isinstance(item, dict)
            or set(item) != {"collision1", "collision2"}
            or not all(isinstance(item[key], str) for key in item)
        ):
            raise ValueError("invalid body-contact pair")
        parsed.append((item["collision1"], item["collision2"]))
    return stamp, parsed


class CollisionDebouncer:
    """Emit one event per external static contact activation."""

    def __init__(self):
        self.active = set()

    @staticmethod
    def eligible_pair(first, second):
        robot = _trunk_robot(first)
        robot_collision = first
        object_collision = second
        if robot is None:
            robot = _trunk_robot(second)
            robot_collision = second
            object_collision = first
        if robot is None:
            return None
        object_parts = _scoped_parts(object_collision)
        object_model = object_parts[0] if object_parts else object_collision
        if (
            object_model in IGNORED_MODELS
            or object_model == robot
            or object_model.startswith("go2_")
        ):
            return None
        key = tuple(sorted((robot_collision, object_collision)))
        return key, {
            "robot": robot,
            "robot_collision": robot_collision,
            "object_model": object_model,
            "object_collision": object_collision,
        }

    def update(self, pairs):
        current = {}
        for first, second in pairs:
            eligible = self.eligible_pair(first, second)
            if eligible is not None:
                key, event = eligible
                current[key] = event
        new_events = [current[key] for key in sorted(current.keys() - self.active)]
        self.active = set(current)
        return new_events
