"""Strict YAML loading and validation for target-test suites."""

import math
from pathlib import Path

import yaml

from go2_test_framework.common.execution import execution_from_mapping


SCENES = ("city", "forest", "airport")
ROUTES = ("straight", "rectangle", "v_shape")
ROBOTS = ("go2_1", "go2_2", "go2_3")


def read_yaml(path):
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to read YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path}: root must be a mapping")
    if value.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    return value


def _required(mapping, key, where):
    if key not in mapping:
        raise ValueError(f"{where}.{key} is required")
    return mapping[key]


def _finite_positive(value, where, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} must be numeric") from error
    minimum_ok = result >= 0.0 if allow_zero else result > 0.0
    if not math.isfinite(result) or not minimum_ok:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{where} must be finite and {relation}")
    return result


def _finite_number(value, where):
    if isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} must be numeric") from error
    if not math.isfinite(result):
        raise ValueError(f"{where} must be finite")
    return result


def load_routes(path):
    root = read_yaml(path)
    scenes = _required(root, "scenes", "root")
    if not isinstance(scenes, dict):
        raise ValueError("root.scenes must be a mapping")
    result = {}
    for scene in SCENES:
        entry = _required(scenes, scene, "scenes")
        base_world = _required(entry, "base_world", f"scenes.{scene}")
        if not isinstance(base_world, str) or not base_world.strip():
            raise ValueError(f"scenes.{scene}.base_world must be a non-empty string")
        route_map = _required(entry, "routes", f"scenes.{scene}")
        result[scene] = {"base_world": base_world, "routes": {}}
        for route_name in ROUTES:
            route = _required(route_map, route_name, f"scenes.{scene}.routes")
            points = _required(route, "points", f"{scene}.{route_name}")
            if not isinstance(points, list) or len(points) < 2:
                raise ValueError(f"{scene}.{route_name}.points must have at least two points")
            parsed_points = []
            for index, point in enumerate(points):
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError(f"{scene}.{route_name}.points[{index}] must be [x, y]")
                parsed_points.append(tuple(
                    _finite_number(v, f"{scene}.{route_name}.points[{index}]")
                    for v in point
                ))
            if any(a == b for a, b in zip(parsed_points, parsed_points[1:])):
                raise ValueError(f"{scene}.{route_name} contains a zero-length segment")
            traversal = _required(route, "traversal", f"{scene}.{route_name}")
            if traversal not in ("closed", "ping_pong"):
                raise ValueError(f"{scene}.{route_name}.traversal must be closed or ping_pong")
            provisional = _required(route, "provisional", f"{scene}.{route_name}")
            if not isinstance(provisional, bool):
                raise ValueError(f"{scene}.{route_name}.provisional must be boolean")
            result[scene]["routes"][route_name] = {
                "points": tuple(parsed_points),
                "traversal": traversal,
                "speed": _finite_positive(_required(route, "speed", f"{scene}.{route_name}"), f"{scene}.{route_name}.speed"),
                "turn_duration": _finite_positive(_required(route, "turn_duration", f"{scene}.{route_name}"), f"{scene}.{route_name}.turn_duration", allow_zero=True),
                "provisional": provisional,
            }
    return result


def _parse_pose_group(name, raw, where):
    if not isinstance(raw, dict):
        raise ValueError(f"{where}.{name} must be a mapping")
    resolved = _required(raw, "resolved", f"{where}.{name}")
    if not isinstance(resolved, bool):
        raise ValueError(f"{where}.{name}.resolved must be boolean")
    robots = _required(raw, "robots", f"{where}.{name}")
    if not isinstance(robots, dict):
        raise ValueError(f"{where}.{name}.robots must be a mapping")
    parsed = {}
    for robot in ROBOTS:
        pose = _required(robots, robot, f"{where}.{name}.robots")
        if not isinstance(pose, dict):
            raise ValueError(f"{where}.{name}.robots.{robot} must be a mapping")
        values = {}
        for field in ("x", "y", "z", "yaw"):
            value = _required(pose, field, f"{where}.{name}.robots.{robot}")
            if value is None and not resolved:
                values[field] = None
                continue
            if isinstance(value, bool):
                raise ValueError(f"{where}.{name}.robots.{robot}.{field} must be numeric")
            try:
                value = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError(f"{where}.{name}.robots.{robot}.{field} must be numeric") from error
            if not math.isfinite(value):
                raise ValueError(f"{where}.{name}.robots.{robot}.{field} must be finite")
            if field == "z" and value <= 0.0:
                raise ValueError(f"{where}.{name}.robots.{robot}.z must be positive")
            values[field] = value
        parsed[robot] = values
    return {"resolved": resolved, "robots": parsed}


def load_pose_groups(path):
    root = read_yaml(path)
    if root.get("coordinate_mode") != "shared_absolute":
        raise ValueError("coordinate_mode must be shared_absolute")
    groups = _required(root, "pose_groups", "root")
    if not isinstance(groups, dict):
        raise ValueError("root.pose_groups must be a mapping")
    expected = tuple(f"group_{index:02d}" for index in range(1, 12))
    if tuple(groups) != expected:
        raise ValueError("pose_groups must contain group_01..group_11 in order")
    return {name: _parse_pose_group(name, groups[name], "pose_groups") for name in expected}


def load_suite(path):
    root = read_yaml(path)
    suite_id = _required(root, "suite_id", "root")
    if not isinstance(suite_id, str) or not suite_id:
        raise ValueError("suite_id must be a non-empty string")
    for key, allowed in (("scenes", SCENES), ("routes", ROUTES)):
        values = _required(root, key, "root")
        if not isinstance(values, list) or not values or len(values) != len(set(values)):
            raise ValueError(f"{key} must be a non-empty unique list")
        unknown = set(values) - set(allowed)
        if unknown:
            raise ValueError(f"{key} contains unsupported values: {sorted(unknown)}")
    groups = _required(root, "pose_groups", "root")
    if not isinstance(groups, list) or not groups or len(groups) != len(set(groups)):
        raise ValueError("pose_groups must be a non-empty unique list")
    for key in ("formal", "consider_occlusion"):
        if not isinstance(_required(root, key, "root"), bool):
            raise ValueError(f"{key} must be boolean")
    if root["visibility_method"] != "camera_projection":
        raise ValueError("visibility_method must be camera_projection")
    if root["consider_occlusion"]:
        raise ValueError("consider_occlusion must be false in phase one")
    if root["perception_mode"] != "selected_robot":
        raise ValueError("perception_mode must be selected_robot")
    numeric_keys = (
        "evaluation_rate_hz", "evaluation_duration_sec", "match_timeout_sec",
        "startup_timeout_sec", "role_timeout_sec", "data_ready_timeout_sec",
        "min_camera_depth_m", "max_camera_depth_m",
    )
    parsed = dict(root)
    for key in numeric_keys:
        parsed[key] = _finite_positive(_required(root, key, "root"), key)
    if parsed["min_camera_depth_m"] >= parsed["max_camera_depth_m"]:
        raise ValueError("min_camera_depth_m must be less than max_camera_depth_m")
    inline = root.get("inline_pose_groups", {})
    if not isinstance(inline, dict):
        raise ValueError("inline_pose_groups must be a mapping")
    parsed["inline_pose_groups"] = {
        name: _parse_pose_group(name, value, "inline_pose_groups")
        for name, value in inline.items()
    }
    parsed["execution"] = execution_from_mapping(
        _required(root, "execution", "root")
    ).to_dict()
    return parsed


def require_resolved_pose(group_name, pose_group):
    if not pose_group["resolved"]:
        raise ValueError(
            f"pose group {group_name} is unresolved; fill all robots x/y/z/yaw and set resolved: true"
        )
    return pose_group["robots"]
