"""Strict configuration loading for the static perception suite."""

import math
from pathlib import Path

import yaml


TARGET_KEYS = ("airplane", "person", "pickup_truck", "ground_robot", "dumpster")
POSE_KEYS = tuple(f"pose_{index:02d}" for index in range(1, 21))


def read_yaml(path):
    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"failed to read YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _number(value, where, *, positive=False, non_negative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{where} must be finite")
    if positive and value <= 0 or non_negative and value < 0:
        raise ValueError(f"{where} has an invalid range")
    return value


def load_targets(path):
    root = read_yaml(path)
    if root.get("schema_version") != 1 or set(root) != {"schema_version", "targets"}:
        raise ValueError("target config must contain schema_version 1 and targets")
    targets = root["targets"]
    if not isinstance(targets, dict) or tuple(targets) != TARGET_KEYS:
        raise ValueError(f"targets must be ordered as {list(TARGET_KEYS)}")
    names = []
    for key, target in targets.items():
        if not isinstance(target, dict) or set(target) != {"model_name", "prompt", "pose"}:
            raise ValueError(f"targets.{key} has invalid fields")
        if not all(isinstance(target[name], str) and target[name].strip()
                   for name in ("model_name", "prompt")):
            raise ValueError(f"targets.{key} model_name/prompt must be non-empty")
        names.append(target["model_name"])
        pose = target["pose"]
        if not isinstance(pose, dict) or set(pose) != {"x", "y", "z", "roll", "pitch", "yaw"}:
            raise ValueError(f"targets.{key}.pose has invalid fields")
        for name, value in pose.items():
            _number(value, f"targets.{key}.pose.{name}")
    if len(names) != len(set(names)):
        raise ValueError("target model_name values must be unique")
    return targets


def load_poses(path):
    root = read_yaml(path)
    expected = {"schema_version", "coordinate_mode", "robot_name", "targets"}
    if set(root) != expected or root["schema_version"] != 1:
        raise ValueError("pose config has invalid root fields or schema_version")
    if root["coordinate_mode"] != "world_absolute" or root["robot_name"] != "go2_1":
        raise ValueError("pose config must use world_absolute coordinates for go2_1")
    if not isinstance(root["targets"], dict) or tuple(root["targets"]) != TARGET_KEYS:
        raise ValueError(f"pose targets must be ordered as {list(TARGET_KEYS)}")
    for target, poses in root["targets"].items():
        if not isinstance(poses, dict) or tuple(poses) != POSE_KEYS:
            raise ValueError(f"poses.{target} must contain ordered pose_01 through pose_20")
        for pose_key, pose in poses.items():
            if not isinstance(pose, dict) or set(pose) != {"x", "y", "z", "yaw"}:
                raise ValueError(f"poses.{target}.{pose_key} has invalid fields")
            for name, value in pose.items():
                _number(value, f"poses.{target}.{pose_key}.{name}")
    return root


def load_suite(path):
    root = read_yaml(path)
    required = {"schema_version", "suite_id", "targets", "poses", "evaluation_rate_hz",
                "evaluation_duration_sec", "match_timeout_sec", "result_settle_delay_sec",
                "startup_timeout_sec", "data_ready_timeout_sec", "min_camera_depth_m", "max_camera_depth_m",
                "consider_occlusion", "execution"}
    if set(root) != required or root["schema_version"] != 1:
        raise ValueError("suite has invalid fields or schema_version")
    if root["suite_id"] != "static_100cases":
        raise ValueError("suite_id must be static_100cases")
    if tuple(root["targets"]) != TARGET_KEYS or tuple(root["poses"]) != POSE_KEYS:
        raise ValueError("suite must select all five targets and pose_01 through pose_20")
    for key in ("evaluation_rate_hz", "evaluation_duration_sec", "match_timeout_sec",
                "startup_timeout_sec", "data_ready_timeout_sec", "min_camera_depth_m",
                "max_camera_depth_m"):
        root[key] = _number(root[key], key, positive=True)
    root["result_settle_delay_sec"] = _number(
        root["result_settle_delay_sec"], "result_settle_delay_sec", non_negative=True)
    if root["min_camera_depth_m"] >= root["max_camera_depth_m"]:
        raise ValueError("camera depth bounds are invalid")
    if root["consider_occlusion"] is not False:
        raise ValueError("consider_occlusion must be false")
    execution = root["execution"]
    fields = {"gazebo_gui", "rqt", "world_to_robot_delay_sec", "attitude_settle_delay_sec",
              "attitude_roll_limit_deg", "attitude_sample_frames", "attitude_timeout_sec",
              "max_restarts", "restart_delay_sec"}
    if not isinstance(execution, dict) or set(execution) != fields:
        raise ValueError("suite execution fields are invalid")
    for key in ("gazebo_gui", "rqt"):
        if not isinstance(execution[key], bool):
            raise ValueError(f"execution.{key} must be boolean")
    for key in ("world_to_robot_delay_sec", "attitude_settle_delay_sec", "restart_delay_sec"):
        execution[key] = _number(execution[key], f"execution.{key}", non_negative=True)
    for key in ("attitude_roll_limit_deg", "attitude_timeout_sec"):
        execution[key] = _number(execution[key], f"execution.{key}", positive=True)
    for key in ("attitude_sample_frames", "max_restarts"):
        if isinstance(execution[key], bool) or not isinstance(execution[key], int) or execution[key] < (1 if key == "attitude_sample_frames" else 0):
            raise ValueError(f"execution.{key} has an invalid range")
    return root


def load_metric(path, expected):
    root = read_yaml(path)
    if set(root) != {"metric", "pass_threshold_percent"} or root["metric"] != expected:
        raise ValueError(f"invalid metric configuration for {expected}")
    root["pass_threshold_percent"] = _number(
        root["pass_threshold_percent"], "pass_threshold_percent", non_negative=True)
    return root
