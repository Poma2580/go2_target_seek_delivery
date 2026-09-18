"""Stable target-major expansion of the 100 static cases."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class StaticCase:
    case_index: int
    case_id: str
    suite_id: str
    target_key: str
    model_name: str
    prompt: str
    target_gt_pose: dict
    robot_pose: dict
    settings: dict
    metrics: dict

    def to_dict(self):
        return asdict(self)


def expand_cases(suite, targets, poses, metrics):
    settings = {key: suite[key] for key in (
        "evaluation_rate_hz", "evaluation_duration_sec", "match_timeout_sec",
        "result_settle_delay_sec", "startup_timeout_sec", "data_ready_timeout_sec",
        "min_camera_depth_m",
        "max_camera_depth_m", "consider_occlusion")}
    result = []
    for target_key in suite["targets"]:
        target = targets[target_key]
        for pose_key in suite["poses"]:
            index = len(result) + 1
            token = target_key.replace("_", "").upper()
            result.append(StaticCase(
                index, f"SP-{token}-P{int(pose_key[-2:]):02d}", suite["suite_id"],
                target_key, target["model_name"], target["prompt"],
                dict(target["pose"]), dict(poses["targets"][target_key][pose_key]),
                dict(settings), dict(metrics)))
    ids = [case.case_id for case in result]
    if len(result) != 100 or len(ids) != len(set(ids)):
        raise ValueError("suite expansion must produce 100 unique cases")
    return result
