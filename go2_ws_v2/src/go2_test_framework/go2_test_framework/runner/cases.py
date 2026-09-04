"""Stable Cartesian expansion of T1 target-test cases."""

from dataclasses import asdict, dataclass

from go2_test_framework.common.config import require_resolved_pose


ROUTE_IDS = {"straight": "STRAIGHT", "rectangle": "RECTANGLE", "v_shape": "V"}


@dataclass(frozen=True)
class TestCase:
    case_index: int
    case_id: str
    suite_id: str
    formal: bool
    scene: str
    route: str
    pose_group: str
    route_config: dict
    robot_poses: dict
    settings: dict

    def to_dict(self):
        return asdict(self)


def expand_cases(suite, routes, pose_groups, require_resolved=False):
    """Expand cases in scene -> route -> pose-group order."""
    groups = dict(pose_groups)
    groups.update(suite.get("inline_pose_groups", {}))
    result = []
    index = 0
    settings = {
        key: suite[key]
        for key in (
            "evaluation_rate_hz", "evaluation_duration_sec", "match_timeout_sec",
            "startup_timeout_sec", "role_timeout_sec", "data_ready_timeout_sec",
            "visibility_method", "consider_occlusion", "perception_mode",
            "min_camera_depth_m", "max_camera_depth_m",
        )
    }
    for scene in suite["scenes"]:
        for route_name in suite["routes"]:
            route = routes[scene]["routes"][route_name]
            for group_name in suite["pose_groups"]:
                index += 1
                if group_name not in groups:
                    raise ValueError(f"pose group {group_name} is not defined")
                pose_group = groups[group_name]
                robot_poses = (
                    require_resolved_pose(group_name, pose_group)
                    if require_resolved else pose_group["robots"]
                )
                group_id = (
                    f"G{int(group_name.rsplit('_', 1)[1]):02d}"
                    if group_name.startswith("group_") else group_name.upper()
                )
                case_id = f"{suite['suite_id']}-{scene.upper()}-{ROUTE_IDS[route_name]}-{group_id}"
                result.append(TestCase(
                    case_index=index,
                    case_id=case_id,
                    suite_id=suite["suite_id"],
                    formal=suite["formal"],
                    scene=scene,
                    route=route_name,
                    pose_group=group_name,
                    route_config=route,
                    robot_poses=robot_poses,
                    settings=settings,
                ))
    return result
