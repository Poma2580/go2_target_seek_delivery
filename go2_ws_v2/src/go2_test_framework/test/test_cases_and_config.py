from pathlib import Path

import pytest

from go2_test_framework.common.config import load_pose_groups, load_routes, load_suite, require_resolved_pose
from go2_test_framework.runner.cases import expand_cases


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_formal_suite_expands_stable_99_cases():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / "config/suites/T1_target_test.yaml")
    cases = expand_cases(suite, routes, poses)
    assert len(cases) == 99
    assert cases[0].case_id == "T1-CITY-STRAIGHT-G01"
    assert cases[0].case_index == 1
    assert cases[-1].case_id == "T1-AIRPORT-V-G11"
    assert cases[-1].case_index == 99


def test_formal_pose_groups_are_resolved_and_accepted():
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    assert all(group["resolved"] for group in poses.values())
    assert require_resolved_pose("group_01", poses["group_01"])["go2_1"]["z"] == 0.8


def test_unresolved_pose_is_still_rejected_before_execution():
    with pytest.raises(ValueError, match="temporary is unresolved"):
        require_resolved_pose("temporary", {"resolved": False, "robots": {}})


def test_smoke_case_has_resolved_current_city_pose():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / "config/suites/T1_smoke_city.yaml")
    case = expand_cases(suite, routes, poses, require_resolved=True)[0]
    assert case.case_id == "T1-SMOKE-CITY-RECTANGLE-SMOKE_DEFAULT"
    assert case.robot_poses["go2_1"]["x"] == -30.0
    assert case.formal is False
    assert int(round(
        case.settings["evaluation_rate_hz"]
        * case.settings["evaluation_duration_sec"]
    )) == 60
