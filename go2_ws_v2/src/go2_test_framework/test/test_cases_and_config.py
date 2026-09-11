from pathlib import Path

import pytest
import yaml

from go2_test_framework.common.config import load_pose_groups, load_routes, load_suite, require_resolved_pose
from go2_test_framework.runner.cases import expand_cases


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def test_formal_suite_expands_stable_99_cases():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / "config/suites/T1_target_test.yaml")
    cases = expand_cases(suite, routes, poses)
    assert len(cases) == 99
    assert all(case.task_type == "perception" for case in cases)
    assert cases[0].case_id == "T1-CITY-STRAIGHT-G01"
    assert cases[0].case_index == 1
    assert cases[-1].case_id == "T1-AIRPORT-V-G11"
    assert cases[-1].case_index == 99


def test_formal_t2_suite_expands_stable_99_tracking_cases():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / "config/suites/T2_tracking_test.yaml")
    cases = expand_cases(suite, routes, poses)
    assert len(cases) == 99
    assert cases[0].case_id == "T2-CITY-STRAIGHT-G01"
    assert cases[-1].case_id == "T2-AIRPORT-V-G11"
    assert {case.task_type for case in cases} == {"tracking"}


def test_formal_t3_suite_preserves_forest_only_selection():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / "config/suites/T3_path_planning_test.yaml")
    cases = expand_cases(suite, routes, poses)
    assert suite["scenes"] == ["forest"]
    assert len(cases) == 33
    assert cases[0].case_id == "T3-FOREST-STRAIGHT-G01"
    assert cases[-1].case_id == "T3-FOREST-V-G11"
    assert {case.task_type for case in cases} == {"path_planning"}


def test_formal_pose_groups_are_resolved_and_accepted():
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    assert tuple(poses) == ("city", "forest", "airport")
    for groups in poses.values():
        assert tuple(groups) == tuple(f"group_{i:02d}" for i in range(1, 12))
        assert all(group["resolved"] for group in groups.values())
        assert all(set(group["robots"]) == {"go2_1", "go2_2", "go2_3"} for group in groups.values())
    assert require_resolved_pose("group_01", poses["city"]["group_01"])["go2_1"]["z"] == 0.6


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


def test_suite_task_type_is_required_and_validated(tmp_path):
    source = (PACKAGE_ROOT / "config/suites/T1_smoke_city.yaml").read_text(
        encoding="utf-8"
    )
    missing = tmp_path / "missing.yaml"
    missing.write_text(source.replace("task_type: perception\n", ""), encoding="utf-8")
    with pytest.raises(ValueError, match="root.task_type is required"):
        load_suite(missing)

    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        source.replace("task_type: perception", "task_type: unsupported"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="task_type must be one of"):
        load_suite(invalid)


@pytest.mark.parametrize("suite_name,prefix", [("T1_target_test", "T1"), ("T2_tracking_test", "T2")])
def test_every_case_id_order_and_scene_pose(suite_name, prefix):
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml")
    suite = load_suite(PACKAGE_ROOT / f"config/suites/{suite_name}.yaml")
    cases = expand_cases(suite, routes, poses, require_resolved=True)
    expected = [
        f"{prefix}-{scene.upper()}-{route}-G{i:02d}"
        for scene in ("city", "forest", "airport")
        for route in ("STRAIGHT", "RECTANGLE", "V")
        for i in range(1, 12)
    ]
    assert [case.case_id for case in cases] == expected
    assert [case.case_index for case in cases] == list(range(1, 100))
    for case in cases:
        assert case.robot_poses == poses[case.scene][case.pose_group]["robots"]
    suite["inline_pose_groups"] = {"group_01": poses["forest"]["group_02"]}
    overridden = expand_cases(suite, routes, poses)
    assert all(c.robot_poses == poses["forest"]["group_02"]["robots"]
               for c in overridden if c.pose_group == "group_01")
    assert poses["city"]["group_01"] != suite["inline_pose_groups"]["group_01"]
    del poses["forest"]["group_02"]
    with pytest.raises(ValueError, match="forest/group_02"):
        expand_cases(suite, routes, poses)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), 1),
    (("schema_version",), True),
    (("coordinate_mode",), "invalid"),
    (("scenes",), []),
    (("scenes", "city"), None),
    (("scenes", "city", "pose_groups"), []),
    (("scenes", "city", "pose_groups", "group_01"), None),
    (("scenes", "city", "pose_groups", "group_01", "resolved"), "true"),
    (("scenes", "city", "pose_groups", "group_01", "robots"), []),
    (("scenes", "city", "pose_groups", "group_01", "robots", "go2_1"), None),
    *((('scenes', 'city', 'pose_groups', 'group_01', 'robots', 'go2_1', field), value)
      for field, value in [("x", True), ("x", float("nan")), ("yaw", float("inf")), ("y", None), ("z", 0), ("z", -1)]),
])
def test_pose_schema_rejects_invalid_values(tmp_path, path, value):
    document = yaml.safe_load((PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml").read_text())
    entry = document
    for key in path[:-1]:
        entry = entry[key]
    entry[path[-1]] = value
    output = tmp_path / "poses.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError):
        load_pose_groups(output)


@pytest.mark.parametrize("path", [
    ("scenes", "airport"),
    ("scenes", "forest", "pose_groups", "group_11"),
    ("scenes", "city", "pose_groups", "group_01", "resolved"),
    ("scenes", "city", "pose_groups", "group_01", "robots", "go2_3"),
    ("scenes", "city", "pose_groups", "group_01", "robots", "go2_1", "yaw"),
])
def test_pose_schema_rejects_missing_fields(tmp_path, path):
    document = yaml.safe_load((PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml").read_text())
    entry = document
    for key in path[:-1]:
        entry = entry[key]
    del entry[path[-1]]
    output = tmp_path / "poses.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError):
        load_pose_groups(output)


def test_pose_group_order_and_unresolved_semantics(tmp_path):
    document = yaml.safe_load((PACKAGE_ROOT / "config/parameters/robot_pose_groups.yaml").read_text())
    groups = document["scenes"]["city"]["pose_groups"]
    groups["group_01"]["resolved"] = False
    groups["group_01"]["robots"]["go2_1"]["x"] = None
    output = tmp_path / "poses.yaml"
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    parsed = load_pose_groups(output)
    with pytest.raises(ValueError, match="unresolved"):
        require_resolved_pose("group_01", parsed["city"]["group_01"])
    groups["group_01"] = groups.pop("group_01")
    output.write_text(yaml.safe_dump(document, sort_keys=False))
    with pytest.raises(ValueError, match="in order"):
        load_pose_groups(output)
