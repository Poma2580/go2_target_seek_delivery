"""Execution policy, retry orchestration, and batch reporting tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from go2_test_framework.common.config import load_pose_groups, load_routes, load_suite
from go2_test_framework.common.execution import (
    apply_execution_overrides,
    execution_from_mapping,
)
from go2_test_framework.runner.cases import expand_cases
from go2_test_framework.runner.main import _batch_summary, _build_parser
from go2_test_framework.runner.orchestration import (
    AttemptResult,
    AttemptStatus,
    CaseResult,
    run_case,
    should_retry,
    spawn_robots,
)
from go2_test_framework.runner.ros_wait import active_controller_names


PACKAGE = Path(__file__).parents[1]


def execution_mapping(startup_overrides=None, **attitude_overrides):
    startup = {
        "world_to_first_delay_sec": 3.0,
        "inter_robot_delay_sec": 3.0,
        "enable_lidar": False,
    }
    startup.update(startup_overrides or {})
    attitude = {
        "enabled": True,
        "settle_delay_sec": 3.0,
        "roll_limit_deg": 90.0,
        "sample_frames": 5,
        "timeout_sec": 10.0,
        "max_restarts": 3,
        "restart_delay_sec": 1.0,
    }
    attitude.update(attitude_overrides)
    return {
        "gazebo_gui": False,
        "rqt": False,
        "robot_startup": startup,
        "attitude_check": attitude,
    }


def smoke_case():
    suite = load_suite(PACKAGE / "config/suites/T1_smoke_city.yaml")
    routes = load_routes(PACKAGE / "config/parameters/target_routes.yaml")
    poses = load_pose_groups(PACKAGE / "config/parameters/robot_pose_groups.yaml")
    return expand_cases(suite, routes, poses, require_resolved=True)[0]


def result(number, status, *, passed=False, reason=None):
    summary = {
        "infrastructure_valid": status is AttemptStatus.COMPLETED,
        "provisional": False,
        "recognition": None,
        "localization": None,
        "pass": passed,
    }
    return AttemptResult(number, status, reason, summary)


def test_suite_execution_defaults_and_cli_overrides():
    suite = load_suite(PACKAGE / "config/suites/T1_target_test.yaml")
    config = execution_from_mapping(suite["execution"])
    assert config.attitude_check.enabled
    assert config.attitude_check.settle_delay_sec == 3.0
    assert config.attitude_check.sample_frames == 5
    assert config.robot_startup.world_to_first_delay_sec == 3.0
    assert config.robot_startup.inter_robot_delay_sec == 3.0

    overridden = apply_execution_overrides(
        config, gazebo_gui=False, rqt=False,
        attitude_enabled=False, max_restarts=1, enable_lidar=True,
    )
    assert not overridden.gazebo_gui
    assert not overridden.rqt
    assert not overridden.attitude_check.enabled
    assert overridden.attitude_check.max_restarts == 1
    assert overridden.robot_startup.enable_lidar


def test_cli_boolean_options_support_positive_and_negative_forms():
    parser = _build_parser(Path("/package"))
    enabled = parser.parse_args(
        ["--gui", "--rqt", "--lidar", "--check-attitude"]
    )
    disabled = parser.parse_args(
        ["--no-gui", "--no-rqt", "--no-lidar", "--no-check-attitude"]
    )
    assert (
        enabled.gui, enabled.rqt, enabled.lidar, enabled.check_attitude
    ) == (True, True, True, True)
    assert (
        disabled.gui, disabled.rqt, disabled.lidar, disabled.check_attitude
    ) == (False, False, False, False)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_restarts": -1}, "max_restarts"),
        ({"settle_delay_sec": -0.1}, "settle_delay_sec"),
        ({"sample_frames": 0}, "sample_frames"),
        ({"timeout_sec": 0}, "timeout_sec"),
        ({"restart_delay_sec": -0.1}, "restart_delay_sec"),
    ],
)
def test_invalid_execution_values_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        execution_from_mapping(execution_mapping(**overrides))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"world_to_first_delay_sec": -0.1}, "world_to_first_delay_sec"),
        ({"inter_robot_delay_sec": -0.1}, "inter_robot_delay_sec"),
        ({"enable_lidar": "false"}, "enable_lidar"),
    ],
)
def test_invalid_robot_startup_values_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        execution_from_mapping(execution_mapping(startup_overrides=overrides))


def test_active_controller_names_filters_inactive_controllers():
    response = SimpleNamespace(controller=[
        SimpleNamespace(name="joint_group_effort_controller", state="active"),
        SimpleNamespace(name="joint_states_controller", state="inactive"),
    ])
    assert active_controller_names(response) == {"joint_group_effort_controller"}


@pytest.mark.parametrize("enable_lidar", [False, True])
def test_robot_spawn_sequence_and_conditional_lidar(enable_lidar):
    events = []

    class FakeProcesses:
        def start(self, name, command):
            events.append(("start", name, command))

    def wait_graph(kind, name, timeout):
        events.append(("graph", kind, name))

    def wait_controllers(robot, timeout, health_check=None):
        events.append(("controllers", robot))

    def sleep(seconds):
        events.append(("sleep", seconds))

    config = execution_from_mapping(execution_mapping(
        startup_overrides={"enable_lidar": enable_lidar}
    ))
    spawn_robots(
        smoke_case(), FakeProcesses(), 120.0, config.robot_startup,
        wait_graph=wait_graph,
        wait_controllers=wait_controllers,
        sleep=sleep,
    )

    assert [event for event in events if event[0] == "sleep"] == [
        ("sleep", 3.0), ("sleep", 3.0), ("sleep", 3.0),
    ]
    assert [event[1] for event in events if event[0] == "start"] == [
        "spawn_go2_1", "spawn_go2_2", "spawn_go2_3",
    ]
    assert [event[1] for event in events if event[0] == "controllers"] == [
        "go2_1", "go2_2", "go2_3",
    ]
    sleep_positions = [
        index for index, event in enumerate(events) if event[0] == "sleep"
    ]
    start_positions = [
        index for index, event in enumerate(events) if event[0] == "start"
    ]
    assert (
        sleep_positions[0] < start_positions[0]
        < sleep_positions[1] < start_positions[1]
        < sleep_positions[2] < start_positions[2]
    )
    graph_topics = [event[2] for event in events if event[0] == "graph"]
    for robot in ("go2_1", "go2_2", "go2_3"):
        assert f"/{robot}/odom" in graph_topics
        assert f"/{robot}/odom/ground_truth" in graph_topics
        assert f"/{robot}/camera/image_raw" in graph_topics
        assert f"/{robot}/camera/depth/image_raw" in graph_topics
        assert f"/{robot}/camera/depth/camera_info" in graph_topics
        assert (f"/{robot}/velodyne_points" in graph_topics) is enable_lidar
    for event in events:
        if event[0] == "start":
            expected = f"enable_lidar:={'true' if enable_lidar else 'false'}"
            assert expected in event[2]


def test_retry_policy_only_accepts_fallen_with_budget():
    assert should_retry(result(1, AttemptStatus.FALLEN), 3)
    assert not should_retry(result(4, AttemptStatus.FALLEN), 3)
    assert not should_retry(result(1, AttemptStatus.INFRASTRUCTURE_FAILED), 3)
    assert not should_retry(result(1, AttemptStatus.COMPLETED), 3)


def run_with_results(tmp_path, results, max_restarts=3):
    calls = []

    def fake_attempt(*args, **kwargs):
        number = args[-1]
        calls.append(number)
        return results[number - 1]

    config = execution_from_mapping(
        execution_mapping(max_restarts=max_restarts, restart_delay_sec=0.0)
    )
    case_result = run_case(
        smoke_case(), PACKAGE, tmp_path / "case_001", Path("model.pt"),
        {}, config, attempt_runner=fake_attempt, sleep=lambda _: None,
    )
    return case_result, calls


def test_fallen_attempt_restarts_then_completed_metric_failure_stops(tmp_path):
    case_result, calls = run_with_results(tmp_path, [
        result(1, AttemptStatus.FALLEN, reason="fallen"),
        result(2, AttemptStatus.COMPLETED, passed=False),
    ])
    assert calls == [1, 2]
    assert case_result.status == "completed"
    assert not case_result.summary["pass"]
    assert case_result.summary["restarts_used"] == 1
    assert not case_result.summary["restart_exhausted"]


def test_repeated_falls_exhaust_restart_budget_and_record_failure(tmp_path):
    falls = [
        result(number, AttemptStatus.FALLEN, reason="fallen")
        for number in range(1, 5)
    ]
    case_result, calls = run_with_results(tmp_path, falls)
    assert calls == [1, 2, 3, 4]
    assert case_result.infrastructure_failed
    assert case_result.summary["restart_exhausted"]
    assert case_result.summary["attempts_used"] == 4
    assert "3 restart(s)" in case_result.summary["reason"]
    assert (tmp_path / "case_001/case_summary.yaml").is_file()


def test_non_fall_infrastructure_failure_is_not_retried(tmp_path):
    case_result, calls = run_with_results(tmp_path, [
        result(1, AttemptStatus.INFRASTRUCTURE_FAILED, reason="checker status 20")
    ])
    assert calls == [1]
    assert case_result.infrastructure_failed
    assert not case_result.summary["restart_exhausted"]


def test_attempt_and_restart_status_are_printed(tmp_path, capsys):
    run_with_results(tmp_path, [
        result(1, AttemptStatus.FALLEN, reason="fallen"),
        result(2, AttemptStatus.COMPLETED, passed=True),
    ])
    output = capsys.readouterr().out
    assert "Attempt 1/4 starting" in output
    assert "fallen; restart budget 1/3; restarting in 0.0s" in output
    assert "Attempt 2/4 starting" in output


def test_disabled_attitude_reports_one_possible_attempt(tmp_path, capsys):
    config = execution_from_mapping(execution_mapping(enabled=False))

    def fake_attempt(*args, **kwargs):
        return result(1, AttemptStatus.COMPLETED, passed=True)

    run_case(
        smoke_case(), PACKAGE, tmp_path / "case_001", Path("model.pt"),
        {}, config, attempt_runner=fake_attempt,
    )
    assert "Attempt 1/1 starting" in capsys.readouterr().out


def test_batch_summary_separates_metrics_from_infrastructure():
    completed_fail = CaseResult(
        "metric-fail", "completed",
        {
            "pass": False,
            "recognition": {"pass": False},
            "localization": {"pass": True},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )
    infrastructure = CaseResult(
        "infra-fail", "infrastructure_failed",
        {
            "pass": False, "attempts_used": 4, "restarts_used": 3,
            "reason": "fallen",
        },
    )
    summary = _batch_summary("batch", [completed_fail, infrastructure])
    assert summary["metric_failed_count"] == 1
    assert summary["infrastructure_failed_count"] == 1
    assert summary["status"] == "infrastructure_failed"


def test_batch_metrics_ignore_provisional_overall_pass_flag():
    provisional = CaseResult(
        "smoke", "completed",
        {
            "pass": False,
            "provisional": True,
            "recognition": {"pass": True},
            "localization": {"pass": True},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )
    summary = _batch_summary("batch", [provisional])
    assert summary["metric_passed_count"] == 1
    assert summary["metric_failed_count"] == 0


def test_batch_summary_aggregates_completed_cases_with_equal_weight():
    metric_pass = CaseResult(
        "metric-pass", "completed",
        {
            "pass": True,
            "recognition": {"accuracy": 90.0, "pass": True},
            "localization": {"mean_relative_error": 10.0, "pass": True},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )
    metric_fail = CaseResult(
        "metric-fail", "completed",
        {
            "pass": False,
            "recognition": {"accuracy": 70.0, "pass": False},
            "localization": {"mean_relative_error": 20.0, "pass": False},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )

    summary = _batch_summary("batch", [metric_pass, metric_fail])

    assert summary["schema_version"] == 2
    assert summary["metric_passed_count"] == 1
    assert summary["metric_failed_count"] == 1
    aggregate = summary["aggregate_metrics"]
    assert aggregate["method"] == "case_mean"
    assert aggregate["eligible_case_count"] == 2
    assert aggregate["recognition"] == {
        "valid_case_count": 2,
        "excluded_case_count": 0,
        "accuracy_sum": 160.0,
        "mean_accuracy": 80.0,
    }
    assert aggregate["localization"] == {
        "valid_case_count": 2,
        "excluded_case_count": 0,
        "mean_relative_error_sum": 30.0,
        "mean_relative_error": 15.0,
    }


def test_batch_summary_aggregates_each_metric_independently():
    recognition_only = CaseResult(
        "recognition-only", "completed",
        {
            "pass": False,
            "recognition": {"accuracy": 75.0, "pass": False},
            "localization": {"mean_relative_error": None, "pass": False},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )
    localization_only = CaseResult(
        "localization-only", "completed",
        {
            "pass": False,
            "recognition": {"accuracy": float("nan"), "pass": False},
            "localization": {"mean_relative_error": 12.5, "pass": True},
            "attempts_used": 1,
            "restarts_used": 0,
        },
    )
    infrastructure = CaseResult(
        "infrastructure", "infrastructure_failed",
        {
            "pass": False,
            "recognition": {"accuracy": 100.0, "pass": True},
            "localization": {"mean_relative_error": 1.0, "pass": True},
            "attempts_used": 1,
            "restarts_used": 0,
            "reason": "failed",
        },
    )

    summary = _batch_summary(
        "batch", [recognition_only, localization_only, infrastructure]
    )

    aggregate = summary["aggregate_metrics"]
    assert aggregate["eligible_case_count"] == 2
    assert aggregate["recognition"] == {
        "valid_case_count": 1,
        "excluded_case_count": 1,
        "accuracy_sum": 75.0,
        "mean_accuracy": 75.0,
    }
    assert aggregate["localization"] == {
        "valid_case_count": 1,
        "excluded_case_count": 1,
        "mean_relative_error_sum": 12.5,
        "mean_relative_error": 12.5,
    }


def test_batch_summary_has_empty_aggregates_without_completed_metrics():
    infrastructure = CaseResult(
        "infrastructure", "infrastructure_failed",
        {
            "pass": False,
            "attempts_used": 1,
            "restarts_used": 0,
            "reason": "failed",
        },
    )

    summary = _batch_summary("batch", [infrastructure])

    assert summary["aggregate_metrics"] == {
        "method": "case_mean",
        "eligible_case_count": 0,
        "recognition": {
            "valid_case_count": 0,
            "excluded_case_count": 0,
            "accuracy_sum": 0.0,
            "mean_accuracy": None,
        },
        "localization": {
            "valid_case_count": 0,
            "excluded_case_count": 0,
            "mean_relative_error_sum": 0.0,
            "mean_relative_error": None,
        },
    }
