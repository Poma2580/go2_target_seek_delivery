import csv
from pathlib import Path

import pytest
import yaml

from go2_static_perception_test.common.config import load_metric, load_poses, load_suite, load_targets
from go2_static_perception_test.evaluators.metrics import localization, recognition
from go2_static_perception_test.reporting.results import write_aggregate_summaries, write_yaml
from go2_static_perception_test.runner.cases import expand_cases
from go2_static_perception_test.runner.orchestration import AttemptResult, run_case
from go2_static_perception_test.recorders.cache import TimeCache


PACKAGE = Path(__file__).parents[1]
CONFIG = PACKAGE / "config"


def cases():
    suite = load_suite(CONFIG / "suites/static_100cases.yaml")
    targets = load_targets(CONFIG / "targets/static_targets.yaml")
    poses = load_poses(CONFIG / "poses/static_robot_pose_cases.yaml")
    metrics = {"recognition_pass_threshold_percent": load_metric(
        CONFIG / "metrics/recognition.yaml", "target_recognition_accuracy")["pass_threshold_percent"],
        "localization_pass_threshold_percent": load_metric(
        CONFIG / "metrics/localization.yaml", "mean_relative_localization_error")["pass_threshold_percent"]}
    return suite, targets, poses, expand_cases(suite, targets, poses, metrics)


def runner_module(monkeypatch):
    try:
        from go2_static_perception_test.runner import main as module
    except ModuleNotFoundError as error:
        if error.name != "ament_index_python":
            raise
        import sys
        import types
        package = types.ModuleType("ament_index_python")
        packages = types.ModuleType("ament_index_python.packages")
        packages.get_package_share_directory = lambda _name: str(PACKAGE)
        monkeypatch.setitem(sys.modules, "ament_index_python", package)
        monkeypatch.setitem(sys.modules, "ament_index_python.packages", packages)
        from go2_static_perception_test.runner import main as module
    return module


def test_expands_exactly_100_source_backed_cases():
    suite, targets, poses, expanded = cases()
    assert len(expanded) == len({case.case_id for case in expanded}) == 100
    assert expanded[0].case_id == "SP-AIRPLANE-P01"
    assert expanded[-1].case_id == "SP-DUMPSTER-P20"
    for target in suite["targets"]:
        selected = [case for case in expanded if case.target_key == target]
        assert len(selected) == 20
        assert all(case.prompt == targets[target]["prompt"] for case in selected)
        assert [case.robot_pose for case in selected] == list(poses["targets"][target].values())


@pytest.mark.parametrize("mutation", ["missing_pose", "nan_pose", "wrong_order"])
def test_rejects_invalid_pose_configuration(tmp_path, mutation):
    value = yaml.safe_load((CONFIG / "poses/static_robot_pose_cases.yaml").read_text())
    if mutation == "missing_pose": value["targets"]["person"].pop("pose_20")
    elif mutation == "nan_pose": value["targets"]["person"]["pose_20"]["x"] = float("nan")
    else: value["targets"] = dict(reversed(value["targets"].items()))
    path = tmp_path / "poses.yaml"; path.write_text(yaml.safe_dump(value, sort_keys=False))
    with pytest.raises(ValueError): load_poses(path)


def sample(**changes):
    row = {"visible": True, "recognition_matched": True, "recognition_success": True,
           "localization_matched": True, "localization_success": True,
           "target_gt_x": 10, "target_gt_y": 0, "target_est_x": 11,
           "target_est_y": 0, "robot_gt_x": 0, "robot_gt_y": 0}
    row.update(changes); return row


def test_metrics_match_t1_formulas_and_empty_failures():
    rows = [sample(), sample(recognition_success=False, localization_success=False,
                             localization_matched=False)]
    assert recognition(rows)["accuracy"] == 50.0
    assert localization(rows)["mean_relative_error"] == pytest.approx(10.0)
    assert recognition([]) == {"valid_frames": 0, "correct_frames": 0, "accuracy": None,
                               "pass": False, "reason": "no visible evaluation frames"}
    assert localization([])["reason"] == "no valid localization samples"
    zero = localization([sample(target_gt_x=0)])
    assert not zero["pass"] and zero["reason"] == "target-to-robot reference distance is zero"


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def test_aggregate_is_frame_sample_weighted_and_excludes_infrastructure(tmp_path):
    a = tmp_path / "a.csv"; b = tmp_path / "b.csv"; failed = tmp_path / "failed.csv"
    write_rows(a, [sample(), sample()])
    write_rows(b, [sample(recognition_success=False, target_est_x=12)])
    write_rows(failed, [sample()] * 20)
    results = [
        {"target_key": "person", "status": "completed", "summary": {"pass": True}, "csv_path": a},
        {"target_key": "person", "status": "completed", "summary": {"pass": False}, "csv_path": b},
        {"target_key": "person", "status": "infrastructure_failed", "summary": {"pass": False}, "csv_path": failed},
    ]
    classes, overall = write_aggregate_summaries(results, tmp_path / "summary", ["person"])
    assert classes[0]["recognition_accuracy"] == pytest.approx(200 / 3)
    assert classes[0]["mean_relative_localization_error"] == pytest.approx(40 / 3)
    assert classes[0]["infrastructure_failed_cases"] == 1
    assert overall["valid_visible_frames"] == 3


def test_algorithm_failure_does_not_retry_and_infrastructure_does(tmp_path):
    suite, _, _, expanded = cases(); case = expanded[20]
    calls = []
    def algorithm(*args, **kwargs):
        calls.append(1); attempt = Path(args[2]); write_rows(attempt / "raw/static_samples.csv", [sample()])
        write_yaml(attempt / "metrics/recognition_summary.yaml", {})
        write_yaml(attempt / "metrics/localization_summary.yaml", {})
        summary = {"status": "completed", "infrastructure_valid": True, "pass": False}
        return AttemptResult(args[-1], "completed", None, summary)
    result = run_case(case, PACKAGE, tmp_path / "algorithm", "model.pt", "cpu",
                      suite["execution"], attempt_runner=algorithm)
    assert len(calls) == 1 and result["status"] == "completed"

    calls.clear()
    def transient(*args, **kwargs):
        calls.append(1); number = args[-1]
        if number == 1:
            return AttemptResult(number, "infrastructure_failed", "camera missing", {})
        attempt = Path(args[2]); write_rows(attempt / "raw/static_samples.csv", [sample()])
        write_yaml(attempt / "metrics/recognition_summary.yaml", {})
        write_yaml(attempt / "metrics/localization_summary.yaml", {})
        return AttemptResult(number, "completed", None,
                             {"status": "completed", "infrastructure_valid": True, "pass": True})
    result = run_case(case, PACKAGE, tmp_path / "retry", "model.pt", "cpu",
                      suite["execution"], attempt_runner=transient, sleep=lambda _: None)
    assert len(calls) == 2 and result["status"] == "completed"

    failed_execution = dict(suite["execution"]); failed_execution["max_restarts"] = 0
    result = run_case(case, PACKAGE, tmp_path / "exhausted", "model.pt", "cpu",
                      failed_execution,
                      attempt_runner=lambda *args, **kwargs: AttemptResult(
                          args[-1], "infrastructure_failed", "world missing", {}),
                      sleep=lambda _: None)
    assert result["status"] == "infrastructure_failed"
    final_attempt = tmp_path / "exhausted/attempts/attempt_01"
    assert (final_attempt / "raw/static_samples.csv").is_file()
    assert (final_attempt / "metrics/recognition_summary.yaml").is_file()
    assert not (tmp_path / "exhausted/raw").exists()
    assert not (tmp_path / "exhausted/metrics").exists()



def test_time_cache_matches_t1_consumable_semantics():
    cache = TimeCache(consumable=True)
    cache.append(1.00, "first")
    cache.append(1.20, "second")
    assert cache.nearest(1.05, 0.4) == (1.0, "first")
    assert cache.nearest(1.05, 0.4) == (1.2, "second")
    assert cache.nearest(1.05, 0.4) is None
    assert cache.latest() == (1.2, "second")


def test_suite_execution_has_t1_style_gui_and_rqt_switches():
    suite, _, _, _ = cases()
    assert suite["execution"]["gazebo_gui"] is True
    assert suite["execution"]["rqt"] is True


def test_default_results_root_uses_delivery_root_environment(monkeypatch, tmp_path):
    module = runner_module(monkeypatch)
    delivery_root = tmp_path / "delivery"
    (delivery_root / "go2_ws_v2").mkdir(parents=True)
    monkeypatch.setenv("DELIVERY_ROOT", str(delivery_root))
    assert module.resolve_results_root(None, PACKAGE) == delivery_root / "TestResults"


@pytest.mark.parametrize("cwd", [PACKAGE.parents[2], PACKAGE.parents[1]])
def test_default_results_root_is_independent_of_cwd(monkeypatch, cwd):
    module = runner_module(monkeypatch)
    monkeypatch.delenv("DELIVERY_ROOT", raising=False)
    monkeypatch.chdir(cwd)
    assert module.resolve_results_root(None, PACKAGE) == PACKAGE.parents[2] / "TestResults"


def test_explicit_results_root_remains_relative_to_cwd(monkeypatch, tmp_path):
    module = runner_module(monkeypatch)
    monkeypatch.setenv("DELIVERY_ROOT", "/does/not/exist")
    monkeypatch.chdir(tmp_path)
    assert module.resolve_results_root(Path("custom"), PACKAGE) == tmp_path / "custom"


def test_missing_delivery_root_has_actionable_error(monkeypatch, tmp_path):
    module = runner_module(monkeypatch)
    isolated = tmp_path / "isolated"
    isolated.mkdir()
    monkeypatch.delenv("DELIVERY_ROOT", raising=False)
    monkeypatch.setattr(module, "__file__", str(isolated / "main.py"))
    monkeypatch.chdir(isolated)
    with pytest.raises(ValueError, match="set DELIVERY_ROOT or pass --results-root"):
        module.resolve_results_root(None, isolated / "share")


def test_dry_run_does_not_create_results_or_launch(monkeypatch, tmp_path, capsys):
    module = runner_module(monkeypatch)
    monkeypatch.setattr(module, "package_share", lambda: PACKAGE)
    monkeypatch.chdir(tmp_path)
    assert module.main(["--dry-run"]) == 0
    assert not (tmp_path / "TestResults").exists()
    assert "case_count: 100" in capsys.readouterr().out
