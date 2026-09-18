#!/usr/bin/env python3
"""Run isolated configuration-driven Go2 test cases."""

import argparse
from dataclasses import replace
from datetime import datetime
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory

from go2_test_framework.common.config import (
    load_pose_groups, load_routes, load_suite, read_yaml, require_resolved_pose,
)
from go2_test_framework.common.execution import (
    apply_execution_overrides, execution_from_mapping,
)
from go2_test_framework.reporting.results import write_yaml
from go2_test_framework.runner.cases import expand_cases
from go2_test_framework.runner.orchestration import run_case
from go2_test_framework.runner.lifecycle import BatchSafetyError
from go2_test_framework.runner.health import startup_health
from go2_test_framework.runner.runtime import (
    BatchLock,
    RunnerAlreadyActive,
    ShutdownRequested,
    cleanup_stale_test_processes,
    controlled_shutdown_signals,
)


TASK_RESULT_DIRECTORIES = {
    "perception": "T1_target_test",
    "tracking": "T2_tracking_test",
    "path_planning": "T3_path_planning_test",
}


def _package_share(package="go2_test_framework"):
    return Path(get_package_share_directory(package))


def _select_cases(cases, requested, run_all):
    if run_all and requested:
        raise ValueError("--all cannot be combined with --case-id")
    if run_all:
        return cases
    if not requested:
        return cases[:1]
    lookup = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested if case_id not in lookup]
    if missing:
        raise ValueError(f"unknown case IDs: {missing}")
    return [lookup[case_id] for case_id in requested]


def _resolved(case, groups, inline):
    all_groups = dict(groups[case.scene])
    all_groups.update(inline)
    poses = require_resolved_pose(case.pose_group, all_groups[case.pose_group])
    return replace(case, robot_poses=poses)


def _build_parser(share):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", type=Path,
        default=share / "config/suites/T1_smoke_city.yaml",
    )
    parser.add_argument(
        "--routes", type=Path,
        default=share / "config/parameters/target_routes.yaml",
    )
    parser.add_argument(
        "--pose-groups", type=Path,
        default=share / "config/parameters/robot_pose_groups.yaml",
    )
    parser.add_argument(
        "--recognition-metrics", type=Path,
        default=share / "config/metrics/recognition.yaml",
    )
    parser.add_argument(
        "--localization-metrics", type=Path,
        default=share / "config/metrics/localization.yaml",
    )
    parser.add_argument(
        "--tracking-metrics", type=Path,
        default=share / "config/metrics/tracking.yaml",
    )
    parser.add_argument(
        "--path-planning-metrics", type=Path,
        default=share / "config/metrics/path_planning.yaml",
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("yolov8s.pt"))
    parser.add_argument("--results-root", type=Path, default=Path("TestResults"))
    parser.add_argument(
        "--dry-run", action="store_true",
        help="validate and resolve without launching ROS",
    )
    parser.add_argument(
        "--gui", action=argparse.BooleanOptionalAction, default=None,
        help="override suite Gazebo GUI setting",
    )
    parser.add_argument(
        "--rqt", action=argparse.BooleanOptionalAction, default=None,
        help="override suite selected-robot rqt image view setting",
    )
    parser.add_argument(
        "--rviz", action=argparse.BooleanOptionalAction, default=None,
        help="override suite unified mapping/navigation RViz setting",
    )
    parser.add_argument(
        "--lidar", action=argparse.BooleanOptionalAction, default=None,
        help="override suite Go2 lidar setting",
    )
    parser.add_argument(
        "--check-attitude", action=argparse.BooleanOptionalAction, default=None,
        help="override suite Go2 attitude gate setting",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=None,
        help="override shared retries after falls or infrastructure failures",
    )
    return parser


def _batch_summary(
    batch_id, case_results, task_type="perception", metrics=None
):
    completed = [result for result in case_results if not result.infrastructure_failed]
    infrastructure_failed = [
        result for result in case_results if result.infrastructure_failed
    ]

    def metrics_passed(result):
        if task_type == "tracking":
            return bool(result.summary.get("tracking_success"))
        if task_type == "path_planning":
            return bool(
                result.summary.get("path_success")
                and result.summary.get("collision_count", 0) == 0
            )
        recognition = result.summary.get("recognition") or {}
        localization = result.summary.get("localization") or {}
        return bool(recognition.get("pass") and localization.get("pass"))

    metric_passed = [
        result for result in completed if metrics_passed(result)
    ]
    success_rate = (
        100.0 * len(metric_passed) / len(completed)
        if completed else None
    )
    default_thresholds = {
        "perception": 100.0, "tracking": 70.0, "path_planning": 90.0,
    }
    threshold = float(
        (metrics or {}).get(
            "batch_pass_threshold_percent", default_thresholds[task_type]
        )
    )
    batch_pass = bool(
        not infrastructure_failed and completed
        and success_rate >= threshold
    )

    def case_mean(section, field, sum_field, mean_field):
        values = []
        for result in completed:
            metric = result.summary.get(section) or {}
            value = metric.get(field)
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
            ):
                values.append(float(value))
        total = math.fsum(values)
        return {
            "valid_case_count": len(values),
            "excluded_case_count": len(completed) - len(values),
            sum_field: total,
            mean_field: total / len(values) if values else None,
        }

    summary = {
        "schema_version": 2,
        "batch_id": batch_id,
        "task_type": task_type,
        "status": (
            "infrastructure_failed" if infrastructure_failed else "completed"
        ),
        "completion_status": (
            "incomplete" if infrastructure_failed else "complete"
        ),
        "incomplete_reason": (
            "infrastructure_failed" if infrastructure_failed else None
        ),
        "case_count": len(case_results),
        "scheduled_case_count": len(case_results),
        "eligible_case_count": len(completed),
        "success_count": len(metric_passed),
        "valid_failure_count": len(completed) - len(metric_passed),
        "infrastructure_failure_count": len(infrastructure_failed),
        "success_rate_percent": success_rate,
        "batch_pass_threshold_percent": threshold,
        "batch_pass": batch_pass,
        "completed_count": len(completed),
        "metric_passed_count": len(metric_passed),
        "metric_failed_count": len(completed) - len(metric_passed),
        "infrastructure_failed_count": len(infrastructure_failed),
        "aggregate_metrics": {
            "method": "case_mean",
            "eligible_case_count": len(completed),
            "recognition": case_mean(
                "recognition", "accuracy", "accuracy_sum", "mean_accuracy"
            ),
            "localization": case_mean(
                "localization", "mean_relative_error",
                "mean_relative_error_sum", "mean_relative_error",
            ),
        },
        "cases": [
            {
                "case_id": result.case_id,
                "status": result.status,
                "pass": bool(result.summary.get("pass", False)),
                "attempts_used": result.summary["attempts_used"],
                "restarts_used": result.summary["restarts_used"],
                "reason": result.summary.get("reason"),
            }
            for result in case_results
        ],
    }
    if task_type == "tracking":
        summary["aggregate_metrics"] = {
            "eligible_case_count": len(completed),
            "tracking_success_count": len(metric_passed),
            "tracking_success_rate_percent": (
                100.0 * len(metric_passed) / len(completed)
                if completed else None
            ),
        }
    elif task_type == "path_planning":
        summary["aggregate_metrics"] = {
            "eligible_case_count": len(completed),
            "path_success_count": len(metric_passed),
            "path_success_rate_percent": success_rate,
            "collision_event_count": sum(
                int(result.summary.get("collision_count", 0))
                for result in completed
            ),
        }
    return summary


def main(argv=None):
    share = _package_share()
    scenario_share = _package_share("go2_scenario_config")
    args = _build_parser(share).parse_args(argv)
    suite = load_suite(args.suite)
    try:
        result_directory = TASK_RESULT_DIRECTORIES[suite["task_type"]]
    except KeyError as error:
        raise ValueError(
            f"no runner registered for task_type {suite['task_type']!r}"
        ) from error
    routes = load_routes(args.routes)
    pose_groups = load_pose_groups(args.pose_groups)
    if suite["task_type"] == "perception":
        recognition_metrics = read_yaml(args.recognition_metrics)
        localization_metrics = read_yaml(args.localization_metrics)
        metrics = {
            "recognition_pass_threshold_percent": float(
                recognition_metrics["pass_threshold_percent"]
            ),
            "localization_pass_threshold_percent": float(
                localization_metrics["pass_threshold_percent"]
            ),
        }
    elif suite["task_type"] == "tracking":
        tracking_metrics = read_yaml(args.tracking_metrics)
        metrics = {
            key: float(tracking_metrics[key])
            for key in (
                "tracking_radius_m", "tracking_min_duration_sec",
                "acquisition_timeout_sec", "batch_pass_threshold_percent",
            )
        }
    else:
        path_metrics = read_yaml(args.path_planning_metrics)
        metrics = {
            key: float(path_metrics[key])
            for key in (
                "endpoint_tolerance_m", "path_timeout_sec",
                "batch_pass_threshold_percent",
            )
        }
    execution = apply_execution_overrides(
        execution_from_mapping(suite["execution"]),
        gazebo_gui=args.gui,
        rqt=args.rqt,
        rviz=args.rviz,
        enable_lidar=args.lidar,
        attitude_enabled=args.check_attitude,
        max_restarts=args.max_restarts,
    )
    if suite["task_type"] == "path_planning":
        execution = replace(
            execution,
            robot_startup=replace(execution.robot_startup, enable_lidar=True),
        )
    cases = expand_cases(suite, routes, pose_groups)
    selected = [
        _resolved(case, pose_groups, suite["inline_pose_groups"])
        for case in _select_cases(cases, args.case_id, args.all)
    ]
    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_root = args.results_root.resolve() / batch_id
    batch_dir = batch_root / result_directory
    batch_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(batch_root / "resolved_cases.yaml", {
        "schema_version": 1,
        "batch_id": batch_id,
        "suite": str(args.suite.resolve()),
        "metrics": metrics,
        "execution": execution.to_dict(),
        "cases": [case.to_dict() for case in selected],
    })
    if args.dry_run:
        print(f"resolved {len(selected)} case(s) in {batch_root}")
        return 0
    if not args.model_path.is_file():
        raise ValueError(f"YOLO model does not exist: {args.model_path}")

    results = []
    started_case_ids = []
    abort_reason = None
    exit_status = None
    try:
        with BatchLock(), controlled_shutdown_signals():
            cleanup_stale_test_processes(
                reporter=lambda message: print(message, flush=True),
                output_dir=batch_dir,
            )
            startup_health(batch_dir, {"GO2_TEST_RUN_ID": batch_id})
            for case in selected:
                print(
                    f"running {case.case_id} "
                    f"({case.case_index}/{len(cases)})",
                    flush=True,
                )
                started_case_ids.append(case.case_id)
                result = run_case(
                    case, share, batch_dir / f"case_{case.case_index:03d}",
                    args.model_path.resolve(), metrics, execution,
                    run_id=batch_id,
                    scene_config_root=scenario_share / "config/scenes",
                )
                results.append(result)
                if result.summary.get("abort_batch"):
                    abort_reason = result.summary["abort_batch"]
                    exit_status = result.summary.get("exit_status")
                    print(f"Batch blocked: {abort_reason}; {result.summary.get('reason')}", flush=True)
                    break
                print(
                    f"finished {case.case_id}: {result.status}, "
                    f"attempts={result.summary['attempts_used']}",
                    flush=True,
                )
    except RunnerAlreadyActive as error:
        print(f"runner refused to start: {error}", flush=True)
        return 2
    except BatchSafetyError as error:
        abort_reason = error.reason
        print(f"Batch blocked: {error.reason}: {error}", flush=True)
    except ShutdownRequested as error:
        print(f"runner stopped: {error}; Attempt cleanup completed", flush=True)
        abort_reason = "interrupted"
        exit_status = 128 + error.signum
    except KeyboardInterrupt:
        print("runner interrupted; Attempt cleanup completed", flush=True)
        abort_reason = "interrupted"
        exit_status = 130

    summary = _batch_summary(
        batch_id, results, suite["task_type"], metrics
    )
    planned_ids = [case.case_id for case in selected]
    summary.update({
        "scheduled_case_count": len(planned_ids),
        "scheduled_case_ids": planned_ids,
        "started_case_count": len(started_case_ids),
        "started_case_ids": started_case_ids,
        "not_run_case_ids": [case_id for case_id in planned_ids if case_id not in started_case_ids],
        "not_run_case_count": len(planned_ids) - len(started_case_ids),
    })
    if abort_reason:
        summary.update(status="infrastructure_failed", completion_status="incomplete",
                       incomplete_reason=abort_reason, batch_pass=False)
    write_yaml(batch_dir / "batch_summary.yaml", summary)
    return exit_status or (1 if abort_reason or summary["infrastructure_failed_count"] else 0)


if __name__ == "__main__":
    raise SystemExit(main())
