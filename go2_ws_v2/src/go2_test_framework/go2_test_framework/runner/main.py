#!/usr/bin/env python3
"""Run isolated T1 target-perception cases."""

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
from go2_test_framework.runner.runtime import (
    BatchLock,
    RunnerAlreadyActive,
    ShutdownRequested,
    cleanup_stale_test_processes,
    controlled_shutdown_signals,
)


def _package_share():
    return Path(get_package_share_directory("go2_test_framework"))


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
    all_groups = dict(groups)
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
        "--lidar", action=argparse.BooleanOptionalAction, default=None,
        help="override suite Go2 lidar setting",
    )
    parser.add_argument(
        "--check-attitude", action=argparse.BooleanOptionalAction, default=None,
        help="override suite Go2 attitude gate setting",
    )
    parser.add_argument(
        "--max-restarts", type=int, default=None,
        help="override retries allowed after confirmed falls",
    )
    return parser


def _batch_summary(batch_id, case_results):
    completed = [result for result in case_results if not result.infrastructure_failed]
    infrastructure_failed = [
        result for result in case_results if result.infrastructure_failed
    ]

    def metrics_passed(result):
        recognition = result.summary.get("recognition") or {}
        localization = result.summary.get("localization") or {}
        return bool(recognition.get("pass") and localization.get("pass"))

    metric_passed = [
        result for result in completed if metrics_passed(result)
    ]

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

    return {
        "schema_version": 2,
        "batch_id": batch_id,
        "status": (
            "infrastructure_failed" if infrastructure_failed else "completed"
        ),
        "case_count": len(case_results),
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


def main(argv=None):
    share = _package_share()
    args = _build_parser(share).parse_args(argv)
    suite = load_suite(args.suite)
    routes = load_routes(args.routes)
    pose_groups = load_pose_groups(args.pose_groups)
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
    execution = apply_execution_overrides(
        execution_from_mapping(suite["execution"]),
        gazebo_gui=args.gui,
        rqt=args.rqt,
        enable_lidar=args.lidar,
        attitude_enabled=args.check_attitude,
        max_restarts=args.max_restarts,
    )
    cases = expand_cases(suite, routes, pose_groups)
    selected = [
        _resolved(case, pose_groups, suite["inline_pose_groups"])
        for case in _select_cases(cases, args.case_id, args.all)
    ]
    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_root = args.results_root.resolve() / batch_id
    batch_dir = batch_root / "T1_target_test"
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
    try:
        with BatchLock(), controlled_shutdown_signals():
            cleanup_stale_test_processes(
                reporter=lambda message: print(message, flush=True)
            )
            for case in selected:
                print(
                    f"running {case.case_id} "
                    f"({case.case_index}/{len(cases)})",
                    flush=True,
                )
                result = run_case(
                    case, share, batch_dir / f"case_{case.case_index:03d}",
                    args.model_path.resolve(), metrics, execution,
                    run_id=batch_id,
                )
                results.append(result)
                print(
                    f"finished {case.case_id}: {result.status}, "
                    f"attempts={result.summary['attempts_used']}",
                    flush=True,
                )
    except RunnerAlreadyActive as error:
        print(f"runner refused to start: {error}", flush=True)
        return 2
    except ShutdownRequested as error:
        print(f"runner stopped: {error}; Attempt cleanup completed", flush=True)
        return 128 + error.signum
    except KeyboardInterrupt:
        print("runner interrupted; Attempt cleanup completed", flush=True)
        return 130

    summary = _batch_summary(batch_id, results)
    write_yaml(batch_dir / "batch_summary.yaml", summary)
    return 1 if summary["infrastructure_failed_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
