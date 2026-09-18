#!/usr/bin/env python3
"""Run the isolated 100-case static perception suite."""

import argparse
from contextlib import contextmanager
from datetime import datetime
import fcntl
import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
import yaml

from go2_static_perception_test.common.config import (
    load_metric,
    load_poses,
    load_suite,
    load_targets,
)
from go2_static_perception_test.reporting.results import (
    write_aggregate_summaries,
    write_yaml,
)
from go2_static_perception_test.runner.cases import expand_cases
from go2_static_perception_test.runner.orchestration import run_case


RESULT_DIRECTORY = "static_target_test"


def package_share():
    return Path(get_package_share_directory("go2_static_perception_test"))


def discover_delivery_root(share):
    env_root = os.environ.get("DELIVERY_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not (root / "go2_ws_v2").is_dir():
            raise ValueError(
                f"DELIVERY_ROOT does not contain go2_ws_v2: {root}"
            )
        return root

    seen = set()
    for origin in (Path(__file__), share, Path.cwd()):
        start = Path(origin).resolve()
        if start.is_file():
            start = start.parent
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "go2_ws_v2").is_dir():
                return candidate

    raise ValueError(
        "cannot discover the delivery root; set DELIVERY_ROOT or pass "
        "--results-root"
    )


def resolve_results_root(value, share):
    if value is not None:
        return Path(value).resolve()
    return discover_delivery_root(share) / "TestResults"


def resolve_config(value, directory, share):
    path = Path(value)
    if path.is_file():
        return path.resolve()
    candidate = share / "config" / directory / path.name
    if candidate.is_file():
        return candidate
    raise ValueError(f"configuration does not exist: {value}")


@contextmanager
def batch_lock():
    path = Path("/tmp/go2_static_perception_test.lock")
    with path.open("w", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("another static_test_runner is active") from error
        yield


def parser(share):
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", default="static_100cases.yaml")
    value.add_argument(
        "--targets",
        type=Path,
        default=share / "config/targets/static_targets.yaml",
    )
    value.add_argument(
        "--poses",
        type=Path,
        default=share / "config/poses/static_robot_pose_cases.yaml",
    )
    value.add_argument(
        "--recognition-metrics",
        type=Path,
        default=share / "config/metrics/recognition.yaml",
    )
    value.add_argument(
        "--localization-metrics",
        type=Path,
        default=share / "config/metrics/localization.yaml",
    )
    selected = value.add_mutually_exclusive_group()
    selected.add_argument("--target")
    selected.add_argument("--case")
    value.add_argument("--model-path", default="yoloe-26s-seg.pt")
    value.add_argument("--device", default="cuda:0")
    value.add_argument(
        "--results-root",
        type=Path,
        help="result root (default: <DELIVERY_ROOT>/TestResults)",
    )
    value.add_argument("--dry-run", action="store_true")
    value.add_argument(
        "--gui",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override suite Gazebo GUI setting",
    )
    value.add_argument(
        "--rqt",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override suite static-perception rqt image view setting",
    )
    return value


def main(argv=None):
    share = package_share()
    args = parser(share).parse_args(argv)
    suite_path = resolve_config(args.suite, "suites", share)
    suite = load_suite(suite_path)
    targets = load_targets(args.targets)
    poses = load_poses(args.poses)
    metrics = {
        "recognition_pass_threshold_percent": load_metric(
            args.recognition_metrics, "target_recognition_accuracy"
        )["pass_threshold_percent"],
        "localization_pass_threshold_percent": load_metric(
            args.localization_metrics, "mean_relative_localization_error"
        )["pass_threshold_percent"],
    }

    all_cases = expand_cases(suite, targets, poses, metrics)
    cases = all_cases
    if args.target:
        if args.target not in suite["targets"]:
            raise ValueError(f"unknown target: {args.target}")
        cases = [case for case in all_cases if case.target_key == args.target]
    elif args.case:
        cases = [case for case in all_cases if case.case_id == args.case]
        if not cases:
            raise ValueError(f"unknown case: {args.case}")

    execution = dict(suite["execution"])
    if args.gui is not None:
        execution["gazebo_gui"] = bool(args.gui)
    if args.rqt is not None:
        execution["rqt"] = bool(args.rqt)

    if args.dry_run:
        print(
            yaml.safe_dump(
                {
                    "case_count": len(cases),
                    "execution": execution,
                    "cases": [case.to_dict() for case in cases],
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            end="",
        )
        return 0

    model = Path(args.model_path).expanduser()
    if model.parent != Path(".") and not model.is_file():
        raise ValueError(f"model path does not exist: {model}")

    batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")
    batch_root = resolve_results_root(args.results_root, share) / batch_id
    batch_dir = batch_root / RESULT_DIRECTORY
    batch_dir.mkdir(parents=True, exist_ok=False)
    write_yaml(
        batch_root / "resolved_cases.yaml",
        {
            "schema_version": 1,
            "batch_id": batch_id,
            "suite": str(suite_path),
            "model_path": args.model_path,
            "device": args.device,
            "execution": execution,
            "cases": [case.to_dict() for case in cases],
        },
    )

    results = []
    failure = None
    try:
        with batch_lock():
            for case in cases:
                print(
                    f"running {case.case_id} "
                    f"({case.case_index}/{len(all_cases)})",
                    flush=True,
                )
                result = run_case(
                    case,
                    share,
                    batch_dir / f"case_{case.case_index:03d}",
                    args.model_path,
                    args.device,
                    execution,
                    reporter=print,
                )
                results.append(result)
                print(
                    f"finished {case.case_id}: {result['status']}, "
                    f"attempts={result['summary']['attempts_used']}",
                    flush=True,
                )
    except KeyboardInterrupt:
        write_yaml(batch_dir / "batch_failure.yaml", {"status": "interrupted"})
        return 130
    except Exception as error:
        failure = str(error)
        write_yaml(
            batch_dir / "batch_failure.yaml",
            {"status": "infrastructure_failed", "reason": failure},
        )

    write_aggregate_summaries(results, batch_dir / "summary", suite["targets"])
    return 2 if failure else (
        1 if any(item["status"] != "completed" for item in results) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
