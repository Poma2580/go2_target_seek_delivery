"""YAML/CSV evaluation and case-percentile batch summaries."""

import argparse
import csv
import math
from pathlib import Path
import shutil

import yaml

from go2_static_perception_test.evaluators.metrics import localization, recognition


# Change this value to 90.0 (or another percentage) to adjust both aggregate metrics.
AGGREGATION_PERCENTILE_PERCENT = 90.0


def write_yaml(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def evaluate_csv(path, output_dir, case, infrastructure_valid=True, reasons=()):
    rows = read_rows(path)
    rec = recognition(rows, float(case["metrics"]["recognition_pass_threshold_percent"]))
    loc = localization(rows, float(case["metrics"]["localization_pass_threshold_percent"]))
    write_yaml(Path(output_dir) / "metrics/recognition_summary.yaml", rec)
    write_yaml(Path(output_dir) / "metrics/localization_summary.yaml", loc)
    summary = {"case_id": case["case_id"], "target_key": case["target_key"],
               "infrastructure_valid": bool(infrastructure_valid),
               "recognition_accuracy": rec["accuracy"], "recognition_pass": rec["pass"],
               "mean_relative_localization_error": loc["mean_relative_error"],
               "localization_pass": loc["pass"],
               "pass": bool(infrastructure_valid and rec["pass"] and loc["pass"]),
               "recognition": rec, "localization": loc}
    if not infrastructure_valid:
        summary["reason"] = "; ".join(reasons) or "test infrastructure data was incomplete"
    elif not summary["pass"]:
        summary["reason"] = "algorithm metrics did not meet thresholds"
    write_yaml(Path(output_dir) / "case_summary.yaml", summary)
    return summary


def promote_attempt(attempt_dir, case_dir):
    for name in ("raw", "metrics"):
        destination = Path(case_dir) / name
        if destination.exists(): shutil.rmtree(destination)
        shutil.copytree(Path(attempt_dir) / name, destination)


def nearest_rank_percentile(values, percentile, reverse=False):
    """Return the nearest-rank percentile value and its one-based rank."""
    if not 0.0 < percentile <= 100.0:
        raise ValueError("percentile must be greater than 0 and at most 100")
    if not values:
        return None, None
    ordered = sorted(values, reverse=reverse)
    rank = math.ceil(percentile / 100.0 * len(ordered))
    return ordered[rank - 1], rank


def _finite_values(results, key):
    values = []
    for item in results:
        value = item["summary"].get(key)
        if value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            values.append(value)
    return values


def _summary_count(item, group, key):
    return int((item["summary"].get(group) or {}).get(key, 0))


def _aggregate(label, results, percentile=AGGREGATION_PERCENTILE_PERCENT):
    eligible = [item for item in results if item["status"] == "completed"]
    recognition_values = _finite_values(eligible, "recognition_accuracy")
    localization_values = _finite_values(eligible, "mean_relative_localization_error")
    recognition_value, recognition_rank = nearest_rank_percentile(
        recognition_values, percentile, reverse=True)
    localization_value, localization_rank = nearest_rank_percentile(
        localization_values, percentile)
    return {"target_key": label, "total_cases": len(results),
            "completed_cases": len(eligible),
            "infrastructure_failed_cases": len(results) - len(eligible),
            "aggregation_percentile": float(percentile),
            "valid_recognition_cases": len(recognition_values),
            "recognition_rank": recognition_rank,
            "valid_visible_frames": sum(_summary_count(item, "recognition", "valid_frames")
                                        for item in eligible),
            "correct_detection_frames": sum(_summary_count(item, "recognition", "correct_frames")
                                            for item in eligible),
            "recognition_accuracy": recognition_value,
            "valid_localization_cases": len(localization_values),
            "localization_rank": localization_rank,
            "valid_localization_samples": sum(_summary_count(item, "localization", "valid_samples")
                                              for item in eligible),
            "mean_relative_localization_error": localization_value,
            "case_pass_count": sum(bool(item["summary"].get("pass")) for item in eligible)}


def build_aggregate_summaries(results, target_order, percentile=AGGREGATION_PERCENTILE_PERCENT):
    classes = [_aggregate(key, [item for item in results if item["target_key"] == key], percentile)
               for key in target_order if any(item["target_key"] == key for item in results)]
    return classes, _aggregate("overall", results, percentile)


def write_aggregate_summaries(results, output_dir, target_order):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    classes, overall = build_aggregate_summaries(results, target_order)
    write_yaml(output_dir / "class_summary.yaml", {"schema_version": 2, "classes": classes})
    write_yaml(output_dir / "overall_summary.yaml", {"schema_version": 2, **overall})
    fields = list(classes[0]) if classes else list(overall)
    with (output_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(classes)
    return classes, overall


def preview_existing_results(static_test_dir):
    """Build summaries from existing case YAML files without writing any files."""
    static_test_dir = Path(static_test_dir)
    paths = sorted(static_test_dir.glob("case_*/case_summary.yaml"))
    if not paths:
        raise ValueError(f"no case summaries found under: {static_test_dir}")

    results = []
    target_order = []
    for path in paths:
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or not summary.get("target_key"):
            raise ValueError(f"invalid case summary: {path}")
        target_key = summary["target_key"]
        if target_key not in target_order:
            target_order.append(target_key)
        status = summary.get("status")
        if status is None:
            status = "completed" if summary.get("infrastructure_valid") else "infrastructure_failed"
        results.append({"target_key": target_key, "status": status, "summary": summary})

    classes, overall = build_aggregate_summaries(results, target_order)
    return {"schema_version": 2, "classes": classes, "overall": overall}


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Preview case-percentile summaries without modifying an existing test batch.")
    parser.add_argument("static_test_dir", type=Path,
                        help="path containing case_*/case_summary.yaml")
    args = parser.parse_args(argv)
    try:
        preview = preview_existing_results(args.static_test_dir)
    except ValueError as error:
        parser.error(str(error))
    print(yaml.safe_dump(preview, sort_keys=False, allow_unicode=True), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
