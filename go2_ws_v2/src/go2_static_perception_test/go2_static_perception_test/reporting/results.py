"""YAML/CSV evaluation and weighted batch summaries."""

import csv
import json
import math
from pathlib import Path
import shutil

import yaml

from go2_static_perception_test.evaluators.metrics import localization, recognition, relative_errors


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


def _aggregate(label, results):
    eligible = [item for item in results if item["status"] == "completed"]
    rows = [row for item in eligible for row in read_rows(item["csv_path"])]
    visible = [row for row in rows if str(row.get("visible", "")).lower() == "true"]
    correct = sum(str(row.get("recognition_matched", "")).lower() == "true" and
                  str(row.get("recognition_success", "")).lower() == "true" for row in visible)
    try: errors = relative_errors(rows)
    except ZeroDivisionError: errors = []
    return {"target_key": label, "total_cases": len(results),
            "completed_cases": len(eligible),
            "infrastructure_failed_cases": len(results) - len(eligible),
            "valid_visible_frames": len(visible), "correct_detection_frames": correct,
            "recognition_accuracy": 100.0 * correct / len(visible) if visible else None,
            "valid_localization_samples": len(errors),
            "mean_relative_localization_error": 100.0 * math.fsum(errors) / len(errors) if errors else None,
            "case_pass_count": sum(bool(item["summary"].get("pass")) for item in eligible)}


def write_aggregate_summaries(results, output_dir, target_order):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    classes = [_aggregate(key, [item for item in results if item["target_key"] == key])
               for key in target_order if any(item["target_key"] == key for item in results)]
    overall = _aggregate("overall", results)
    write_yaml(output_dir / "class_summary.yaml", {"schema_version": 1, "classes": classes})
    write_yaml(output_dir / "overall_summary.yaml", {"schema_version": 1, **overall})
    fields = list(classes[0]) if classes else list(overall)
    with (output_dir / "class_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(classes)
    return classes, overall
