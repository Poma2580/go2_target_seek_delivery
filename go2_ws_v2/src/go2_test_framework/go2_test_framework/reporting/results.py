"""CSV evaluation and YAML result writing."""

import csv
from pathlib import Path

import yaml

from go2_test_framework.evaluators.localization import evaluate_localization
from go2_test_framework.evaluators.recognition import evaluate_recognition


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def evaluate_csv(csv_path, metrics_dir, infrastructure_valid=True, provisional=False,
                 recognition_threshold=80.0, localization_threshold=15.0):
    with Path(csv_path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    recognition = evaluate_recognition(rows, recognition_threshold)
    localization = evaluate_localization(rows, localization_threshold)
    metrics_dir = Path(metrics_dir)
    write_yaml(metrics_dir / "recognition_summary.yaml", recognition)
    write_yaml(metrics_dir / "localization_summary.yaml", localization)
    summary = {
        "infrastructure_valid": bool(infrastructure_valid),
        "provisional": bool(provisional),
        "recognition": recognition,
        "localization": localization,
        "pass": bool(
            infrastructure_valid and not provisional
            and recognition["pass"] and localization["pass"]
        ),
    }
    if provisional:
        summary["reason"] = "provisional configuration is not an official result"
    elif not infrastructure_valid:
        summary["reason"] = "test infrastructure data was incomplete"
    return summary
