"""Exercise map-mode selection without starting Gazebo or Nav2."""

import os
from pathlib import Path
import subprocess

import pytest


PACKAGE_ROOT = Path(__file__).parents[1]
DELIVERY_ROOT = PACKAGE_ROOT.parents[2]
GOAL_SCRIPT = DELIVERY_ROOT / "Scripts" / "send_go2_1_static_goal.sh"


@pytest.fixture
def fake_ros2(tmp_path):
    executable = tmp_path / "ros2"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
if [[ $1 == param && $2 == get ]]; then
    if [[ ${FAKE_PARAM_FAIL:-false} == true ]]; then
        echo "node not found" >&2
        exit 1
    fi
    echo "String value is: ${FAKE_GLOBAL_FRAME:-go2_1/map}"
    exit 0
fi
if [[ $1 == run ]]; then
    printf '%s\\n' "$*"
    exit 0
fi
echo "unexpected ros2 invocation: $*" >&2
exit 2
""",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable


def run_goal(fake_ros2, *arguments, **environment):
    env = os.environ.copy()
    env.update(environment)
    env["ROS2_BIN"] = str(fake_ros2)
    return subprocess.run(
        [str(GOAL_SCRIPT), *arguments],
        cwd=DELIVERY_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_auto_selects_robot_local_map(fake_ros2):
    result = run_goal(
        fake_ros2,
        "--robot",
        "go2_2",
        "1",
        "2",
        FAKE_GLOBAL_FRAME="go2_2/map",
    )
    assert result.returncode == 0, result.stderr
    assert "map_topic:=/go2_2/map" in result.stdout
    assert "global_frame:=go2_2/map" in result.stdout


def test_auto_selects_merged_map(fake_ros2):
    result = run_goal(
        fake_ros2,
        "--robot",
        "go2_3",
        "1",
        "2",
        FAKE_GLOBAL_FRAME="merged_map",
    )
    assert result.returncode == 0, result.stderr
    assert "map_topic:=/merged_map" in result.stdout
    assert "global_frame:=merged_map" in result.stdout


def test_auto_rejects_unknown_frame(fake_ros2):
    result = run_goal(
        fake_ros2,
        "1",
        "2",
        FAKE_GLOBAL_FRAME="world",
    )
    assert result.returncode != 0
    assert "unsupported global_frame 'world'" in result.stderr


def test_auto_reports_missing_nav2_node(fake_ros2):
    result = run_goal(fake_ros2, "1", "2", FAKE_PARAM_FAIL="true")
    assert result.returncode != 0
    assert "cannot read global_frame" in result.stderr


@pytest.mark.parametrize(
    ("mode", "expected_frame", "expected_topic"),
    [
        ("local", "go2_1/map", "/go2_1/map"),
        ("merged", "merged_map", "/merged_map"),
    ],
)
def test_explicit_mode_skips_auto_detection(
    fake_ros2, mode, expected_frame, expected_topic
):
    result = run_goal(
        fake_ros2,
        "--map-mode",
        mode,
        "1",
        "2",
        FAKE_PARAM_FAIL="true",
    )
    assert result.returncode == 0, result.stderr
    assert f"global_frame:={expected_frame}" in result.stdout
    assert f"map_topic:={expected_topic}" in result.stdout
