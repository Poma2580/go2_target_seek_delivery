"""Batch runtime guard tests."""

import os

import pytest

from go2_test_framework.runner.runtime import (
    BatchLock, ProcessInfo, RunnerAlreadyActive, cleanup_stale_test_processes,
    stale_test_processes,
)


def record(pid, group, command, environment=None):
    return ProcessInfo(pid, group, command, environment or {})


def test_stale_match_is_marked_or_narrow_legacy_only():
    records = [
        record(10, 10, "anything", {"GO2_TEST_RUN_ID": "old"}),
        record(
            20, 20,
            "ros2 launch go2_config gazebo_target_seek_world.launch.py "
            "world:=/ws/install/go2_test_framework/share/"
            "go2_test_framework/worlds/city_rectangle.world",
        ),
        record(
            30, 30,
            "ros2 launch go2_config spawn_go2_velodyne_2.launch.py scene:=city",
        ),
        record(40, 40, "gzserver /some/other/project.world"),
        record(50, 99, "anything", {"GO2_TEST_RUN_ID": "current-shell"}),
    ]
    matches = stale_test_processes(records, current_process_group=99)
    assert [item.pid for item in matches] == [10, 20, 30]


def test_batch_lock_refuses_a_second_runner(tmp_path):
    path = tmp_path / "runner.lock"
    with BatchLock(path):
        with pytest.raises(RunnerAlreadyActive):
            with BatchLock(path):
                pass
    with BatchLock(path):
        assert f"pid={os.getpid()}" in path.read_text()


def test_cleanup_uses_term_then_reports_completion(monkeypatch):
    stale = record(10, 20, "marked", {"GO2_TEST_RUN_ID": "old"})
    signals = []
    reports = []
    alive_checks = iter([True, False])
    monkeypatch.setattr(
        "go2_test_framework.runner.runtime.scan_processes", lambda: [stale]
    )
    monkeypatch.setattr(
        "go2_test_framework.runner.runtime.os.killpg",
        lambda group, signum: signals.append((group, signum)),
    )
    monkeypatch.setattr(
        "go2_test_framework.runner.runtime._group_exists",
        lambda _group: next(alive_checks),
    )
    monkeypatch.setattr("go2_test_framework.runner.runtime.time.sleep", lambda _: None)
    cleanup_stale_test_processes(timeout=1.0, reporter=reports.append)
    assert signals[0][0] == 20
    assert signals[0][1].name == "SIGTERM"
    assert all(item[1].name != "SIGKILL" for item in signals)
    assert any("PID=10 PGID=20" in message for message in reports)


def test_cleanup_escalates_only_a_still_live_group(monkeypatch):
    stale = record(10, 20, "marked", {"GO2_TEST_RUN_ID": "old"})
    signals = []
    monkeypatch.setattr(
        "go2_test_framework.runner.runtime.scan_processes", lambda: [stale]
    )
    monkeypatch.setattr(
        "go2_test_framework.runner.runtime.os.killpg",
        lambda group, signum: signals.append((group, signum)),
    )
    cleanup_stale_test_processes(timeout=0.0, reporter=lambda _: None)
    assert [item[1].name for item in signals] == ["SIGTERM", "SIGKILL"]
