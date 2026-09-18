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


def test_stale_cleanup_uses_shared_shutdown(monkeypatch, tmp_path):
    from go2_test_framework.runner import lifecycle
    stale = lifecycle.Identity(123456, 10, 123456, "S", "test", run="old")
    monkeypatch.setattr(lifecycle, "identities", lambda: ([stale], []))
    monkeypatch.setattr("go2_test_framework.runner.runtime.scan_processes", lambda: [])
    captured = []
    def shutdown(self, timeouts):
        captured.append((list(self.known), timeouts))
        return {"success": True}
    monkeypatch.setattr(lifecycle.OwnedProcesses, "shutdown", shutdown)
    assert cleanup_stale_test_processes(output_dir=tmp_path) == [123456]
    assert captured == [([(123456, 10)], (15.0, 5.0, 3.0))]
    assert (tmp_path / "cleanup_summary.yaml").exists()


def test_stale_cleanup_failure_blocks(monkeypatch, tmp_path):
    from go2_test_framework.runner import lifecycle
    monkeypatch.setattr(lifecycle, "identities", lambda: ([], []))
    monkeypatch.setattr("go2_test_framework.runner.runtime.scan_processes", lambda: [])
    monkeypatch.setattr(lifecycle.OwnedProcesses, "shutdown", lambda *a: {"success": False})
    with pytest.raises(lifecycle.BatchSafetyError, match="could not confirm"):
        cleanup_stale_test_processes(output_dir=tmp_path)
