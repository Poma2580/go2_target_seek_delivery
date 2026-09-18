"""Owned process-group cleanup and tee tests."""

import io
import subprocess

import pytest

from go2_test_framework.runner.processes import ProcessGroupManager


class FakeProcess:
    next_pid = 100

    def __init__(self):
        self.pid = FakeProcess.next_pid
        FakeProcess.next_pid += 1
        self.running = True
        self.stdout = None

    def poll(self):
        return None if self.running else 0

    def wait(self, timeout=None):
        if self.running:
            raise subprocess.TimeoutExpired("fake", timeout)
        return 0



def test_environment_markers_are_added_to_children(monkeypatch, tmp_path):
    captured = {}

    def fake_popen(*args, **kwargs):
        captured.update(kwargs["env"])
        process = FakeProcess()
        process.running = False
        return process

    monkeypatch.setattr("go2_test_framework.runner.processes.subprocess.Popen", fake_popen)
    monkeypatch.setattr("go2_test_framework.runner.lifecycle.OwnedProcesses.add", lambda *a: None)
    manager = ProcessGroupManager(
        tmp_path / "logs", environment={"GO2_TEST_RUN_ID": "batch-1"}
    )
    manager.start("child", ["child"])
    manager.stop()
    assert captured["GO2_TEST_RUN_ID"] == "batch-1"


def test_console_tee_preserves_log_and_prefix(monkeypatch, tmp_path, capsys):
    class TeeProcess(FakeProcess):
        def __init__(self):
            super().__init__()
            self.stdout = io.StringIO("frame 1/5\nFALLEN: go2_1\n")
            self.running = False

    monkeypatch.setattr(
        "go2_test_framework.runner.processes.subprocess.Popen",
        lambda *args, **kwargs: TeeProcess(),
    )
    monkeypatch.setattr("go2_test_framework.runner.lifecycle.OwnedProcesses.add", lambda *a: None)
    manager = ProcessGroupManager(tmp_path / "logs")
    manager.start(
        "attitude_check", ["checker"], mirror_to_console=True,
        console_prefix="[Attempt 1/4] ",
    )
    manager.stop()
    assert (tmp_path / "logs/attitude_check.log").read_text() == (
        "frame 1/5\nFALLEN: go2_1\n"
    )
    output = capsys.readouterr().out
    assert "[Attempt 1/4] frame 1/5" in output
    assert "[Attempt 1/4] FALLEN: go2_1" in output


def test_world_exit_255_is_rejected_before_stale_graph_can_be_used(
    monkeypatch, tmp_path,
):
    process = FakeProcess()
    process.running = False
    process.poll = lambda: 255
    monkeypatch.setattr(
        "go2_test_framework.runner.processes._group_command_pids",
        lambda *_: [],
    )
    manager = ProcessGroupManager(tmp_path / "logs")
    with pytest.raises(RuntimeError, match="status 255"):
        manager.wait_for_group_command(process, "gzserver", timeout=0.1)
