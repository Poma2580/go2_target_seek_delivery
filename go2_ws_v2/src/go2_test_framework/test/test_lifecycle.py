"""Shutdown ordering, identity safety and bounded startup diagnostics."""

from dataclasses import replace
import os
from pathlib import Path
import signal
import time

import pytest

from go2_test_framework.runner import lifecycle
from go2_test_framework.runner.lifecycle import Identity, OwnedProcesses, BatchSafetyError
from go2_test_framework.runner.processes import ProcessGroupManager


def fake_record(pid=900001, **kwargs):
    return Identity(pid, 100, pid, "S", "fake", **kwargs)


def test_shutdown_waits_for_children_and_world_is_last(monkeypatch):
    worker = fake_record(owner="test")
    child = replace(worker, pid=900002, start_ticks=101)
    world = fake_record(900003, owner="test", phase="world")
    live = [worker, child, world]
    signals = []
    monkeypatch.setattr(lifecycle, "identities", lambda: (list(live), []))
    def send(record, sig):
        signals.append((record.pid, sig))
        if record is child and sig != signal.SIGTERM:
            return
        live.remove(record)
    owned = OwnedProcesses("test", [worker, world])
    monkeypatch.setattr(owned, "send", send)
    report = owned.shutdown((0.01, 0.01, 0.01))
    assert report["success"]
    assert signals == [(worker.pid, signal.SIGINT), (child.pid, signal.SIGINT),
                       (child.pid, signal.SIGTERM), (world.pid, signal.SIGINT)]


def test_escalation_and_remaining_block(monkeypatch):
    record = fake_record(owner="test")
    monkeypatch.setattr(lifecycle, "identities", lambda: ([record], []))
    owned = OwnedProcesses("test")
    seen = []
    monkeypatch.setattr(owned, "send", lambda r, sig: seen.append(sig))
    report = owned.shutdown((0, 0, 0))
    assert seen == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert not report["success"] and report["remaining"][0]["pid"] == record.pid


def test_escaped_marked_child_and_pid_reuse(monkeypatch):
    parent = fake_record(owner="test")
    escaped = replace(parent, pid=900004, group=900004, start_ticks=101)
    reused = replace(parent, owner="", start_ticks=200)
    unrelated = fake_record(900005)
    monkeypatch.setattr(lifecycle, "identities", lambda: ([escaped, reused, unrelated], []))
    owned = OwnedProcesses("test", [parent])
    assert owned.scan()[0] == [escaped]


def test_zombie_and_inaccessible_known_process(monkeypatch):
    record = fake_record(owner="test")
    owned = OwnedProcesses("test", [record])
    monkeypatch.setattr(lifecycle, "identities", lambda: ([replace(record, state="Z")], []))
    report = owned.shutdown((0, 0, 0))
    assert report["success"] and report["zombies"]
    monkeypatch.setattr(lifecycle, "identities", lambda: ([], [{"pid": record.pid, "error": "denied"}]))
    assert not owned.shutdown((0, 0, 0))["success"]


def test_pidfd_signal_rechecks_start_time(monkeypatch):
    original = fake_record()
    calls = []
    monkeypatch.setattr(os, "pidfd_open", lambda pid: 999)
    monkeypatch.setattr(lifecycle, "read_identity", lambda pid: replace(original, start_ticks=200))
    monkeypatch.setattr(signal, "pidfd_send_signal", lambda *a: calls.append(a))
    monkeypatch.setattr(os, "close", lambda fd: None)
    OwnedProcesses.send(original, signal.SIGINT)
    assert calls == []


def test_repeated_signal_is_deferred():
    received = []
    previous = signal.getsignal(signal.SIGINT)
    with lifecycle.defer_shutdown_signals(received):
        os.kill(os.getpid(), signal.SIGINT)
        os.kill(os.getpid(), signal.SIGTERM)
    assert received == ["SIGINT", "SIGTERM"]
    assert signal.getsignal(signal.SIGINT) == previous


def test_real_child_outlives_parent_and_cleanup_is_idempotent(tmp_path):
    manager = ProcessGroupManager(tmp_path / "logs", shutdown_timeout=0.3, term_timeout=0.2, kill_timeout=1)
    ready = tmp_path / "ready"
    script = """
import os, signal, time, sys
pid = os.fork()
if pid:
    sys.exit(0)
os.setsid()
signal.signal(signal.SIGINT, signal.SIG_IGN)
signal.signal(signal.SIGTERM, signal.SIG_IGN)
open(sys.argv[1], 'w').write(str(os.getpid()))
while True: time.sleep(1)
"""
    parent = manager.start("worker", ["/usr/bin/python3", "-c", script, str(ready)])
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        parent.wait(timeout=2)
        report = manager.stop()
        assert report["success"]
        assert {item["signal"] for item in report["signals"]} == {"SIGINT", "SIGTERM", "SIGKILL"}
        assert manager.stop() is report
        assert not manager.owned.scan()[0]
    finally:
        manager.stop()


def test_hung_probe_is_bounded_and_gate_blocks(tmp_path, monkeypatch):
    from go2_test_framework.runner import health
    class HungManager(ProcessGroupManager):
        def __init__(self, *a, **k):
            super().__init__(*a, **k, shutdown_timeout=.2, term_timeout=.2, kill_timeout=1)
        def start(self, name, command, **kwargs):
            return super().start(name, ["/usr/bin/python3", "-c", "import time; time.sleep(60)"])
    monkeypatch.setattr(health, "identities", lambda: ([], []))
    started = time.monotonic()
    with pytest.raises(BatchSafetyError) as error:
        health.startup_health(tmp_path, {}, timeout=.2, manager_factory=HungManager)
    assert error.value.reason == "startup_health_failed"
    assert time.monotonic() - started < 4
    import yaml
    report = yaml.safe_load((tmp_path / "startup_health.yaml").read_text())
    assert not report["success"] and report["cleanup"]["success"]
    assert not list(tmp_path.glob("*.rviz"))


def test_shm_diagnostics_ignore_unrelated_files():
    assert lifecycle.shm_diagnostics([]) == []


def test_shm_reports_only_referenced_data_and_never_mutates(tmp_path, monkeypatch):
    shm = tmp_path / "shm"
    proc = tmp_path / "proc"
    record = fake_record()
    fds = proc / str(record.pid) / "fd"
    fds.mkdir(parents=True)
    shm.mkdir()
    paths = [shm / name for name in ("fastrtps_port123", "fastrtps_port123_el", "fastrtps_port123_sl", "fastrtps_port456")]
    for path in paths:
        path.touch()
    for index, path in enumerate(paths[:3]):
        (fds / str(index)).symlink_to(path)
    (fds.parent / "maps").write_text("")
    before = {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths}
    monkeypatch.setattr(lifecycle, "read_identity", lambda *a: record)
    report = lifecycle.shm_diagnostics([record], proc, shm)
    assert len(report) == 1 and report[0]["path"] == str(paths[0]) and report[0]["size"] == 0
    assert before == {p.name: (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) for p in paths}


def test_real_normal_sigint_shutdown(tmp_path):
    manager = ProcessGroupManager(tmp_path / "logs", shutdown_timeout=2, term_timeout=.2, kill_timeout=1)
    ready = tmp_path / "ready"
    process = manager.start("worker", ["/usr/bin/python3", "-c", """
import signal, time, sys
signal.signal(signal.SIGINT, lambda *a: sys.exit(0))
open(sys.argv[1], 'w').write('ready')
while True: time.sleep(1)
""", str(ready)])
    try:
        deadline = time.monotonic() + 3
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        report = manager.stop()
        assert report["success"] and process.poll() == 0
        assert [s["signal"] for s in report["signals"]] == ["SIGINT"]
    finally:
        manager.stop()


def test_cleanup_failure_is_cached_and_reported(tmp_path, monkeypatch):
    manager = ProcessGroupManager(tmp_path / "logs")
    calls = []
    def shutdown(*a, **k):
        calls.append(1)
        return {"success": False, "remaining": [{"pid": 123}], "signals": []}
    monkeypatch.setattr(manager.owned, "shutdown", shutdown)
    for _ in range(2):
        with pytest.raises(BatchSafetyError) as error:
            manager.stop()
        assert error.value.reason == "cleanup_failed"
    assert calls == [1]
    assert (tmp_path / "cleanup_summary.yaml").exists()


def test_unreadable_group_child_blocks_confirmation(monkeypatch):
    parent = fake_record(owner="test")
    owned = OwnedProcesses("test", [parent])
    monkeypatch.setattr(lifecycle, "identities", lambda: (
        [], [{"pid": 900002, "group": parent.group, "error": "denied"}],
    ))
    assert not owned.shutdown((0, 0, 0))["success"]


def test_probe_cleanup_does_not_overwrite_batch_cleanup(tmp_path, monkeypatch):
    from go2_test_framework.runner import health
    existing = tmp_path / "cleanup_summary.yaml"
    existing.write_text("startup stale cleanup evidence\n")
    monkeypatch.setattr(health, "identities", lambda: ([], []))
    monkeypatch.setattr(health, "bounded_worker", lambda *a: None)
    health.startup_health(tmp_path, {})
    assert existing.read_text() == "startup stale cleanup evidence\n"
    assert (tmp_path / "health_logs/cleanup_summary.yaml").exists()


def test_transient_exit_permission_error_is_not_cleanup_failure(monkeypatch):
    record = fake_record(owner="test")
    owned = OwnedProcesses("test", [record])
    scans = iter([([record], []), ([], [{"pid": record.pid, "error": "exiting"}]), ([], []), ([], [])])
    monkeypatch.setattr(lifecycle, "identities", lambda: next(scans, ([], [])))
    monkeypatch.setattr(owned, "send", lambda *a: None)
    report = owned.shutdown((.1, .1, .1))
    assert report["success"]
    assert report["observed_errors"] and not report["errors"]
