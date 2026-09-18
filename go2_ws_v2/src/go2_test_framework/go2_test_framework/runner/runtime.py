"""Batch mutual exclusion, stale-process recovery, and shutdown signals."""

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import os
from pathlib import Path
import signal
import time


RUN_MARKER = "GO2_TEST_RUN_ID"
CASE_MARKER = "GO2_TEST_CASE_ID"
ATTEMPT_MARKER = "GO2_TEST_ATTEMPT"
DEFAULT_LOCK_PATH = Path("/tmp/go2_test_framework.lock")


class RunnerAlreadyActive(RuntimeError):
    pass


class ShutdownRequested(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"received signal {signal.Signals(signum).name}")


class BatchLock:
    def __init__(self, path=DEFAULT_LOCK_PATH):
        self.path = Path(path)
        self._file = None

    def __enter__(self):
        self._file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._file.close()
            self._file = None
            raise RunnerAlreadyActive(
                "another target_test_runner is already using Gazebo"
            ) from error
        self._file.seek(0)
        self._file.truncate()
        self._file.write(f"pid={os.getpid()}\n")
        self._file.flush()
        return self

    def __exit__(self, *_):
        if self._file is not None:
            fcntl.flock(self._file.fileno(), fcntl.LOCK_UN)
            self._file.close()


@dataclass(frozen=True)
class ProcessInfo:
    pid: int
    process_group: int
    command: str
    environment: dict


def scan_processes(proc_root=Path("/proc")):
    records = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            ).strip()
            environment = {}
            for item in (entry / "environ").read_bytes().split(b"\0"):
                key, separator, value = item.partition(b"=")
                if separator:
                    environment[key.decode(errors="replace")] = value.decode(
                        errors="replace"
                    )
            process_group = os.getpgid(pid)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        records.append(ProcessInfo(pid, process_group, command, environment))
    return records


def _is_legacy_test_process(record):
    command = record.command
    if (
        "gazebo_target_seek_world.launch.py" in command
        and "go2_test_framework" in command
        and "/worlds/" in command
    ):
        return True
    return any(
        f"spawn_go2_velodyne_{number}.launch.py" in command
        for number in (1, 2, 3)
    ) and "ros2 launch go2_config" in command


def stale_test_processes(records, current_process_group=None):
    current_process_group = (
        os.getpgrp() if current_process_group is None else current_process_group
    )
    return [
        record for record in records
        if record.process_group != current_process_group
        and (record.environment.get(RUN_MARKER) or _is_legacy_test_process(record))
    ]


def _group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def cleanup_stale_test_processes(timeout=15.0, *, reporter=print, output_dir=None):
    from dataclasses import replace
    from go2_test_framework.reporting.results import write_yaml
    from go2_test_framework.runner.lifecycle import BatchSafetyError, OwnedProcesses, identities

    records, errors = identities()
    marked = [r for r in records if r.run and r.pid != os.getpid() and r.state != "Z"]
    # Unmarked historical commands are diagnostic evidence, not ownership proof.
    legacy = [r for r in scan_processes() if _is_legacy_test_process(r) and not r.environment.get(RUN_MARKER)]
    world_groups = {r.group for r in marked if r.phase == "world" or "gzserver" in r.command
                    or "gazebo_target_seek_world.launch.py" in r.command}
    seeds = [replace(r, phase="world" if r.group in world_groups else "workers") for r in marked]
    for record in seeds:
        reporter(f"[startup] stale PID={record.pid} PGID={record.group}: {record.command[:180]}")
    owned = OwnedProcesses(seeds=seeds)
    report = owned.shutdown((timeout, 5.0, 3.0))
    if legacy:
        report["success"] = False
        report["unowned_legacy"] = [{"pid": r.pid, "command": r.command} for r in legacy]
    if output_dir is not None:
        write_yaml(Path(output_dir) / "cleanup_summary.yaml", report)
    if not report["success"]:
        raise BatchSafetyError("cleanup_failed", "startup cleanup could not confirm a clean environment")
    if report.get("received_signals"):
        raise ShutdownRequested(getattr(signal, report["received_signals"][0]))
    reporter(f"[startup] stale cleanup complete; processes={len(seeds)}")
    return sorted({r.group for r in seeds})


@contextmanager
def controlled_shutdown_signals():
    previous = {}

    def request_shutdown(signum, _frame):
        raise ShutdownRequested(signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, request_shutdown)
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
