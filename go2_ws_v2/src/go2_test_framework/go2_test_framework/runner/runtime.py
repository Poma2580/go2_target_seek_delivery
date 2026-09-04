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


def cleanup_stale_test_processes(timeout=5.0, *, reporter=print):
    matches = stale_test_processes(scan_processes())
    if not matches:
        reporter("[startup] no stale test-framework process groups found")
        return []
    groups = sorted({record.process_group for record in matches})
    for record in matches:
        reporter(
            f"[startup] stale PID={record.pid} PGID={record.process_group}: "
            f"{record.command[:180]}"
        )
    for process_group in groups:
        try:
            os.killpg(process_group, signal.SIGTERM)
            reporter(f"[startup] SIGTERM sent to PGID={process_group}")
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    remaining = groups
    while remaining and time.monotonic() < deadline:
        remaining = [group for group in remaining if _group_exists(group)]
        if remaining:
            time.sleep(0.1)
    for process_group in remaining:
        try:
            os.killpg(process_group, signal.SIGKILL)
            reporter(f"[startup] SIGKILL sent to PGID={process_group}")
        except ProcessLookupError:
            pass
    reporter(f"[startup] stale cleanup complete; groups={len(groups)}")
    return groups


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
