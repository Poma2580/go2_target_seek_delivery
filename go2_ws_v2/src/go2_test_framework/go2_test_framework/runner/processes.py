"""Owned process-group lifecycle and selective console log mirroring."""

from dataclasses import dataclass, field
import os
from pathlib import Path
import signal
import subprocess
import threading
import time


def _group_command_pids(process_group, command_fragment, proc_root=Path("/proc")):
    """Return PIDs in an owned group whose command contains a fragment."""
    matches = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            if os.getpgid(pid) != process_group:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if command_fragment in command:
            matches.append(pid)
    return matches


def _process_group_exists(process_group):
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class _ManagedProcess:
    name: str
    process: object
    log: object
    tee_thread: threading.Thread | None = None
    mirrored_lines: list[str] = field(default_factory=list)


class ProcessGroupManager:
    """Start commands in isolated sessions and clean up only owned groups."""

    def __init__(self, log_dir, shutdown_timeout=8.0, environment=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.shutdown_timeout = shutdown_timeout
        self.environment = dict(environment or {})
        self.processes = []

    def start(self, name, command, *, mirror_to_console=False, console_prefix=""):
        log = (self.log_dir / f"{name}.log").open("w", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(self.environment)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE if mirror_to_console else log,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=environment,
                start_new_session=True,
            )
        except Exception:
            log.close()
            raise
        managed = _ManagedProcess(name, process, log)
        if mirror_to_console:
            managed.tee_thread = threading.Thread(
                target=self._tee_output,
                args=(managed, console_prefix),
                name=f"go2-test-tee-{name}",
                daemon=True,
            )
            managed.tee_thread.start()
        self.processes.append(managed)
        return process

    @staticmethod
    def _tee_output(managed, prefix):
        stream = managed.process.stdout
        if stream is None:
            return
        for line in stream:
            managed.mirrored_lines.append(line.rstrip("\n"))
            managed.log.write(line)
            managed.log.flush()
            print(f"{prefix}{line.rstrip()}", flush=True)

    def wait_for_group_command(self, process, command_fragment, timeout=15.0):
        """Wait for a real command in this newly-created process group."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            pids = _group_command_pids(process.pid, command_fragment)
            if pids:
                return pids
            status = process.poll()
            if status is not None:
                raise RuntimeError(
                    f"world launch exited with status {status} before "
                    f"starting {command_fragment}"
                )
            time.sleep(0.1)
        raise RuntimeError(f"timed out waiting for owned {command_fragment} process")

    def require_group_command(self, process_group, command_fragment):
        """Reject stale graph state if this Attempt's process has disappeared."""
        if not _group_command_pids(process_group, command_fragment):
            raise RuntimeError(
                f"owned {command_fragment} process disappeared from process "
                f"group {process_group}"
            )

    def stop(self):
        process_groups = [item.process.pid for item in self.processes]
        for item in reversed(self.processes):
            if item.process.poll() is None:
                try:
                    os.killpg(item.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        deadline = time.monotonic() + self.shutdown_timeout
        for item in reversed(self.processes):
            remaining = max(0.0, deadline - time.monotonic())
            try:
                item.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                pass

        # A ros2 launch parent can exit before descendants. Sweep the exact
        # process groups created here without touching unrelated ROS processes.
        for process_group in reversed(process_groups):
            if _process_group_exists(process_group):
                try:
                    os.killpg(process_group, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for item in self.processes:
            if item.tee_thread is not None:
                item.tee_thread.join(timeout=2.0)
            item.log.close()
