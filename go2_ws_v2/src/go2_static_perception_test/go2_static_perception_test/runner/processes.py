"""Attempt-scoped subprocess groups with T1-style selective console mirroring."""

import os
from pathlib import Path
import signal
import subprocess
import threading
import time


class ManagedProcess:
    def __init__(self, name, process, stream, phase, tee_thread=None):
        self.name = name
        self.process = process
        self.stream = stream
        self.phase = phase
        self.tee_thread = tee_thread

    @property
    def pid(self):
        return self.process.pid

    @property
    def returncode(self):
        return self.process.poll()

    def poll(self):
        return self.process.poll()


class ProcessManager:
    """Own Attempt subprocess groups and keep full logs under the Attempt."""

    def __init__(self, log_dir, environment=None):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.environment = dict(environment or {})
        self.items = []

    @staticmethod
    def _tee_output(item, prefix):
        source = item.process.stdout
        if source is None:
            return
        for line in source:
            item.stream.write(line)
            item.stream.flush()
            print(f"{prefix}{line.rstrip()}", flush=True)

    def start(self, name, command, phase="workers", *, mirror_to_console=False,
              console_prefix=""):
        stream = (self.log_dir / f"{name}.log").open("w", encoding="utf-8")
        env = os.environ.copy()
        env.update(self.environment)
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE if mirror_to_console else stream,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
                env=env,
            )
        except Exception:
            stream.close()
            raise
        item = ManagedProcess(name, process, stream, phase)
        if mirror_to_console:
            item.tee_thread = threading.Thread(
                target=self._tee_output,
                args=(item, console_prefix),
                name=f"go2-static-tee-{name}",
                daemon=True,
            )
            item.tee_thread.start()
        self.items.append(item)
        return item

    def require_alive(self, *items):
        for item in items:
            status = item.poll()
            if status is not None:
                raise RuntimeError(f"{item.name} exited with status {status}")

    def stop(self):
        remaining = []
        for phase in ("workers", "world"):
            targets = [
                item for item in reversed(self.items)
                if item.phase == phase and item.poll() is None
            ]
            for sig, timeout in (
                (signal.SIGINT, 15.0),
                (signal.SIGTERM, 5.0),
                (signal.SIGKILL, 3.0),
            ):
                live = [item for item in targets if item.poll() is None]
                if not live:
                    break
                for item in live:
                    try:
                        os.killpg(item.pid, sig)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + timeout
                while (
                    any(item.poll() is None for item in live)
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
            remaining.extend(item.name for item in targets if item.poll() is None)

        for item in self.items:
            if item.tee_thread is not None:
                item.tee_thread.join(timeout=2.0)
            if item.tee_thread is None or not item.tee_thread.is_alive():
                item.stream.close()
        if remaining:
            raise RuntimeError(f"failed to clean processes: {remaining}")
