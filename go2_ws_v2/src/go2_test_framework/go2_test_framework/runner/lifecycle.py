"""Linux process identity, bounded shutdown, and read-only DDS diagnostics."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import signal
import threading
import time


OWNER_MARKER = "GO2_TEST_PROCESS_OWNER"
PHASE_MARKER = "GO2_TEST_PROCESS_PHASE"


class BatchSafetyError(RuntimeError):
    def __init__(self, reason, detail):
        self.reason = reason
        super().__init__(detail)


@dataclass(frozen=True)
class Identity:
    pid: int
    start_ticks: int
    group: int
    state: str
    command: str
    owner: str = ""
    phase: str = "workers"
    run: str = ""
    case: str = ""
    attempt: str = ""

    @property
    def key(self):
        return (self.pid, self.start_ticks)

    def diagnostic(self):
        return {key: value for key, value in asdict(self).items() if key != "owner"}


def read_identity(pid, proc_root=Path("/proc")):
    path = proc_root / str(pid)
    fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
    state, group, ticks = fields[0], int(fields[2]), int(fields[19])
    command = (path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    # Zombie processes no longer hold DDS resources and have no readable environ.
    env = {}
    if state != "Z":
        for item in (path / "environ").read_bytes().split(b"\0"):
            key, _, value = item.partition(b"=")
            if key.startswith(b"GO2_TEST_"):
                env[key.decode()] = value.decode(errors="replace")
    return Identity(pid, ticks, group, state, command,
                    env.get(OWNER_MARKER, ""), env.get(PHASE_MARKER, "workers"),
                    env.get("GO2_TEST_RUN_ID", ""), env.get("GO2_TEST_CASE_ID", ""),
                    env.get("GO2_TEST_ATTEMPT", ""))


def identities():
    records, errors = [], []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            records.append(read_identity(int(path.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError) as error:
            detail = {"pid": int(path.name), "error": str(error)}
            try:
                detail["group"] = os.getpgid(int(path.name))
            except (ProcessLookupError, PermissionError):
                pass
            errors.append(detail)
    return records, errors


@contextmanager
def defer_shutdown_signals(received):
    """Finish resource release even if the user repeats Ctrl-C/termination."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, lambda sig, _frame: received.append(signal.Signals(sig).name))
    try:
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


class OwnedProcesses:
    """Track identities, including marked descendants which start new sessions."""

    def __init__(self, owner="", seeds=()):
        self.owner = owner
        self.known = {record.key: record for record in seeds}
        self.groups = {}
        for record in seeds:
            self.groups[record.group] = min(self.groups.get(record.group, record.start_ticks), record.start_ticks)
        self.seed_owners = {record.owner for record in seeds if record.owner}
        self.zombies = {}
        self.unverified = []
        self.seed_markers = {(r.run, r.case, r.attempt) for r in seeds if r.run}

    def add(self, pid):
        try:
            record = read_identity(pid)
        except (FileNotFoundError, ProcessLookupError):
            return
        except OSError as error:
            self.unverified.append({"pid": pid, "error": str(error)})
            return
        self.known[record.key] = record
        self.groups[record.group] = record.start_ticks

    def scan(self):
        records, errors = identities()
        by_pid = {r.pid: r for r in records}
        # A newly reused group leader invalidates the old group membership claim.
        for group, ticks in list(self.groups.items()):
            leader = by_pid.get(group)
            if leader is not None and leader.start_ticks != ticks:
                self.groups.pop(group)
        found = []
        for record in records:
            if record.pid == os.getpid():
                continue
            known = self.known.get(record.key)
            marked = ((self.owner and record.owner == self.owner)
                      or record.owner in self.seed_owners
                      or (record.run, record.case, record.attempt) in self.seed_markers)
            in_group = record.group in self.groups and record.start_ticks >= self.groups[record.group]
            if not (known or marked or in_group):
                continue
            # Preserve the phase for unmarked legacy children.
            if known and record.phase != known.phase:
                from dataclasses import replace
                record = replace(record, phase=known.phase)
            if not record.owner and in_group and not known:
                from dataclasses import replace
                phase = next((r.phase for r in self.known.values() if r.group == record.group), "workers")
                record = replace(record, phase=phase)
            self.known[record.key] = record
            if record.state == "Z":
                self.zombies[record.key] = record.diagnostic()
            else:
                found.append(record)
        self.unverified = [e for e in self.unverified if Path(f"/proc/{e['pid']}").exists()]
        critical = self.unverified + [
            e for e in errors if e.get("group") in self.groups
            or any(r.pid == e["pid"] for r in self.known.values())
        ]
        return found, critical

    @staticmethod
    def send(record, signum):
        # pidfd pins the process across exit/PID reuse, including between check and signal.
        try:
            fd = os.pidfd_open(record.pid)
        except ProcessLookupError:
            return
        try:
            current = read_identity(record.pid)
            if current.key == record.key and current.state != "Z":
                signal.pidfd_send_signal(fd, signum)
        except (FileNotFoundError, ProcessLookupError):
            pass
        finally:
            os.close(fd)

    def shutdown(self, timeouts=(15.0, 5.0, 3.0), reap=lambda: None):
        started = time.monotonic()
        report = {"success": False, "signals": [], "received_signals": [], "errors": []}
        with defer_shutdown_signals(report["received_signals"]):
            for phase in ("workers", "world"):
                for signum, timeout in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), timeouts):
                    deadline = time.monotonic() + timeout
                    signaled = set()
                    while True:
                        reap()
                        live, errors = self.scan()
                        report["errors"].extend(e for e in errors if e not in report["errors"])
                        targets = [r for r in live if r.phase == phase]
                        if not targets:
                            break
                        for record in targets:
                            if record.key in signaled:
                                continue
                            try:
                                self.send(record, signum)
                                report["signals"].append({"signal": signum.name, "phase": phase,
                                    "elapsed_sec": time.monotonic() - started, **record.diagnostic()})
                                signaled.add(record.key)
                            except OSError as error:
                                report["errors"].append({"pid": record.pid, "error": str(error)})
                        if time.monotonic() >= deadline:
                            break
                        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
                    if not targets:
                        break
            reap()
            live, errors = self.scan()
            report["observed_errors"] = report["errors"]
            # /proc/environ can become unreadable during exit. Only unresolved
            # final errors or signaling failures for still-live identities block.
            live_pids = {r.pid for r in live}
            report["errors"] = errors + [e for e in report["observed_errors"]
                                         if e["pid"] in live_pids and e not in errors]
            report["remaining"] = [r.diagnostic() for r in live]
            report["zombies"] = list(self.zombies.values())
            report["success"] = not live and not report["errors"]
            report["elapsed_sec"] = time.monotonic() - started
        return report


def shm_diagnostics(records, proc_root=Path("/proc"), shm_root=Path("/dev/shm")):
    """Inspect only port data files actually referenced by these processes."""
    files = {}
    for record in records:
        base = proc_root / str(record.pid)
        try:
            if read_identity(record.pid, proc_root).key != record.key:
                continue
            names = set()
            for fd in (base / "fd").iterdir():
                try:
                    names.add(os.readlink(fd))
                except OSError:
                    pass
            for line in (base / "maps").read_text().splitlines():
                if str(shm_root) + "/" in line:
                    names.add(line[line.index(str(shm_root) + "/"):])
            for name in names:
                if not re.fullmatch(re.escape(str(shm_root)) + r"/fastrtps_port[0-9]+", name):
                    continue
                try:
                    stat = Path(name).stat()
                except OSError:
                    continue
                item = files.setdefault(name, {"path": name, "inode": stat.st_ino,
                    "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "pids": []})
                item["pids"].append(record.pid)
        except (OSError, ProcessLookupError):
            continue
    return list(files.values())
