"""Bounded ROS workers and startup health gate; no shared memory mutations."""

from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import time

from go2_test_framework.reporting.results import write_yaml
from go2_test_framework.runner.lifecycle import BatchSafetyError, identities, shm_diagnostics
from go2_test_framework.runner.processes import ProcessGroupManager


def bounded_worker(manager, operation, arguments, timeout, health_check=None):
    """Wall-clock timeout includes imports, node construction and ROS shutdown."""
    sequence = len(manager.processes)
    name = f"{operation}_{sequence}"
    result_path = manager.log_dir / f"{name}.json"
    command = ["/usr/bin/python3", "-m", "go2_test_framework.runner.health",
               operation, json.dumps(arguments), str(result_path)]
    process = manager.start(name, command)
    deadline = time.monotonic() + timeout
    while process.poll() is None:
        if health_check is not None:
            health_check()
        if time.monotonic() >= deadline:
            records, _ = manager.owned.scan()
            write_yaml(manager.log_dir / f"{name}_diagnostic.yaml", {
                "operation": operation, "reason": "wall_clock_timeout",
                "processes": [r.diagnostic() for r in records],
                "shared_memory": shm_diagnostics(records),
            })
            raise RuntimeError(f"{operation} wall-clock timeout after {timeout}s; see {name}.log")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    if not result_path.is_file():
        raise RuntimeError(f"{operation} exited with {process.returncode} without a result; see {name}.log")
    result = json.loads(result_path.read_text())
    if not result["success"]:
        if operation == "walking_target" and "observation" in result:
            from go2_test_framework.runner.ros_wait import WalkingTargetStartError, WalkingTargetStartObservation
            raise WalkingTargetStartError(result["error"], WalkingTargetStartObservation(**result["observation"]))
        raise RuntimeError(result["error"])
    if process.returncode != 0:
        raise RuntimeError(f"{operation} exited with {process.returncode}")
    return result.get("value")


def startup_health(directory, environment, *, timeout=10.0, manager_factory=ProcessGroupManager):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"success": False, "reason": None}
    manager = manager_factory(directory / "health_logs" / "logs", environment=environment)
    started = time.monotonic()
    try:
        records, _ = identities()
        remaining = [r for r in records if r.run and r.state != "Z"]
        if remaining:
            report["remaining"] = [r.diagnostic() for r in remaining]
            raise RuntimeError("test processes remain before startup")
        bounded_worker(manager, "probe", {}, timeout)
        report["success"] = True
    except Exception as error:
        report["reason"] = str(error)
        records, _ = manager.owned.scan()
        report["shared_memory"] = shm_diagnostics(records)
    finally:
        try:
            report["cleanup"] = manager.stop()
        except BatchSafetyError as error:
            report["success"] = False
            report["reason"] = str(error)
            report["abort_reason"] = "cleanup_failed"
        report["elapsed_sec"] = time.monotonic() - started
        report["stages"] = {}
        for path in manager.log_dir.glob("*.log"):
            report["stages"][path.name] = path.read_text(errors="replace")[-4000:]
        write_yaml(directory / "startup_health.yaml", report)
    if report.get("cleanup", {}).get("received_signals"):
        import signal
        from go2_test_framework.runner.runtime import ShutdownRequested
        raise ShutdownRequested(getattr(signal, report["cleanup"]["received_signals"][0]))
    if not report["success"]:
        raise BatchSafetyError(report.get("abort_reason", "startup_health_failed"), report["reason"])
    return report


def main():
    import sys
    operation, raw, result_file = sys.argv[1:]
    args = json.loads(raw)
    result = {"success": False}
    try:
        print("worker: importing ROS", flush=True)
        if operation == "probe":
            import rclpy
            print("probe: initializing context", flush=True)
            rclpy.init()
            print("probe: creating node", flush=True)
            node = rclpy.create_node("go2_test_startup_probe")
            print("probe: destroying node", flush=True)
            node.destroy_node()
            print("probe: shutting down context", flush=True)
            rclpy.shutdown()
            print("probe: complete", flush=True)
            value = None
        else:
            from go2_test_framework.runner import ros_wait
            functions = {"controllers": ros_wait.wait_for_controllers_active,
                         "role": ros_wait.wait_for_perception_role,
                         "walking_target": ros_wait.start_walking_target}
            print(f"{operation}: starting ROS wait", flush=True)
            value = functions[operation](**args)
            if operation == "walking_target":
                value = asdict(value)
        result.update(success=True, value=value)
    except Exception as error:
        result["error"] = str(error)
        if hasattr(error, "observation"):
            result["observation"] = asdict(error.observation)
    Path(result_file).write_text(json.dumps(result))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
