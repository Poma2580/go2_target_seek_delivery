"""Static Attempt execution, retry classification, and cleanup."""

from dataclasses import dataclass
import csv
from pathlib import Path
import subprocess
import time

import yaml

from go2_static_perception_test.reporting.results import evaluate_csv, write_yaml
from go2_static_perception_test.runner.processes import ProcessManager


@dataclass(frozen=True)
class AttemptResult:
    number: int
    status: str
    reason: str | None
    summary: dict


def _list(kind):
    result = subprocess.run(
        ["ros2", kind, "list"],
        text=True,
        capture_output=True,
        timeout=5.0,
        check=False,
    )
    return result.stdout.splitlines() if result.returncode == 0 else []


def wait_graph(kind, name, timeout, health=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health()
        if name in _list(kind):
            return
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for ROS {kind} {name}")


def wait_topic_message(topic, message_type, timeout, health=lambda: None, manager=None):
    """Wait for one real message, not only a ROS graph publisher."""
    command = ["ros2", "topic", "echo", topic, message_type, "--once"]
    if manager is None:
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        poll = process.poll
    else:
        item = manager.start(
            "topic_wait_" + topic.strip("/").replace("/", "_"), command
        )
        process = item.process
        poll = item.poll

    deadline = time.monotonic() + timeout
    try:
        while poll() is None:
            health()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for message on {topic}")
            time.sleep(0.2)
        if process.returncode != 0:
            raise RuntimeError(
                f"ROS topic waiter for {topic} exited with {process.returncode}"
            )
    finally:
        if manager is None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def wait_controllers(timeout, health=lambda: None):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        health()
        result = subprocess.run(
            [
                "ros2",
                "control",
                "list_controllers",
                "-c",
                "/go2_1/controller_manager",
            ],
            text=True,
            capture_output=True,
            timeout=5.0,
            check=False,
        )
        lines = result.stdout.splitlines()
        if result.returncode == 0 and all(
            any(name in line and "active" in line for line in lines)
            for name in ("joint_group_effort_controller", "joint_states_controller")
        ):
            return
        time.sleep(0.5)
    raise RuntimeError("timed out waiting for go2_1 controllers")


def failure_summary(case, reason):
    return {
        "case_id": case.case_id,
        "target_key": case.target_key,
        "status": "infrastructure_failed",
        "infrastructure_valid": False,
        "recognition_accuracy": None,
        "recognition_pass": False,
        "mean_relative_localization_error": None,
        "localization_pass": False,
        "pass": False,
        "reason": reason,
    }


def ensure_artifacts(case, attempt_dir, reason):
    """Give even pre-recorder infrastructure failures the stable Attempt shape."""
    raw = Path(attempt_dir) / "raw/static_samples.csv"
    if not raw.exists():
        raw.parent.mkdir(parents=True, exist_ok=True)
        fields = (
            "case_id",
            "eval_index",
            "eval_time",
            "target_key",
            "prompt",
            "infrastructure_valid",
            "visible",
            "recognition_matched",
            "recognition_success",
            "confidence",
            "bbox",
            "localization_matched",
            "localization_success",
            "target_gt_x",
            "target_gt_y",
            "target_est_x",
            "target_est_y",
            "robot_gt_x",
            "robot_gt_y",
        )
        with raw.open("w", newline="", encoding="utf-8") as stream:
            csv.DictWriter(stream, fieldnames=fields).writeheader()
    return evaluate_csv(raw, attempt_dir, case.to_dict(), False, [reason])


def _spawn_progress(manager, reporter, health):
    """Report the same useful go2_1 spawn stages exposed by the T1 runner."""
    stage = None

    def check():
        nonlocal stage
        health()
        path = manager.log_dir / "spawn_go2_1.log"
        content = path.read_text(errors="replace") if path.exists() else ""
        current = "spawn node initialization"
        if "Spawn Entity started" in content:
            current = "spawn request pending"
        if "Calling service /spawn_entity" in content:
            current = "model generation pending"
        if "Successfully spawned entity [go2_1]" in content:
            current = "model generated; controllers activation pending"
        if current != stage:
            stage = current
            reporter(f"go2_1: {stage}")

    return check


def run_attempt(
    case,
    share,
    attempt_dir,
    case_config,
    model_path,
    device,
    execution,
    number,
    *,
    total_attempts=None,
    reporter=print,
    process_factory=ProcessManager,
):
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    total_attempts = total_attempts or (execution["max_restarts"] + 1)
    prefix = f"[Attempt {number}/{total_attempts}]"

    def report(message):
        reporter(f"{prefix} {message}", flush=True)

    manager = process_factory(
        attempt_dir / "logs",
        {
            "GO2_STATIC_CASE_ID": case.case_id,
            "GO2_STATIC_ATTEMPT": str(number),
        },
    )
    world = perception = None
    timeout = case.settings["startup_timeout_sec"]
    try:
        report("world starting")
        world = manager.start(
            "world",
            [
                "ros2",
                "launch",
                "go2_config",
                "gazebo_target_seek_world.launch.py",
                f"gui:={'true' if execution['gazebo_gui'] else 'false'}",
                f"world:={Path(share) / 'worlds/city_static_objects.world'}",
            ],
            phase="world",
        )
        health = lambda: manager.require_alive(world)
        for kind, name in (
            ("service", "/spawn_entity"),
            ("topic", "/clock"),
            ("topic", "/gazebo/model_states"),
        ):
            wait_graph(kind, name, timeout, health)

        report(
            f"world ready; waiting {execution['world_to_robot_delay_sec']:.1f}s "
            "before go2_1"
        )
        time.sleep(execution["world_to_robot_delay_sec"])
        health()

        pose = case.robot_pose
        spawn = manager.start(
            "spawn_go2_1",
            [
                "ros2",
                "launch",
                "go2_config",
                "spawn_go2_velodyne_1.launch.py",
                "scene:=city",
                "use_sim_time:=true",
                "enable_camera:=true",
                "enable_lidar:=false",
                f"spawn_x:={pose['x']}",
                f"spawn_y:={pose['y']}",
                f"spawn_z:={pose['z']}",
                f"spawn_yaw:={pose['yaw']}",
            ],
        )
        spawn_health = _spawn_progress(
            manager,
            report,
            lambda: manager.require_alive(world, spawn),
        )
        spawn_health()
        wait_controllers(timeout, spawn_health)
        report("go2_1: controllers active")

        health = lambda: manager.require_alive(world, spawn)
        for topic in (
            "/go2_1/odom",
            "/go2_1/odom/ground_truth",
            "/go2_1/camera/image_raw",
            "/go2_1/camera/depth/image_raw",
            "/go2_1/camera/depth/camera_info",
        ):
            wait_graph("topic", topic, timeout, health)
        report("go2_1 controllers/topics ready; lidar skipped")

        report(
            f"go2_1 ready; settling {execution['attitude_settle_delay_sec']:.1f}s "
            f"before {execution['attitude_sample_frames']}-frame attitude check"
        )
        time.sleep(execution["attitude_settle_delay_sec"])
        health()
        attitude = manager.start(
            "attitude",
            [
                "ros2",
                "run",
                "go2_static_perception_test",
                "check_static_go2_attitude",
                "--ros-args",
                "-p",
                f"roll_limit_deg:={execution['attitude_roll_limit_deg']}",
                "-p",
                f"sample_frames:={execution['attitude_sample_frames']}",
                "-p",
                f"timeout_sec:={execution['attitude_timeout_sec']}",
            ],
            mirror_to_console=True,
            console_prefix=f"{prefix} ",
        )
        try:
            attitude_status = attitude.process.wait(
                timeout=execution["attitude_timeout_sec"] + 5.0
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("attitude checker timed out") from error
        if attitude_status == 10:
            report("FALLEN: attitude gate returned exit code 10")
            summary = failure_summary(case, "go2_1 fell")
            write_yaml(attempt_dir / "case_summary.yaml", summary)
            return AttemptResult(number, "fallen", "go2_1 fell", summary)
        if attitude_status != 0:
            raise RuntimeError(
                f"attitude checker exited with status {attitude_status}"
            )
        report(f"attitude PASSED: {execution['attitude_sample_frames']} complete frames")

        report(f"starting YOLOE: prompt={case.prompt}")
        perception = manager.start(
            "perception",
            [
                "ros2",
                "launch",
                "go2_static_perception_test",
                "single_go2_static_yoloe.launch.py",
                f"target_prompt:={case.prompt}",
                f"model_path:={model_path}",
                f"device:={device}",
                "use_sim_time:=true",
            ],
        )
        perception_health = lambda: manager.require_alive(world, spawn, perception)
        for topic in (
            "/go2_1/static_perception/result_status",
            "/go2_1/static_perception/target_pose_estimated",
        ):
            wait_graph("topic", topic, timeout, perception_health)

        # T1-style Recorder semantics are kept unchanged.  The only static-YOLOE
        # timing guard is to wait for one real result before starting Recorder,
        # so graph registration is not mistaken for a warmed-up perception stream.
        wait_topic_message(
            "/go2_1/static_perception/result_status",
            "std_msgs/msg/String",
            timeout,
            perception_health,
            manager,
        )
        report("YOLOE result stream ready")

        if execution["rqt"]:
            manager.start(
                "rqt",
                [
                    "ros2",
                    "run",
                    "rqt_image_view",
                    "rqt_image_view",
                    "/go2_1/static_perception/debug_image",
                ],
            )
            report("rqt started: /go2_1/static_perception/debug_image")

        report("recorder started")
        recorder = manager.start(
            "recorder",
            [
                "ros2",
                "run",
                "go2_static_perception_test",
                "static_perception_recorder",
                "--ros-args",
                "-p",
                "use_sim_time:=true",
                "-p",
                f"case_config:={case_config}",
                "-p",
                f"output_dir:={attempt_dir}",
            ],
        )
        deadline = (
            time.monotonic()
            + timeout
            + case.settings["evaluation_duration_sec"]
            + 30.0
        )
        while recorder.poll() is None and time.monotonic() < deadline:
            manager.require_alive(world, spawn, perception)
            time.sleep(0.2)
        if recorder.poll() is None:
            raise RuntimeError("static recorder timed out")
        if recorder.returncode != 0:
            raise RuntimeError(
                f"static recorder exited with status {recorder.returncode}"
            )
        report("recorder completed")

        path = attempt_dir / "case_summary.yaml"
        if not path.is_file():
            raise RuntimeError("recorder did not write case_summary.yaml")
        summary = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not summary.get("infrastructure_valid"):
            return AttemptResult(
                number,
                "infrastructure_failed",
                summary.get("reason"),
                summary,
            )
        summary["status"] = "completed"
        write_yaml(path, summary)
        return AttemptResult(number, "completed", None, summary)
    except Exception as error:
        report(f"INFRASTRUCTURE FAILED: {error}")
        summary = failure_summary(case, str(error))
        write_yaml(attempt_dir / "case_summary.yaml", summary)
        return AttemptResult(number, "infrastructure_failed", str(error), summary)
    finally:
        manager.stop()
        report("cleanup complete")


def run_case(
    case,
    share,
    case_dir,
    model_path,
    device,
    execution,
    attempt_runner=run_attempt,
    sleep=time.sleep,
    reporter=print,
):
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=False)
    case_config = case_dir / "case_config.yaml"
    write_yaml(
        case_config,
        {"schema_version": 1, **case.to_dict(), "execution": execution},
    )

    attempts = []
    total_attempts = execution["max_restarts"] + 1
    while True:
        number = len(attempts) + 1
        reporter(
            f"[{case.case_id}] Attempt {number}/{total_attempts} starting",
            flush=True,
        )
        result = attempt_runner(
            case,
            share,
            case_dir / f"attempts/attempt_{number:02d}",
            case_config,
            model_path,
            device,
            execution,
            number,
            total_attempts=total_attempts,
            reporter=reporter,
        )
        attempts.append(result)
        if result.status == "completed" or number > execution["max_restarts"]:
            break
        delay = execution["restart_delay_sec"]
        reporter(
            f"[Attempt {number}/{total_attempts}] {result.status}; "
            f"restarting in {delay:.1f}s",
            flush=True,
        )
        sleep(delay)

    final = attempts[-1]
    final_dir = case_dir / f"attempts/attempt_{final.number:02d}"
    if not (final_dir / "raw/static_samples.csv").is_file():
        final.summary.update(
            ensure_artifacts(
                case,
                final_dir,
                final.reason or "infrastructure failure",
            )
        )

    summary = dict(final.summary)
    if final.status != "completed":
        summary.update(
            {
                "status": "infrastructure_failed",
                "infrastructure_valid": False,
                "pass": False,
                "reason": (
                    "infrastructure failure persisted after "
                    f"{execution['max_restarts']} restart(s): {final.reason}"
                ),
            }
        )
    summary.update(
        {
            "attempts_used": len(attempts),
            "restarts_used": len(attempts) - 1,
            "restart_exhausted": final.status != "completed",
            "final_attempt": final.number,
            "attempts": [
                {
                    "attempt": item.number,
                    "status": item.status,
                    "reason": item.reason,
                    "path": f"attempts/attempt_{item.number:02d}",
                }
                for item in attempts
            ],
        }
    )
    write_yaml(case_dir / "case_summary.yaml", summary)
    return {
        "case_id": case.case_id,
        "target_key": case.target_key,
        "status": summary["status"],
        "summary": summary,
        "csv_path": final_dir / "raw/static_samples.csv",
    }
