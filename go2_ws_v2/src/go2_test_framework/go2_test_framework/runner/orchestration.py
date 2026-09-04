"""Attempt execution and Case-level retry orchestration."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import time

import yaml

from go2_test_framework.reporting.results import write_yaml
from go2_test_framework.runner.processes import ProcessGroupManager
from go2_test_framework.runner.runtime import (
    ATTEMPT_MARKER, CASE_MARKER, RUN_MARKER,
)
from go2_test_framework.runner.ros_wait import (
    wait_for_controllers_active,
    wait_for_perception_role,
)


ATTITUDE_EXIT_FALLEN = 10


class AttemptStatus(str, Enum):
    COMPLETED = "completed"
    FALLEN = "fallen"
    INFRASTRUCTURE_FAILED = "infrastructure_failed"


@dataclass(frozen=True)
class AttemptResult:
    number: int
    status: AttemptStatus
    reason: str | None
    summary: dict

    @property
    def retryable(self):
        return self.status is AttemptStatus.FALLEN

    def to_dict(self):
        value = {
            "attempt": self.number,
            "status": self.status.value,
            "path": f"attempts/attempt_{self.number:02d}",
        }
        if self.reason:
            value["reason"] = self.reason
        if self.status is AttemptStatus.COMPLETED:
            value["pass"] = bool(self.summary.get("pass", False))
        return value


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    status: str
    summary: dict

    @property
    def infrastructure_failed(self):
        return self.status == AttemptStatus.INFRASTRUCTURE_FAILED.value


def _wait_graph(kind, name, timeout, health_check=None):
    command = ["ros2", kind, "list"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if health_check is not None:
            health_check()
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode == 0 and name in result.stdout.splitlines():
            if health_check is not None:
                health_check()
            return
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for ROS {kind} {name}")


def _load_mapping(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid YAML summary: {path}")
    return value


def _failure_summary(status, reason, case):
    return {
        "status": status.value,
        "infrastructure_valid": False,
        "provisional": bool(case.route_config["provisional"] or not case.formal),
        "recognition": None,
        "localization": None,
        "pass": False,
        "reason": reason,
    }


def _attitude_command(config):
    return [
        "ros2", "run", "go2_scenario_config", "check_three_go2_attitude",
        "--ros-args",
        "-p", "model_states_topic:=/gazebo/model_states",
        "-p", "robot_names:=[go2_1,go2_2,go2_3]",
        "-p", f"roll_limit_deg:={config.roll_limit_deg}",
        "-p", f"sample_frames:={config.sample_frames}",
        "-p", f"timeout_seconds:={config.timeout_sec}",
    ]


def spawn_robots(
    case, processes, startup_timeout, startup, *,
    wait_graph=_wait_graph,
    wait_controllers=wait_for_controllers_active,
    sleep=time.sleep,
    reporter=lambda _message: None,
    health_check=None,
):
    """Spawn three robots sequentially and verify each one before continuing."""
    sleep(startup.world_to_first_delay_sec)
    for index, robot in enumerate(("go2_1", "go2_2", "go2_3"), start=1):
        pose = case.robot_poses[robot]
        processes.start(f"spawn_{robot}", [
            "ros2", "launch", "go2_config",
            f"spawn_go2_velodyne_{index}.launch.py",
            f"scene:={case.scene}", "use_sim_time:=true",
            f"enable_lidar:={'true' if startup.enable_lidar else 'false'}",
            "enable_camera:=true",
            f"spawn_x:={pose['x']}", f"spawn_y:={pose['y']}",
            f"spawn_z:={pose['z']}", f"spawn_yaw:={pose['yaw']}",
        ])
        wait_controllers(robot, startup_timeout, health_check=health_check)
        for topic in (
            f"/{robot}/odom",
            f"/{robot}/odom/ground_truth",
            f"/{robot}/camera/image_raw",
            f"/{robot}/camera/depth/image_raw",
            f"/{robot}/camera/depth/camera_info",
        ):
            wait_graph("topic", topic, startup_timeout)
        if startup.enable_lidar:
            wait_graph("topic", f"/{robot}/velodyne_points", startup_timeout)
        lidar_state = "lidar ready" if startup.enable_lidar else "lidar skipped"
        reporter(f"{robot} controllers/topics ready; {lidar_state}")
        if index < 3:
            sleep(startup.inter_robot_delay_sec)


def run_attempt(
    case, share, attempt_dir, case_config, model_path, execution, attempt_number,
    *, run_id="unmarked", total_attempts=None, reporter=print,
    process_manager_factory=ProcessGroupManager,
):
    """Run one isolated attempt and convert every terminal path to a result."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    total_attempts = total_attempts or (
        execution.attitude_check.max_restarts + 1
        if execution.attitude_check.enabled else 1
    )
    prefix = f"[Attempt {attempt_number}/{total_attempts}]"

    def report(message):
        reporter(f"{prefix} {message}", flush=True)

    processes = process_manager_factory(
        attempt_dir / "logs",
        environment={
            RUN_MARKER: str(run_id),
            CASE_MARKER: case.case_id,
            ATTEMPT_MARKER: str(attempt_number),
        },
    )
    world = Path(share) / "worlds" / f"{case.scene}_{case.route}.world"
    startup_timeout = case.settings["startup_timeout_sec"]
    try:
        world_process = processes.start("world", [
            "ros2", "launch", "go2_config", "gazebo_target_seek_world.launch.py",
            f"gui:={'true' if execution.gazebo_gui else 'false'}",
            f"world:={world}",
        ])
        processes.wait_for_group_command(
            world_process, "gzserver", min(startup_timeout, 15.0)
        )

        def require_current_world():
            processes.require_group_command(world_process.pid, "gzserver")

        def wait_current_graph(kind, name, timeout):
            return _wait_graph(kind, name, timeout, require_current_world)

        wait_current_graph("service", "/spawn_entity", startup_timeout)
        wait_current_graph("service", "/walking_target/start", startup_timeout)
        wait_current_graph("topic", "/gazebo/model_states", startup_timeout)
        wait_current_graph("topic", "/clock", startup_timeout)
        startup = execution.robot_startup
        report(
            f"world ready; waiting {startup.world_to_first_delay_sec:.1f}s "
            "before go2_1"
        )
        spawn_robots(
            case, processes, startup_timeout, startup,
            wait_graph=wait_current_graph,
            reporter=report,
            health_check=require_current_world,
        )

        attitude = execution.attitude_check
        if attitude.enabled:
            report(
                f"all robots ready; settling {attitude.settle_delay_sec:.1f}s "
                f"before {attitude.sample_frames}-frame attitude check"
            )
            time.sleep(attitude.settle_delay_sec)
            require_current_world()
            checker = processes.start(
                "attitude_check", _attitude_command(attitude),
                mirror_to_console=True,
                console_prefix=f"{prefix} ",
            )
            try:
                checker_status = checker.wait(timeout=attitude.timeout_sec + 5.0)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("attitude checker process timed out") from error
            if checker_status == ATTITUDE_EXIT_FALLEN:
                report("FALLEN: attitude gate returned exit code 10")
                reason = "attitude check confirmed a fallen Go2"
                summary = _failure_summary(AttemptStatus.FALLEN, reason, case)
                write_yaml(attempt_dir / "case_summary.yaml", summary)
                return AttemptResult(
                    attempt_number, AttemptStatus.FALLEN, reason, summary
                )
            if checker_status != 0:
                raise RuntimeError(
                    f"attitude checker exited with status {checker_status}"
                )
            report(f"attitude PASSED: {attitude.sample_frames} complete frames")

        processes.start("actor_state", [
            "ros2", "run", "walking_target_controller", "actor_state_publisher",
            "--ros-args", "-p", "use_sim_time:=true",
        ])
        wait_current_graph("topic", "/walking_target/odom", startup_timeout)
        processes.start("perception", [
            "ros2", "launch", "go2_target_perception",
            "three_go2_target_tracking.launch.py", "use_sim_time:=true",
            f"model_path:={model_path}",
        ])
        selected_robot = wait_for_perception_role(
            case.settings["role_timeout_sec"], require_current_world
        )
        if execution.rqt:
            processes.start("rqt", [
                "ros2", "run", "rqt_image_view", "rqt_image_view",
                f"/{selected_robot}/target_perception/debug_image",
            ])

        recorder = processes.start("recorder", [
            "ros2", "run", "go2_test_framework", "target_test_recorder",
            "--ros-args", "-p", "use_sim_time:=true",
            "-p", f"case_config:={case_config}",
            "-p", f"output_dir:={attempt_dir}",
        ])
        require_current_world()
        service = subprocess.run(
            [
                "ros2", "service", "call", "/walking_target/start",
                "std_srvs/srv/Trigger", "{}",
            ],
            text=True,
            capture_output=True,
            timeout=15.0,
            check=False,
        )
        if service.returncode != 0 or "success=True" not in service.stdout:
            raise RuntimeError(
                f"failed to start walking target: {service.stdout}{service.stderr}"
            )
        recorder_timeout = (
            case.settings["startup_timeout_sec"]
            + case.settings["evaluation_duration_sec"]
            + 30.0
        )
        try:
            recorder_status = recorder.wait(timeout=recorder_timeout)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("target recorder timed out") from error
        if recorder_status != 0:
            raise RuntimeError(
                f"target recorder exited with status {recorder_status}"
            )
        summary_path = attempt_dir / "case_summary.yaml"
        if not summary_path.is_file():
            raise RuntimeError("target recorder did not write case_summary.yaml")
        summary = _load_mapping(summary_path)
        if not summary.get("infrastructure_valid", False):
            reason = summary.get("reason") or "; ".join(
                summary.get("infrastructure_errors", [])
            ) or "target recorder reported invalid infrastructure"
            summary["status"] = AttemptStatus.INFRASTRUCTURE_FAILED.value
            write_yaml(summary_path, summary)
            return AttemptResult(
                attempt_number,
                AttemptStatus.INFRASTRUCTURE_FAILED,
                reason,
                summary,
            )
        summary["status"] = AttemptStatus.COMPLETED.value
        write_yaml(summary_path, summary)
        return AttemptResult(
            attempt_number, AttemptStatus.COMPLETED, None, summary
        )
    except Exception as error:
        report(f"INFRASTRUCTURE FAILED: {error}")
        reason = str(error)
        summary = _failure_summary(
            AttemptStatus.INFRASTRUCTURE_FAILED, reason, case
        )
        write_yaml(attempt_dir / "case_summary.yaml", summary)
        return AttemptResult(
            attempt_number, AttemptStatus.INFRASTRUCTURE_FAILED, reason, summary
        )
    finally:
        processes.stop()
        report("cleanup complete")


def should_retry(result, max_restarts):
    """Return whether another attempt is allowed for this result."""
    return result.retryable and result.number <= max_restarts


def run_case(
    case, share, case_dir, model_path, metrics, execution, *,
    run_id="unmarked", attempt_runner=run_attempt, sleep=time.sleep,
    reporter=print,
):
    """Run a Case, retrying only confirmed falls, and write its root summary."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    case_value = {
        "schema_version": 1,
        **case.to_dict(),
        "provisional": case.route_config["provisional"],
        "metrics": metrics,
        "execution": execution.to_dict(),
    }
    case_config = case_dir / "case_config.yaml"
    write_yaml(case_config, case_value)
    attempts_dir = case_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)

    attempts = []
    total_attempts = (
        execution.attitude_check.max_restarts + 1
        if execution.attitude_check.enabled else 1
    )
    while True:
        number = len(attempts) + 1
        reporter(
            f"[{case.case_id}] Attempt {number}/{total_attempts} starting",
            flush=True,
        )
        result = attempt_runner(
            case,
            share,
            attempts_dir / f"attempt_{number:02d}",
            case_config,
            model_path,
            execution,
            number,
            run_id=run_id,
            total_attempts=total_attempts,
            reporter=reporter,
        )
        attempts.append(result)
        if not should_retry(result, execution.attitude_check.max_restarts):
            break
        delay = execution.attitude_check.restart_delay_sec
        reporter(
            f"[Attempt {number}/{total_attempts}] fallen; restart budget "
            f"{number}/{execution.attitude_check.max_restarts}; "
            f"restarting in {delay:.1f}s",
            flush=True,
        )
        sleep(delay)

    final = attempts[-1]
    restart_exhausted = (
        final.status is AttemptStatus.FALLEN
        and final.number > execution.attitude_check.max_restarts
    )
    if final.status is AttemptStatus.COMPLETED:
        summary = dict(final.summary)
        case_status = AttemptStatus.COMPLETED.value
    else:
        reason = final.reason or "unknown infrastructure failure"
        if restart_exhausted:
            reason = (
                f"attitude check remained fallen after "
                f"{execution.attitude_check.max_restarts} restart(s)"
            )
        summary = dict(final.summary)
        summary.update({
            "status": AttemptStatus.INFRASTRUCTURE_FAILED.value,
            "infrastructure_valid": False,
            "pass": False,
            "reason": reason,
        })
        case_status = AttemptStatus.INFRASTRUCTURE_FAILED.value
        if restart_exhausted:
            reporter(
                f"[{case.case_id}] restart exhausted after "
                f"{len(attempts)} Attempt(s)",
                flush=True,
            )
    summary.update({
        "status": case_status,
        "attempts_used": len(attempts),
        "restarts_used": len(attempts) - 1,
        "restart_exhausted": restart_exhausted,
        "final_attempt": final.number,
        "attempts": [attempt.to_dict() for attempt in attempts],
    })
    write_yaml(case_dir / "case_summary.yaml", summary)
    return CaseResult(case.case_id, case_status, summary)
