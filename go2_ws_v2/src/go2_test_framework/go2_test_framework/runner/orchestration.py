"""Attempt execution and Case-level retry orchestration."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import subprocess
import time

import yaml

from go2_test_framework.reporting.results import write_yaml
from go2_test_framework.common.scene_resolution import write_resolved_scene_config
from go2_test_framework.runner.processes import ProcessGroupManager
from go2_test_framework.runner.lifecycle import BatchSafetyError
from go2_test_framework.runner.health import startup_health, bounded_worker
from go2_test_framework.runner.runtime import (
    ATTEMPT_MARKER, CASE_MARKER, RUN_MARKER, ShutdownRequested,
)
from go2_test_framework.runner.ros_wait import (
    WalkingTargetStartError,
    start_walking_target,
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
        return self.status in {
            AttemptStatus.FALLEN,
            AttemptStatus.INFRASTRUCTURE_FAILED,
        }

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
        try:
            result = subprocess.run(
                command, text=True, capture_output=True, check=False,
                timeout=max(0.01, min(5.0, deadline - time.monotonic())),
            )
        except subprocess.TimeoutExpired:
            continue
        if result.returncode == 0 and name in result.stdout.splitlines():
            if health_check is not None:
                health_check()
            return
        time.sleep(0.5)
    raise RuntimeError(f"timed out waiting for ROS {kind} {name}")


def _wait_topic_message(topic, message_type, timeout, health_check=None, processes=None):
    """Wait for one real message, not merely a ROS graph endpoint."""
    command = ["ros2", "topic", "echo", topic, message_type, "--once"]
    process = (processes.start("topic_wait_" + topic.strip("/").replace("/", "_"), command)
               if processes is not None else subprocess.Popen(
                   command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
               ))
    deadline = time.monotonic() + timeout
    try:
        while process.poll() is None:
            if health_check is not None:
                health_check()
            if time.monotonic() >= deadline:
                raise RuntimeError(f"timed out waiting for message on {topic}")
            time.sleep(0.2)
        if process.returncode != 0:
            raise RuntimeError(
                f"ROS topic waiter for {topic} exited with {process.returncode}"
            )
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def _load_mapping(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"invalid YAML summary: {path}")
    return value


def _failure_summary(status, reason, case):
    summary = {
        "status": status.value,
        "infrastructure_valid": False,
        "provisional": bool(case.route_config["provisional"] or not case.formal),
        "recognition": None,
        "localization": None,
        "pass": False,
        "reason": reason,
    }
    if case.task_type == "tracking":
        summary.update({
            "tracking_success": False,
            "acquisition_time_sec": None,
            "continuous_tracking_time_sec": 0.0,
            "failure_reason": None,
        })
    elif case.task_type == "path_planning":
        summary.update({
            "path_success": False,
            "all_reached": False,
            "collision_count": 0,
            "latest_generation": None,
            "failure_reason": None,
        })
    return summary


def _failure_summary_preserving_metrics(attempt_dir, reason, case):
    summary_path = Path(attempt_dir) / "case_summary.yaml"
    try:
        summary = _load_mapping(summary_path)
    except (FileNotFoundError, RuntimeError, OSError, yaml.YAMLError):
        summary = _failure_summary(
            AttemptStatus.INFRASTRUCTURE_FAILED, reason, case
        )
    summary.update({
        "status": AttemptStatus.INFRASTRUCTURE_FAILED.value,
        "infrastructure_valid": False,
        "pass": False,
        "reason": reason,
    })
    return summary


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
        stage = None

        def spawn_health():
            nonlocal stage
            if health_check is not None:
                health_check()
            log_dir = getattr(processes, "log_dir", None)
            if log_dir is None:
                return
            path = log_dir / f"spawn_{robot}.log"
            content = path.read_text(errors="replace") if path.exists() else ""
            current = "spawn node initialization"
            if "Spawn Entity started" in content:
                current = "spawn request pending"
            if "Calling service /spawn_entity" in content:
                current = "model generation pending"
            if f"Successfully spawned entity [{robot}]" in content:
                current = "model generated; controllers activation pending"
            if current != stage:
                stage = current
                reporter(f"{robot}: {stage}")

        try:
            spawn_health()
            wait_controllers(robot, startup_timeout, health_check=spawn_health)
        except Exception as error:
            raise RuntimeError(f"{robot}: {error}; last startup stage: {stage}") from error
        reporter(f"{robot}: controllers active")
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
    process_manager_factory=ProcessGroupManager, health_gate=startup_health,
):
    """Run one isolated attempt and convert every terminal path to a result."""
    attempt_dir = Path(attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=False)
    total_attempts = total_attempts or (
        execution.attitude_check.max_restarts + 1
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
    target_start_observation = None
    world = Path(share) / "worlds" / f"{case.scene}_{case.route}.world"
    startup_timeout = case.settings["startup_timeout_sec"]
    case_metrics = _load_mapping(case_config).get("metrics", {})
    try:
        health_gate(attempt_dir, {
            RUN_MARKER: str(run_id), CASE_MARKER: case.case_id,
            ATTEMPT_MARKER: str(attempt_number),
        })
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
            wait_controllers=lambda robot, timeout, health_check=None: bounded_worker(
                processes, "controllers", {"robot_name": robot, "timeout_sec": timeout},
                timeout, health_check,
            ),
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
        if case.task_type == "path_planning":
            processes.start("body_contact_monitor", [
                "ros2", "run", "champ_gazebo", "body_contact_monitor",
                "--ros-args", "-p", "use_sim_time:=true",
            ])
            _wait_topic_message(
                "/go2_test/body_contacts", "std_msgs/msg/String",
                startup_timeout, require_current_world, processes,
            )
            report("received body-contact state stream")
            processes.start("map_merger", [
                "ros2", "launch", "go2_mapping_nav",
                "three_go2_map_merge.launch.py",
                "use_sim_time:=true", "use_rviz:=false",
            ])
            processes.start("follower_cmd_vel_mux", [
                "ros2", "run", "go2_dynamic_encircle",
                "follower_cmd_vel_mux", "--ros-args",
                "-p", "use_sim_time:=true",
            ])
            for index, robot in enumerate(
                ("go2_1", "go2_2", "go2_3"), start=1
            ):
                processes.start(f"mapping_nav_{robot}", [
                    "ros2", "launch", "go2_test_framework",
                    "t3_mapping_nav.launch.py",
                    f"robot_name:={robot}",
                    f"scene:={case.scene}",
                    "use_sim_time:=true", "use_merged_map:=true",
                    "use_rviz:=false", "delete_db_on_start:=true",
                    f"cmd_vel_topic:=/{robot}/nav_cmd_vel",
                ])
                if index == 1:
                    _wait_topic_message(
                        "/merged_map", "nav_msgs/msg/OccupancyGrid",
                        startup_timeout, require_current_world, processes,
                    )
                    report("received first valid /merged_map message")
                wait_current_graph(
                    "action", f"/{robot}/navigate_to_pose", startup_timeout
                )
                report(f"{robot} NavigateToPose action ready")
        if execution.rviz:
            from ament_index_python.packages import get_package_share_directory

            rviz_config = (
                Path(get_package_share_directory("go2_mapping_nav"))
                / "rviz/three_go2_mapping_nav.rviz"
            )
            processes.start("rviz", [
                "rviz2", "-d", str(rviz_config),
                "--ros-args", "-p", "use_sim_time:=true",
            ])
        processes.start("perception", [
            "ros2", "launch", "go2_target_perception",
            "three_go2_target_tracking.launch.py", "use_sim_time:=true",
            f"model_path:={model_path}",
        ])
        selected_robot = bounded_worker(
            processes, "role", {"timeout_sec": case.settings["role_timeout_sec"]},
            case.settings["role_timeout_sec"], require_current_world,
        )
        if execution.rqt:
            processes.start("rqt", [
                "ros2", "run", "rqt_image_view", "rqt_image_view",
                f"/{selected_robot}/target_perception/debug_image",
            ])

        if case.task_type == "tracking":
            resolved_scene = Path(case_config).with_name(
                "resolved_scene_config.yaml"
            )
            processes.start("dynamic_encircle", [
                "ros2", "run", "go2_dynamic_encircle", "dynamic_encircle",
                "--ros-args", "-p", "use_sim_time:=true",
                "-p", f"scene:={case.scene}",
                "-p", f"scene_config:={resolved_scene}",
                "-p", "perception_robot_topic:=/target_role/perception_robot",
                "-p", "robot_names:=[go2_1,go2_2,go2_3]",
            ])
            recorder_executable = "tracking_test_recorder"
        elif case.task_type == "path_planning":
            resolved_scene = Path(case_config).with_name(
                "resolved_scene_config.yaml"
            )
            recorder_executable = "path_planning_recorder"
            recorder = processes.start("recorder", [
                "ros2", "run", "go2_test_framework", recorder_executable,
                "--ros-args", "-p", "use_sim_time:=true",
                "-p", f"case_config:={case_config}",
                "-p", f"output_dir:={attempt_dir}",
            ])
            if case.settings.get("inject_collision_probe", False):
                collision_probe = processes.start(
                    "collision_probe", [
                        "ros2", "run", "go2_test_framework",
                        "collision_probe_spawner", "--ros-args",
                        "-p", "use_sim_time:=true",
                        "-p", "robot:=go2_3",
                        "-p", "model_name:=phase6_test_box",
                    ],
                    mirror_to_console=True,
                    console_prefix=f"{prefix} ",
                )
                try:
                    collision_probe_status = collision_probe.wait(timeout=30.0)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError("collision probe timed out") from error
                if collision_probe_status != 0:
                    raise RuntimeError(
                        "collision probe exited with status "
                        f"{collision_probe_status}"
                    )
                report("temporary trunk-overlap collision box spawned")
            processes.start("dynamic_encircle", [
                "ros2", "run", "go2_dynamic_encircle", "dynamic_encircle",
                "--ros-args", "-p", "use_sim_time:=true",
                "-p", f"scene:={case.scene}",
                "-p", f"scene_config:={resolved_scene}",
                "-p", "perception_robot_topic:=/target_role/perception_robot",
                "-p", "robot_names:=[go2_1,go2_2,go2_3]",
                "-p", (
                    "arrival_hold_duration:="
                    f"{float(case_metrics['path_timeout_sec']) + 1.0}"
                ),
            ])
        elif case.task_type == "perception":
            recorder_executable = "target_test_recorder"
        else:
            raise RuntimeError(
                f"no Attempt workflow registered for {case.task_type!r}"
            )
        if case.task_type != "path_planning":
            recorder = processes.start("recorder", [
                "ros2", "run", "go2_test_framework", recorder_executable,
                "--ros-args", "-p", "use_sim_time:=true",
                "-p", f"case_config:={case_config}",
                "-p", f"output_dir:={attempt_dir}",
            ])
        require_current_world()
        from go2_test_framework.runner.ros_wait import WalkingTargetStartObservation
        target_start_observation = WalkingTargetStartObservation(**bounded_worker(
            processes, "walking_target", {}, 15.0, require_current_world,
        ))
        report(
            "walking target movement confirmed: displacement="
            f"{target_start_observation.displacement_m:.3f}m, requests="
            f"{target_start_observation.start_requests_sent}"
        )
        if case.task_type == "tracking":
            recorder_timeout = (
                case.settings["startup_timeout_sec"]
                + float(case_metrics["acquisition_timeout_sec"])
                + float(case_metrics["tracking_min_duration_sec"])
                + 30.0
            )
        elif case.task_type == "path_planning":
            recorder_timeout = (
                case.settings["startup_timeout_sec"]
                + float(case_metrics["path_timeout_sec"])
                + 30.0
            )
        else:
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
        summary["walking_target_start"] = target_start_observation.to_dict()
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
    except BatchSafetyError:
        raise
    except Exception as error:
        report(f"INFRASTRUCTURE FAILED: {error}")
        reason = str(error)
        summary = _failure_summary_preserving_metrics(
            attempt_dir, reason, case
        )
        if isinstance(error, WalkingTargetStartError):
            target_start_observation = error.observation
        if target_start_observation is not None:
            summary["walking_target_start"] = (
                target_start_observation.to_dict()
            )
        write_yaml(attempt_dir / "case_summary.yaml", summary)
        return AttemptResult(
            attempt_number, AttemptStatus.INFRASTRUCTURE_FAILED, reason, summary
        )
    finally:
        cleanup = processes.stop()
        report("cleanup complete")
        if cleanup and cleanup.get("received_signals"):
            import signal
            from go2_test_framework.runner.runtime import ShutdownRequested
            raise ShutdownRequested(getattr(signal, cleanup["received_signals"][0]))


def should_retry(result, max_restarts):
    """Return whether another attempt is allowed for this result."""
    return result.retryable and result.number <= max_restarts


def run_case(
    case, share, case_dir, model_path, metrics, execution, *,
    run_id="unmarked", attempt_runner=run_attempt, sleep=time.sleep,
    reporter=print, scene_config_root=None,
):
    """Run a Case with a shared retry budget and write its root summary."""
    case_dir = Path(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    if scene_config_root is None:
        from ament_index_python.packages import get_package_share_directory
        scene_config_root = (
            Path(get_package_share_directory("go2_scenario_config"))
            / "config/scenes"
        )
    resolved_scene_config = case_dir / "resolved_scene_config.yaml"
    write_resolved_scene_config(
        case,
        Path(scene_config_root) / f"{case.scene}.yaml",
        resolved_scene_config,
    )
    case_value = {
        "schema_version": 1,
        **case.to_dict(),
        "resolved_scene_config": str(resolved_scene_config.resolve()),
        "provisional": case.route_config["provisional"],
        "metrics": metrics,
        "execution": execution.to_dict(),
    }
    case_config = case_dir / "case_config.yaml"
    write_yaml(case_config, case_value)
    attempts_dir = case_dir / "attempts"
    attempts_dir.mkdir(exist_ok=True)

    attempts = []
    total_attempts = execution.attitude_check.max_restarts + 1
    while True:
        number = len(attempts) + 1
        reporter(
            f"[{case.case_id}] Attempt {number}/{total_attempts} starting",
            flush=True,
        )
        try:
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
        except BatchSafetyError as error:
            summary = _failure_summary_preserving_metrics(
                attempts_dir / f"attempt_{number:02d}", str(error), case,
            )
            summary["abort_batch"] = error.reason
            write_yaml(attempts_dir / f"attempt_{number:02d}" / "case_summary.yaml", summary)
            result = AttemptResult(number, AttemptStatus.INFRASTRUCTURE_FAILED, str(error), summary)
        except (KeyboardInterrupt, ShutdownRequested) as error:
            summary = _failure_summary_preserving_metrics(
                attempts_dir / f"attempt_{number:02d}", "runner interrupted", case,
            )
            summary["abort_batch"] = "interrupted"
            summary["exit_status"] = 128 + (error.signum if isinstance(error, ShutdownRequested) else 2)
            write_yaml(attempts_dir / f"attempt_{number:02d}" / "case_summary.yaml", summary)
            result = AttemptResult(number, AttemptStatus.INFRASTRUCTURE_FAILED, "runner interrupted", summary)
        attempts.append(result)
        if result.summary.get("abort_batch") or not should_retry(result, execution.attitude_check.max_restarts):
            break
        delay = execution.attitude_check.restart_delay_sec
        reporter(
            f"[Attempt {number}/{total_attempts}] {result.status.value}; "
            "restart budget "
            f"{number}/{execution.attitude_check.max_restarts}; "
            f"restarting in {delay:.1f}s",
            flush=True,
        )
        sleep(delay)

    final = attempts[-1]
    restart_exhausted = (
        final.retryable
        and not final.summary.get("abort_batch")
        and final.number > execution.attitude_check.max_restarts
    )
    if final.status is AttemptStatus.COMPLETED:
        summary = dict(final.summary)
        case_status = AttemptStatus.COMPLETED.value
    else:
        reason = final.reason or "unknown infrastructure failure"
        if restart_exhausted:
            if final.status is AttemptStatus.FALLEN:
                reason = (
                    "attitude check remained fallen after "
                    f"{execution.attitude_check.max_restarts} restart(s)"
                )
            else:
                reason = (
                    "infrastructure failure persisted after "
                    f"{execution.attitude_check.max_restarts} restart(s): "
                    f"{reason}"
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
