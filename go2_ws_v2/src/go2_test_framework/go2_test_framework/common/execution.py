"""Validated runner execution policy and CLI override handling."""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class AttitudeCheckConfig:
    enabled: bool
    settle_delay_sec: float
    roll_limit_deg: float
    sample_frames: int
    timeout_sec: float
    max_restarts: int
    restart_delay_sec: float


@dataclass(frozen=True)
class RobotStartupConfig:
    world_to_first_delay_sec: float
    inter_robot_delay_sec: float
    enable_lidar: bool


@dataclass(frozen=True)
class ExecutionConfig:
    gazebo_gui: bool
    rqt: bool
    robot_startup: RobotStartupConfig
    attitude_check: AttitudeCheckConfig

    def to_dict(self):
        return asdict(self)


def _boolean(value, where):
    if not isinstance(value, bool):
        raise ValueError(f"{where} must be boolean")
    return value


def _positive_float(value, where, *, allow_zero=False):
    if isinstance(value, bool):
        raise ValueError(f"{where} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where} must be numeric") from error
    valid = parsed >= 0.0 if allow_zero else parsed > 0.0
    if not math.isfinite(parsed) or not valid:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{where} must be finite and {relation}")
    return parsed


def _non_negative_int(value, where):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative integer")
    return value


def execution_from_mapping(raw):
    """Parse a complete suite ``execution`` mapping."""
    if not isinstance(raw, dict):
        raise ValueError("execution must be a mapping")
    expected = {"gazebo_gui", "rqt", "robot_startup", "attitude_check"}
    if set(raw) != expected:
        raise ValueError(
            f"execution must contain exactly {sorted(expected)}"
        )
    startup = raw["robot_startup"]
    if not isinstance(startup, dict):
        raise ValueError("execution.robot_startup must be a mapping")
    startup_expected = {
        "world_to_first_delay_sec", "inter_robot_delay_sec", "enable_lidar",
    }
    if set(startup) != startup_expected:
        raise ValueError(
            "execution.robot_startup must contain exactly "
            f"{sorted(startup_expected)}"
        )

    attitude = raw["attitude_check"]
    if not isinstance(attitude, dict):
        raise ValueError("execution.attitude_check must be a mapping")
    attitude_expected = {
        "enabled", "settle_delay_sec", "roll_limit_deg", "sample_frames",
        "timeout_sec", "max_restarts", "restart_delay_sec",
    }
    if set(attitude) != attitude_expected:
        raise ValueError(
            "execution.attitude_check must contain exactly "
            f"{sorted(attitude_expected)}"
        )
    sample_frames = attitude["sample_frames"]
    if (
        isinstance(sample_frames, bool)
        or not isinstance(sample_frames, int)
        or sample_frames <= 0
    ):
        raise ValueError(
            "execution.attitude_check.sample_frames must be a positive integer"
        )
    return ExecutionConfig(
        gazebo_gui=_boolean(raw["gazebo_gui"], "execution.gazebo_gui"),
        rqt=_boolean(raw["rqt"], "execution.rqt"),
        robot_startup=RobotStartupConfig(
            world_to_first_delay_sec=_positive_float(
                startup["world_to_first_delay_sec"],
                "execution.robot_startup.world_to_first_delay_sec",
                allow_zero=True,
            ),
            inter_robot_delay_sec=_positive_float(
                startup["inter_robot_delay_sec"],
                "execution.robot_startup.inter_robot_delay_sec",
                allow_zero=True,
            ),
            enable_lidar=_boolean(
                startup["enable_lidar"],
                "execution.robot_startup.enable_lidar",
            ),
        ),
        attitude_check=AttitudeCheckConfig(
            enabled=_boolean(
                attitude["enabled"], "execution.attitude_check.enabled"
            ),
            settle_delay_sec=_positive_float(
                attitude["settle_delay_sec"],
                "execution.attitude_check.settle_delay_sec",
                allow_zero=True,
            ),
            roll_limit_deg=_positive_float(
                attitude["roll_limit_deg"],
                "execution.attitude_check.roll_limit_deg",
            ),
            sample_frames=sample_frames,
            timeout_sec=_positive_float(
                attitude["timeout_sec"],
                "execution.attitude_check.timeout_sec",
            ),
            max_restarts=_non_negative_int(
                attitude["max_restarts"],
                "execution.attitude_check.max_restarts",
            ),
            restart_delay_sec=_positive_float(
                attitude["restart_delay_sec"],
                "execution.attitude_check.restart_delay_sec",
                allow_zero=True,
            ),
        ),
    )


def apply_execution_overrides(
    config, *, gazebo_gui=None, rqt=None, attitude_enabled=None,
    max_restarts=None, enable_lidar=None,
):
    """Return a new config with explicitly supplied CLI values applied."""
    if max_restarts is not None:
        _non_negative_int(max_restarts, "--max-restarts")
    attitude = config.attitude_check
    return ExecutionConfig(
        gazebo_gui=config.gazebo_gui if gazebo_gui is None else gazebo_gui,
        rqt=config.rqt if rqt is None else rqt,
        robot_startup=RobotStartupConfig(
            world_to_first_delay_sec=(
                config.robot_startup.world_to_first_delay_sec
            ),
            inter_robot_delay_sec=config.robot_startup.inter_robot_delay_sec,
            enable_lidar=(
                config.robot_startup.enable_lidar
                if enable_lidar is None
                else enable_lidar
            ),
        ),
        attitude_check=AttitudeCheckConfig(
            enabled=(
                attitude.enabled
                if attitude_enabled is None
                else attitude_enabled
            ),
            settle_delay_sec=attitude.settle_delay_sec,
            roll_limit_deg=attitude.roll_limit_deg,
            sample_frames=attitude.sample_frames,
            timeout_sec=attitude.timeout_sec,
            max_restarts=(
                attitude.max_restarts if max_restarts is None else max_restarts
            ),
            restart_delay_sec=attitude.restart_delay_sec,
        ),
    )
