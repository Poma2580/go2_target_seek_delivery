"""Unit tests for the ROS-independent Go2 attitude-checking logic."""

import math

import pytest
from geometry_msgs.msg import Quaternion


from go2_scenario_config.check_three_go2_attitude import (
    EulerDegrees,
    RollObservationChecker,
    quaternion_to_euler_degrees,
)

ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


def attitude(roll: float = 0.0, pitch: float = 0.0, yaw: float = 0.0):
    return EulerDegrees(roll=roll, pitch=pitch, yaw=yaw)


def frame(**rolls: float):
    return {name: attitude(roll=rolls.get(name, 0.0)) for name in ROBOT_NAMES}


@pytest.mark.parametrize("roll", [0.0, 89.9, 90.0, -89.9, -90.0])
def test_roll_at_or_below_limit_passes(roll):
    checker = RollObservationChecker(ROBOT_NAMES, 90.0, 5)
    for _ in range(5):
        checker.add_frame(frame(go2_1=roll))
    assert checker.complete
    assert not checker.fallen_robots


@pytest.mark.parametrize("fallen_frame", range(1, 6))
@pytest.mark.parametrize("roll", [90.1, -90.1])
def test_any_over_limit_frame_fails_immediately(fallen_frame, roll):
    checker = RollObservationChecker(ROBOT_NAMES, 90.0, 5)
    for _ in range(fallen_frame - 1):
        checker.add_frame(frame())
    checker.add_frame(frame(go2_2=roll))
    assert checker.fallen_robots == {"go2_2"}
    assert checker.valid_frames == fallen_frame
    assert checker.complete


def test_multiple_fallen_robots_in_one_frame_are_reported():
    checker = RollObservationChecker(ROBOT_NAMES, 90.0, 5)
    checker.add_frame(frame(go2_1=91.0, go2_3=-91.0))
    assert checker.fallen_robots == {"go2_1", "go2_3"}


def test_pitch_and_yaw_do_not_affect_roll_decision():
    checker = RollObservationChecker(ROBOT_NAMES, 90.0, 5)
    angled = {name: attitude(pitch=120.0, yaw=180.0) for name in ROBOT_NAMES}
    for _ in range(5):
        checker.add_frame(angled)
    assert checker.complete
    assert not checker.fallen_robots


def test_incomplete_frame_is_rejected():
    checker = RollObservationChecker(ROBOT_NAMES, 90.0, 5)
    with pytest.raises(ValueError, match="missing"):
        checker.add_frame({"go2_1": attitude()})


@pytest.mark.parametrize(
    ("roll", "pitch", "yaw"),
    [(45.0, 0.0, 0.0), (0.0, -30.0, 0.0), (0.0, 0.0, 135.0)],
)
def test_quaternion_conversion(roll, pitch, yaw):
    half_roll = math.radians(roll) / 2.0
    half_pitch = math.radians(pitch) / 2.0
    half_yaw = math.radians(yaw) / 2.0
    quaternion = Quaternion(
        x=(
            math.sin(half_roll) * math.cos(half_pitch) * math.cos(half_yaw)
            - math.cos(half_roll) * math.sin(half_pitch) * math.sin(half_yaw)
        ),
        y=(
            math.cos(half_roll) * math.sin(half_pitch) * math.cos(half_yaw)
            + math.sin(half_roll) * math.cos(half_pitch) * math.sin(half_yaw)
        ),
        z=(
            math.cos(half_roll) * math.cos(half_pitch) * math.sin(half_yaw)
            - math.sin(half_roll) * math.sin(half_pitch) * math.cos(half_yaw)
        ),
        w=(
            math.cos(half_roll) * math.cos(half_pitch) * math.cos(half_yaw)
            + math.sin(half_roll) * math.sin(half_pitch) * math.sin(half_yaw)
        ),
    )
    result = quaternion_to_euler_degrees(quaternion)
    assert result.roll == pytest.approx(roll)
    assert result.pitch == pytest.approx(pitch)
    assert result.yaw == pytest.approx(yaw)
