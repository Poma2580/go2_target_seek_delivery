"""Role mapping tests for the command mux."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts/follower_cmd_vel_mux.py"
SPEC = importlib.util.spec_from_file_location("follower_cmd_vel_mux", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    "perception, followers",
    [("go2_1", ("go2_2", "go2_3")), ("go2_2", ("go2_1", "go2_3")),
     ("go2_3", ("go2_1", "go2_2"))],
)
def test_mux_excludes_perception_dog(perception, followers):
    assert MODULE.navigation_dogs(MODULE.ROBOT_NAMES, perception) == followers
