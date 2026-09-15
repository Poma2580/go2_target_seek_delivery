"""Role mapping tests for the command mux."""

import pytest

from go2_dynamic_encircle import follower_cmd_vel_mux as MODULE


@pytest.mark.parametrize(
    "perception, followers",
    [("go2_1", ("go2_2", "go2_3")), ("go2_2", ("go2_1", "go2_3")),
     ("go2_3", ("go2_1", "go2_2"))],
)
def test_mux_excludes_perception_dog(perception, followers):
    assert MODULE.navigation_dogs(MODULE.ROBOT_NAMES, perception) == followers
