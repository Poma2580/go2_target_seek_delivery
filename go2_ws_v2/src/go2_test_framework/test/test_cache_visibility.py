import math

import pytest

from go2_test_framework.ground_truth.visibility import CameraIntrinsics, project_camera_point, quaternion_conjugate_rotate
from go2_test_framework.recorders.cache import TimeCache


def test_consumable_cache_uses_nearest_sample_only_once():
    cache = TimeCache(consumable=True)
    cache.append(1.0, "first")
    cache.append(1.1, "second")
    assert cache.nearest(1.04, 0.15)[1] == "first"
    assert cache.nearest(1.04, 0.15)[1] == "second"
    assert cache.nearest(1.04, 0.15) is None


def test_camera_projection_uses_camera_info_bounds_and_depth():
    info = CameraIntrinsics(100.0, 100.0, 50.0, 40.0, 100, 80)
    assert project_camera_point((0.0, 0.0, 2.0), info, 0.3, 25.0)[0] is True
    assert project_camera_point((1.0, 0.0, 1.0), info, 0.3, 25.0)[0] is False
    assert project_camera_point((0.0, 0.0, 0.2), info, 0.3, 25.0)[0] is False


def test_inverse_quaternion_rotation_moves_world_x_into_robot_right_frame():
    half = math.pi / 4.0
    rotated = quaternion_conjugate_rotate((1.0, 0.0, 0.0), (0.0, 0.0, math.sin(half), math.cos(half)))
    assert rotated == pytest.approx((0.0, -1.0, 0.0), abs=1e-7)
