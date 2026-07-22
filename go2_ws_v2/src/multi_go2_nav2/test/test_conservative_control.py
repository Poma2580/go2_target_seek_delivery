from pathlib import Path
import xml.etree.ElementTree as ET

import pytest
import yaml


PACKAGE_ROOT = Path(__file__).parents[1]


def test_go2_controller_limits_are_conservative():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'nav2_params.yaml').read_text())
    controller = config['controller_server']['ros__parameters']
    goal = controller['general_goal_checker']
    pursuit = controller['FollowPath']

    assert controller['controller_frequency'] <= 10.0
    assert controller['progress_checker']['movement_time_allowance'] >= 20.0
    assert goal['xy_goal_tolerance'] >= 0.4
    assert goal['yaw_goal_tolerance'] >= 0.4
    assert pursuit['desired_linear_vel'] <= 0.25
    assert pursuit['rotate_to_heading_angular_vel'] <= 0.25
    assert pursuit['max_angular_accel'] <= 0.5


def test_velocity_smoother_is_open_loop_and_rate_limited():
    config = yaml.safe_load(
        (PACKAGE_ROOT / 'config' / 'nav2_params.yaml').read_text())
    smoother = config['velocity_smoother']['ros__parameters']

    assert smoother['feedback'] == 'OPEN_LOOP'
    assert smoother['max_velocity'] == pytest.approx([0.30, 0.0, 0.35])
    assert smoother['min_velocity'] == pytest.approx([-0.10, 0.0, -0.35])
    assert smoother['max_accel'] == pytest.approx([0.30, 0.0, 0.50])
    assert smoother['max_decel'] == pytest.approx([-0.40, 0.0, -0.60])
    assert smoother['velocity_timeout'] <= 0.5


def test_default_navigation_tree_has_no_motion_recovery_actions():
    tree_file = PACKAGE_ROOT.joinpath(
        'behavior_trees', 'navigate_to_pose_no_recovery.xml')
    tags = {element.tag for element in ET.parse(tree_file).iter()}

    assert 'ComputePathToPose' in tags
    assert 'FollowPath' in tags
    assert 'RecoveryNode' not in tags
    assert 'Spin' not in tags
    assert 'BackUp' not in tags
