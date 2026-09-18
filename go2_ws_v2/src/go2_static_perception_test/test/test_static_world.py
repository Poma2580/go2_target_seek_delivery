from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose
import pytest
import yaml

from go2_static_perception_test.validate_world import (
    load_targets, PoseCheck, quaternion, errors, WorldValidator,
)

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE / 'config/targets/static_targets.yaml'
WORLD = PACKAGE / 'worlds/city_static_objects.world'


def state_message(targets):
    message = ModelStates()
    for target in targets.values():
        message.name.append(target['model_name'])
        data = target['pose']
        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = (data[k] for k in ('x','y','z'))
        q = quaternion(*(data[k] for k in ('roll','pitch','yaw')))
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q
        message.pose.append(pose)
    return message


def test_world_yaml_and_static_contract():
    targets = load_targets(CONFIG)
    world = ET.parse(WORLD).getroot().find('world')
    assert world.find('state') is None
    assert world.find('actor') is None
    assert 'walking_target' not in WORLD.read_text()
    assert world.find("model[@name='dumpster_94']") is not None
    for target in targets.values():
        model = world.find(f"model[@name='{target['model_name']}']")
        assert model is not None
        assert model.findtext('static') in ('true', '1')
        assert list(map(float, model.findtext('pose').split())) == list(target['pose'].values())
        assert not model.findall('.//sensor') and not model.findall('.//plugin')
    assert world.find("plugin[@filename='libgazebo_ros_state.so']") is not None
    person = world.find("model[@name='static_person']")
    assert person.findtext('static') in ('true', '1')
    assert person.findtext('link/visual/pose') == '0 0 -0.02 0 0 0'
    person_uris = [node.text for node in person.findall('.//mesh/uri')]
    assert person_uris == [
        'model://person_walking/meshes/walking.dae',
        'model://person_walking/meshes/walking.dae',
    ]


@pytest.mark.parametrize('fault', ['missing', 'prompt', 'duplicate', 'nan', 'pose_missing', 'version'])
def test_invalid_target_config(tmp_path, fault):
    value = yaml.safe_load(CONFIG.read_text())
    targets = value['targets']
    if fault == 'missing':
        targets.pop('person')
    elif fault == 'prompt':
        targets['ground_robot']['prompt'] = 'robot'
    elif fault == 'duplicate':
        targets['person']['model_name'] = targets['construction_barrel']['model_name']
    elif fault == 'nan':
        targets['person']['pose']['z'] = float('nan')
    elif fault == 'pose_missing':
        targets['person']['pose'].pop('yaw')
    else:
        value['schema_version'] = 2
    file = tmp_path / 'targets.yaml'
    file.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError):
        load_targets(file)


def test_still_poses_and_equivalent_quaternion_sign():
    targets = load_targets(CONFIG)
    message = state_message(targets)
    check = PoseCheck(targets)
    check.update(message)
    pose = message.pose[0]
    for key in ('x', 'y', 'z', 'w'):
        setattr(pose.orientation, key, -getattr(pose.orientation, key))
    check.update(message)
    assert check.samples == 2
    assert max(v['max_displacement_m'] for v in check.report.values()) == 0


@pytest.mark.parametrize('fault', ['walker', 'missing', 'duplicate', 'position', 'rotation', 'nan'])
def test_validator_rejects_bad_world(fault):
    targets = load_targets(CONFIG)
    message = state_message(targets)
    check = PoseCheck(targets)
    check.update(message)
    if fault == 'walker':
        message.name.append('walking_target')
        message.pose.append(Pose())
    elif fault == 'missing':
        message.name.pop()
        message.pose.pop()
    elif fault == 'duplicate':
        message.name.append(message.name[0])
        message.pose.append(message.pose[0])
    elif fault == 'position':
        message.pose[0].position.x += .02
    elif fault == 'rotation':
        message.pose[0].orientation.w = .9
    elif fault == 'nan':
        message.pose[0].position.x = float('nan')
    with pytest.raises(ValueError):
        check.update(message)


def test_motion_within_yaml_tolerance_still_compared_to_initial():
    targets = load_targets(CONFIG)
    message = state_message(targets)
    check = PoseCheck(targets)
    message.pose[0].position.x -= .009
    check.update(message)
    message.pose[0].position.x += .018
    with pytest.raises(ValueError, match='moved'):
        check.update(message)


@pytest.mark.parametrize('case,expected', [('timeout', 'wall timeout'),
                                          ('messages', 'messages stopped'),
                                          ('clock', 'clock stopped')])
def test_wall_watchdogs(monkeypatch, case, expected):
    from types import SimpleNamespace
    from go2_static_perception_test import validate_world as module
    node = WorldValidator.__new__(WorldValidator)
    node.args = SimpleNamespace(timeout=120., stall_timeout=10.)
    node.started = 0. if case == 'timeout' else 100.
    node.last_message = 100. if case == 'messages' else 125.
    node.last_progress = 100. if case == 'clock' else 125.
    node.first_sim = 1.
    node.done, node.failure = False, None
    monkeypatch.setattr(module.time, 'monotonic', lambda: 125.)
    node.watchdog()
    assert node.done and expected in node.failure
