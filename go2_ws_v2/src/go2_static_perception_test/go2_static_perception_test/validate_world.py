"""Check five model poses over simulation time, with wall-clock watchdogs."""
import argparse
import json
import math
from pathlib import Path
import time

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.msg import ModelStates
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
import yaml


PROMPTS = {'airplane': 'airplane', 'person': 'person', 'pickup_truck': 'pickup truck',
           'ground_robot': 'ground robot', 'dumpster': 'dumpster'}
POSE_FIELDS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw')


def load_targets(path):
    config = yaml.safe_load(Path(path).read_text())
    if not isinstance(config, dict) or config.get('schema_version') != 1:
        raise ValueError('expected schema_version: 1')
    targets = config.get('targets')
    if not isinstance(targets, dict) or set(targets) != set(PROMPTS):
        raise ValueError('exactly five official target keys required')
    names = []
    for key, prompt in PROMPTS.items():
        value = targets[key]
        if not isinstance(value, dict) or value.get('prompt') != prompt:
            raise ValueError(f'{key}: incorrect prompt')
        name = value.get('model_name')
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f'{key}: model_name required')
        names.append(name)
        pose = value.get('pose')
        if not isinstance(pose, dict) or set(pose) != set(POSE_FIELDS):
            raise ValueError(f'{key}: six pose components required')
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                   and math.isfinite(v) for v in pose.values()):
            raise ValueError(f'{key}: pose must be finite numeric values')
    if len(set(names)) != 5:
        raise ValueError('model_name values must be unique')
    return targets


def quaternion(roll, pitch, yaw):
    sr, cr = math.sin(roll/2), math.cos(roll/2)
    sp, cp = math.sin(pitch/2), math.cos(pitch/2)
    sy, cy = math.sin(yaw/2), math.cos(yaw/2)
    return (sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy,
            cr*cp*sy-sr*sp*cy, cr*cp*cy+sr*sp*sy)


def pose_values(pose):
    p, q = pose.position, pose.orientation
    result = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(v) for v in result):
        raise ValueError('non-finite model pose')
    return result


def errors(actual, reference):
    distance = math.dist(actual[:3], reference[:3])
    a, b = actual[3:], reference[3:]
    norm = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    if norm < 1e-12:
        raise ValueError('zero quaternion')
    dot = abs(sum(x*y for x, y in zip(a, b))) / norm
    return distance, 2*math.acos(min(1.0, dot))


class PoseCheck:
    """Pure validation logic, also used by the live subscriber."""
    def __init__(self, targets, position_tolerance=0.01, angle_tolerance=0.01):
        self.targets = targets
        self.position_tolerance = position_tolerance
        self.angle_tolerance = angle_tolerance
        self.initial = None
        self.report = {}
        self.samples = 0

    def update(self, message):
        if len(message.name) != len(message.pose):
            raise ValueError('malformed model_states')
        if 'walking_target' in message.name:
            raise ValueError('walking_target must not exist')
        actual = {}
        for key, value in self.targets.items():
            name = value['model_name']
            if message.name.count(name) != 1:
                raise ValueError(f'{name}: missing or duplicate model')
            actual[key] = pose_values(message.pose[message.name.index(name)])
            pose = value['pose']
            expected = tuple(pose[k] for k in ('x', 'y', 'z')) + quaternion(
                pose['roll'], pose['pitch'], pose['yaw'])
            position, angle = errors(actual[key], expected)
            if position > self.position_tolerance or angle > self.angle_tolerance:
                raise ValueError(f'{name}: YAML pose mismatch: {position:.6g}m, {angle:.6g}rad')
            if self.initial is not None:
                position, angle = errors(actual[key], self.initial[key])
            else:
                position, angle = 0.0, 0.0
            self.report.setdefault(key, {'model_name': name, 'initial_pose_xyz_xyzw': actual[key],
                                        'max_displacement_m': 0.0, 'max_rotation_rad': 0.0})
            entry = self.report[key]
            entry['max_displacement_m'] = max(entry['max_displacement_m'], position)
            entry['max_rotation_rad'] = max(entry['max_rotation_rad'], angle)
            if position > self.position_tolerance or angle > self.angle_tolerance:
                raise ValueError(f'{name}: moved: {position:.6g}m, {angle:.6g}rad')
        if self.initial is None:
            self.initial = actual
        self.samples += 1


class WorldValidator(Node):
    def __init__(self, args, targets):
        super().__init__('validate_static_world', parameter_overrides=[
            Parameter('use_sim_time', value=True)])
        self.args = args
        self.check = PoseCheck(targets, args.position_tolerance, args.angle_tolerance)
        self.started = time.monotonic()
        self.last_message = self.started
        self.last_progress = self.started
        self.first_sim = None
        self.last_sim = None
        self.elapsed_sim = 0.0
        self.done = False
        self.failure = None
        self.subscription = self.create_subscription(ModelStates, args.topic, self.callback, 10)

    def callback(self, message):
        if self.done:
            return
        now = time.monotonic()
        self.last_message = now
        names = [v['model_name'] for v in self.check.targets.values()]
        # Allow Gazebo to finish initial model insertion, but never ignore a walker.
        if self.first_sim is None and 'walking_target' not in message.name:
            if not all(name in message.name for name in names):
                return
        try:
            self.check.update(message)
            sim = self.get_clock().now().nanoseconds * 1e-9
            if sim <= 0:
                return
            if self.last_sim is not None and sim < self.last_sim:
                raise ValueError('simulation clock reset during validation')
            if self.first_sim is None:
                self.first_sim = sim
            if self.last_sim is None or sim > self.last_sim:
                self.last_progress = now
            self.last_sim = sim
            self.elapsed_sim = sim-self.first_sim
            self.done = self.elapsed_sim >= self.args.duration
        except ValueError as error:
            self.failure, self.done = str(error), True

    def watchdog(self):
        now = time.monotonic()
        if now-self.started > self.args.timeout:
            self.failure = 'wall timeout waiting for models/validation completion'
        elif self.first_sim is not None and now-self.last_message > self.args.stall_timeout:
            self.failure = 'model_states messages stopped'
        elif self.first_sim is not None and now-self.last_progress > self.args.stall_timeout:
            self.failure = 'simulation clock stopped'
        self.done = self.done or self.failure is not None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets', default=str(Path(get_package_share_directory(
        'go2_static_perception_test')) / 'config/targets/static_targets.yaml'))
    parser.add_argument('--topic', default='/gazebo/model_states')
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--timeout', type=float, default=120.0)
    parser.add_argument('--stall-timeout', type=float, default=10.0)
    parser.add_argument('--position-tolerance', type=float, default=0.01)
    parser.add_argument('--angle-tolerance', type=float, default=0.01)
    args = parser.parse_args(argv)
    for key in ('duration', 'timeout', 'stall_timeout', 'position_tolerance', 'angle_tolerance'):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f'{key} must be finite and positive')
    targets = load_targets(args.targets)
    rclpy.init(args=[])
    node = WorldValidator(args, targets)
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
            node.watchdog()
        print(json.dumps({'success': node.done and node.failure is None,
                          'failure': node.failure, 'samples': node.check.samples,
                          'simulation_duration': node.elapsed_sim,
                          'targets': node.check.report}, indent=2, allow_nan=False))
        result = 0 if node.done and node.failure is None else 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(result)


if __name__ == '__main__':
    main()
