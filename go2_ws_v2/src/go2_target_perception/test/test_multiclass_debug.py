"""Exercise debug rendering and the pedestrian path without loading YOLO."""

from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest
from cv_bridge import CvBridge
from geometry_msgs.msg import Point

from go2_target_perception.target_perception import TargetPerception


NAMES = {0: 'person', 2: 'car', 4: 'airplane', 5: 'bus', 7: 'truck',
         10: 'fire hydrant', 11: 'stop sign', 13: 'bench'}


def box(cls, coords=(10, 40, 30, 80)):
    return NS(cls=np.array([cls]), conf=np.array([0.85]),
              xyxy=np.array([coords]))


def probe(boxes):
    node = TargetPerception.__new__(TargetPerception)
    images, positions, statuses, warnings, calls = [], [], [], [], []
    node.bridge = CvBridge()
    node.dbg_pub = NS(publish=images.append)
    node._warn = warnings.append
    node.get_logger = lambda: NS(info=lambda *a, **kw: None)
    node.K = (100., 100., 50., 50.)
    node._yolo_kwargs = {}
    results = [NS(boxes=boxes, names=NAMES)]

    def infer(*args, **kwargs):
        calls.append(1)
        return results

    node.model = infer
    node._sample_depth_from_bbox = lambda *a: (2., 'roi')
    node.tf_buffer = NS(transform=lambda *a, **kw: NS(point=Point(x=1., y=2., z=3.)))
    node.target_frame = 'go2_1/odom'
    node.tf_timeout = 0.1
    node.vx = node.vy = 0.
    node._publish = lambda *a: positions.append(a)
    node._next_sample_id = 0
    node._publish_result_status = lambda *a, **kw: statuses.append(kw)
    rgb = node.bridge.cv2_to_imgmsg(np.zeros((100, 100, 3), np.uint8), 'bgr8')
    depth = node.bridge.cv2_to_imgmsg(np.ones((100, 100), np.float32), '32FC1')
    rgb.header.frame_id = depth.header.frame_id = 'camera'
    return NS(node=node, images=images, positions=positions, statuses=statuses,
              warnings=warnings, calls=calls, results=results, rgb=rgb, depth=depth)


def test_all_classes_drawn_and_target_overlaid_last(monkeypatch):
    p = probe([box(cls) for cls in NAMES])
    labels, colors = [], []
    monkeypatch.setattr(cv2, 'putText', lambda img, label, *a: labels.append(label))
    monkeypatch.setattr(cv2, 'rectangle', lambda img, p1, p2, color, *a: colors.append(color))
    p.node._process_rgbd_pair(p.rgb, p.depth)
    assert labels[:-1] == [f'{name} 0.85' for name in NAMES.values()]
    assert labels[-1] == 'person 0.85 Z=2.00m roi (1.0,2.0)'
    assert colors == [(255, 255, 0)] * 8 + [(0, 255, 0)]
    assert len(p.images) == len(p.positions) == len(p.statuses) == len(p.calls) == 1
    assert p.images[0].header == p.rgb.header
    assert p.images[0].encoding == 'bgr8'
    assert p.statuses[0]['localization_success'] is True


def test_larger_object_never_selected_over_largest_person():
    p = probe([box(0), box(2, (0, 0, 100, 100)), box(0, (5, 5, 40, 80))])
    assert p.node._pick_person(p.results) == (5, 5, 40, 80, 0.85)


def test_objects_without_person_only_publish_debug(monkeypatch):
    p = probe([box(2)])
    labels = []
    monkeypatch.setattr(cv2, 'putText', lambda img, label, *a: labels.append(label))
    p.node._process_rgbd_pair(p.rgb, p.depth)
    assert labels == ['car 0.85', 'no person']
    assert len(p.images) == len(p.statuses) == len(p.calls) == 1
    assert not p.positions
    assert p.statuses[0]['recognition_success'] is False
    assert p.statuses[0]['localization_success'] is False


@pytest.mark.parametrize('failure', ['depth', 'tf'])
def test_localization_failure_still_draws_objects(failure, monkeypatch):
    p = probe([box(0), box(2)])
    labels = []
    monkeypatch.setattr(cv2, 'putText', lambda img, label, *a: labels.append(label))

    def fail(*a, **kw):
        raise RuntimeError('TF unavailable')

    if failure == 'depth':
        p.node._sample_depth_from_bbox = lambda *a: (None, 'center')
    else:
        p.node.tf_buffer = NS(transform=fail, lookup_transform=fail)
    p.node._process_rgbd_pair(p.rgb, p.depth)
    assert 'car 0.85' in labels
    assert len(p.images) == len(p.statuses) == 1
    assert not p.positions
    assert p.statuses[0]['recognition_success'] is True
    assert p.statuses[0]['localization_success'] is False


@pytest.mark.parametrize('failure', ['draw', 'publish'])
def test_debug_failure_preserves_successful_localization(failure, monkeypatch):
    p = probe([box(0), box(2)])

    def fail(*a, **kw):
        raise RuntimeError('debug failed')

    if failure == 'draw':
        monkeypatch.setattr(cv2, 'rectangle', fail)
    else:
        p.node.dbg_pub = NS(publish=fail)
    p.node._process_rgbd_pair(p.rgb, p.depth)
    assert len(p.positions) == len(p.statuses) == 1
    assert p.statuses[0]['recognition_success'] is True
    assert p.statuses[0]['localization_success'] is True
    assert any('debug failed' in warning for warning in p.warnings)
