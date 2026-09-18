"""Test the independent node with fake inference, real ROS messages and TF types."""
import json
from types import SimpleNamespace as NS
from unittest.mock import Mock

import numpy as np
import pytest
import rclpy
from builtin_interfaces.msg import Time as Stamp
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from rclpy.parameter import Parameter
from sensor_msgs.msg import CameraInfo, Image

from go2_static_perception_test import static_yoloe_perception as module


def bare_node():
    node = module.StaticYoloePerception.__new__(module.StaticYoloePerception)
    for name, value in module.DEFAULTS.items():
        setattr(node, name, value)
    node.target_frame = 'go2_1/odom'
    node._next_sample_id = 0
    node._yolo_kwargs = {}
    node.status_pub, node.pose_pub = Mock(), Mock()
    node.dbg_pub = None
    node._warn = Mock()
    return node


def inputs():
    rgb, depth = Image(), Image()
    for message in (rgb, depth):
        message.header.frame_id = 'go2_1/camera_depth_optical_frame'
        message.header.stamp = Stamp(sec=10)
        message.height, message.width = 20, 20
    depth.encoding = '32FC1'
    info = CameraInfo()
    info.header = depth.header
    info.height, info.width = 20, 20
    info.k = [10., 0., 10., 0., 10., 10., 0., 0., 1.]
    return np.zeros((20, 20, 3), dtype=np.uint8), rgb, depth, info


def box(xyxy, confidence=0.8):
    return NS(xyxy=np.array([xyxy]), conf=[confidence], cls=[99])


def results(*boxes):
    return [NS(boxes=list(boxes))]


@pytest.mark.parametrize('prompt', ['', '  ', None, []])
def test_empty_prompt(prompt):
    with pytest.raises(ValueError):
        module.validate_prompt(prompt)


def test_load_model_encodes_exactly_one_prompt(monkeypatch, tmp_path):
    import sys
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'ultralytics', NS(YOLOE=factory))
    weight = tmp_path / 'model.pt'
    weight.touch()
    assert module.load_model(str(weight), 'ground robot') is model
    model.get_text_pe.assert_called_once_with(['ground robot'])
    model.set_classes.assert_called_once_with(['ground robot'], model.get_text_pe.return_value)


def test_load_model_reuses_cache_from_arbitrary_cwd(monkeypatch, tmp_path):
    import sys
    start = tmp_path / 'start'
    cache = tmp_path / 'cache'
    start.mkdir()
    cache.mkdir()
    cached_weight = cache / 'model.pt'
    cached_weight.touch()
    monkeypatch.chdir(start)

    observed = {}
    model = Mock()

    def factory(path):
        observed['model_path'] = path
        observed['model_cwd'] = module.Path.cwd()
        return model

    def encode(_prompts):
        observed['encoder_cwd'] = module.Path.cwd()
        return Mock()

    model.get_text_pe.side_effect = encode
    monkeypatch.setitem(sys.modules, 'ultralytics', NS(YOLOE=factory))

    module.load_model('model.pt', 'person', str(cache))

    assert observed == {
        'model_path': str(cached_weight.resolve()),
        'model_cwd': cache,
        'encoder_cwd': cache,
    }
    assert module.Path.cwd() == start


def test_load_model_downloads_bare_assets_into_cache(monkeypatch, tmp_path):
    import sys
    start = tmp_path / 'start'
    cache = tmp_path / 'missing-cache'
    start.mkdir()
    monkeypatch.chdir(start)
    model = Mock()

    def factory(path):
        assert path == 'download.pt'
        assert module.Path.cwd() == cache
        (cache / path).touch()
        return model

    def encode(_prompts):
        assert module.Path.cwd() == cache
        (cache / 'mobileclip2_b.ts').touch()
        return Mock()

    model.get_text_pe.side_effect = encode
    monkeypatch.setitem(sys.modules, 'ultralytics', NS(YOLOE=factory))

    module.load_model('download.pt', 'person', str(cache))

    assert (cache / 'download.pt').is_file()
    assert (cache / 'mobileclip2_b.ts').is_file()
    assert module.Path.cwd() == start


def test_load_model_prefers_explicit_path_and_restores_cwd_on_error(
        monkeypatch, tmp_path):
    import sys
    start = tmp_path / 'start'
    cache = tmp_path / 'cache'
    start.mkdir()
    cache.mkdir()
    explicit = tmp_path / 'explicit.pt'
    explicit.touch()
    (cache / explicit.name).touch()
    monkeypatch.chdir(start)

    def fail(path):
        assert path == str(explicit.resolve())
        assert module.Path.cwd() == cache
        raise RuntimeError('load failed')

    monkeypatch.setitem(sys.modules, 'ultralytics', NS(YOLOE=fail))
    with pytest.raises(RuntimeError, match='load failed'):
        module.load_model(str(explicit), 'person', str(cache))
    assert module.Path.cwd() == start


def test_prompt_is_read_only(monkeypatch):
    monkeypatch.setattr(module, 'load_model', Mock(return_value=Mock()))
    rclpy.init(args=[])
    node = module.StaticYoloePerception()
    try:
        result = node.set_parameters([Parameter('target_prompt', value='dumpster')])
        assert not result[0].successful
        assert node.target_prompt == 'person'
        module.load_model.assert_called_once()
    finally:
        node.destroy_node()
        rclpy.shutdown()


def test_largest_area_tie_and_invalid_boxes():
    selected = module.StaticYoloePerception._pick_target(results(
        box([0, 0, 5, 5]), box([0, 0, 10, 10], .6), box([0, 0, 10, 10], .9),
        box([0, 0, float('inf'), 20]), box([4, 4, 2, 2]), box([0, 0, 50, 50], float('nan'))))
    assert selected == (0, 0, 10, 10, .6)
    assert module.StaticYoloePerception._pick_target([]) is None
    assert module.StaticYoloePerception._pick_target([NS(boxes=None)]) is None


@pytest.mark.parametrize('recognized,localized', [(False, False), (True, False), (True, True)])
def test_status_schema(recognized, localized):
    value = json.loads(module.result_status_json(
        Stamp(sec=4, nanosec=5), 6, 'ground robot', recognized, .8, [1, 2, 3, 4], localized))
    assert set(value) == {'schema_version', 'stamp', 'sample_id', 'prompt',
                          'recognition_success', 'confidence', 'bbox', 'localization_success'}
    assert value['stamp'] == {'sec': 4, 'nanosec': 5}
    assert value['prompt'] == 'ground robot' and value['sample_id'] == 6
    assert value['recognition_success'] is recognized
    assert value['localization_success'] is localized
    assert (value['confidence'], value['bbox']) == ((.8, [1., 2., 3., 4.]) if recognized else (None, None))


@pytest.mark.parametrize('confidence,bbox', [(float('nan'), [1, 2, 3, 4]), (.5, [1, 2, float('inf'), 4])])
def test_nonfinite_json_rejected(confidence, bbox):
    with pytest.raises(ValueError):
        module.result_status_json(Stamp(), 0, 'person', True, confidence, bbox)


@pytest.mark.parametrize('case', ['empty', 'inference_error', 'depth_error', 'success', 'debug_error'])
def test_each_inference_has_one_status(case):
    node = bare_node()
    image, rgb, depth, _ = inputs()
    node.model = Mock(return_value=[] if case == 'empty' else results(box([1, 1, 15, 15])))
    node._localize = Mock(return_value=(Point(x=2., y=3., z=4.), 5.))
    if case == 'inference_error':
        node.model.side_effect = RuntimeError('model failure')
    if case == 'depth_error':
        node._localize.side_effect = ValueError('no valid depth')
    if case == 'debug_error':
        node.dbg_pub = Mock()
        node._publish_debug = Mock(side_effect=RuntimeError('debug failure'))
    node._infer_sample(image, rgb, depth)
    node.status_pub.publish.assert_called_once()
    value = json.loads(node.status_pub.publish.call_args.args[0].data)
    assert value['sample_id'] == 0 and node._next_sample_id == 1
    assert value['recognition_success'] == (case not in ('empty', 'inference_error'))
    assert value['localization_success'] == (case in ('success', 'debug_error'))
    assert node.pose_pub.publish.call_count == int(value['localization_success'])
    if value['localization_success']:
        pose = node.pose_pub.publish.call_args.args[0]
        assert pose.header.stamp == depth.header.stamp
        assert pose.header.frame_id == 'go2_1/odom'
        assert pose.pose.orientation.w == 1


def localization_node():
    node = bare_node()
    image, rgb, depth, info = inputs()
    node.camera_info = info
    node.bridge = Mock()
    node.bridge.imgmsg_to_cv2.return_value = np.full((20, 20), 5., dtype=np.float32)
    node.tf_buffer = Mock()
    node.tf_buffer.transform.side_effect = lambda point, *a, **kw: point
    return node, image, rgb, depth


def test_depth_units_and_pinhole():
    node, image, rgb, depth = localization_node()
    node.bridge.imgmsg_to_cv2.return_value[:] = 5000
    depth.encoding = '16UC1'
    point, z = node._localize(image, rgb, depth, [5, 5, 15, 15])
    assert z == 5 and (point.x, point.y, point.z) == (0, 0, 5)
    source = node.tf_buffer.transform.call_args.args[0]
    assert source.header.stamp == depth.header.stamp


@pytest.mark.parametrize('failure', ['no_info', 'fx_zero', 'nan', 'info_shape', 'depth_shape', 'frame', 'no_depth'])
def test_localization_failures(failure):
    node, image, rgb, depth = localization_node()
    if failure == 'no_info':
        node.camera_info = None
    elif failure in ('fx_zero', 'nan'):
        node.camera_info.k[0] = 0 if failure == 'fx_zero' else float('nan')
    elif failure == 'info_shape':
        node.camera_info.width = 10
    elif failure == 'depth_shape':
        node.bridge.imgmsg_to_cv2.return_value = np.ones((10, 10))
    elif failure == 'frame':
        depth.header.frame_id = ''
    elif failure == 'no_depth':
        node.bridge.imgmsg_to_cv2.return_value[:] = np.nan
    with pytest.raises(ValueError):
        node._localize(image, rgb, depth, [5, 5, 15, 15])


def test_roi_percentile_and_center_fallback():
    node = bare_node()
    depth = np.full((20, 20), 5.)
    depth[4:14, 5:15] = np.arange(100).reshape(10, 10)/100 + 2
    assert node._sample_depth_from_bbox(depth, [0, 0, 20, 20], 10, 10) == pytest.approx(2.297)
    node.min_depth_samples = 1000
    assert node._sample_depth_from_bbox(depth, [0, 0, 20, 20], 10, 10) == node._sample_depth(depth, 10, 10)
    depth[:] = 0
    assert node._sample_depth_from_bbox(depth, [0, 0, 20, 20], 10, 10) is None


def test_latest_tf_fallback_and_total_failure():
    node, image, rgb, depth = localization_node()
    node.tf_buffer.transform.side_effect = RuntimeError('history unavailable')
    transform = TransformStamped()
    transform.transform.rotation.w = 1.
    transform.transform.translation.x = 2.
    node.tf_buffer.lookup_transform.return_value = transform
    point, _ = node._localize(image, rgb, depth, [5, 5, 15, 15])
    assert point.x == 2
    assert node.tf_buffer.lookup_transform.call_args.args[2].nanoseconds == 0
    node.tf_buffer.lookup_transform.side_effect = RuntimeError('no TF')
    with pytest.raises(RuntimeError, match='no TF'):
        node._localize(image, rgb, depth, [5, 5, 15, 15])


def test_stale_duplicate_and_repeated_pair_do_not_infer():
    node = bare_node()
    image, rgb, depth, _ = inputs()
    node._latest_rgbd = (1, rgb, depth)
    node._processed_pair_id = -1
    node._last_clock = node._last_sensor_stamp = None
    node.get_clock = lambda: NS(now=lambda: NS(nanoseconds=11_000_000_000))
    node.bridge = Mock()
    node.bridge.imgmsg_to_cv2.return_value = image
    node._infer_sample = Mock()
    node._process_latest_rgbd()
    node._infer_sample.assert_not_called()
    rgb.header.stamp.sec = depth.header.stamp.sec = 11
    node._latest_rgbd = (2, rgb, depth)
    node._process_latest_rgbd()
    node._process_latest_rgbd()
    node._latest_rgbd = (3, rgb, depth)
    node._process_latest_rgbd()
    node._infer_sample.assert_called_once()


def test_empty_startup_prompt_fails_before_loading(monkeypatch):
    monkeypatch.setattr(module, 'load_model', Mock())
    rclpy.init(args=['--ros-args', '-p', "target_prompt:=' '"])
    try:
        with pytest.raises(ValueError, match='target_prompt'):
            module.StaticYoloePerception()
        module.load_model.assert_not_called()
    finally:
        rclpy.shutdown()


def test_different_rgb_optical_frame_requires_identity_tf():
    node, image, rgb, depth = localization_node()
    rgb.header.frame_id = 'go2_1/camera_rgb_optical_frame'
    alignment = TransformStamped()
    alignment.transform.rotation.w = 1.
    node.tf_buffer.lookup_transform.return_value = alignment
    assert node._localize(image, rgb, depth, [5, 5, 15, 15])[1] == 5
    alignment.transform.translation.x = .02
    with pytest.raises(ValueError, match='not registered'):
        node._localize(image, rgb, depth, [5, 5, 15, 15])


def test_actual_missing_depth_retains_recognition():
    node, image, rgb, depth = localization_node()
    node.model = Mock(return_value=results(box([2, 2, 18, 18])))
    node.bridge.imgmsg_to_cv2.return_value[:] = 0
    node._infer_sample(image, rgb, depth)
    value = json.loads(node.status_pub.publish.call_args.args[0].data)
    assert value['recognition_success'] is True and value['localization_success'] is False
    assert value['bbox'] == [2, 2, 18, 18]
    node.pose_pub.publish.assert_not_called()
