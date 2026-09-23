"""Unit tests for switchable YOLOv8 and YOLOE detector construction."""

import sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from go2_target_perception import detector_backend as module


CLASSES = [
    'person', 'car', 'airplane', 'bus', 'truck',
    'fire hydrant', 'stop sign', 'bench',
]


def test_default_yaml_selects_yoloe_with_expected_classes_and_limits():
    import yaml

    config = yaml.safe_load(
        (Path(__file__).parents[1] / 'config/target_perception.yaml').read_text()
    )['/**/target_perception']['ros__parameters']
    assert config['detector_backend'] == 'yoloe'
    assert config['detection_classes'] == CLASSES
    assert config['target_class'] == 'person'
    assert (config['conf'], config['imgsz'], config['inference_rate']) == (
        0.15, 640, 8.0)


def create(tmp_path, monkeypatch, backend, model):
    weight = tmp_path / f'{backend}.pt'
    weight.touch()
    factory = Mock(return_value=model)
    key = 'YOLO' if backend == 'yolov8' else 'YOLOE'
    monkeypatch.setitem(sys.modules, 'ultralytics', NS(**{key: factory}))
    detector = module.create_detector(
        backend=backend,
        yolov8_model_path=str(weight),
        yoloe_model_path=str(weight),
        model_cache_dir=str(tmp_path / 'cache'),
        detection_classes=CLASSES,
        target_class='person',
        conf=0.15,
        imgsz=640,
        device='cuda:0',
    )
    return detector, factory


def test_yolov8_maps_configured_names_to_coco_ids(tmp_path, monkeypatch):
    names = {
        0: 'person', 2: 'car', 4: 'airplane', 5: 'bus', 7: 'truck',
        10: 'fire hydrant', 11: 'stop sign', 13: 'bench',
    }
    model = Mock(names=names)
    detector, factory = create(tmp_path, monkeypatch, 'yolov8', model)

    assert detector.target_class_id == 0
    assert detector.inference_kwargs['classes'] == [0, 2, 4, 5, 7, 10, 11, 13]
    assert detector.inference_kwargs['conf'] == 0.15
    factory.assert_called_once()


def test_yoloe_sets_ordered_prompts_without_coco_filter(tmp_path, monkeypatch):
    model = Mock()
    embeddings = object()
    model.get_text_pe.return_value = embeddings
    detector, _ = create(tmp_path, monkeypatch, 'yoloe', model)

    model.get_text_pe.assert_called_once_with(CLASSES)
    model.set_classes.assert_called_once_with(CLASSES, embeddings)
    assert detector.target_class_id == 0
    assert 'classes' not in detector.inference_kwargs


@pytest.mark.parametrize('backend', ['', 'v8', 'unknown'])
def test_invalid_backend_rejected(tmp_path, backend):
    with pytest.raises(ValueError, match='detector_backend'):
        module.create_detector(
            backend, 'v8.pt', 'e.pt', str(tmp_path), CLASSES, 'person',
            0.15, 640, 'cpu')


@pytest.mark.parametrize(
    'classes,target,error',
    [
        ([], 'person', 'non-empty'),
        (['person', ''], 'person', 'empty'),
        (['person', 'person'], 'person', 'duplicates'),
        (['car'], 'person', 'included'),
    ],
)
def test_invalid_class_configuration_rejected(classes, target, error):
    with pytest.raises(ValueError, match=error):
        module.validate_detection_classes(classes, target)


def test_yolov8_rejects_class_missing_from_model(tmp_path, monkeypatch):
    weight = tmp_path / 'v8.pt'
    weight.touch()
    model = Mock(names={0: 'person'})
    monkeypatch.setitem(sys.modules, 'ultralytics', NS(YOLO=Mock(return_value=model)))
    with pytest.raises(ValueError, match='configured classes'):
        module.create_detector(
            'yolov8', str(weight), 'unused.pt', str(tmp_path / 'cache'),
            CLASSES, 'person', 0.15, 640, 'cpu')


def test_explicit_missing_model_path_rejected(tmp_path):
    missing = tmp_path / 'missing' / 'model.pt'
    with pytest.raises(ValueError, match='does not exist'):
        module._model_path(str(missing), tmp_path)
