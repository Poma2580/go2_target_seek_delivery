"""Switchable Ultralytics detection backends for target perception."""

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_BACKENDS = ("yolov8", "yoloe")


@contextmanager
def working_directory(path):
    """Temporarily change cwd for Ultralytics asset resolution."""
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def validate_detection_classes(classes, target_class):
    """Return normalized class names and reject ambiguous configurations."""
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError("detection_classes must be a non-empty string array")
    normalized = []
    for value in classes:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("detection_classes cannot contain empty values")
        normalized.append(value.strip())
    if len(set(normalized)) != len(normalized):
        raise ValueError("detection_classes cannot contain duplicates")
    if not isinstance(target_class, str) or not target_class.strip():
        raise ValueError("target_class must be a non-empty string")
    target_class = target_class.strip()
    if target_class not in normalized:
        raise ValueError("target_class must be included in detection_classes")
    return tuple(normalized), target_class


def _model_path(model_path, cache_dir):
    """Resolve explicit paths and cached bare model names."""
    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("model path must be a non-empty string")
    source = Path(model_path.strip()).expanduser()
    cached = cache_dir / source.name
    if source.is_file():
        return str(source.resolve())
    if source.parent == Path("."):
        return str(cached.resolve()) if cached.is_file() else source.name
    raise ValueError(f"model file does not exist: {source}")


def _class_names(model):
    names = model.names
    if isinstance(names, dict):
        return {int(index): str(name) for index, name in names.items()}
    return {index: str(name) for index, name in enumerate(names)}


@dataclass
class UltralyticsDetector:
    """Small adapter exposing one inference contract to the ROS node."""

    backend: str
    model: object
    target_class_id: int
    inference_kwargs: dict

    def predict(self, image):
        return self.model(image, **self.inference_kwargs)


def create_detector(
    backend,
    yolov8_model_path,
    yoloe_model_path,
    model_cache_dir,
    detection_classes,
    target_class,
    conf,
    imgsz,
    device,
    model_path_override="",
):
    """Build a YOLOv8 or YOLOE detector with equivalent result semantics."""
    backend = str(backend).strip().lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"detector_backend must be one of {SUPPORTED_BACKENDS}, got {backend!r}"
        )
    classes, target_class = validate_detection_classes(
        detection_classes, target_class
    )
    cache_dir = Path(model_cache_dir).expanduser().resolve()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise ValueError(
            f"cannot create model cache directory {cache_dir}: {error}"
        ) from error

    chosen_path = model_path_override or (
        yolov8_model_path if backend == "yolov8" else yoloe_model_path
    )
    resolved_path = _model_path(chosen_path, cache_dir)

    if backend == "yolov8":
        from ultralytics import YOLO

        with working_directory(cache_dir):
            model = YOLO(resolved_path)
        names = _class_names(model)
        ids_by_name = {name: index for index, name in names.items()}
        missing = [name for name in classes if name not in ids_by_name]
        if missing:
            raise ValueError(
                f"YOLOv8 model does not provide configured classes: {missing}"
            )
        class_ids = [ids_by_name[name] for name in classes]
        target_class_id = ids_by_name[target_class]
        kwargs = {
            "classes": class_ids,
            "conf": float(conf),
            "imgsz": int(imgsz),
            "device": device,
            "verbose": False,
        }
    else:
        from ultralytics import YOLOE

        with working_directory(cache_dir):
            model = YOLOE(resolved_path)
            embeddings = model.get_text_pe(list(classes))
            model.set_classes(list(classes), embeddings)
        target_class_id = classes.index(target_class)
        kwargs = {
            "conf": float(conf),
            "imgsz": int(imgsz),
            "device": device,
            "verbose": False,
        }

    return UltralyticsDetector(backend, model, target_class_id, kwargs)
