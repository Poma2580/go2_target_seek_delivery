"""Shared occupancy-map image readers."""

from pathlib import Path

import numpy as np
from PIL import Image


def read_grayscale_image(path: Path) -> np.ndarray:
    """Read PNG or P2/P5 PGM without relying on Pillow's P2 support."""
    data = path.read_bytes()
    if not data.startswith((b"P2", b"P5")):
        with Image.open(path) as image:
            return np.asarray(image.convert("L"), dtype=np.uint8)

    index = 0

    def token() -> bytes:
        nonlocal index
        while index < len(data):
            if data[index:index + 1] == b"#":
                newline = data.find(b"\n", index)
                index = len(data) if newline < 0 else newline + 1
            elif data[index:index + 1].isspace():
                index += 1
            else:
                break
        start = index
        while index < len(data) and not data[index:index + 1].isspace():
            index += 1
        if start == index:
            raise ValueError(f"truncated PGM header: {path}")
        return data[start:index]

    try:
        magic = token()
        width, height, max_value = int(token()), int(token()), int(token())
    except ValueError as error:
        raise ValueError(f"invalid PGM header in {path}: {error}") from error
    if width <= 0 or height <= 0 or not 0 < max_value <= 255:
        raise ValueError(f"unsupported PGM dimensions or max value in {path}")
    count = width * height
    if magic == b"P2":
        try:
            values = np.fromiter((int(token()) for _ in range(count)), dtype=np.int64)
        except ValueError as error:
            raise ValueError(f"invalid P2 pixels in {path}: {error}") from error
    else:
        if index >= len(data) or not data[index:index + 1].isspace():
            raise ValueError(f"missing P5 header separator in {path}")
        index += 2 if data[index:index + 2] == b"\r\n" else 1
        values = np.frombuffer(data[index:index + count], dtype=np.uint8).astype(np.int64)
        if values.size != count:
            raise ValueError(f"truncated P5 pixels in {path}")
    if np.any(values < 0) or np.any(values > max_value):
        raise ValueError(f"PGM pixel outside declared range in {path}")
    if max_value != 255:
        values = np.rint(values * (255.0 / max_value)).astype(np.int64)
    return values.astype(np.uint8).reshape((height, width))
