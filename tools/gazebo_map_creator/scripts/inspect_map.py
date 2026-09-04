#!/usr/bin/env python3
"""Normalize and verify one gazebo_map_creator 2-D map."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", required=True, type=Path)
    parser.add_argument("--png", required=True, type=Path)
    parser.add_argument("--image-name", required=True)
    parser.add_argument("--expected-size", nargs=2, required=True, type=int, metavar=("W", "H"))
    parser.add_argument(
        "--expected-origin", nargs=2, required=True, type=float, metavar=("X", "Y")
    )
    parser.add_argument("--expected-resolution", required=True, type=float)
    args = parser.parse_args()

    yaml_path = args.yaml.resolve()
    with yaml_path.open("r", encoding="utf-8") as stream:
        metadata = yaml.safe_load(stream)
    metadata["image"] = args.image_name
    with yaml_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False)

    image_path = yaml_path.parent / args.image_name
    pixels = read_grayscale_image(image_path)
    # Store the normalized occupancy map as compact binary PGM (P5).  The
    # upstream generator emits whitespace-heavy ASCII PGM (P2).
    Image.fromarray(pixels, mode="L").save(image_path)
    pgm_size = (pixels.shape[1], pixels.shape[0])
    with Image.open(args.png) as png_image:
        png_size = png_image.size

    expected_size = tuple(args.expected_size)
    if pgm_size != expected_size or png_size != expected_size:
        raise ValueError(
            f"image dimensions differ: expected={expected_size}, pgm={pgm_size}, png={png_size}"
        )
    resolution = float(metadata["resolution"])
    origin = tuple(float(value) for value in metadata["origin"][:2])
    if abs(resolution - args.expected_resolution) > 1e-9:
        raise ValueError(f"resolution differs: expected={args.expected_resolution}, actual={resolution}")
    if any(abs(actual - expected) > 1e-6 for actual, expected in zip(origin, args.expected_origin)):
        raise ValueError(f"origin differs: expected={tuple(args.expected_origin)}, actual={origin}")

    occupied = int(np.count_nonzero(pixels < 128))
    free = int(np.count_nonzero(pixels >= 128))
    if occupied == 0 or free == 0:
        raise ValueError(f"map must contain occupied and free cells: occupied={occupied}, free={free}")
    print(
        f"Map OK: size={pgm_size[0]}x{pgm_size[1]} resolution={resolution:g} "
        f"origin=({origin[0]:g},{origin[1]:g}) occupied={occupied} free={free} "
        f"occupied_ratio={occupied / pixels.size:.4%}"
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(2)
