#!/usr/bin/env python3
"""Convert WebM/VP8-like videos to real H.264 MP4 files.

Usage:

cd /home/bit/go2_target_seek_delivery
python3 Video/convert_webm_to_mp4.py single_go2_nav2_city.webm

By default, the output path is:
  - same name when input already ends with .mp4
  - same stem plus .mp4 for other extensions

The script writes to a temporary file first, then replaces the output only
after ffmpeg succeeds.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def resolve_video_path(name: str) -> Path:
    path = Path(name).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def default_output_path(input_path: Path) -> Path:
    if input_path.suffix.lower() == ".mp4":
        return input_path
    return input_path.with_suffix(".mp4")


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def convert(input_path: Path, output_path: Path, crf: int, preset: str) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found. Please install ffmpeg first.")

    if not input_path.is_file():
        raise FileNotFoundError(f"Input file does not exist: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    if tmp_path.exists():
        tmp_path.unlink()

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        str(tmp_path),
    ]

    try:
        run(cmd)
        tmp_path.replace(output_path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert WebM/VP8 videos to browser-friendly H.264 MP4."
    )
    parser.add_argument("input", help="Input video name or path.")
    parser.add_argument(
        "-o",
        "--output",
        help="Output MP4 name or path. Defaults to replacing/saving beside input.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="H.264 quality, lower is better/larger. Default: 23.",
    )
    parser.add_argument(
        "--preset",
        default="medium",
        help="x264 preset, e.g. veryfast, medium, slow. Default: medium.",
    )
    args = parser.parse_args()

    input_path = resolve_video_path(args.input)
    output_path = resolve_video_path(args.output) if args.output else default_output_path(input_path)

    try:
        convert(input_path, output_path, args.crf, args.preset)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Saved: {output_path} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
