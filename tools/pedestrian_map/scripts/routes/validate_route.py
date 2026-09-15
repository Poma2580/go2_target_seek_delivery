#!/usr/bin/env python3
"""Command entry point for pedestrian route validation."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from pedestrian_map.route_validation import main


if __name__ == "__main__":
    raise SystemExit(main())
