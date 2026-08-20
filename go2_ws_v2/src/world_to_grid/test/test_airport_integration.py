#!/usr/bin/env python3
"""Repository integration checks for the generated airport occupancy map."""

from collections import deque
import csv
import hashlib
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv[1]).resolve()
AIRPORT_WORLD = Path(sys.argv[2]).resolve()
sys.argv = [sys.argv[0]]


class AirportIntegrationTest(unittest.TestCase):
    def test_airport_generation_is_valid_and_repeatable(self):
        if not AIRPORT_WORLD.is_file():
            self.skipTest("airport world is not present in this source checkout")
        if not (Path.home() / ".gazebo" / "models").is_dir():
            self.skipTest("official Gazebo models are not installed in ~/.gazebo/models")

        with tempfile.TemporaryDirectory(prefix="world_to_grid_airport_") as temporary:
            prefix = Path(temporary) / "airport"
            command = [
                str(BINARY), "--world", str(AIRPORT_WORLD),
                "--output-prefix", str(prefix),
                "--bounds", "-180", "-75", "180", "75",
                "--resolution", "0.20", "--z-min", "0.03", "--z-max", "0.80",
                "--border-cells", "1", "--ignore-model", "uav1_iris_depth_camera",
            ]
            subprocess.run(command, check=True, text=True, capture_output=True)
            paths = [Path(str(prefix) + suffix) for suffix in
                     (".pgm", ".yaml", "_collisions.csv", "_preview.svg")]
            first_hashes = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
            subprocess.run(command, check=True, text=True, capture_output=True)
            second_hashes = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
            self.assertEqual(first_hashes, second_hashes)

            width, height, pixels = self.read_pgm(paths[0])
            self.assertEqual((width, height), (1800, 750))
            yaml = paths[1].read_text(encoding="utf-8")
            self.assertIn("resolution: 0.2\n", yaml)
            self.assertIn("origin: [-180, -75, 0.0]\n", yaml)

            expected = {
                (0, -4): 254, (2, -4): 254, (0, -6): 254,
                (0, 18): 254, (0, -10): 254,
                (15, -60): 0, (80, -25): 0,
            }
            for point, value in expected.items():
                self.assertEqual(self.value_at(pixels, width, height, *point), value)

            with paths[2].open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            statuses = lambda name: {row["status"] for row in rows if row["model"] == name}
            self.assertEqual(statuses("airport_flat_field"), {"ignored_below_slice"})
            self.assertEqual(statuses("runway_04_22"), {"ignored_below_slice"})
            self.assertIn("occupied", statuses("terminal_building"))
            self.assertIn("occupied", statuses("airpor_cart_target"))
            self.assertEqual(statuses("uav1_iris_depth_camera"), {"ignored_model"})

            reachable = self.reachable_free_cells(pixels, width, height, self.cell(0, -4))
            self.assertIn(self.cell(2, -4), reachable)
            self.assertIn(self.cell(0, -6), reachable)
            target_ring_reachable = any(
                2.5 <= math.hypot(-180 + (gx + 0.5) * 0.2 - 80,
                                  -75 + (gy + 0.5) * 0.2 + 25) <= 5.0
                for gx, gy in reachable)
            self.assertTrue(target_ring_reachable)

    @staticmethod
    def read_pgm(path):
        with path.open("rb") as stream:
            assert stream.readline().strip() == b"P5"
            line = stream.readline()
            while line.startswith(b"#"):
                line = stream.readline()
            width, height = map(int, line.split())
            assert stream.readline().strip() == b"255"
            pixels = stream.read()
        assert len(pixels) == width * height
        return width, height, pixels

    @staticmethod
    def cell(x, y):
        return math.floor((x + 180) / 0.2), math.floor((y + 75) / 0.2)

    @classmethod
    def value_at(cls, pixels, width, height, x, y):
        gx, gy = cls.cell(x, y)
        return pixels[(height - 1 - gy) * width + gx]

    @staticmethod
    def reachable_free_cells(pixels, width, height, start):
        queue = deque([start])
        seen = {start}
        while queue:
            gx, gy = queue.popleft()
            for cell in ((gx - 1, gy), (gx + 1, gy), (gx, gy - 1), (gx, gy + 1)):
                nx, ny = cell
                if (0 <= nx < width and 0 <= ny < height and cell not in seen and
                        pixels[(height - 1 - ny) * width + nx] == 254):
                    seen.add(cell)
                    queue.append(cell)
        return seen


if __name__ == "__main__":
    unittest.main(verbosity=2)
