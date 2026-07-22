#!/usr/bin/env python3
"""Black-box tests for the world_to_grid command line converter."""

import csv
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


BINARY = Path(sys.argv[1]).resolve()
# Prevent unittest from interpreting the executable path as a test name.
sys.argv = [sys.argv[0]]


def world(models: str) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="default">
    {models}
  </world>
</sdf>
"""


def model_with_geometry(geometry: str, pose: str = "1.5 2.5 0.5 0 0 0") -> str:
    return f"""
<model name="fixture">
  <static>true</static>
  <link name="link">
    <collision name="collision">
      <pose>{pose}</pose>
      <geometry>{geometry}</geometry>
    </collision>
  </link>
</model>
"""


def read_pgm(path: Path):
    with path.open("rb") as stream:
        assert stream.readline().strip() == b"P5"
        line = stream.readline()
        while line.startswith(b"#"):
            line = stream.readline()
        width, height = (int(value) for value in line.split())
        assert stream.readline().strip() == b"255"
        pixels = stream.read()
    assert len(pixels) == width * height
    return width, height, pixels


class WorldToGridTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="world_to_grid_test_")
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_converter(self, sdf: str, name: str = "map", extra=(), check=True):
        world_path = self.root / f"{name}.sdf"
        output_prefix = self.root / name
        world_path.write_text(sdf, encoding="utf-8")
        command = [
            str(BINARY), "--world", str(world_path),
            "--output-prefix", str(output_prefix),
            "--bounds", "0", "0", "10", "10",
            "--resolution", "1", "--z-min", "0.03", "--z-max", "0.8",
            "--border-cells", "1", *extra,
        ]
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if check and result.returncode != 0:
            self.fail(f"converter failed:\nstdout={result.stdout}\nstderr={result.stderr}")
        return result, output_prefix

    def records(self, prefix: Path):
        with Path(str(prefix) + "_collisions.csv").open(newline="", encoding="utf-8") as stream:
            return {row["collision"]: row for row in csv.DictReader(stream)}

    def test_pose_composition_geometry_and_slice_statuses(self):
        sdf = world("""
<model name="composed">
  <pose>2 2 0 0 0 1.5707963267948966</pose>
  <static>false</static>
  <link name="body">
    <pose>1 0 0 0 0 0</pose>
    <collision name="box"><pose>1 0 0.5 0 0 0</pose>
      <geometry><box><size>2 1 1</size></box></geometry></collision>
    <collision name="model_relative"><pose relative_to="__model__">3 0 0.5 0 0 0</pose>
      <geometry><sphere><radius>0.25</radius></sphere></geometry></collision>
    <collision name="cylinder"><pose>0 1 0.4 0.2 0.1 0</pose>
      <geometry><cylinder><radius>0.2</radius><length>0.6</length></cylinder></geometry></collision>
    <collision name="below"><pose>0 0 0 0 0 0</pose>
      <geometry><box><size>1 1 0.02</size></box></geometry></collision>
    <collision name="above"><pose>0 0 2 0 0 0</pose>
      <geometry><sphere><radius>0.1</radius></sphere></geometry></collision>
  </link>
</model>
<model name="outside"><pose>20 20 0.5 0 0 0</pose><link name="link">
  <collision name="outside"><geometry><box><size>1 1 1</size></box></geometry></collision>
</link></model>
<model name="skip_me"><link name="link"><collision name="never_read">
  <geometry><plane><normal>0 0 1</normal><size>1 1</size></plane></geometry>
</collision></link></model>
""")
        _, prefix = self.run_converter(sdf, extra=("--ignore-model", "skip_me"))
        rows = self.records(prefix)
        self.assertEqual(rows["box"]["static"], "false")
        self.assertEqual(rows["box"]["status"], "occupied")
        for field, expected in {
            "min_x": 1.5, "min_y": 3.0, "min_z": 0.0,
            "max_x": 2.5, "max_y": 5.0, "max_z": 1.0,
        }.items():
            # Gazebo's `gz sdf -p` currently prints RPY with about six
            # significant decimal digits, so allow its small round-trip loss.
            self.assertAlmostEqual(float(rows["box"][field]), expected, places=4)
        self.assertEqual(rows["model_relative"]["status"], "occupied")
        self.assertEqual(rows["cylinder"]["geometry"], "cylinder")
        self.assertEqual(rows["below"]["status"], "ignored_below_slice")
        self.assertEqual(rows["above"]["status"], "ignored_above_slice")
        self.assertEqual(rows["outside"]["status"], "ignored_outside_bounds")
        ignored = [row for row in rows.values() if row["status"] == "ignored_model"]
        self.assertEqual(len(ignored), 1)

    def test_assimp_node_transform_and_sdf_mesh_scale(self):
        dae = self.root / "triangle.dae"
        dae.write_text("""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="triangle"><mesh>
    <source id="positions"><float_array id="positions-array" count="9">0 0 0 1 0 0 0 1 1</float_array>
      <technique_common><accessor source="#positions-array" count="3" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <vertices id="vertices"><input semantic="POSITION" source="#positions"/></vertices>
    <triangles count="1"><input semantic="VERTEX" source="#vertices" offset="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="scene"><node id="translated">
    <translate>2 0 0</translate><instance_geometry url="#triangle"/>
  </node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
""", encoding="utf-8")
        geometry = f"<mesh><uri>{dae}</uri><scale>2 3 1</scale></mesh>"
        _, prefix = self.run_converter(world(model_with_geometry(geometry, "1 1 0.2 0 0 0")))
        row = self.records(prefix)["collision"]
        self.assertEqual(row["geometry"], "mesh")
        for field, expected in {
            "min_x": 5.0, "min_y": 1.0, "min_z": 0.2,
            "max_x": 7.0, "max_y": 4.0, "max_z": 1.2,
        }.items():
            self.assertAlmostEqual(float(row[field]), expected, places=6)

    def test_golden_pgm_yaml_border_and_y_flip(self):
        sdf = world(model_with_geometry(
            "<box><size>1 1 1</size></box>", "1.5 2.5 0.5 0 0 0"))
        _, prefix = self.run_converter(
            sdf, name="golden",
            extra=("--bounds", "0", "0", "4", "4"))
        pgm_path = Path(str(prefix) + ".pgm")
        expected = (
            b"P5\n# generated by world_to_grid from golden.sdf\n4 4\n255\n" +
            bytes([
                0, 0, 0, 0,
                0, 0, 254, 0,
                0, 254, 254, 0,
                0, 0, 0, 0,
            ])
        )
        self.assertEqual(pgm_path.read_bytes(), expected)
        yaml = Path(str(prefix) + ".yaml").read_text(encoding="utf-8")
        self.assertIn("image: golden.pgm\n", yaml)
        self.assertIn("resolution: 1\n", yaml)
        self.assertIn("origin: [0, 0, 0.0]\n", yaml)
        self.assertIn("mode: trinary\n", yaml)

    def test_output_is_byte_deterministic(self):
        sdf = world(model_with_geometry("<sphere><radius>0.4</radius></sphere>"))
        _, prefix = self.run_converter(sdf, name="repeat")
        paths = [Path(str(prefix) + suffix) for suffix in
                 (".pgm", ".yaml", "_collisions.csv", "_preview.svg")]
        first = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
        self.run_converter(sdf, name="repeat")
        second = [hashlib.sha256(path.read_bytes()).digest() for path in paths]
        self.assertEqual(first, second)

    def test_invalid_inputs_fail_without_outputs(self):
        invalid_geometries = {
            "missing_mesh": "<mesh><uri>does_not_exist.obj</uri></mesh>",
            "bad_size": "<box><size>1 0 1</size></box>",
            "plane": "<plane><normal>0 0 1</normal><size>1 1</size></plane>",
            "heightmap": "<heightmap><uri>missing.png</uri></heightmap>",
            "submesh": "<mesh><uri>missing.obj</uri><submesh><name>x</name></submesh></mesh>",
        }
        for index, (label, geometry) in enumerate(invalid_geometries.items()):
            with self.subTest(label=label):
                result, prefix = self.run_converter(
                    world(model_with_geometry(geometry)), name=f"invalid_{index}", check=False)
                self.assertNotEqual(result.returncode, 0)
                for suffix in (".pgm", ".yaml", "_collisions.csv", "_preview.svg"):
                    self.assertFalse(Path(str(prefix) + suffix).exists())

        bad_frame = """
<model name="bad_frame"><link name="link"><collision name="collision">
  <pose relative_to="unknown_frame">1 1 0.5 0 0 0</pose>
  <geometry><box><size>1 1 1</size></box></geometry>
</collision></link></model>
"""
        result, prefix = self.run_converter(world(bad_frame), name="bad_frame", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(Path(str(prefix) + ".pgm").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
