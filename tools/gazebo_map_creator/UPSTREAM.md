# gazebo_map_creator provenance

The source under `tools/gazebo_map_creator/vendor` is vendored from:

- Repository: <https://github.com/arshadlab/gazebo_map_creator>
- Commit: `93120acd8ceed9d0a4b2f04cb0ee313f196609f4`
- License: Apache-2.0

The local patch changes the requested XY sampling from inclusive endpoints to
occupancy-grid cell centres. This makes image dimensions, world-to-grid
conversion, and the YAML resolution agree.
