"""Resolve one test route into a dynamic-encircle scene configuration."""

from copy import deepcopy
from pathlib import Path

from go2_scenario_config.scene_config import load_scene_config

from go2_test_framework.common.config import read_yaml
from go2_test_framework.reporting.results import write_yaml


def resolved_route_points(route_config):
    """Convert test traversal semantics to a closed scene-config route."""
    points = [tuple(point) for point in route_config["points"]]
    traversal = route_config["traversal"]
    if traversal == "closed":
        return tuple(points)
    if traversal != "ping_pong":
        raise ValueError(f"unsupported route traversal {traversal!r}")
    if len(points) == 2:
        start, end = points
        midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
        return start, midpoint, end
    return tuple(points + list(reversed(points[1:-1])))


def write_resolved_scene_config(case, source_path, output_path):
    """Copy a scene YAML and override only its dynamic-target route settings."""
    source = read_yaml(source_path)
    if source.get("scene") != case.scene:
        raise ValueError(
            f"scene config declares {source.get('scene')!r}, expected {case.scene!r}"
        )
    resolved = deepcopy(source)
    target = resolved.get("dynamic_target")
    if not isinstance(target, dict):
        raise ValueError("scene config dynamic_target must be a mapping")
    target.update({
        "speed": float(case.route_config["speed"]),
        "turn_duration": float(case.route_config["turn_duration"]),
        "route": [
            {"x": float(x), "y": float(y)}
            for x, y in resolved_route_points(case.route_config)
        ],
    })
    output_path = Path(output_path)
    write_yaml(output_path, resolved)
    load_scene_config(case.scene, output_path)
    return resolved
