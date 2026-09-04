import re
from pathlib import Path

from go2_test_framework.common.config import load_routes
from go2_test_framework.world_generator import ACTOR_PATTERN, TRAJECTORY_PATTERN, generate_all, route_waypoints


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[2]


def test_route_waypoint_times_are_strictly_increasing():
    routes = load_routes(PACKAGE_ROOT / "config/parameters/target_routes.yaml")
    for scene in routes.values():
        for route in scene["routes"].values():
            timestamps = [value[0] for value in route_waypoints(route)]
            assert timestamps == sorted(timestamps)
            assert all(right > left for left, right in zip(timestamps, timestamps[1:]))


def test_all_nine_committed_worlds_match_generator_and_keep_actor_plugin():
    generated = generate_all(
        PACKAGE_ROOT / "config/parameters/target_routes.yaml",
        REPO_ROOT,
        PACKAGE_ROOT / "worlds",
        check=True,
    )
    assert len(generated) == 9
    for path in generated:
        text = path.read_text(encoding="utf-8")
        actor = ACTOR_PATTERN.search(text)
        assert actor is not None
        assert 'filename="libwalking_target_controller.so"' in actor.group(0)
        trajectory = TRAJECTORY_PATTERN.search(actor.group(2))
        assert trajectory is not None
        times = [float(value) for value in re.findall(r"<time>([^<]+)</time>", trajectory.group(2))]
        assert all(right > left for left, right in zip(times, times[1:]))
