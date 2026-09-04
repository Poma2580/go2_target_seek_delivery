"""Canonical repository and artifact paths for the pedestrian-map toolkit."""

from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = TOOL_ROOT.parents[1]
ARTIFACTS_ROOT = TOOL_ROOT / "artifacts"
MAPS_ROOT = REPO_ROOT / "tools/gazebo_map_creator/artifacts/maps"
ROUTE_VALIDATION_ROOT = ARTIFACTS_ROOT / "route_validation"
ROBOT_POSE_REPORT_ROOT = ARTIFACTS_ROOT / "robot_pose_generation"
TARGET_ROUTES = (
    REPO_ROOT
    / "go2_ws_v2/src/go2_test_framework/config/parameters/target_routes.yaml"
)
ROBOT_POSE_GROUPS = (
    REPO_ROOT
    / "go2_ws_v2/src/go2_test_framework/config/parameters/robot_pose_groups.yaml"
)
