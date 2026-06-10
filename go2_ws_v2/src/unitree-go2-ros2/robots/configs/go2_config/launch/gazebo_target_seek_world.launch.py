"""Start only the target_seek Gazebo world (single gzserver), no robots.

Sets the same GAZEBO_MODEL_PATH / world as the single-dog target_seek launch,
but does NOT spawn any robot. Physics runs (not paused). Spawn each Go2 with
spawn_go2_i.launch.py afterwards, then teleop_three_go2.launch.py for keyboards.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration


def _prepend_paths(*paths):
    existing = os.environ.get("GAZEBO_MODEL_PATH", "")
    values = [p for p in paths if p and os.path.isdir(p)]
    if existing:
        values.append(existing)
    return ":".join(values)


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    paused = LaunchConfiguration("paused")

    qy_model_root = os.environ.get(
        "QY_MODEL_ROOT", os.path.abspath(os.path.join(os.getcwd(), "..", "QY_MODEL"))
    )
    default_world_path = os.path.join(qy_model_root, "target_seek")
    qy_model_path = os.path.join(qy_model_root, "models")
    gazebo_model_cache = os.path.expanduser("~/.gazebo/models")

    gazebo_config = os.path.join(
        get_package_share_directory("champ_gazebo"), "config", "gazebo.yaml"
    )

    common_gz_args = [
        "-s", "libgazebo_ros_init.so",
        "-s", "libgazebo_ros_factory.so",
        LaunchConfiguration("world"),
        "--ros-args", "--params-file", gazebo_config,
    ]

    start_gazebo_server = ExecuteProcess(
        condition=UnlessCondition(paused),
        cmd=["gzserver"] + common_gz_args,
        output="screen",
    )
    # Debug-only paused start.
    start_paused_gazebo_server = ExecuteProcess(
        condition=IfCondition(paused),
        cmd=["gzserver", "-u"] + common_gz_args,
        output="screen",
    )
    start_gazebo_client = ExecuteProcess(
        condition=IfCondition(gui),
        cmd=["gzclient"],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true", description="Use Gazebo GUI"),
            DeclareLaunchArgument(
                "paused", default_value="false",
                description="Start Gazebo physics paused (debug only)",
            ),
            DeclareLaunchArgument(
                "world", default_value=default_world_path,
                description="Gazebo world path (default: $QY_MODEL_ROOT/target_seek)",
            ),
            SetEnvironmentVariable(
                "GAZEBO_MODEL_PATH",
                _prepend_paths(qy_model_path, gazebo_model_cache),
            ),
            SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
            start_gazebo_server,
            start_paused_gazebo_server,
            start_gazebo_client,
        ]
    )
