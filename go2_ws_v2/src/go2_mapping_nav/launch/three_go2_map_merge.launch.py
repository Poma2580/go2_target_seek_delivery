"""Launch known-pose map fusion, shared TF roots, and optional unified RViz."""

import math
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


def _scene_spawn_log(context):
    scene = LaunchConfiguration("scene").perform(context).strip()
    scene_config_override = LaunchConfiguration("scene_config").perform(
        context
    ).strip()
    config_path = (
        Path(scene_config_override).expanduser()
        if scene_config_override
        else Path(get_package_share_directory("multi_go2_waypoint"))
        / "config"
        / "scenes"
        / f"{scene}.yaml"
    )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(
            f"failed to read scene config {config_path}: {error}"
        ) from error
    if not isinstance(config, dict) or config.get("scene") != scene:
        raise RuntimeError(
            f"scene config {config_path} does not describe scene {scene!r}"
        )
    robots = config.get("robots")
    if not isinstance(robots, dict):
        raise RuntimeError(f"scene config {config_path} has no robots mapping")

    descriptions = []
    for robot_name in ROBOT_NAMES:
        robot = robots.get(robot_name)
        spawn = robot.get("spawn") if isinstance(robot, dict) else None
        if not isinstance(spawn, dict):
            raise RuntimeError(
                f"scene config has no robots.{robot_name}.spawn mapping"
            )
        values = []
        for key in ("x", "y", "yaw"):
            value = spawn.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise RuntimeError(
                    f"robots.{robot_name}.spawn.{key} must be finite"
                )
            values.append(float(value))
        descriptions.append(
            f"{robot_name}=({values[0]:g}, {values[1]:g}, "
            f"yaw={values[2]:g})"
        )

    return [
        LogInfo(
            msg=(
                f"Scene {scene} spawn pose check: "
                + "; ".join(descriptions)
                + ". Current Gazebo ground-truth odometry is already in "
                "world coordinates, so map fusion deliberately uses identity "
                "transforms instead of applying these poses again."
            )
        )
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_parameter = ParameterValue(use_sim_time, value_type=bool)
    use_rviz = LaunchConfiguration("use_rviz")
    publish_rate = LaunchConfiguration("publish_rate")
    stale_warning_sec = LaunchConfiguration("stale_warning_sec")
    package_share = get_package_share_directory("go2_mapping_nav")
    rviz_config = os.path.join(
        package_share, "rviz", "three_go2_mapping_nav.rviz"
    )

    actions = [
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="false"),
        DeclareLaunchArgument("scene", default_value="city"),
        DeclareLaunchArgument("scene_config", default_value=""),
        DeclareLaunchArgument("publish_rate", default_value="1.0"),
        DeclareLaunchArgument("stale_warning_sec", default_value="5.0"),
        OpaqueFunction(function=_scene_spawn_log),
        Node(
            package="go2_mapping_nav",
            executable="known_pose_map_merger.py",
            name="known_pose_map_merger",
            output="screen",
            parameters=[
                {
                    "use_sim_time": use_sim_time_parameter,
                    "map_topics": [
                        "/go2_1/map",
                        "/go2_2/map",
                        "/go2_3/map",
                    ],
                    "output_topic": "/merged_map",
                    "output_frame": "merged_map",
                    "publish_rate": ParameterValue(
                        publish_rate, value_type=float
                    ),
                    "stale_warning_sec": ParameterValue(
                        stale_warning_sec, value_type=float
                    ),
                }
            ],
        ),
    ]

    for robot_name in ROBOT_NAMES:
        actions.append(
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"merged_map_to_{robot_name}_map",
                output="screen",
                arguments=[
                    "--x",
                    "0",
                    "--y",
                    "0",
                    "--z",
                    "0",
                    "--yaw",
                    "0",
                    "--pitch",
                    "0",
                    "--roll",
                    "0",
                    "--frame-id",
                    "merged_map",
                    "--child-frame-id",
                    f"{robot_name}/map",
                ],
            )
        )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="three_go2_mapping_nav_rviz",
            output="screen",
            condition=IfCondition(use_rviz),
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time_parameter}],
            additional_env={"GTK_PATH": ""},
        )
    )
    return LaunchDescription(actions)
