"""Launch known-pose map fusion, shared TF roots, and optional unified RViz."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


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
        DeclareLaunchArgument("publish_rate", default_value="1.0"),
        DeclareLaunchArgument("stale_warning_sec", default_value="5.0"),
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
