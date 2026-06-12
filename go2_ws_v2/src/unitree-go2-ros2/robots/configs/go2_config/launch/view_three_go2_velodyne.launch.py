"""Open RViz for the three-Go2 Velodyne point clouds."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def static_world_to_odom(robot_name, x, y):
    return Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name=f"world_to_{robot_name}_odom",
        arguments=[x, y, "0", "0", "0", "0", "world", f"{robot_name}/odom"],
    )


def generate_launch_description():
    config_pkg_share = get_package_share_directory("go2_config")
    rviz_config = os.path.join(config_pkg_share, "rviz", "three_go2_velodyne.rviz")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_three_go2_velodyne",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            static_world_to_odom("go2_1", "-3.5", "1.5"),
            static_world_to_odom("go2_2", "-3.5", "2.7"),
            static_world_to_odom("go2_3", "-3.5", "3.9"),
            rviz,
        ]
    )
