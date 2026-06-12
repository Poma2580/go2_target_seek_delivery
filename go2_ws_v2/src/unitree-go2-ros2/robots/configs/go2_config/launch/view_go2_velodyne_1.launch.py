"""Open RViz for the go2_1 Velodyne point cloud."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    config_pkg_share = get_package_share_directory("go2_config")
    rviz_config = os.path.join(config_pkg_share, "rviz", "go2_velodyne_1.rviz")

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2_go2_velodyne_1",
        output="screen",
        arguments=["-d", rviz_config],
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([rviz])
