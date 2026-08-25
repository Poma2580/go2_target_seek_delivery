"""Launch three namespaced target perception nodes and one role selector."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOT_NAMES = ("go2_1", "go2_2", "go2_3")


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)
    model_path = LaunchConfiguration("model_path")
    package_share = get_package_share_directory("go2_target_perception")
    perception_config = os.path.join(
        package_share, "config", "target_perception.yaml"
    )

    perception_nodes = [
        Node(
            package="go2_target_perception",
            executable="target_perception",
            namespace=name,
            name="target_perception",
            output="screen",
            respawn=False,
            parameters=[
                perception_config,
                {
                    "use_sim_time": use_sim_time_param,
                    "robot_namespace": name,
                    "model_path": model_path,
                },
            ],
        )
        for name in ROBOT_NAMES
    ]

    selector = Node(
        package="go2_target_perception",
        executable="target_role_selector",
        name="target_role_selector",
        output="screen",
        respawn=False,
        parameters=[
            {
                "use_sim_time": use_sim_time_param,
                "robot_names": list(ROBOT_NAMES),
                "confirmation_count": 3,
                "confirmation_window": 1.0,
                "max_message_age": 0.5,
                "role_topic": "/target_role/perception_robot",
            }
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("model_path", default_value="yolov8s.pt"),
            *perception_nodes,
            selector,
        ]
    )
