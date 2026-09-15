"""Launch the recorder for a world and robots prepared by target_test_runner."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("use_sim_time", default_value="true"),
        DeclareLaunchArgument("case_config"),
        DeclareLaunchArgument("output_dir"),
        Node(
            package="go2_test_framework",
            executable="target_test_recorder",
            name="target_test_recorder",
            output="screen",
            parameters=[{
                "use_sim_time": ParameterValue(LaunchConfiguration("use_sim_time"), value_type=bool),
                "case_config": LaunchConfiguration("case_config"),
                "output_dir": LaunchConfiguration("output_dir"),
            }],
        ),
    ])
