"""Start only the single-prompt go2_1 static perception node."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    defaults = {'model_path': 'yoloe-26s-seg.pt', 'target_prompt': 'person',
                'model_cache_dir': '~/.cache/go2_static_yolo_smoke',
                'use_sim_time': 'true', 'device': 'cpu', 'target_frame': 'go2_1/odom'}
    parameters = {key: ParameterValue(LaunchConfiguration(key),
                                     value_type=bool if key == 'use_sim_time' else str)
                  for key in defaults}
    config = os.path.join(get_package_share_directory('go2_static_perception_test'),
                          'config', 'static_yoloe_perception.yaml')
    return LaunchDescription([
        *[DeclareLaunchArgument(key, default_value=value) for key, value in defaults.items()],
        Node(package='go2_static_perception_test', executable='static_yoloe_perception',
             namespace='go2_1', name='static_yoloe_perception', output='screen',
             parameters=[config, parameters]),
    ])
