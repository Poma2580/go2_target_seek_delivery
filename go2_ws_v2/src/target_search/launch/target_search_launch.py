import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    package_dir = get_package_share_directory('target_search')
    config_file = os.path.join(package_dir, 'config', 'camera_params.yaml')

    target_detector_node = Node(
        package='target_search',
        executable='target_detector',
        name='target_detector',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('/camera/rgb/image_raw', '/go2_1/camera/image_raw'),
            ('/camera/depth/image_raw', '/go2_1/camera/depth/image_raw'),
            ('/camera/odom', '/go2_1/odom'),
        ],
    )

    return LaunchDescription([
        target_detector_node,
    ])