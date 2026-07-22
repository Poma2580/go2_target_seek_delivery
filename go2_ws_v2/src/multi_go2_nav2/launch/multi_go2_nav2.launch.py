"""Launch one shared map and three independent, namespaced Nav2 stacks."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import ReplaceString, RewrittenYaml

from multi_go2_nav2.scene_config import ROBOT_NAMES, load_scene_config


NAVIGATION_LIFECYCLE_NODES = [
    'controller_server',
    'smoother_server',
    'planner_server',
    'behavior_server',
    'bt_navigator',
    'velocity_smoother',
]


def _navigation_group(
        robot_name, params_file, use_sim_time, nav_to_pose_bt_xml):
    replaced = ReplaceString(
        source_file=params_file,
        replacements={'<robot_name>': robot_name},
    )
    configured = ParameterFile(
        RewrittenYaml(
            source_file=replaced,
            root_key=robot_name,
            param_rewrites={
                'use_sim_time': use_sim_time,
                'default_nav_to_pose_bt_xml': nav_to_pose_bt_xml,
            },
            convert_types=True,
        ),
        allow_substs=True,
    )
    common = {
        'output': 'screen',
        'parameters': [configured],
    }
    return GroupAction([
        PushRosNamespace(robot_name),
        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            remappings=[('cmd_vel', 'cmd_vel_nav')],
            **common,
        ),
        Node(
            package='nav2_smoother',
            executable='smoother_server',
            name='smoother_server',
            **common,
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            **common,
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            # Keep explicitly requested behaviors in the same smoothing
            # pipeline instead of publishing competing final commands.
            remappings=[('cmd_vel', 'cmd_vel_nav')],
            **common,
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            **common,
        ),
        Node(
            package='nav2_velocity_smoother',
            executable='velocity_smoother',
            name='velocity_smoother',
            remappings=[
                ('cmd_vel', 'cmd_vel_nav'),
                ('cmd_vel_smoothed', 'cmd_vel'),
            ],
            **common,
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                # The package sequencer starts the three managers one at a
                # time.  Letting each manager autostart here can overload DDS
                # while all three 1800x750 costmaps are being configured.
                'autostart': False,
                'node_names': NAVIGATION_LIFECYCLE_NODES,
            }],
        ),
    ])


def _launch_setup(context):
    package_share = Path(get_package_share_directory('multi_go2_nav2'))
    config_file = LaunchConfiguration('scene_config').perform(context)
    config = load_scene_config(config_file)
    map_yaml = config.resolve_map_yaml()
    if not map_yaml.is_file():
        raise RuntimeError(f'map YAML does not exist: {map_yaml}')

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    autostart = LaunchConfiguration('autostart')
    lifecycle_bringup_timeout = LaunchConfiguration(
        'lifecycle_bringup_timeout')
    nav_to_pose_bt_xml = str(package_share.joinpath(
        'behavior_trees', 'navigate_to_pose_no_recovery.xml'))
    actions = [
        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'yaml_filename': str(map_yaml),
                'frame_id': config.map.frame_id,
            }],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_map',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'autostart': False,
                'node_names': ['map_server'],
            }],
        ),
    ]

    # Gazebo ground-truth odometry already uses world-coordinate values.  The
    # identity transforms therefore make that world coordinate system the map
    # frame without AMCL or another localization source fighting the truth TF.
    for robot_name in ROBOT_NAMES:
        actions.append(Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name=f'{robot_name}_map_to_odom',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', config.map.frame_id,
                '--child-frame-id', f'{robot_name}/odom',
            ],
            output='screen',
        ))
        actions.append(_navigation_group(
            robot_name, params_file, use_sim_time, nav_to_pose_bt_xml))

    actions.extend([
        Node(
            package='multi_go2_nav2',
            executable='nav2_bringup_sequencer',
            name='nav2_bringup_sequencer',
            output='screen',
            condition=IfCondition(autostart),
            parameters=[{
                'use_sim_time': use_sim_time,
                'manager_timeout': lifecycle_bringup_timeout,
            }],
        ),
        Node(
            package='multi_go2_nav2',
            executable='trajectory_recorder',
            name='trajectory_recorder',
            output='screen',
            parameters=[{
                'use_sim_time': use_sim_time,
                'scene_config': config_file,
            }],
        ),
        Node(
            package='multi_go2_nav2',
            executable='encircle_coordinator',
            name='encircle_coordinator',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_coordinator')),
            parameters=[{
                'use_sim_time': use_sim_time,
                'scene_config': config_file,
                'autostart': autostart,
            }],
        ),
        Node(
            package='rviz2',
            executable='rviz2',
            name='multi_go2_rviz',
            output='screen',
            condition=IfCondition(LaunchConfiguration('use_rviz')),
            arguments=['-d', str(package_share / 'rviz' / 'multi_go2_nav2.rviz')],
            parameters=[{'use_sim_time': use_sim_time}],
        ),
    ])
    return actions


def generate_launch_description():
    package_share = Path(get_package_share_directory('multi_go2_nav2'))
    return LaunchDescription([
        DeclareLaunchArgument(
            'scene_config',
            default_value=str(package_share / 'config' / 'scenes' / 'airport.yaml'),
            description='Absolute path to the unified scene YAML.',
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=str(package_share / 'config' / 'nav2_params.yaml'),
            description='Common Nav2 parameters containing <robot_name>.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument(
            'lifecycle_bringup_timeout',
            default_value='60.0',
            description='Wall-clock timeout for starting each managed stack.',
        ),
        DeclareLaunchArgument('start_coordinator', default_value='true'),
        DeclareLaunchArgument('use_rviz', default_value='true'),
        OpaqueFunction(function=_launch_setup),
    ])
