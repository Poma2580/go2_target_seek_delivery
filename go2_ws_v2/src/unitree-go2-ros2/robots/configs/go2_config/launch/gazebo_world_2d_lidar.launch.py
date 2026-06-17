"""Start a selectable Gazebo world and spawn one Go2 with 2D lidar.

scene:=qy loads QY_MODEL/target_seek with QY_MODEL/models.
scene:=forest loads KD_MODEL/world/forestV3.world with KD_MODEL/models.
scene:=airport loads KD_MODEL/world/airport with KD_MODEL/models.
"""

import os

import launch_ros
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    SetEnvironmentVariable,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _prepend_paths(*paths):
    existing = os.environ.get("GAZEBO_MODEL_PATH", "")
    values = [path for path in paths if path and os.path.isdir(path)]
    if existing:
        values.append(existing)
    return ":".join(values)


def _scene_paths(scene):
    delivery_root = os.environ.get(
        "DELIVERY_ROOT", os.path.abspath(os.path.join(os.getcwd(), ".."))
    )
    qy_model_root = os.environ.get(
        "QY_MODEL_ROOT", os.path.join(delivery_root, "QY_MODEL")
    )
    kd_model_root = os.environ.get(
        "KD_MODEL_ROOT", os.path.join(delivery_root, "KD_MODEL")
    )

    scenes = {
        "qy": {
            "world": os.path.join(qy_model_root, "target_seek"),
            "models": os.path.join(qy_model_root, "models"),
        },
        "forest": {
            "world": os.path.join(kd_model_root, "world", "forestV3.world"),
            "models": os.path.join(kd_model_root, "models"),
        },
        "airport": {
            "world": os.path.join(kd_model_root, "world", "airport"),
            "models": os.path.join(kd_model_root, "models"),
        },
    }

    if scene not in scenes:
        available = ", ".join(sorted(scenes))
        raise RuntimeError(f"Unsupported scene '{scene}'. Use one of: {available}")

    return scenes[scene]


def _launch_setup(context):
    scene = LaunchConfiguration("scene").perform(context)
    selected_scene = _scene_paths(scene)
    default_world_path = selected_scene["world"]
    model_path = selected_scene["models"]
    gazebo_model_cache = os.path.expanduser("~/.gazebo/models")

    config_pkg_share = launch_ros.substitutions.FindPackageShare(
        package="go2_config"
    ).find("go2_config")
    descr_pkg_share = launch_ros.substitutions.FindPackageShare(
        package="go2_description"
    ).find("go2_description")

    joints_config = os.path.join(config_pkg_share, "config/joints/joints.yaml")
    gait_config = os.path.join(config_pkg_share, "config/gait/gait.yaml")
    links_config = os.path.join(config_pkg_share, "config/links/links.yaml")
    default_model_path = os.path.join(descr_pkg_share, "xacro/robot_2d_lidar.xacro")

    declare_gazebo_world = DeclareLaunchArgument(
        "world",
        default_value=default_world_path,
        description="Gazebo world path. Defaults to the world selected by scene.",
    )
    declare_use_ground_truth_odom = DeclareLaunchArgument(
        "use_ground_truth_odom",
        default_value="true",
        description="Use Gazebo ground-truth odometry for simulation /odom",
    )
    declare_world_init_x = DeclareLaunchArgument("world_init_x", default_value="-3.5")
    declare_world_init_y = DeclareLaunchArgument("world_init_y", default_value="2.7")
    declare_world_init_z = DeclareLaunchArgument("world_init_z", default_value="0.40")
    declare_world_init_heading = DeclareLaunchArgument(
        "world_init_heading", default_value="0.0"
    )

    bringup_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("champ_bringup"),
                "launch",
                "bringup.launch.py",
            )
        ),
        launch_arguments={
            "description_path": default_model_path,
            "joints_map_path": joints_config,
            "links_map_path": links_config,
            "gait_config_path": gait_config,
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "robot_name": LaunchConfiguration("robot_name"),
            "gazebo": "true",
            "lite": LaunchConfiguration("lite"),
            "rviz": LaunchConfiguration("rviz"),
            "joint_controller_topic": "joint_group_effort_controller/joint_trajectory",
            "hardware_connected": "false",
            "publish_foot_contacts": "false",
            "publish_odom_tf": "false",
            "close_loop_odom": "true",
        }.items(),
    )

    gazebo_ld = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("champ_gazebo"),
                "launch",
                "gazebo.launch.py",
            )
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "robot_name": LaunchConfiguration("robot_name"),
            "world": LaunchConfiguration("world"),
            "lite": LaunchConfiguration("lite"),
            "world_init_x": LaunchConfiguration("world_init_x"),
            "world_init_y": LaunchConfiguration("world_init_y"),
            "world_init_z": LaunchConfiguration("world_init_z"),
            "world_init_heading": LaunchConfiguration("world_init_heading"),
            "gui": LaunchConfiguration("gui"),
            "headless": PythonExpression(
                ["'", LaunchConfiguration("gui"), "' == 'false'"]
            ),
            "close_loop_odom": "true",
        }.items(),
    )

    ground_truth_odom_relay = Node(
        package="go2_config",
        executable="ground_truth_odom_relay.py",
        name="ground_truth_odom_relay",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_ground_truth_odom")),
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "input_topic": "/odom/ground_truth",
                "output_topic": "/odom",
                "odom_frame": "odom",
                "child_frame": "base_footprint",
                "publish_tf": True,
                "project_to_2d": True,
            }
        ],
    )

    return [
        declare_gazebo_world,
        declare_use_ground_truth_odom,
        declare_world_init_x,
        declare_world_init_y,
        declare_world_init_z,
        declare_world_init_heading,
        SetEnvironmentVariable(
            "GAZEBO_MODEL_PATH",
            _prepend_paths(model_path, gazebo_model_cache),
        ),
        SetEnvironmentVariable("GAZEBO_MODEL_DATABASE_URI", ""),
        bringup_ld,
        gazebo_ld,
        ground_truth_odom_relay,
    ]


def generate_launch_description():
    declare_scene = DeclareLaunchArgument(
        "scene",
        default_value="forest",
        choices=["qy", "forest", "airport"],
        description="Gazebo scene to load: qy, forest, or airport",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock if true",
    )
    declare_rviz = DeclareLaunchArgument(
        "rviz", default_value="false", description="Launch rviz"
    )
    declare_robot_name = DeclareLaunchArgument(
        "robot_name", default_value="go2", description="Robot name"
    )
    declare_lite = DeclareLaunchArgument("lite", default_value="false")
    declare_gui = DeclareLaunchArgument(
        "gui", default_value="true", description="Use Gazebo GUI"
    )

    return LaunchDescription(
        [
            declare_scene,
            declare_use_sim_time,
            declare_rviz,
            declare_robot_name,
            declare_lite,
            declare_gui,
            OpaqueFunction(function=_launch_setup),
        ]
    )
