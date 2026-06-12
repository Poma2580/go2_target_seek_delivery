"""Spawn a single namespaced Go2 (go2_3) into an already-running Gazebo world.

namespace is baked into the URDF plugins at xacro time (robot_namespace:=/go2_3),
so gazebo_ros2_control's controller_manager lands at /go2_3/controller_manager.
3D Velodyne lidar (no camera) -> /go2_3/velodyne_points. Run gazebo_target_seek_world.launch.py first.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOT_NAME = "go2_3"
SPAWN_X = "-3.5"
SPAWN_Y = "3.9"
SPAWN_Z = "0.50"
SPAWN_YAW = "0.0"


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)

    config_pkg_share = get_package_share_directory("go2_config")
    descr_pkg_share = get_package_share_directory("go2_description")

    model_path = os.path.join(descr_pkg_share, "xacro", "robot_3d_lidar_nocam.xacro")
    joints_config = os.path.join(config_pkg_share, "config", "joints", "joints.yaml")
    links_config = os.path.join(config_pkg_share, "config", "links", "links.yaml")
    gait_config = os.path.join(config_pkg_share, "config", "gait", "gait.yaml")
    ros_control_config = os.path.join(
        config_pkg_share, "config", "ros_control", f"ros_control_{ROBOT_NAME}.yaml"
    )

    controller_manager = f"/{ROBOT_NAME}/controller_manager"
    effort_controller = "joint_group_effort_controller"
    joint_states_controller = "joint_states_controller"

    xacro_cmd = [
        "xacro ",
        model_path,
        " robot_namespace:=/",
        ROBOT_NAME,
        " frame_prefix:=",
        f"{ROBOT_NAME}/",
        " points_topic:=",
        f"/{ROBOT_NAME}/velodyne_points",
        " ros_control_file:=",
        ros_control_config,
        " name_suffix:=",
        ROBOT_NAME,
    ]
    robot_description = {"robot_description": Command(xacro_cmd)}
    robot_urdf = Command(xacro_cmd)

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        namespace=ROBOT_NAME,
        output="screen",
        parameters=[
            robot_description,
            {"use_tf_static": False},
            {"publish_frequency": 200.0},
            {"ignore_timestamp": True},
            {"use_sim_time": use_sim_time_param},
            {"frame_prefix": f"{ROBOT_NAME}/"},
        ],
    )

    quadruped_controller = Node(
        package="champ_base",
        executable="quadruped_controller_node",
        namespace=ROBOT_NAME,
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time_param},
            {"gazebo": True},
            {"publish_joint_states": True},
            {"publish_joint_control": True},
            {"publish_foot_contacts": False},
            {"joint_controller_topic": f"{effort_controller}/joint_trajectory"},
            {"urdf": robot_urdf},
            joints_config,
            links_config,
            gait_config,
        ],
        remappings=[("cmd_vel/smooth", "cmd_vel")],
    )

    state_estimator = Node(
        package="champ_base",
        executable="state_estimation_node",
        namespace=ROBOT_NAME,
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time_param},
            {"orientation_from_imu": False},
            {"urdf": robot_urdf},
            joints_config,
            links_config,
            gait_config,
        ],
    )

    base_to_footprint_ekf = Node(
        package="robot_localization",
        executable="ekf_node",
        namespace=ROBOT_NAME,
        name="base_to_footprint_ekf",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time_param,
                "frequency": 50.0,
                "publish_tf": True,
                "transform_timeout": 0.01,
                "transform_time_offset": 0.045,
                "two_d_mode": False,
                "pose0": "base_to_footprint_pose",
                "pose0_config": [
                    True, True, True,
                    True, True, True,
                    False, False, False,
                    False, False, False,
                    False, False, False,
                ],
                "imu0": "imu/data",
                "imu0_config": [
                    False, False, False,
                    False, False, False,
                    True, True, True,
                    False, False, False,
                    False, False, False,
                ],
                "world_frame": f"{ROBOT_NAME}/base_footprint",
                "odom_frame": f"{ROBOT_NAME}/base_footprint",
                "base_link_frame": f"{ROBOT_NAME}/base_link",
            }
        ],
        remappings=[("odometry/filtered", "odom/local")],
    )

    ground_truth_odom_relay = Node(
        package="go2_config",
        executable="ground_truth_odom_relay.py",
        namespace=ROBOT_NAME,
        name="ground_truth_odom_relay",
        output="screen",
        condition=IfCondition(LaunchConfiguration("use_ground_truth_odom")),
        parameters=[
            {
                "use_sim_time": use_sim_time_param,
                "input_topic": "odom/ground_truth",
                "output_topic": "odom",
                "odom_frame": f"{ROBOT_NAME}/odom",
                "child_frame": f"{ROBOT_NAME}/base_footprint",
                "publish_tf": True,
                "project_to_2d": True,
            }
        ],
    )

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        output="screen",
        arguments=[
            "-entity",
            ROBOT_NAME,
            "-robot_namespace",
            f"/{ROBOT_NAME}",
            "-topic",
            f"/{ROBOT_NAME}/robot_description",
            "-x",
            SPAWN_X,
            "-y",
            SPAWN_Y,
            "-z",
            SPAWN_Z,
            "-R",
            "0",
            "-P",
            "0",
            "-Y",
            SPAWN_YAW,
        ],
    )

    # Use the controller_manager spawner: it patiently waits for the namespaced
    # CM (up to the timeout) and load+configure+activates reliably, even while
    # other robots' plugins are still initializing in the shared gzserver.
    load_effort_controller = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            effort_controller,
            "--controller-manager", controller_manager,
            "--controller-manager-timeout", "120",
        ],
    )

    load_joint_states_controller = Node(
        package="controller_manager",
        executable="spawner",
        output="screen",
        arguments=[
            joint_states_controller,
            "--controller-manager", controller_manager,
            "--controller-manager-timeout", "120",
        ],
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation (Gazebo) clock if true",
            ),
            DeclareLaunchArgument(
                "use_ground_truth_odom",
                default_value="true",
                description="Use Gazebo ground-truth odometry for this robot",
            ),
            robot_state_publisher,
            quadruped_controller,
            state_estimator,
            base_to_footprint_ekf,
            ground_truth_odom_relay,
            spawn_robot,
            load_effort_controller,
            load_joint_states_controller,
        ]
    )
