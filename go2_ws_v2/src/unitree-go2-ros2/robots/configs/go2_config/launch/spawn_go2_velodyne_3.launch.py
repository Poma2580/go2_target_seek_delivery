"""Spawn a single namespaced Go2 (go2_3) into an already-running Gazebo world.

namespace is baked into the URDF plugins at xacro time (robot_namespace:=/go2_3),
so gazebo_ros2_control's controller_manager lands at /go2_3/controller_manager.
3D Velodyne lidar (no camera) -> /go2_3/velodyne_points.
Run gazebo_target_seek_world.launch.py first.
"""

import math
import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    OpaqueFunction,
    SetLaunchConfiguration,
)
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOT_NAME = "go2_3"
VALID_SCENES = ("city", "forest", "airport")


def _mapping(value, field):
    if not isinstance(value, dict):
        raise RuntimeError(f"{field} must be a YAML mapping")
    return value


def _finite_number(value, field):
    if isinstance(value, bool):
        raise RuntimeError(f"{field} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{field} must be numeric") from error
    if not math.isfinite(number):
        raise RuntimeError(f"{field} must be finite")
    return number


def _configure_scene(context):
    scene = LaunchConfiguration("scene").perform(context).strip().lower()
    if scene not in VALID_SCENES:
        raise RuntimeError(
            f"scene must be one of {', '.join(VALID_SCENES)}, got {scene!r}"
        )

    configured_path = LaunchConfiguration("scene_config").perform(context).strip()
    if configured_path:
        config_path = Path(configured_path).expanduser().resolve()
    else:
        config_path = Path(
            get_package_share_directory("go2_scenario_config")
        ) / "config" / "scenes" / f"{scene}.yaml"

    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(
            f"failed to read scene config {config_path}: {error}"
        ) from error

    root = _mapping(config, "root")
    if root.get("schema_version") != 1:
        raise RuntimeError("scene config schema_version must be 1")
    if root.get("scene") != scene:
        raise RuntimeError(
            f"scene config declares {root.get('scene')!r}, expected {scene!r}"
        )

    robots = _mapping(root.get("robots"), "robots")
    robot = _mapping(robots.get(ROBOT_NAME), f"robots.{ROBOT_NAME}")
    spawn = _mapping(robot.get("spawn"), f"robots.{ROBOT_NAME}.spawn")
    sensors = _mapping(robot.get("sensors"), f"robots.{ROBOT_NAME}.sensors")

    resolved = {}
    for argument, key in (
        ("spawn_x", "x"),
        ("spawn_y", "y"),
        ("spawn_z", "z"),
        ("spawn_yaw", "yaw"),
    ):
        yaml_value = _finite_number(
            spawn.get(key), f"robots.{ROBOT_NAME}.spawn.{key}"
        )
        override = LaunchConfiguration(argument).perform(context).strip()
        value = (
            _finite_number(override, argument)
            if override
            else yaml_value
        )
        if argument == "spawn_z" and value <= 0.0:
            raise RuntimeError("spawn_z must be greater than zero")
        resolved[argument] = f"{value:g}"

    for argument, key in (
        ("enable_lidar", "lidar"),
        ("enable_camera", "camera"),
    ):
        yaml_value = sensors.get(key)
        if not isinstance(yaml_value, bool):
            raise RuntimeError(
                f"robots.{ROBOT_NAME}.sensors.{key} must be boolean"
            )
        override = LaunchConfiguration(argument).perform(context).strip().lower()
        if override == "auto":
            value = yaml_value
        elif override in ("true", "false"):
            value = override == "true"
        else:
            raise RuntimeError(f"{argument} must be auto, true or false")
        resolved[argument] = "true" if value else "false"

    return [
        SetLaunchConfiguration(name, value)
        for name, value in resolved.items()
    ]


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_param = ParameterValue(use_sim_time, value_type=bool)
    spawn_x = LaunchConfiguration("spawn_x")
    spawn_y = LaunchConfiguration("spawn_y")
    spawn_z = LaunchConfiguration("spawn_z")
    spawn_yaw = LaunchConfiguration("spawn_yaw")

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
        " enable_velodyne:=",
        LaunchConfiguration("enable_lidar"),
        " enable_camera:=",
        LaunchConfiguration("enable_camera"),
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
            spawn_x,
            "-y",
            spawn_y,
            "-z",
            spawn_z,
            "-R",
            "0",
            "-P",
            "0",
            "-Y",
            spawn_yaw,
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
            DeclareLaunchArgument(
                "scene",
                default_value="city",
                description="Scene whose YAML provides spawn and sensor defaults",
            ),
            DeclareLaunchArgument(
                "scene_config",
                default_value="",
                description="Optional scene YAML path overriding the installed file",
            ),
            DeclareLaunchArgument(
                "spawn_x",
                default_value="",
                description="Gazebo spawn x override (m); empty uses scene YAML",
            ),
            DeclareLaunchArgument(
                "spawn_y",
                default_value="",
                description="Gazebo spawn y override (m); empty uses scene YAML",
            ),
            DeclareLaunchArgument(
                "spawn_z",
                default_value="",
                description="Gazebo spawn z override (m); empty uses scene YAML",
            ),
            DeclareLaunchArgument(
                "spawn_yaw",
                default_value="",
                description="Gazebo spawn yaw override; empty uses scene YAML",
            ),
            DeclareLaunchArgument(
                "enable_lidar",
                default_value="auto",
                description="auto reads scene YAML; true/false overrides it",
            ),
            DeclareLaunchArgument(
                "enable_camera",
                default_value="auto",
                description="auto reads scene YAML; true/false overrides it",
            ),
            OpaqueFunction(function=_configure_scene),
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
