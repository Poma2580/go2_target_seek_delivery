"""T3-only scene-aware RTAB-Map + Nav2 launcher for one Go2 robot.

This launcher keeps go2_mapping_nav as the source of the complete RTAB-Map/Nav2
base configuration, and only overrides the small set of parameters that differ
between T3 scenes.
"""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, LogInfo, OpaqueFunction, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml


ROBOTS = ("go2_1", "go2_2", "go2_3")
SCENES = ("city", "forest", "airport")


def _arg(context, name):
    return LaunchConfiguration(name).perform(context).strip()


def _bool_arg(context, name):
    value = _arg(context, name).lower()
    if value not in ("true", "false"):
        raise RuntimeError(f"{name} must be true or false")
    return value == "true"


def _load_scene_profile(scene):
    profile_path = (
        Path(get_package_share_directory("go2_test_framework"))
        / "config"
        / "parameters"
        / "nav2_scene_profiles.yaml"
    )
    try:
        root = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise RuntimeError(
            f"failed to read T3 Nav2 scene profile {profile_path}: {error}"
        ) from error

    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise RuntimeError(f"invalid T3 Nav2 scene profile: {profile_path}")

    scenes = root.get("scenes")
    if not isinstance(scenes, dict) or scene not in scenes:
        raise RuntimeError(f"Nav2 scene profile for {scene!r} is not defined")

    profile = scenes[scene]
    required = (
        "min_height",
        "max_height",
        "cost_scaling_factor",
        "inflation_radius",
    )
    if not isinstance(profile, dict) or any(key not in profile for key in required):
        raise RuntimeError(f"incomplete Nav2 scene profile for {scene!r}")

    try:
        resolved = {key: float(profile[key]) for key in required}
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            f"Nav2 scene profile for {scene!r} contains non-numeric values"
        ) from error

    if resolved["min_height"] >= resolved["max_height"]:
        raise RuntimeError(
            f"invalid height range for scene {scene!r}: "
            f"{resolved['min_height']} >= {resolved['max_height']}"
        )
    if resolved["cost_scaling_factor"] <= 0.0:
        raise RuntimeError("cost_scaling_factor must be greater than zero")
    if resolved["inflation_radius"] <= 0.0:
        raise RuntimeError("inflation_radius must be greater than zero")

    return resolved


def _default_database_path(robot):
    delivery_root = os.environ.get("DELIVERY_ROOT")
    if delivery_root:
        return (
            Path(delivery_root)
            / "go2_ws_v2"
            / "src"
            / "go2_mapping_nav"
            / "runtime"
            / "maps"
            / f"{robot}_mapping.db"
        )
    return (
        Path.home()
        / ".go2_target_seek_delivery"
        / "runtime"
        / "maps"
        / f"{robot}_mapping.db"
    )


def _value_or_default(context, name, default):
    value = _arg(context, name)
    return value if value else default


def _launch_setup(context):
    robot = _arg(context, "robot_name")
    scene = _arg(context, "scene")

    if robot not in ROBOTS:
        raise RuntimeError(
            f"robot_name must be one of {', '.join(ROBOTS)}, got {robot!r}"
        )
    if scene not in SCENES:
        raise RuntimeError(
            f"scene must be one of {', '.join(SCENES)}, got {scene!r}"
        )

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_sim_time_bool = _bool_arg(context, "use_sim_time")
    use_merged_map = _bool_arg(context, "use_merged_map")
    delete_db_on_start = _bool_arg(context, "delete_db_on_start")

    profile = _load_scene_profile(scene)

    mapping_share = Path(get_package_share_directory("go2_mapping_nav"))
    rtabmap_config = mapping_share / "config" / "rtabmap" / f"{robot}_mapping.yaml"
    nav2_config = mapping_share / "config" / "nav2" / f"{robot}_nav2.yaml"
    rviz_config = mapping_share / "rviz" / f"{robot}_mapping_nav.rviz"

    if not rtabmap_config.is_file():
        raise RuntimeError(f"RTAB-Map config not found: {rtabmap_config}")
    if not nav2_config.is_file():
        raise RuntimeError(f"Nav2 config not found: {nav2_config}")

    cloud_topic = _value_or_default(
        context, "cloud_topic", f"/{robot}/velodyne_points"
    )
    scan_topic = _value_or_default(context, "scan_topic", f"/{robot}/scan")
    odom_topic = _value_or_default(context, "odom_topic", f"/{robot}/odom")
    map_topic = _value_or_default(context, "map_topic", f"/{robot}/map")
    cmd_vel_topic = _value_or_default(
        context, "cmd_vel_topic", f"/{robot}/cmd_vel"
    )
    raw_cmd_vel_topic = _value_or_default(
        context, "raw_cmd_vel_topic", f"/{robot}/raw_cmd_nav_vel"
    )

    database_value = _arg(context, "database_path")
    database_path = (
        Path(database_value).expanduser()
        if database_value
        else _default_database_path(robot)
    )
    if database_path.exists() and database_path.is_dir():
        raise RuntimeError(
            f"database_path must be a file, got directory: {database_path}"
        )
    database_path.parent.mkdir(parents=True, exist_ok=True)

    actions = [
        LogInfo(
            msg=(
                f"[T3 nav] robot={robot}, scene={scene}, "
                f"min_height={profile['min_height']}, "
                f"max_height={profile['max_height']}, "
                f"cost_scaling_factor={profile['cost_scaling_factor']}, "
                f"inflation_radius={profile['inflation_radius']}"
            )
        ),
        LogInfo(msg=f"RTAB-Map database: {database_path}"),
    ]

    if delete_db_on_start and database_path.exists():
        database_path.unlink()
        actions.append(LogInfo(msg=f"Deleted RTAB-Map database: {database_path}"))

    nav_global_frame = "merged_map" if use_merged_map else f"{robot}/map"
    nav_map_topic = "/merged_map" if use_merged_map else f"/{robot}/map"

    configured_nav2_params = ParameterFile(
        RewrittenYaml(
            source_file=str(nav2_config),
            root_key=robot,
            param_rewrites={
                # Keep the original go2_mapping_nav runtime rewrites.
                "use_sim_time": use_sim_time,
                "bt_navigator.ros__parameters.global_frame": nav_global_frame,
                "global_costmap.global_costmap.ros__parameters.global_frame": (
                    nav_global_frame
                ),
                "global_costmap.global_costmap.ros__parameters.static_layer.map_topic": (
                    nav_map_topic
                ),

                # T3 scene-specific overrides.
                "local_costmap.local_costmap.ros__parameters."
                "inflation_layer.cost_scaling_factor": str(
                    profile["cost_scaling_factor"]
                ),
                "local_costmap.local_costmap.ros__parameters."
                "inflation_layer.inflation_radius": str(
                    profile["inflation_radius"]
                ),
                "global_costmap.global_costmap.ros__parameters."
                "inflation_layer.cost_scaling_factor": str(
                    profile["cost_scaling_factor"]
                ),
                "global_costmap.global_costmap.ros__parameters."
                "inflation_layer.inflation_radius": str(
                    profile["inflation_radius"]
                ),
            },
            convert_types=True,
        ),
        allow_substs=True,
    )

    lifecycle_nodes = [
        "controller_server",
        "smoother_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
        "velocity_smoother",
    ]

    nav2_nodes = GroupAction(
        [
            PushRosNamespace(robot),
            Node(
                package="nav2_controller",
                executable="controller_server",
                name="controller_server",
                output="screen",
                parameters=[configured_nav2_params],
                remappings=[("cmd_vel", raw_cmd_vel_topic), ("odom", odom_topic)],
            ),
            Node(
                package="nav2_smoother",
                executable="smoother_server",
                name="smoother_server",
                output="screen",
                parameters=[configured_nav2_params],
                remappings=[("odom", odom_topic)],
            ),
            Node(
                package="nav2_planner",
                executable="planner_server",
                name="planner_server",
                output="screen",
                parameters=[configured_nav2_params],
            ),
            Node(
                package="nav2_behaviors",
                executable="behavior_server",
                name="behavior_server",
                output="screen",
                parameters=[configured_nav2_params],
                remappings=[("cmd_vel", raw_cmd_vel_topic), ("odom", odom_topic)],
            ),
            Node(
                package="nav2_bt_navigator",
                executable="bt_navigator",
                name="bt_navigator",
                output="screen",
                parameters=[configured_nav2_params],
            ),
            Node(
                package="nav2_waypoint_follower",
                executable="waypoint_follower",
                name="waypoint_follower",
                output="screen",
                parameters=[configured_nav2_params],
            ),
            Node(
                package="nav2_velocity_smoother",
                executable="velocity_smoother",
                name="velocity_smoother",
                output="screen",
                parameters=[configured_nav2_params],
                remappings=[
                    ("cmd_vel", raw_cmd_vel_topic),
                    ("cmd_vel_smoothed", cmd_vel_topic),
                    ("odom", odom_topic),
                ],
            ),
            TimerAction(
                period=10.0,
                actions=[
                    Node(
                        package="nav2_lifecycle_manager",
                        executable="lifecycle_manager",
                        namespace=robot,
                        name="lifecycle_manager_navigation",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time_bool,
                                "autostart": True,
                                "bond_timeout": 30.0,
                                "node_names": lifecycle_nodes,
                            }
                        ],
                    )
                ],
            ),
        ]
    )

    actions.extend(
        [
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name=f"{robot}_pointcloud_to_laserscan",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time_bool,
                        "target_frame": f"{robot}/velodyne",
                        "transform_tolerance": 0.1,
                        "min_height": profile["min_height"],
                        "max_height": profile["max_height"],
                        "angle_min": -3.14159,
                        "angle_max": 3.14159,
                        "angle_increment": 0.0143,
                        "scan_time": 0.1,
                        "range_min": 0.55,
                        "range_max": 20.0,
                        "use_inf": True,
                        "inf_epsilon": 1.0,
                    }
                ],
                remappings=[("cloud_in", cloud_topic), ("scan", scan_topic)],
            ),
            Node(
                package="rtabmap_slam",
                executable="rtabmap",
                namespace=robot,
                name="rtabmap",
                output="screen",
                parameters=[
                    str(rtabmap_config),
                    {
                        "use_sim_time": use_sim_time_bool,
                        "database_path": str(database_path),
                    },
                ],
                remappings=[
                    ("scan", scan_topic),
                    ("odom", odom_topic),
                    ("map", map_topic),
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{robot}_map_to_odom",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("publish_map_to_odom_tf")
                ),
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    f"{robot}/map",
                    f"{robot}/odom",
                ],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name=f"{robot}_base_footprint_to_base_link",
                output="screen",
                condition=IfCondition(
                    LaunchConfiguration("publish_base_footprint_tf")
                ),
                arguments=[
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    "0",
                    f"{robot}/base_footprint",
                    f"{robot}/base_link",
                ],
            ),
            nav2_nodes,
            Node(
                package="rviz2",
                executable="rviz2",
                name=f"{robot}_mapping_nav_rviz",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                arguments=["-d", str(rviz_config)],
                parameters=[{"use_sim_time": use_sim_time_bool}],
                additional_env={"GTK_PATH": ""},
            ),
        ]
    )

    return actions


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_name"),
            DeclareLaunchArgument("scene"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_merged_map", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("delete_db_on_start", default_value="true"),
            DeclareLaunchArgument("database_path", default_value=""),
            DeclareLaunchArgument("cloud_topic", default_value=""),
            DeclareLaunchArgument("scan_topic", default_value=""),
            DeclareLaunchArgument("odom_topic", default_value=""),
            DeclareLaunchArgument("map_topic", default_value=""),
            DeclareLaunchArgument("cmd_vel_topic", default_value=""),
            DeclareLaunchArgument("raw_cmd_vel_topic", default_value=""),
            DeclareLaunchArgument("publish_map_to_odom_tf", default_value="true"),
            DeclareLaunchArgument(
                "publish_base_footprint_tf", default_value="false"
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
