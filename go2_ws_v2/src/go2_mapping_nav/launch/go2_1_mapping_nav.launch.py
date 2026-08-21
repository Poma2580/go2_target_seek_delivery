"""Launch online RTAB-Map mapping and Nav2 static-goal navigation for go2_1.

The Gazebo world and robot are intentionally started by a separate launcher so
this file never starts teleoperation, perception, tracking, or other robots.
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import IfElseSubstitution, LaunchConfiguration
from launch_ros.actions import Node, PushRosNamespace
from launch_ros.parameter_descriptions import ParameterFile, ParameterValue
from nav2_common.launch import RewrittenYaml


def _prepare_database(context):
    database_path = Path(
        LaunchConfiguration("database_path").perform(context)
    ).expanduser()
    delete_on_start = LaunchConfiguration("delete_db_on_start").perform(
        context
    ).strip().lower()

    if delete_on_start not in ("true", "false"):
        raise RuntimeError("delete_db_on_start must be true or false")
    if database_path.exists() and database_path.is_dir():
        raise RuntimeError(f"database_path must be a file, got directory: {database_path}")

    database_path.parent.mkdir(parents=True, exist_ok=True)
    actions = [LogInfo(msg=f"RTAB-Map database: {database_path}")]
    if delete_on_start == "true" and database_path.exists():
        database_path.unlink()
        actions.append(LogInfo(msg=f"Deleted RTAB-Map database: {database_path}"))
    return actions


def generate_launch_description():
    delivery_root = os.environ.get("DELIVERY_ROOT")
    if delivery_root:
        runtime_root = Path(delivery_root)
        missing_root_warning = []
    else:
        runtime_root = Path.home() / ".go2_target_seek_delivery"
        missing_root_warning = [
            LogInfo(
                msg=(
                    "WARNING: DELIVERY_ROOT is not set; using "
                    f"{runtime_root} for runtime data."
                )
            )
        ]

    config_share = get_package_share_directory("go2_mapping_nav")
    rtabmap_config = os.path.join(
        config_share, "config", "rtabmap", "go2_1_mapping.yaml"
    )
    nav2_config = os.path.join(config_share, "config", "nav2", "go2_1_nav2.yaml")
    rviz_config = os.path.join(config_share, "rviz", "go2_1_mapping_nav.rviz")

    use_sim_time = LaunchConfiguration("use_sim_time")
    use_merged_map = LaunchConfiguration("use_merged_map")
    use_sim_time_parameter = ParameterValue(use_sim_time, value_type=bool)
    cloud_topic = LaunchConfiguration("cloud_topic")
    scan_topic = LaunchConfiguration("scan_topic")
    odom_topic = LaunchConfiguration("odom_topic")
    map_topic = LaunchConfiguration("map_topic")
    cmd_vel_topic = LaunchConfiguration("cmd_vel_topic")
    raw_cmd_vel_topic = LaunchConfiguration("raw_cmd_vel_topic")
    database_path = LaunchConfiguration("database_path")
    nav_global_frame = IfElseSubstitution(
        condition=use_merged_map,
        if_value="merged_map",
        else_value="go2_1/map",
    )
    nav_map_topic = IfElseSubstitution(
        condition=use_merged_map,
        if_value="/merged_map",
        else_value="/go2_1/map",
    )
    configured_nav2_params = ParameterFile(
        RewrittenYaml(
            source_file=nav2_config,
            root_key="go2_1",
            param_rewrites={
                "use_sim_time": use_sim_time,
                "bt_navigator.ros__parameters.global_frame": nav_global_frame,
                "global_costmap.global_costmap.ros__parameters.global_frame": (
                    nav_global_frame
                ),
                "global_costmap.global_costmap.ros__parameters.static_layer.map_topic": (
                    nav_map_topic
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
            PushRosNamespace("go2_1"),
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
                        namespace="go2_1",
                        name="lifecycle_manager_navigation",
                        output="screen",
                        parameters=[
                            {
                                "use_sim_time": use_sim_time_parameter,
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

    return LaunchDescription(
        missing_root_warning
        + [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("use_merged_map", default_value="false"),
            DeclareLaunchArgument("use_rviz", default_value="false"),
            DeclareLaunchArgument("delete_db_on_start", default_value="true"),
            DeclareLaunchArgument(
                "database_path",
                default_value=str(
                    runtime_root
                    / "go2_ws_v2"
                    / "src"
                    / "go2_mapping_nav"
                    / "runtime"
                    / "maps"
                    / "go2_1_mapping.db"
                )
                if delivery_root
                else str(runtime_root / "runtime" / "maps" / "go2_1_mapping.db"),
            ),
            DeclareLaunchArgument(
                "cloud_topic", default_value="/go2_1/velodyne_points"
            ),
            DeclareLaunchArgument("scan_topic", default_value="/go2_1/scan"),
            DeclareLaunchArgument("odom_topic", default_value="/go2_1/odom"),
            DeclareLaunchArgument("map_topic", default_value="/go2_1/map"),
            DeclareLaunchArgument("cmd_vel_topic", default_value="/go2_1/cmd_vel"),
            DeclareLaunchArgument("raw_cmd_vel_topic", default_value="/go2_1/raw_cmd_nav_vel"),
            DeclareLaunchArgument("publish_map_to_odom_tf", default_value="true"),
            DeclareLaunchArgument(
                "publish_base_footprint_tf", default_value="false"
            ),
            OpaqueFunction(function=_prepare_database),
            Node(
                package="pointcloud_to_laserscan",
                executable="pointcloud_to_laserscan_node",
                name="go2_1_pointcloud_to_laserscan",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": use_sim_time_parameter,
                        "target_frame": "go2_1/velodyne",
                        "transform_tolerance": 0.1,
                        "min_height": 0.10,
                        "max_height": 0.50,
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
                namespace="go2_1",
                name="rtabmap",
                output="screen",
                parameters=[
                    rtabmap_config,
                    {
                        "use_sim_time": use_sim_time_parameter,
                        "database_path": database_path,
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
                name="go2_1_map_to_odom",
                output="screen",
                condition=IfCondition(LaunchConfiguration("publish_map_to_odom_tf")),
                arguments=["0", "0", "0", "0", "0", "0", "go2_1/map", "go2_1/odom"],
            ),
            Node(
                package="tf2_ros",
                executable="static_transform_publisher",
                name="go2_1_base_footprint_to_base_link",
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
                    "go2_1/base_footprint",
                    "go2_1/base_link",
                ],
            ),
            nav2_nodes,
            Node(
                package="rviz2",
                executable="rviz2",
                name="go2_1_mapping_nav_rviz",
                output="screen",
                condition=IfCondition(LaunchConfiguration("use_rviz")),
                arguments=["-d", rviz_config],
                parameters=[{"use_sim_time": use_sim_time_parameter}],
                # VS Code installed via Snap exports a GTK_PATH under /snap/code.
                # Loading those GTK modules into the host ROS process pulls in an
                # incompatible core20 libpthread and makes RViz exit immediately.
                additional_env={"GTK_PATH": ""},
            ),
        ]
    )
