"""Launch the MADDPG selector; Nav2 remains the only velocity controller."""

import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    root = Path(os.environ.get("DELIVERY_ROOT", Path.home() / "go2_target_seek_delivery-main"))
    default_model = (
        root
        / "waypoint_maddpg_v0/runs/two_obstacles_108rays_final_gpu_20260826/best_model.pt"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("enabled", default_value="true"),
            DeclareLaunchArgument("model_path", default_value=str(default_model)),
            DeclareLaunchArgument("leader_name", default_value="go2_1"),
            DeclareLaunchArgument("follower_1", default_value="go2_2"),
            DeclareLaunchArgument("follower_2", default_value="go2_3"),
            DeclareLaunchArgument("global_frame", default_value="merged_map"),
            DeclareLaunchArgument("decision_period", default_value="1.0"),
            DeclareLaunchArgument("nav_goal_update_period", default_value="3.0"),
            DeclareLaunchArgument("track_pedestrian", default_value="true"),
            DeclareLaunchArgument("leader_speed_tolerance", default_value="0.15"),
            DeclareLaunchArgument("follower_speed_tolerance", default_value="0.20"),
            DeclareLaunchArgument("initial_formation_tolerance", default_value="0.5"),
            Node(
                package="multi_go2_waypoint",
                executable="actor_state_publisher",
                name="actor_state_publisher",
                output="screen",
                condition=IfCondition(LaunchConfiguration("track_pedestrian")),
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        )
                    }
                ],
            ),
            Node(
                package="go2_mapping_nav",
                executable="leader_pedestrian_tracker.py",
                name="leader_pedestrian_tracker",
                output="screen",
                condition=IfCondition(LaunchConfiguration("track_pedestrian")),
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "leader_name": LaunchConfiguration("leader_name"),
                        "auto_start_target": True,
                        "catch_speed": 0.20,
                        "max_linear": 0.20,
                    }
                ],
            ),
            Node(
                package="go2_mapping_nav",
                executable="maddpg_waypoint_selector.py",
                name="maddpg_waypoint_selector",
                output="screen",
                parameters=[
                    {
                        "use_sim_time": ParameterValue(
                            LaunchConfiguration("use_sim_time"), value_type=bool
                        ),
                        "dry_run": ParameterValue(
                            LaunchConfiguration("dry_run"), value_type=bool
                        ),
                        "enabled": ParameterValue(
                            LaunchConfiguration("enabled"), value_type=bool
                        ),
                        "model_path": LaunchConfiguration("model_path"),
                        "leader_name": LaunchConfiguration("leader_name"),
                        "follower_1": LaunchConfiguration("follower_1"),
                        "follower_2": LaunchConfiguration("follower_2"),
                        "global_frame": LaunchConfiguration("global_frame"),
                        "decision_period": ParameterValue(
                            LaunchConfiguration("decision_period"), value_type=float
                        ),
                        "nav_goal_update_period": ParameterValue(
                            LaunchConfiguration("nav_goal_update_period"),
                            value_type=float,
                        ),
                        "leader_speed_tolerance": ParameterValue(
                            LaunchConfiguration("leader_speed_tolerance"),
                            value_type=float,
                        ),
                        "speed_tolerance": ParameterValue(
                            LaunchConfiguration("follower_speed_tolerance"),
                            value_type=float,
                        ),
                        "initial_formation_tolerance": ParameterValue(
                            LaunchConfiguration("initial_formation_tolerance"),
                            value_type=float,
                        ),
                    }
                ],
            ),
        ]
    )
