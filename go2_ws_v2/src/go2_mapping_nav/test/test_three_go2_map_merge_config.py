"""Static integration checks for local/shared-map Nav2 launch configuration."""

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).parents[1]
DELIVERY_ROOT = PACKAGE_ROOT.parents[2]


def test_all_nav2_stacks_default_to_their_local_map():
    for index in (1, 2, 3):
        robot = f"go2_{index}"
        config = yaml.safe_load(
            (
                PACKAGE_ROOT
                / "config"
                / "nav2"
                / f"{robot}_nav2.yaml"
            ).read_text(encoding="utf-8")
        )
        assert config["bt_navigator"]["ros__parameters"]["global_frame"] == (
            f"{robot}/map"
        )
        global_parameters = config["global_costmap"]["global_costmap"][
            "ros__parameters"
        ]
        assert global_parameters["global_frame"] == f"{robot}/map"
        assert global_parameters["static_layer"]["map_topic"] == f"/{robot}/map"

        local_parameters = config["local_costmap"]["local_costmap"][
            "ros__parameters"
        ]
        assert local_parameters["global_frame"] == f"{robot}/odom"
        assert (
            local_parameters["obstacle_layer"]["scan"]["topic"]
            == f"/{robot}/scan"
        )


def test_map_merge_launch_has_one_merger_and_three_unique_root_transforms():
    launch_text = (
        PACKAGE_ROOT / "launch" / "three_go2_map_merge.launch.py"
    ).read_text(encoding="utf-8")
    assert launch_text.count('executable="known_pose_map_merger.py"') == 1
    assert "ROBOT_NAMES = (\"go2_1\", \"go2_2\", \"go2_3\")" in launch_text
    assert 'f"merged_map_to_{robot_name}_map"' in launch_text
    assert '"merged_map",' in launch_text
    assert 'f"{robot_name}/map",' in launch_text


def test_individual_launches_default_to_local_map_and_rewrite_exact_paths():
    for index in (1, 2, 3):
        launch_text = (
            PACKAGE_ROOT
            / "launch"
            / f"go2_{index}_mapping_nav.launch.py"
        ).read_text(encoding="utf-8")
        assert (
            'DeclareLaunchArgument("use_rviz", default_value="false")'
            in launch_text
        )
        assert (
            'DeclareLaunchArgument("use_merged_map", default_value="false")'
            in launch_text
        )
        assert '"bt_navigator.ros__parameters.global_frame"' in launch_text
        assert (
            '"global_costmap.global_costmap.ros__parameters.global_frame"'
            in launch_text
        )
        assert (
            '"global_costmap.global_costmap.ros__parameters.static_layer.map_topic"'
            in launch_text
        )
        assert '"global_frame": nav_global_frame' not in launch_text


def test_unified_rviz_uses_merged_map_and_contains_all_robots():
    config = yaml.safe_load(
        (
            PACKAGE_ROOT / "rviz" / "three_go2_mapping_nav.rviz"
        ).read_text(encoding="utf-8")
    )
    manager = config["Visualization Manager"]
    assert manager["Global Options"]["Fixed Frame"] == "merged_map"
    serialized = yaml.safe_dump(config)
    assert "/merged_map" in serialized
    for index in (1, 2, 3):
        assert f"/go2_{index}/robot_description" in serialized
        assert f"/go2_{index}/plan" in serialized


def test_goal_and_start_scripts_select_the_requested_map_mode():
    goal_script = (
        DELIVERY_ROOT / "Scripts" / "send_go2_1_static_goal.sh"
    ).read_text(encoding="utf-8")
    assert "--map-mode auto|local|merged" in goal_script
    assert '"${ROS2_BIN}" param get "${BT_NAVIGATOR}" global_frame' in goal_script
    assert 'MAP_TOPIC="/${ROBOT_NAME}/map"' in goal_script
    assert 'MAP_TOPIC="/merged_map"' in goal_script

    start_script = (
        DELIVERY_ROOT / "Scripts" / "start_three_go2_velodyne.sh"
    ).read_text(encoding="utf-8")
    assert start_script.count("use_merged_map:=true") == 1
    assert "three_go2_map_merge.launch.py" in start_script
    assert 'wait_for_topic_message "/merged_map"' in start_script
    assert "three_go2_mapping_nav.rviz" in start_script

    single_start_script = (
        DELIVERY_ROOT / "Scripts" / "start_go2_1_mapping_nav.sh"
    ).read_text(encoding="utf-8")
    assert "use_merged_map:=false" in single_start_script
