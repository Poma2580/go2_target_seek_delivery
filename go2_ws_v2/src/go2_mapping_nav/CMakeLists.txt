cmake_minimum_required(VERSION 3.5)
project(go2_mapping_nav)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/diagnose_lidar_ground.py
    scripts/gazebo_maddpg_finetuner.py
    scripts/known_pose_map_merger.py
    scripts/leader_pedestrian_tracker.py
    scripts/maddpg_waypoint_selector.py
    scripts/maddpg_waypoint_monitor.py
    scripts/static_goal_nav.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    test_maddpg_waypoint_contract
    test/test_maddpg_waypoint_contract.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;DELIVERY_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  )
  ament_add_pytest_test(
    test_gazebo_maddpg_finetuner
    test/test_gazebo_maddpg_finetuner.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1;DELIVERY_ROOT=${CMAKE_CURRENT_SOURCE_DIR}/../../.."
  )
  ament_add_pytest_test(
    test_known_pose_map_merger
    test/test_known_pose_map_merger.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_three_go2_map_merge_config
    test/test_three_go2_map_merge_config.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
  ament_add_pytest_test(
    test_send_static_goal_modes
    test/test_send_static_goal_modes.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
endif()

ament_package()
