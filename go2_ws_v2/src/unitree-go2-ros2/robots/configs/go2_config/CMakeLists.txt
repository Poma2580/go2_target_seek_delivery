cmake_minimum_required(VERSION 3.5)
project(go2_config)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/ground_truth_odom_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
