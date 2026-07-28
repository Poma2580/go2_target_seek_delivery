cmake_minimum_required(VERSION 3.5)
project(go2_mapping_nav)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/static_goal_nav.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
