cmake_minimum_required(VERSION 3.8)
project(walking_target_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(gazebo_dev REQUIRED)
find_package(gazebo_ros REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_srvs REQUIRED)

add_library(walking_target_controller SHARED
  src/walking_target_controller.cpp
)

target_include_directories(walking_target_controller PRIVATE
  ${GAZEBO_INCLUDE_DIRS}
)

ament_target_dependencies(walking_target_controller
  gazebo_dev
  gazebo_ros
  rclcpp
  std_srvs
)

target_link_libraries(walking_target_controller
  ${GAZEBO_LIBRARIES}
)

install(TARGETS walking_target_controller
  LIBRARY DESTINATION lib
)

ament_package()
