cmake_minimum_required(VERSION 3.8)
project(walking_target_controller)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
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

ament_python_install_package(${PROJECT_NAME})

install(TARGETS walking_target_controller
  LIBRARY DESTINATION lib
)

install(
  PROGRAMS scripts/actor_state_publisher
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    test_actor_state_publisher
    test/test_actor_state_publisher.py
    ENV "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1"
  )
endif()

ament_package()
