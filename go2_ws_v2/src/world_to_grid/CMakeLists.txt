cmake_minimum_required(VERSION 3.8)
project(world_to_grid)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(assimp REQUIRED)
find_package(ignition-math6 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(TINYXML2 REQUIRED IMPORTED_TARGET tinyxml2)

add_executable(world_to_grid src/world_to_grid.cpp)
target_compile_options(world_to_grid PRIVATE -Wall -Wextra -Wpedantic)
target_link_libraries(world_to_grid PRIVATE
  assimp::assimp
  ignition-math6::ignition-math6
  PkgConfig::TINYXML2
)

install(TARGETS world_to_grid
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  enable_testing()
  add_test(
    NAME world_to_grid_cli_tests
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/test/test_world_to_grid.py
      $<TARGET_FILE:world_to_grid>
  )
  set(AIRPORT_WORLD "${CMAKE_CURRENT_SOURCE_DIR}/../../../KD_MODEL/world/airport")
  if(EXISTS "${AIRPORT_WORLD}")
    add_test(
      NAME world_to_grid_airport_integration
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/test/test_airport_integration.py
        $<TARGET_FILE:world_to_grid>
        ${AIRPORT_WORLD}
    )
    set_tests_properties(world_to_grid_airport_integration PROPERTIES TIMEOUT 60)
  endif()
endif()

ament_package()
