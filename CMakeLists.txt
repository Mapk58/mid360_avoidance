cmake_minimum_required(VERSION 3.0.2)
project(mid360_avoidance)

add_compile_options(-O3 -march=native -flto -std=c++14)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  livox_ros_driver2
)

catkin_package()

include_directories(${catkin_INCLUDE_DIRS})

add_executable(livox_plane_filter src/livox_plane_filter.cpp)
target_link_libraries(livox_plane_filter ${catkin_LIBRARIES})
