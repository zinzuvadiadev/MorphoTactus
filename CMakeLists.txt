cmake_minimum_required(VERSION 3.8)
project(MorphoTactus)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(message_generation REQUIRED)
find_package(message_runtime REQUIRED)

# Generate custom messages
add_message_files(
  FILES
  DefectInfo.msg
  DefectMetadata.msg
)

# Generate message interfaces
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
  sensor_msgs
)

# Install Python modules
# ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/image_source_node.py
  scripts/preprocessing_node.py
  scripts/hybrid_detector_node.py
  scripts/video_viewer_node.py
  scripts/simple_video_viewer.py
  scripts/realtime_video_viewer.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  msg
  DESTINATION share/${PROJECT_NAME}
)

# Install Python modules
install(DIRECTORY
  MorphoTactus
  DESTINATION lib/python3/dist-packages/
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 