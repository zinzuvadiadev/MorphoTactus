#!/bin/bash

# MorphoTactus Build Script
# Builds the ROS 2 package with proper message generation

echo "Building MorphoTactus package..."

# Check if we're in a ROS 2 workspace
if [ ! -f "package.xml" ]; then
    echo "Error: This script must be run from the MorphoTactus package directory"
    exit 1
fi

# Get the workspace root (assuming we're in src/workspace_name/MorphoTactus)
WORKSPACE_ROOT=$(pwd | sed 's|/src/.*||')
echo "Workspace root: $WORKSPACE_ROOT"

# Build the package
echo "Building package..."
cd "$WORKSPACE_ROOT"
colcon build --packages-select MorphoTactus --cmake-args -DCMAKE_BUILD_TYPE=Release

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Source the workspace: source install/setup.bash"
    echo "Run the system: ros2 launch MorphoTactus defect_detection_system.launch.py"
else
    echo "Build failed!"
    exit 1
fi 