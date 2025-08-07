#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get package directory
    pkg_share = FindPackageShare('MorphoTactus')
    
    # Launch arguments
    config_file = LaunchConfiguration('config_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    enable_debug = LaunchConfiguration('enable_debug')
    
    # Declare launch arguments
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'detector_params.yaml']),
        description='Path to configuration file'
    )
    
    declare_use_sim_time_cmd = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation time'
    )
    
    declare_enable_debug_cmd = DeclareLaunchArgument(
        'enable_debug',
        default_value='false',
        description='Enable debug mode'
    )
    
    # Jetson Nano optimizations
    jetson_optimizations = {
        'OMP_NUM_THREADS': '4',
        'CUDA_VISIBLE_DEVICES': '0',
        'TF_CPP_MIN_LOG_LEVEL': '2',
        'OPENCV_OPENCL_RUNTIME': '0'
    }
    
    # Image Source Node
    image_source_node = Node(
        package='MorphoTactus',
        executable='image_source_node',
        name='image_source_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
        ],
        env=jetson_optimizations
    )
    
    # Preprocessing Node
    preprocessing_node = Node(
        package='MorphoTactus',
        executable='preprocessing_node',
        name='preprocessing_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('image_raw', '/camera/image_raw'),
            ('preprocessed_image', '/preprocessed/image'),
        ],
        env=jetson_optimizations
    )
    
    # Hybrid Detector Node
    hybrid_detector_node = Node(
        package='MorphoTactus',
        executable='hybrid_detector_node',
        name='hybrid_detector_node',
        output='screen',
        parameters=[config_file],
        remappings=[
            ('preprocessed_image', '/preprocessed/image'),
            ('annotated_image', '/defect_detection/annotated_image'),
            ('defect_metadata', '/defect_detection/metadata'),
        ],
        env=jetson_optimizations
    )
    
    # RViz for visualization (optional)
    rviz_config_file = PathJoinSubstitution([pkg_share, 'config', 'morphotactus.rviz'])
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_file],
        parameters=[{'use_sim_time': use_sim_time}],
        condition=LaunchConfiguration('enable_rviz')
    )
    
    # Declare RViz argument
    declare_enable_rviz_cmd = DeclareLaunchArgument(
        'enable_rviz',
        default_value='false',
        description='Enable RViz visualization'
    )
    
    # Image viewer for debugging
    image_view_node = Node(
        package='image_view',
        executable='image_view',
        name='image_view',
        parameters=[{'autosize': True}],
        remappings=[
            ('image', '/defect_detection/annotated_image'),
        ],
        condition=LaunchConfiguration('enable_image_view')
    )
    
    # Declare image viewer argument
    declare_enable_image_view_cmd = DeclareLaunchArgument(
        'enable_image_view',
        default_value='true',
        description='Enable image viewer for annotated images'
    )
    
    # Performance monitoring
    performance_monitor_node = Node(
        package='diagnostic_aggregator',
        executable='aggregator_node',
        name='diagnostic_aggregator',
        parameters=[config_file],
        condition=LaunchConfiguration('enable_diagnostics')
    )
    
    # Declare diagnostics argument
    declare_enable_diagnostics_cmd = DeclareLaunchArgument(
        'enable_diagnostics',
        default_value='true',
        description='Enable performance diagnostics'
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_config_file_cmd,
        declare_use_sim_time_cmd,
        declare_enable_debug_cmd,
        declare_enable_rviz_cmd,
        declare_enable_image_view_cmd,
        declare_enable_diagnostics_cmd,
        
        # Nodes
        image_source_node,
        preprocessing_node,
        hybrid_detector_node,
        rviz_node,
        image_view_node,
        performance_monitor_node,
    ]) 