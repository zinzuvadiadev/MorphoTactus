#!/usr/bin/env python3

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Get package directory
    pkg_share = FindPackageShare('MorphoTactus')
    
    # Launch arguments
    config_file = LaunchConfiguration('config_file')
    
    # Declare launch arguments
    declare_config_file_cmd = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([pkg_share, 'config', 'detector_params.yaml']),
        description='Path to configuration file'
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
    
    # Image viewer for debugging
    image_view_node = Node(
        package='image_view',
        executable='image_view',
        name='image_view',
        parameters=[{'autosize': True}],
        remappings=[
            ('image', '/defect_detection/annotated_image'),
        ]
    )
    
    return LaunchDescription([
        # Launch arguments
        declare_config_file_cmd,
        
        # Nodes
        image_source_node,
        preprocessing_node,
        hybrid_detector_node,
        image_view_node,
    ]) 