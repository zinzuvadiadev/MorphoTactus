#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import os


class SimpleVideoViewerNode(Node):
    """
    Simple ROS node for viewing video streams by saving images to files.
    This avoids Qt/OpenCV window issues on some systems.
    """
    
    def __init__(self):
        super().__init__('simple_video_viewer_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Create output directory
        self.output_dir = '/tmp/morphotactus_viewer'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Create subscribers for different image topics
        self.create_subscribers()
        
        self.get_logger().info('Simple video viewer node initialized successfully')
        self.get_logger().info(f'Images will be saved to: {self.output_dir}')
    
    def declare_node_parameters(self):
        """Declare node parameters for configuration."""
        self.declare_parameter('save_raw_images', True)
        self.declare_parameter('save_preprocessed_images', True)
        self.declare_parameter('save_annotated_images', True)
        self.declare_parameter('save_interval', 1.0)  # Save every N seconds
        self.declare_parameter('max_images', 10)  # Keep only N most recent images
        
        # Get parameter values
        self.save_raw = self.get_parameter('save_raw_images').value
        self.save_preprocessed = self.get_parameter('save_preprocessed_images').value
        self.save_annotated = self.get_parameter('save_annotated_images').value
        self.save_interval = self.get_parameter('save_interval').value
        self.max_images = self.get_parameter('max_images').value
        
        # Track last save times
        self.last_save_times = {}
    
    def create_subscribers(self):
        """Create subscribers for image topics."""
        if self.save_raw:
            self.create_subscription(
                Image,
                '/image_raw',
                lambda msg: self.image_callback(msg, 'raw'),
                10
            )
            self.last_save_times['raw'] = 0
        
        if self.save_preprocessed:
            self.create_subscription(
                Image,
                '/preprocessed_image',
                lambda msg: self.image_callback(msg, 'preprocessed'),
                10
            )
            self.last_save_times['preprocessed'] = 0
        
        if self.save_annotated:
            self.create_subscription(
                Image,
                '/annotated_image',
                lambda msg: self.image_callback(msg, 'annotated'),
                10
            )
            self.last_save_times['annotated'] = 0
    
    def image_callback(self, msg, image_type):
        """Callback for image messages."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Check if we should save this image
            current_time = time.time()
            if current_time - self.last_save_times.get(image_type, 0) >= self.save_interval:
                self.save_image(cv_image, image_type)
                self.last_save_times[image_type] = current_time
                
        except Exception as e:
            self.get_logger().error(f'Error processing {image_type} image: {str(e)}')
    
    def save_image(self, image, image_type):
        """Save image to file."""
        try:
            # Create filename with timestamp
            timestamp = int(time.time())
            filename = f"{image_type}_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, image)
            
            # Add overlay text to saved image
            self.add_overlay_to_saved_image(filepath, image_type, timestamp)
            
            self.get_logger().info(f'Saved {image_type} image: {filename}')
            
            # Clean up old images
            self.cleanup_old_images(image_type)
            
        except Exception as e:
            self.get_logger().error(f'Error saving {image_type} image: {str(e)}')
    
    def add_overlay_to_saved_image(self, filepath, image_type, timestamp):
        """Add overlay information to saved image."""
        try:
            # Read the saved image
            image = cv2.imread(filepath)
            
            # Add text overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            color = (255, 255, 255)  # White text
            bg_color = (0, 0, 0)     # Black background
            
            # Create overlay text
            text_lines = [
                f'Type: {image_type.upper()}',
                f'Time: {time.strftime("%H:%M:%S", time.localtime(timestamp))}',
                f'Timestamp: {timestamp}'
            ]
            
            y_offset = 30
            for i, line in enumerate(text_lines):
                # Get text size
                (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
                
                # Draw background rectangle
                cv2.rectangle(image, 
                             (10, y_offset + i * 30 - text_height - 5),
                             (10 + text_width + 10, y_offset + i * 30 + 5),
                             bg_color, -1)
                
                # Draw text
                cv2.putText(image, line, (15, y_offset + i * 30), 
                           font, font_scale, color, thickness)
            
            # Save the image with overlay
            cv2.imwrite(filepath, image)
            
        except Exception as e:
            self.get_logger().error(f'Error adding overlay to {filepath}: {str(e)}')
    
    def cleanup_old_images(self, image_type):
        """Remove old images to prevent disk space issues."""
        try:
            import glob
            
            # Get all files of this type
            pattern = os.path.join(self.output_dir, f"{image_type}_*.jpg")
            files = glob.glob(pattern)
            
            # Sort by modification time (newest first)
            files.sort(key=os.path.getmtime, reverse=True)
            
            # Remove old files
            if len(files) > self.max_images:
                for old_file in files[self.max_images:]:
                    os.remove(old_file)
                    self.get_logger().debug(f'Removed old file: {old_file}')
                    
        except Exception as e:
            self.get_logger().error(f'Error cleaning up old {image_type} images: {str(e)}')
    
    def print_status(self):
        """Print current status."""
        self.get_logger().info("=== Simple Video Viewer Status ===")
        self.get_logger().info(f"Output directory: {self.output_dir}")
        self.get_logger().info(f"Save interval: {self.save_interval} seconds")
        self.get_logger().info(f"Max images per type: {self.max_images}")
        
        # List current files
        try:
            files = os.listdir(self.output_dir)
            if files:
                self.get_logger().info("Current saved images:")
                for file in sorted(files):
                    filepath = os.path.join(self.output_dir, file)
                    size = os.path.getsize(filepath)
                    mtime = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(filepath)))
                    self.get_logger().info(f"  {file} ({size} bytes, {mtime})")
            else:
                self.get_logger().info("No images saved yet")
        except Exception as e:
            self.get_logger().error(f"Error listing files: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    
    try:
        viewer_node = SimpleVideoViewerNode()
        
        # Print initial status
        viewer_node.print_status()
        
        # Set up periodic status updates
        timer = viewer_node.create_timer(10.0, viewer_node.print_status)
        
        rclpy.spin(viewer_node)
    except KeyboardInterrupt:
        viewer_node.get_logger().info('Keyboard interrupt received')
    except Exception as e:
        viewer_node.get_logger().error(f'Error in simple video viewer: {str(e)}')
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main() 