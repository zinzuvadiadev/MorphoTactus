#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
import time
import os


class VideoViewerNode(Node):
    """
    ROS node for viewing video streams from the defect detection system.
    Supports multiple image topics and provides real-time visualization.
    """
    
    def __init__(self):
        super().__init__('video_viewer_node')
        
        # Set OpenCV backend to avoid Qt issues
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Initialize window management
        self.windows = {}
        self.latest_images = {}
        self.window_threads = {}
        self.running = True
        
        # Create subscribers for different image topics
        self.create_subscribers()
        
        # Start display threads
        self.start_display_threads()
        
        self.get_logger().info('Video viewer node initialized successfully')
        self.get_logger().info(f'Viewing topics: {list(self.windows.keys())}')
    
    def declare_node_parameters(self):
        """Declare node parameters for configuration."""
        self.declare_parameter('view_raw_images', True)
        self.declare_parameter('view_preprocessed_images', True)
        self.declare_parameter('view_annotated_images', True)
        self.declare_parameter('window_width', 640)
        self.declare_parameter('window_height', 480)
        self.declare_parameter('fps_limit', 30)
        self.declare_parameter('show_timestamp', True)
        self.declare_parameter('show_topic_name', True)
        
        # Get parameter values
        self.view_raw = self.get_parameter('view_raw_images').value
        self.view_preprocessed = self.get_parameter('view_preprocessed_images').value
        self.view_annotated = self.get_parameter('view_annotated_images').value
        self.window_width = self.get_parameter('window_width').value
        self.window_height = self.get_parameter('window_height').value
        self.fps_limit = self.get_parameter('fps_limit').value
        self.show_timestamp = self.get_parameter('show_timestamp').value
        self.show_topic_name = self.get_parameter('show_topic_name').value
    
    def create_subscribers(self):
        """Create subscribers for image topics."""
        if self.view_raw:
            self.create_subscription(
                Image,
                '/image_raw',
                lambda msg: self.image_callback(msg, 'Raw Images'),
                10
            )
            self.windows['Raw Images'] = None
            self.latest_images['Raw Images'] = None
        
        if self.view_preprocessed:
            self.create_subscription(
                Image,
                '/preprocessed_image',
                lambda msg: self.image_callback(msg, 'Preprocessed Images'),
                10
            )
            self.windows['Preprocessed Images'] = None
            self.latest_images['Preprocessed Images'] = None
        
        if self.view_annotated:
            self.create_subscription(
                Image,
                '/annotated_image',
                lambda msg: self.image_callback(msg, 'Defect Detection Results'),
                10
            )
            self.windows['Defect Detection Results'] = None
            self.latest_images['Defect Detection Results'] = None
    
    def image_callback(self, msg, window_name):
        """Callback for image messages."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store the latest image
            self.latest_images[window_name] = cv_image
            
            # Create window if it doesn't exist
            if self.windows[window_name] is None:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window_name, self.window_width, self.window_height)
                    self.windows[window_name] = True
                    self.get_logger().info(f'Created window: {window_name}')
                except Exception as e:
                    self.get_logger().error(f'Failed to create window {window_name}: {str(e)}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image for {window_name}: {str(e)}')
    
    def start_display_threads(self):
        """Start display threads for each window."""
        for window_name in self.windows.keys():
            thread = threading.Thread(
                target=self.display_loop,
                args=(window_name,),
                daemon=True
            )
            thread.start()
            self.window_threads[window_name] = thread
    
    def display_loop(self, window_name):
        """Display loop for a specific window."""
        frame_time = 1.0 / self.fps_limit if self.fps_limit > 0 else 0
        
        while self.running:
            try:
                if window_name in self.latest_images and self.latest_images[window_name] is not None:
                    # Get the image
                    image = self.latest_images[window_name].copy()
                    
                    # Add overlay information
                    if self.show_timestamp or self.show_topic_name:
                        image = self.add_overlay(image, window_name)
                    
                    # Display the image
                    cv2.imshow(window_name, image)
                    
                    # Handle window events
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.get_logger().info(f'ESC pressed, closing {window_name}')
                        cv2.destroyWindow(window_name)
                        break
                    elif key == ord('q'):
                        self.get_logger().info('Q pressed, shutting down viewer')
                        self.running = False
                        break
                
                # Frame rate limiting
                if frame_time > 0:
                    time.sleep(frame_time)
                    
            except Exception as e:
                self.get_logger().error(f'Error in display loop for {window_name}: {str(e)}')
                time.sleep(0.1)
    
    def add_overlay(self, image, window_name):
        """Add overlay information to the image."""
        overlay = image.copy()
        
        # Get image dimensions
        height, width = image.shape[:2]
        
        # Create overlay text
        text_lines = []
        
        if self.show_topic_name:
            text_lines.append(f'Topic: {window_name}')
        
        if self.show_timestamp:
            timestamp = time.strftime('%H:%M:%S')
            text_lines.append(f'Time: {timestamp}')
        
        # Add text to image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)     # Black background
        
        y_offset = 30
        for i, line in enumerate(text_lines):
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(overlay, 
                         (10, y_offset + i * 25 - text_height - 5),
                         (10 + text_width + 10, y_offset + i * 25 + 5),
                         bg_color, -1)
            
            # Draw text
            cv2.putText(overlay, line, (15, y_offset + i * 25), 
                       font, font_scale, color, thickness)
        
        return overlay
    
    def cleanup(self):
        """Clean up resources."""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.window_threads.values():
            thread.join(timeout=1.0)
        
        # Close all windows
        cv2.destroyAllWindows()
        
        self.get_logger().info('Video viewer node cleaned up')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        viewer_node = VideoViewerNode()
        rclpy.spin(viewer_node)
    except KeyboardInterrupt:
        viewer_node.get_logger().info('Keyboard interrupt received')
    except Exception as e:
        viewer_node.get_logger().error(f'Error in video viewer: {str(e)}')
    finally:
        if 'viewer_node' in locals():
            viewer_node.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 