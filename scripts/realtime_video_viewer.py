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
import subprocess
import tempfile


class RealtimeVideoViewerNode(Node):
    """
    Real-time video viewer that displays live video streams.
    Uses alternative methods to avoid Qt backend issues.
    """
    
    def __init__(self):
        super().__init__('realtime_video_viewer_node')
        
        # Set environment variables to avoid Qt issues
        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
        os.environ['OPENCV_VIDEOIO_PRIORITY_INTEL_MFX'] = '0'
        os.environ['QT_QPA_PLATFORM'] = 'xcb'  # Use XCB instead of Wayland
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_node_parameters()
        
        # Initialize window management
        self.windows = {}
        self.latest_images = {}
        self.window_threads = {}
        self.running = True
        self.fps_counters = {}
        self.last_fps_time = {}
        
        # Create subscribers for different image topics
        self.create_subscribers()
        
        # Start display threads
        self.start_display_threads()
        
        self.get_logger().info('Real-time video viewer node initialized successfully')
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
        self.declare_parameter('show_fps', True)
        self.declare_parameter('use_external_viewer', True)  # Default to external viewer
        
        # Get parameter values
        self.view_raw = self.get_parameter('view_raw_images').value
        self.view_preprocessed = self.get_parameter('view_preprocessed_images').value
        self.view_annotated = self.get_parameter('view_annotated_images').value
        self.window_width = self.get_parameter('window_width').value
        self.window_height = self.get_parameter('window_height').value
        self.fps_limit = self.get_parameter('fps_limit').value
        self.show_timestamp = self.get_parameter('show_timestamp').value
        self.show_topic_name = self.get_parameter('show_topic_name').value
        self.show_fps = self.get_parameter('show_fps').value
        self.use_external_viewer = self.get_parameter('use_external_viewer').value
    
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
            self.fps_counters['Raw Images'] = 0
            self.last_fps_time['Raw Images'] = time.time()
        
        if self.view_preprocessed:
            self.create_subscription(
                Image,
                '/preprocessed_image',
                lambda msg: self.image_callback(msg, 'Preprocessed Images'),
                10
            )
            self.windows['Preprocessed Images'] = None
            self.latest_images['Preprocessed Images'] = None
            self.fps_counters['Preprocessed Images'] = 0
            self.last_fps_time['Preprocessed Images'] = time.time()
        
        if self.view_annotated:
            self.create_subscription(
                Image,
                '/annotated_image',
                lambda msg: self.image_callback(msg, 'Defect Detection Results'),
                10
            )
            self.windows['Defect Detection Results'] = None
            self.latest_images['Defect Detection Results'] = None
            self.fps_counters['Defect Detection Results'] = 0
            self.last_fps_time['Defect Detection Results'] = time.time()
    
    def image_callback(self, msg, window_name):
        """Callback for image messages."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store the latest image
            self.latest_images[window_name] = cv_image
            
            # Update FPS counter
            current_time = time.time()
            self.fps_counters[window_name] += 1
            
            # Calculate FPS every second
            if current_time - self.last_fps_time[window_name] >= 1.0:
                fps = self.fps_counters[window_name] / (current_time - self.last_fps_time[window_name])
                self.get_logger().info(f'{window_name} FPS: {fps:.1f}')
                self.fps_counters[window_name] = 0
                self.last_fps_time[window_name] = current_time
            
            # Create window if it doesn't exist
            if self.windows[window_name] is None:
                try:
                    if self.use_external_viewer:
                        self.create_external_viewer(window_name)
                    else:
                        self.create_opencv_window(window_name)
                except Exception as e:
                    self.get_logger().error(f'Failed to create window {window_name}: {str(e)}')
                
        except Exception as e:
            self.get_logger().error(f'Error processing image for {window_name}: {str(e)}')
    
    def create_opencv_window(self, window_name):
        """Create OpenCV window."""
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, self.window_width, self.window_height)
            self.windows[window_name] = True
            self.get_logger().info(f'Created OpenCV window: {window_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to create OpenCV window: {str(e)}')
            # Fallback to external viewer
            self.create_external_viewer(window_name)
    
    def create_external_viewer(self, window_name):
        """Create external image viewer process."""
        try:
            # Create a temporary directory for this window
            temp_dir = f'/tmp/morphotactus_realtime_{window_name.replace(" ", "_")}'
            os.makedirs(temp_dir, exist_ok=True)
            
            # Start external viewer process
            if self.check_command_exists('feh'):
                cmd = ['feh', '--reload', '1', '--auto-zoom', temp_dir]
            elif self.check_command_exists('eog'):
                cmd = ['eog', '--new-instance', temp_dir]
            elif self.check_command_exists('gthumb'):
                cmd = ['gthumb', temp_dir]
            else:
                cmd = ['xdg-open', temp_dir]
            
            process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            self.windows[window_name] = {
                'type': 'external',
                'temp_dir': temp_dir,
                'process': process
            }
            
            self.get_logger().info(f'Created external viewer for: {window_name}')
            
        except Exception as e:
            self.get_logger().error(f'Failed to create external viewer: {str(e)}')
    
    def check_command_exists(self, command):
        """Check if a command exists in the system."""
        try:
            subprocess.run(['which', command], capture_output=True, check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
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
                    if self.show_timestamp or self.show_topic_name or self.show_fps:
                        image = self.add_overlay(image, window_name)
                    
                    # Display the image
                    if self.windows[window_name] == True:  # OpenCV window
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
                    
                    elif isinstance(self.windows[window_name], dict) and self.windows[window_name]['type'] == 'external':
                        # External viewer - save image to temp directory
                        temp_dir = self.windows[window_name]['temp_dir']
                        timestamp = int(time.time() * 1000)  # milliseconds
                        filename = f'frame_{timestamp}.jpg'
                        filepath = os.path.join(temp_dir, filename)
                        
                        # Save image
                        cv2.imwrite(filepath, image)
                        
                        # Clean up old files
                        self.cleanup_temp_files(temp_dir, max_files=5)
                
                # Frame rate limiting
                if frame_time > 0:
                    time.sleep(frame_time)
                    
            except Exception as e:
                self.get_logger().error(f'Error in display loop for {window_name}: {str(e)}')
                time.sleep(0.1)
    
    def cleanup_temp_files(self, temp_dir, max_files=5):
        """Clean up old temporary files."""
        try:
            files = [f for f in os.listdir(temp_dir) if f.endswith('.jpg')]
            if len(files) > max_files:
                files.sort(key=lambda x: os.path.getmtime(os.path.join(temp_dir, x)))
                for old_file in files[:-max_files]:
                    os.remove(os.path.join(temp_dir, old_file))
        except Exception as e:
            self.get_logger().debug(f'Error cleaning temp files: {str(e)}')
    
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
        
        if self.show_fps:
            current_time = time.time()
            if window_name in self.last_fps_time:
                time_diff = current_time - self.last_fps_time[window_name]
                if time_diff > 0:
                    fps = self.fps_counters[window_name] / time_diff
                    text_lines.append(f'FPS: {fps:.1f}')
        
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
        
        # Clean up external viewers
        for window_name, window_info in self.windows.items():
            if isinstance(window_info, dict) and window_info['type'] == 'external':
                try:
                    if 'process' in window_info:
                        window_info['process'].terminate()
                    if 'temp_dir' in window_info:
                        import shutil
                        shutil.rmtree(window_info['temp_dir'], ignore_errors=True)
                except Exception as e:
                    self.get_logger().error(f'Error cleaning up {window_name}: {str(e)}')
        
        self.get_logger().info('Real-time video viewer node cleaned up')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        viewer_node = RealtimeVideoViewerNode()
        rclpy.spin(viewer_node)
    except KeyboardInterrupt:
        viewer_node.get_logger().info('Keyboard interrupt received')
    except Exception as e:
        viewer_node.get_logger().error(f'Error in real-time video viewer: {str(e)}')
    finally:
        if 'viewer_node' in locals():
            viewer_node.cleanup()
        rclpy.shutdown()


if __name__ == '__main__':
    main() 