#!/usr/bin/env python3
"""
Image Source Node for MorphoTactus
Captures images from industrial camera and publishes to ROS topics
Optimized for Jetson Nano deployment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import logging
from typing import Optional
import threading

class ImageSourceNode(Node):
    """
    ROS node for capturing and publishing images from industrial camera
    Supports both USB camera and network camera sources
    """
    
    def __init__(self):
        super().__init__('image_source_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Get parameters
        self.declare_node_parameters()
        self.load_parameters()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize camera
        self.camera = None
        self.camera_thread = None
        self.is_capturing = False
        
        # Initialize publishers
        self.image_pub = self.create_publisher(Image, 'image_raw', 10)
        self.camera_info_pub = self.create_publisher(CameraInfo, 'camera_info', 10)
        
        # Initialize camera info
        self.camera_info = self.create_camera_info()
        
        # Performance tracking
        self.frame_count = 0
        self.start_time = time.time()
        
        # Initialize camera
        self.initialize_camera()
        
        # Start camera thread
        self.start_camera_thread()
        
        self.get_logger().info("Image source node initialized successfully")
    
    def declare_node_parameters(self):
        """Declare ROS parameters"""
        self.declare_parameter('camera_source', 'usb')
        self.declare_parameter('camera_index', 0)
        self.declare_parameter('camera_url', '')
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('frame_rate', 45.0)
        self.declare_parameter('auto_exposure', True)
        self.declare_parameter('exposure_time', 100)
        self.declare_parameter('gain', 1.0)
        self.declare_parameter('enable_undistortion', False)
        self.declare_parameter('camera_matrix', [])
        self.declare_parameter('distortion_coeffs', [])
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        self.camera_source = self.get_parameter('camera_source').value
        self.camera_index = self.get_parameter('camera_index').value
        self.camera_url = self.get_parameter('camera_url').value
        self.frame_width = self.get_parameter('frame_width').value
        self.frame_height = self.get_parameter('frame_height').value
        self.frame_rate = self.get_parameter('frame_rate').value
        self.auto_exposure = self.get_parameter('auto_exposure').value
        self.exposure_time = self.get_parameter('exposure_time').value
        self.gain = self.get_parameter('gain').value
        self.enable_undistortion = self.get_parameter('enable_undistortion').value
        self.camera_matrix = self.get_parameter('camera_matrix').value
        self.distortion_coeffs = self.get_parameter('distortion_coeffs').value
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_camera(self):
        """Initialize camera based on source type"""
        try:
            if self.camera_source == 'usb':
                self.camera = cv2.VideoCapture(self.camera_index)
            elif self.camera_source == 'network':
                if not self.camera_url:
                    raise ValueError("Camera URL not provided for network camera")
                self.camera = cv2.VideoCapture(self.camera_url)
            else:
                raise ValueError(f"Unsupported camera source: {self.camera_source}")
            
            if not self.camera.isOpened():
                raise RuntimeError(f"Failed to open camera: {self.camera_source}")
            
            # Configure camera properties
            self.configure_camera()
            
            self.logger.info(f"Camera initialized successfully: {self.camera_source}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize camera: {e}")
            # Fallback to test image generation
            self.camera = None
            self.logger.warning("Falling back to test image generation")
    
    def configure_camera(self):
        """Configure camera properties for optimal performance"""
        try:
            # Set resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            # Set frame rate
            self.camera.set(cv2.CAP_PROP_FPS, self.frame_rate)
            
            # Configure exposure
            if not self.auto_exposure:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
                self.camera.set(cv2.CAP_PROP_EXPOSURE, self.exposure_time)
            
            # Set gain
            self.camera.set(cv2.CAP_PROP_GAIN, self.gain)
            
            # Set buffer size for real-time performance
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verify settings
            actual_width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            self.logger.info(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps}fps")
            
        except Exception as e:
            self.logger.error(f"Error configuring camera: {e}")
    
    def create_camera_info(self) -> CameraInfo:
        """Create camera info message"""
        camera_info = CameraInfo()
        camera_info.header.frame_id = "camera_link"
        camera_info.width = self.frame_width
        camera_info.height = self.frame_height
        
        # Set camera matrix if provided
        if len(self.camera_matrix) == 9:
            camera_info.k = self.camera_matrix
        
        # Set distortion coefficients if provided
        if len(self.distortion_coeffs) == 5:
            camera_info.d = self.distortion_coeffs
        
        return camera_info
    
    def start_camera_thread(self):
        """Start camera capture thread"""
        self.is_capturing = True
        self.camera_thread = threading.Thread(target=self.camera_capture_loop)
        self.camera_thread.daemon = True
        self.camera_thread.start()
        self.logger.info("Camera capture thread started")
    
    def camera_capture_loop(self):
        """Main camera capture loop"""
        frame_interval = 1.0 / self.frame_rate
        
        while self.is_capturing and rclpy.ok():
            try:
                start_time = time.time()
                
                # Capture frame
                if self.camera is not None:
                    ret, frame = self.camera.read()
                    if not ret:
                        self.logger.warning("Failed to capture frame from camera")
                        frame = self.generate_test_image()
                else:
                    frame = self.generate_test_image()
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Publish frame
                self.publish_frame(processed_frame)
                
                # Publish camera info
                self.publish_camera_info()
                
                # Update performance stats
                self.update_performance_stats()
                
                # Frame rate control
                elapsed_time = time.time() - start_time
                sleep_time = max(0, frame_interval - elapsed_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"Error in camera capture loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process captured frame"""
        try:
            # Resize if necessary
            if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
            
            # Apply undistortion if enabled
            if self.enable_undistortion and len(self.camera_matrix) == 9:
                camera_matrix = np.array(self.camera_matrix).reshape(3, 3)
                dist_coeffs = np.array(self.distortion_coeffs)
                frame = cv2.undistort(frame, camera_matrix, dist_coeffs)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return frame
    
    def generate_test_image(self) -> np.ndarray:
        """Generate test image for development/debugging"""
        # Create a test image with some geometric shapes
        image = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
        
        # Add some geometric shapes
        cv2.rectangle(image, (100, 100), (200, 200), (255, 255, 255), 2)
        cv2.circle(image, (400, 300), 50, (255, 255, 255), 2)
        cv2.line(image, (50, 50), (300, 400), (255, 255, 255), 2)
        
        # Add some noise to simulate real conditions
        noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def publish_frame(self, frame: np.ndarray):
        """Publish frame to ROS topic"""
        try:
            # Convert OpenCV image to ROS message
            ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = "camera_link"
            
            # Publish image
            self.image_pub.publish(ros_image)
            
        except Exception as e:
            self.logger.error(f"Error publishing frame: {e}")
    
    def publish_camera_info(self):
        """Publish camera info"""
        try:
            self.camera_info.header.stamp = self.get_clock().now().to_msg()
            self.camera_info_pub.publish(self.camera_info)
            
        except Exception as e:
            self.logger.error(f"Error publishing camera info: {e}")
    
    def update_performance_stats(self):
        """Update performance statistics"""
        self.frame_count += 1
        
        if self.frame_count % 100 == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.logger.info(f"Performance: {fps:.1f} FPS, {self.frame_count} frames")
    
    def cleanup(self):
        """Cleanup resources"""
        self.is_capturing = False
        
        if self.camera is not None:
            self.camera.release()
        
        if self.camera_thread is not None:
            self.camera_thread.join(timeout=1.0)
        
        self.logger.info("Image source node cleanup completed")

def main(args=None):
    rclpy.init(args=args)
    
    node = ImageSourceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in image source node: {e}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 