#!/usr/bin/env python3
"""
Preprocessing Node for MorphoTactus
Performs image enhancement, noise reduction, and ROI extraction
Optimized for Jetson Nano deployment
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import logging
from typing import Optional, Tuple

class PreprocessingNode(Node):
    """
    ROS node for image preprocessing including enhancement, noise reduction, and ROI extraction
    Optimized for real-time processing on Jetson Nano
    """
    
    def __init__(self):
        super().__init__('preprocessing_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Get parameters
        self.declare_node_parameters()
        self.load_parameters()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 'image_raw', self.image_callback, 10)
        self.preprocessed_pub = self.create_publisher(Image, 'preprocessed_image', 10)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        
        self.get_logger().info("Preprocessing node initialized successfully")
    
    def declare_node_parameters(self):
        """Declare ROS parameters"""
        # ROI parameters
        self.declare_parameter('roi_x_min', 50)
        self.declare_parameter('roi_y_min', 50)
        self.declare_parameter('roi_x_max', 590)
        self.declare_parameter('roi_y_max', 430)
        
        # Image processing parameters
        self.declare_parameter('gaussian_blur_kernel', 5)
        self.declare_parameter('bilateral_filter_d', 9)
        self.declare_parameter('bilateral_filter_sigma_color', 75)
        self.declare_parameter('bilateral_filter_sigma_space', 75)
        self.declare_parameter('enable_histogram_equalization', True)
        self.declare_parameter('enable_noise_reduction', True)
        self.declare_parameter('enable_contrast_enhancement', True)
        
        # Performance parameters
        self.declare_parameter('enable_profiling', True)
        self.declare_parameter('publish_debug_images', False)
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # ROI parameters
        self.roi_x_min = self.get_parameter('roi_x_min').value
        self.roi_y_min = self.get_parameter('roi_y_min').value
        self.roi_x_max = self.get_parameter('roi_x_max').value
        self.roi_y_max = self.get_parameter('roi_y_max').value
        
        # Image processing parameters
        self.gaussian_blur_kernel = self.get_parameter('gaussian_blur_kernel').value
        self.bilateral_filter_d = self.get_parameter('bilateral_filter_d').value
        self.bilateral_filter_sigma_color = self.get_parameter('bilateral_filter_sigma_color').value
        self.bilateral_filter_sigma_space = self.get_parameter('bilateral_filter_sigma_space').value
        self.enable_histogram_equalization = self.get_parameter('enable_histogram_equalization').value
        self.enable_noise_reduction = self.get_parameter('enable_noise_reduction').value
        self.enable_contrast_enhancement = self.get_parameter('enable_contrast_enhancement').value
        
        # Performance parameters
        self.enable_profiling = self.get_parameter('enable_profiling').value
        self.publish_debug_images = self.get_parameter('publish_debug_images').value
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def image_callback(self, msg: Image):
        """Callback for incoming image messages"""
        try:
            start_time = time.time()
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Apply preprocessing pipeline
            processed_image = self.preprocess_image(cv_image)
            
            # Publish processed image
            self.publish_processed_image(processed_image, msg.header)
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.update_performance_stats(processing_time)
            
        except Exception as e:
            self.logger.error(f"Error in image callback: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply complete preprocessing pipeline
        """
        try:
            # Step 1: Extract ROI
            roi_image = self.extract_roi(image)
            
            # Step 2: Noise reduction
            if self.enable_noise_reduction:
                roi_image = self.reduce_noise(roi_image)
            
            # Step 3: Contrast enhancement
            if self.enable_contrast_enhancement:
                roi_image = self.enhance_contrast(roi_image)
            
            # Step 4: Histogram equalization
            if self.enable_histogram_equalization:
                roi_image = self.apply_histogram_equalization(roi_image)
            
            return roi_image
            
        except Exception as e:
            self.logger.error(f"Error in preprocessing pipeline: {e}")
            return image
    
    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Extract Region of Interest from image
        """
        try:
            height, width = image.shape[:2]
            
            # Ensure ROI bounds are within image dimensions
            x_min = max(0, min(self.roi_x_min, width - 1))
            y_min = max(0, min(self.roi_y_min, height - 1))
            x_max = max(x_min + 1, min(self.roi_x_max, width))
            y_max = max(y_min + 1, min(self.roi_y_max, height))
            
            # Extract ROI
            roi = image[y_min:y_max, x_min:x_max]
            
            return roi
            
        except Exception as e:
            self.logger.error(f"Error extracting ROI: {e}")
            return image
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply noise reduction using bilateral filtering
        """
        try:
            # Apply bilateral filter for edge-preserving noise reduction
            denoised = cv2.bilateralFilter(
                image,
                self.bilateral_filter_d,
                self.bilateral_filter_sigma_color,
                self.bilateral_filter_sigma_space
            )
            
            return denoised
            
        except Exception as e:
            self.logger.error(f"Error in noise reduction: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
        """
        try:
            # Convert to LAB color space for better contrast enhancement
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge channels and convert back to BGR
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                
                return enhanced
            else:
                # For grayscale images
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                return clahe.apply(image)
                
        except Exception as e:
            self.logger.error(f"Error in contrast enhancement: {e}")
            return image
    
    def apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram equalization for better image quality
        """
        try:
            if len(image.shape) == 3:
                # For color images, apply to each channel separately
                ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(ycrcb)
                
                # Equalize Y channel
                y = cv2.equalizeHist(y)
                
                # Merge channels and convert back to BGR
                equalized = cv2.merge([y, cr, cb])
                equalized = cv2.cvtColor(equalized, cv2.COLOR_YCrCb2BGR)
                
                return equalized
            else:
                # For grayscale images
                return cv2.equalizeHist(image)
                
        except Exception as e:
            self.logger.error(f"Error in histogram equalization: {e}")
            return image
    
    def apply_gaussian_blur(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian blur for additional smoothing
        """
        try:
            # Ensure kernel size is odd
            kernel_size = self.gaussian_blur_kernel
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            return blurred
            
        except Exception as e:
            self.logger.error(f"Error in Gaussian blur: {e}")
            return image
    
    def publish_processed_image(self, image: np.ndarray, original_header):
        """Publish processed image to ROS topic"""
        try:
            # Convert OpenCV image to ROS message
            ros_image = self.bridge.cv2_to_imgmsg(image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = original_header.frame_id
            
            # Publish image
            self.preprocessed_pub.publish(ros_image)
            
        except Exception as e:
            self.logger.error(f"Error publishing processed image: {e}")
    
    def update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.frame_count += 1
        self.processing_times.append(processing_time)
        
        # Log performance every 100 frames
        if self.frame_count % 100 == 0:
            avg_time = np.mean(self.processing_times[-100:])
            self.logger.info(f"Preprocessing performance: {avg_time:.2f}ms avg, {self.frame_count} frames")
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'total_frames': self.frame_count,
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'fps': self.frame_count / (time.time() - self.start_time) if hasattr(self, 'start_time') else 0
        }

def main(args=None):
    rclpy.init(args=args)
    
    node = PreprocessingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in preprocessing node: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 