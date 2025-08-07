#!/usr/bin/env python3
"""
Test script for MorphoTactus system
Verifies functionality and performance of the hybrid defect detection system
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from MorphoTactus.msg import DefectMetadata
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import threading
from typing import Optional

class SystemTester(Node):
    """
    Test node for verifying MorphoTactus system functionality
    """
    
    def __init__(self):
        super().__init__('system_tester')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Test results
        self.test_results = {
            'image_received': False,
            'metadata_received': False,
            'processing_times': [],
            'defect_detections': 0,
            'total_frames': 0
        }
        
        # Initialize subscribers
        self.image_sub = self.create_subscription(
            Image, 'defect_detection/annotated_image', self.image_callback, 10)
        self.metadata_sub = self.create_subscription(
            DefectMetadata, 'defect_detection/metadata', self.metadata_callback, 10)
        
        # Test timer
        self.test_timer = self.create_timer(5.0, self.run_tests)
        
        self.get_logger().info("System tester initialized")
    
    def image_callback(self, msg: Image):
        """Callback for annotated images"""
        try:
            # Convert to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Mark as received
            self.test_results['image_received'] = True
            self.test_results['total_frames'] += 1
            
            # Log first few frames
            if self.test_results['total_frames'] <= 5:
                self.get_logger().info(f"Received annotated image {self.test_results['total_frames']}")
            
        except Exception as e:
            self.get_logger().error(f"Error in image callback: {e}")
    
    def metadata_callback(self, msg: DefectMetadata):
        """Callback for defect metadata"""
        try:
            # Mark as received
            self.test_results['metadata_received'] = True
            
            # Track processing time
            if msg.processing_time_ms > 0:
                self.test_results['processing_times'].append(msg.processing_time_ms)
            
            # Track defect detections
            if msg.defect_detected:
                self.test_results['defect_detections'] += 1
            
            # Log metadata
            self.get_logger().info(f"Part {msg.part_id}: {'FAIL' if msg.defect_detected else 'PASS'} "
                                 f"(Confidence: {msg.overall_confidence:.3f}, "
                                 f"Processing: {msg.processing_time_ms:.1f}ms)")
            
        except Exception as e:
            self.get_logger().error(f"Error in metadata callback: {e}")
    
    def run_tests(self):
        """Run system tests"""
        try:
            self.get_logger().info("=== System Test Results ===")
            
            # Check if topics are receiving data
            if self.test_results['image_received']:
                self.get_logger().info("✓ Image topic is working")
            else:
                self.get_logger().error("✗ Image topic not receiving data")
            
            if self.test_results['metadata_received']:
                self.get_logger().info("✓ Metadata topic is working")
            else:
                self.get_logger().error("✗ Metadata topic not receiving data")
            
            # Performance analysis
            if self.test_results['processing_times']:
                avg_time = np.mean(self.test_results['processing_times'])
                min_time = np.min(self.test_results['processing_times'])
                max_time = np.max(self.test_results['processing_times'])
                
                self.get_logger().info(f"Performance: {avg_time:.1f}ms avg ({min_time:.1f}-{max_time:.1f}ms)")
                
                # Check if performance is within acceptable range for Jetson Nano
                if avg_time < 150:  # 150ms threshold for real-time processing
                    self.get_logger().info("✓ Performance is acceptable for real-time processing")
                else:
                    self.get_logger().warning("⚠ Performance may be too slow for real-time processing")
            
            # Defect detection analysis
            if self.test_results['total_frames'] > 0:
                detection_rate = self.test_results['defect_detections'] / self.test_results['total_frames']
                self.get_logger().info(f"Defect detection rate: {detection_rate:.1%}")
            
            self.get_logger().info(f"Total frames processed: {self.test_results['total_frames']}")
            self.get_logger().info("=== End Test Results ===")
            
        except Exception as e:
            self.get_logger().error(f"Error in test execution: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    tester = SystemTester()
    
    try:
        rclpy.spin(tester)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        tester.get_logger().error(f"Error in system tester: {e}")
    finally:
        tester.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 