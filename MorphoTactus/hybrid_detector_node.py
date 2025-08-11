#!/usr/bin/env python3
"""
Hybrid Detector Node for MorphoTactus
Main node implementing parallel contouring and SIFT analysis with decision fusion
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
from typing import List, Dict, Optional
import threading
from concurrent.futures import ThreadPoolExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Import custom messages (temporarily commented out)
# from MorphoTactus.msg import DefectMetadata, DefectInfo
# from geometry_msgs.msg import Point2D, Polygon, Point32

# Import utility modules
from .utils.contour_analyzer import ContourAnalyzer, ContourDefect
from .utils.sift_analyzer import SIFTAnalyzer, SIFTDefect
from .utils.fusion_engine import FusionEngine, FusionResult

class HybridDetectorNode(Node):
    """
    Main hybrid detector node that combines contour and SIFT analysis
    with weighted decision fusion and visual annotation
    """
    
    def __init__(self):
        super().__init__('hybrid_detector_node')
        
        # Initialize CV bridge
        self.bridge = CvBridge()
        
        # Get parameters
        self.declare_node_parameters()
        self.load_parameters()
        
        # Setup logging
        self.setup_logging()
        
        # Initialize analyzers
        self.contour_analyzer = ContourAnalyzer(self.config)
        self.sift_analyzer = SIFTAnalyzer(self.config)
        self.fusion_engine = FusionEngine(self.config)
        
        # Initialize subscribers and publishers
        self.image_sub = self.create_subscription(
            Image, 'preprocessed_image', self.image_callback, 10)
        annotated_qos = QoSProfile(depth=10)
        annotated_qos.reliability = ReliabilityPolicy.BEST_EFFORT
        annotated_qos.history = HistoryPolicy.KEEP_LAST
        self.annotated_pub = self.create_publisher(Image, 'annotated_image', annotated_qos)
        # self.metadata_pub = self.create_publisher(DefectMetadata, 'defect_metadata', 10)
        
        # Performance tracking
        self.frame_count = 0
        self.processing_times = []
        self.part_id_counter = 0
        self._last_pub_time_s = 0.0
        
        # Thread pool for parallel processing
        self.thread_pool = ThreadPoolExecutor(max_workers=2)
        
        self.get_logger().info("Hybrid detector node initialized successfully")
    
    def declare_node_parameters(self):
        """Declare ROS parameters"""
        # Visualization parameters (smaller defaults)
        self.declare_parameter('text_scale', 1.0)
        self.declare_parameter('text_thickness', 2)
        self.declare_parameter('line_thickness', 2)
        self.declare_parameter('hud_scale', 1.0)
        self.declare_parameter('hud_bg', True)
        self.declare_parameter('bounding_box_color_red', 255)
        self.declare_parameter('bounding_box_color_green', 0)
        self.declare_parameter('bounding_box_color_blue', 0)
        self.declare_parameter('circle_color_red', 255)
        self.declare_parameter('circle_color_green', 165)
        self.declare_parameter('circle_color_blue', 0)
        self.declare_parameter('pass_color_red', 0)
        self.declare_parameter('pass_color_green', 255)
        self.declare_parameter('pass_color_blue', 0)
        self.declare_parameter('fail_color_red', 255)
        self.declare_parameter('fail_color_green', 0)
        self.declare_parameter('fail_color_blue', 0)
        
        # Throughput control
        self.declare_parameter('publish_rate_hz', 15.0)  # throttle annotated output
        
        # Performance parameters
        self.declare_parameter('enable_profiling', True)
        self.declare_parameter('publish_debug_info', False)
        
        # Jetson optimizations
        self.declare_parameter('use_parallel_processing', True)
        self.declare_parameter('max_processing_time_ms', 100)
    
    def load_parameters(self):
        """Load parameters from ROS parameter server"""
        # Visualization parameters
        self.text_scale = float(self.get_parameter('text_scale').value)
        self.text_thickness = int(self.get_parameter('text_thickness').value)
        self.line_thickness = int(self.get_parameter('line_thickness').value)
        self.hud_scale = float(self.get_parameter('hud_scale').value)
        self.hud_bg = bool(self.get_parameter('hud_bg').value)
        self.bounding_box_color = (
            int(self.get_parameter('bounding_box_color_red').value),
            int(self.get_parameter('bounding_box_color_green').value),
            int(self.get_parameter('bounding_box_color_blue').value)
        )
        self.circle_color = (
            int(self.get_parameter('circle_color_red').value),
            int(self.get_parameter('circle_color_green').value),
            int(self.get_parameter('circle_color_blue').value)
        )
        self.pass_color = (
            int(self.get_parameter('pass_color_red').value),
            int(self.get_parameter('pass_color_green').value),
            int(self.get_parameter('pass_color_blue').value)
        )
        self.fail_color = (
            int(self.get_parameter('fail_color_red').value),
            int(self.get_parameter('fail_color_green').value),
            int(self.get_parameter('fail_color_blue').value)
        )
        
        # Throughput control
        self.publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        
        # Performance parameters
        self.enable_profiling = bool(self.get_parameter('enable_profiling').value)
        self.publish_debug_info = bool(self.get_parameter('publish_debug_info').value)
        self.use_parallel_processing = bool(self.get_parameter('use_parallel_processing').value)
        self.max_processing_time_ms = float(self.get_parameter('max_processing_time_ms').value)
        
        # Load configuration for analyzers
        self.config = {
            'contour_detection': {
                'canny_low_threshold': 50,
                'canny_high_threshold': 150,
                'min_contour_area': 100,
                'max_contour_area': 50000,
                'anomaly_score_threshold': 0.7
            },
            'sift_analysis': {
                'max_features': 500,
                'match_ratio_threshold': 0.75
            },
            'fusion_engine': {
                'contour_weight': 0.6,
                'sift_weight': 0.4,
                'overall_confidence_threshold': 0.75
            },
            'jetson_optimizations': {
                'use_orb_fallback': True
            }
        }
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def image_callback(self, msg: Image):
        """Callback for incoming preprocessed image messages"""
        try:
            # Simple throttle to avoid backlog/lag
            now = time.time()
            if self.publish_rate_hz > 0.0:
                min_dt = 1.0 / self.publish_rate_hz
                if (now - self._last_pub_time_s) < min_dt:
                    return
            
            start_time = now
            
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Perform hybrid defect detection
            detection_result = self.detect_defects(cv_image)
            
            # Create annotated image
            annotated_image = self.create_annotated_image(cv_image, detection_result)
            
            # Publish results (metadata temporarily disabled)
            self.publish_annotated_image(annotated_image, msg.header)
            self._last_pub_time_s = now
            
            # Update performance stats
            processing_time = (time.time() - start_time) * 1000
            self.update_performance_stats(processing_time)
            
        except Exception as e:
            self.logger.error(f"Error in image callback: {e}")
    
    def detect_defects(self, image: np.ndarray) -> Dict:
        """
        Perform hybrid defect detection using contour and SIFT analysis
        """
        try:
            # Start timing
            processing_start_time = time.time()
            
            if self.use_parallel_processing:
                # Parallel processing
                contour_future = self.thread_pool.submit(self.contour_analyzer.detect_defects, image)
                sift_future = self.thread_pool.submit(self.sift_analyzer.analyze_textural_defects, image, [])
                
                # Get results
                contour_defects = contour_future.result()
                sift_defects = sift_future.result()
            else:
                # Sequential processing
                contour_defects = self.contour_analyzer.detect_defects(image)
                sift_defects = self.sift_analyzer.analyze_textural_defects(image, [])
            
            # Fuse results
            fusion_result = self.fusion_engine.fuse_results(
                contour_defects, sift_defects, processing_start_time
            )
            
            # Create detection result
            detection_result = {
                'contour_defects': contour_defects,
                'sift_defects': sift_defects,
                'fusion_result': fusion_result,
                'processing_time_ms': (time.time() - processing_start_time) * 1000
            }
            
            return detection_result
            
        except Exception as e:
            self.logger.error(f"Error in defect detection: {e}")
            return {
                'contour_defects': [],
                'sift_defects': [],
                'fusion_result': None,
                'processing_time_ms': 0
            }
    
    def create_annotated_image(self, image: np.ndarray, detection_result: Dict) -> np.ndarray:
        """
        Create annotated image with defect visualizations
        """
        try:
            # Create copy of original image
            annotated = image.copy()
            h, w = annotated.shape[:2]
            base = min(h, w)
            # Increase lower bound for small frames and apply user hud_scale
            scale = max(0.8, base / 900.0) * self.hud_scale
            thin  = max(1, int(self.text_thickness * scale))
            line  = max(1, int(self.line_thickness * scale))
            rad   = max(8, int(0.02 * base))
            s     = self.text_scale * scale

            # Draw contour defects (red bounding boxes)
            for defect in detection_result['contour_defects']:
                x, y, w_b, h_b = defect.bounding_box
                cv2.rectangle(annotated, (x, y), (x + w_b, y + h_b), self.bounding_box_color, line)
                label = f"{defect.defect_type}: {defect.severity_score:.2f}"
                cv2.putText(annotated, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                            s * 0.55, self.bounding_box_color, thin, cv2.LINE_AA)
            
            # Draw SIFT defects (smaller circles)
            for defect in detection_result['sift_defects']:
                x, y = int(defect.location[0]), int(defect.location[1])
                cv2.circle(annotated, (x, y), rad, self.circle_color, line)
                label = f"SIFT: {defect.severity_score:.2f}"
                cv2.putText(annotated, label, (x + rad + 4, y), cv2.FONT_HERSHEY_SIMPLEX,
                            s * 0.55, self.circle_color, thin, cv2.LINE_AA)
            
            # Compact status overlay (top-left)
            fusion_result = detection_result['fusion_result']
            if fusion_result:
                status_color = self.fail_color if fusion_result.defect_detected else self.pass_color
                status_text  = "FAIL" if fusion_result.defect_detected else "PASS"

                def draw_text(bg_img, text, org, color, sc, th):
                    (tw, th_text), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
                    x0, y0 = org
                    if self.hud_bg:
                        pad = int(6 * scale)
                        overlay = bg_img.copy()
                        cv2.rectangle(overlay, (x0 - pad, y0 - th_text - pad), (x0 + tw + pad, y0 + pad), (0, 0, 0), -1)
                        cv2.addWeighted(overlay, 0.35, bg_img, 0.65, 0, bg_img)
                    cv2.putText(bg_img, text, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, sc, color, th, cv2.LINE_AA)

                draw_text(annotated, status_text, (10, int(20 + 22 * s)), status_color, s * 1.3, max(1, int(thin * 1.2)))
                draw_text(annotated, f"C:{fusion_result.overall_confidence:.3f} D:{fusion_result.total_defects}",
                          (10, int(10 + 60 * s)), status_color, s * 0.9, thin)
            
            # Compact footer (bottom-right): time | processing
            timestamp = time.strftime("%H:%M:%S")
            proc = f"{detection_result['processing_time_ms']:.1f}ms"
            footer = f"{timestamp} | {proc}"
            (tsw, tsh), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, s * 0.85, thin)
            x0 = w - tsw - 10
            y0 = h - 10
            if self.hud_bg:
                overlay = annotated.copy()
                cv2.rectangle(overlay, (x0 - 6, y0 - tsh - 8), (x0 + tsw + 6, y0 + 4), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0, annotated)
            cv2.putText(annotated, footer, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, s * 0.85, (255, 255, 255), thin, cv2.LINE_AA)
            
            return annotated
            
        except Exception as e:
            self.logger.error(f"Error creating annotated image: {e}")
            return image
    
    def create_defect_metadata(self, detection_result: Dict):  # -> DefectMetadata:
        """
        Create defect metadata message (temporarily disabled)
        """
        try:
            metadata = DefectMetadata()
            metadata.header.stamp = self.get_clock().now().to_msg()
            metadata.header.frame_id = "camera_link"
            
            # Set basic information
            self.part_id_counter += 1
            metadata.part_id = f"PART_{self.part_id_counter:06d}"
            
            fusion_result = detection_result['fusion_result']
            if fusion_result:
                metadata.defect_detected = fusion_result.defect_detected
                metadata.overall_confidence = fusion_result.overall_confidence
                metadata.total_defects = fusion_result.total_defects
                metadata.processing_time_ms = fusion_result.decision_time_ms
            else:
                metadata.defect_detected = False
                metadata.overall_confidence = 0.0
                metadata.total_defects = 0
                metadata.processing_time_ms = detection_result['processing_time_ms']
            
            # Create defect details
            defect_details = []
            
            # Add contour defects
            for defect in detection_result['contour_defects']:
                defect_info = DefectInfo()
                defect_info.defect_type = defect.defect_type
                defect_info.location = Point2D(x=float(defect.location[0]), y=float(defect.location[1]))
                defect_info.severity_score = defect.severity_score
                defect_info.description = defect.description
                
                # Create bounding area polygon
                x, y, w, h = defect.bounding_box
                polygon = Polygon()
                polygon.points = [
                    Point32(x=float(x), y=float(y)),
                    Point32(x=float(x + w), y=float(y)),
                    Point32(x=float(x + w), y=float(y + h)),
                    Point32(x=float(x), y=float(y + h))
                ]
                defect_info.bounding_area = polygon
                
                defect_details.append(defect_info)
            
            # Add SIFT defects
            for defect in detection_result['sift_defects']:
                defect_info = DefectInfo()
                defect_info.defect_type = defect.defect_type
                defect_info.location = Point2D(x=float(defect.location[0]), y=float(defect.location[1]))
                defect_info.severity_score = defect.severity_score
                defect_info.description = defect.description
                
                # Create circular bounding area for SIFT defects
                x, y = defect.location
                radius = 20
                polygon = Polygon()
                for i in range(8):
                    angle = 2 * np.pi * i / 8
                    px = x + radius * np.cos(angle)
                    py = y + radius * np.sin(angle)
                    polygon.points.append(Point32(x=float(px), y=float(py)))
                
                defect_info.bounding_area = polygon
                defect_details.append(defect_info)
            
            metadata.defect_details = defect_details
            
            return None  # metadata
            
        except Exception as e:
            self.logger.error(f"Error creating defect metadata: {e}")
            return None  # DefectMetadata()
    
    def publish_annotated_image(self, annotated_image: np.ndarray, original_header):
        """Publish annotated image"""
        try:
            # Publish annotated image
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = original_header.frame_id
            self.annotated_pub.publish(ros_image)
            
        except Exception as e:
            self.logger.error(f"Error publishing annotated image: {e}")
    
    def publish_results(self, annotated_image: np.ndarray, detection_result: Dict, original_header):
        """Publish annotated image and metadata"""
        try:
            # Publish annotated image
            ros_image = self.bridge.cv2_to_imgmsg(annotated_image, "bgr8")
            ros_image.header.stamp = self.get_clock().now().to_msg()
            ros_image.header.frame_id = original_header.frame_id
            self.annotated_pub.publish(ros_image)
            
            # Publish metadata (temporarily disabled)
            # metadata = self.create_defect_metadata(detection_result)
            # self.metadata_pub.publish(metadata)
            
        except Exception as e:
            self.logger.error(f"Error publishing results: {e}")
    
    def update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        self.frame_count += 1
        self.processing_times.append(processing_time)
        
        # Log performance every 50 frames
        if self.frame_count % 50 == 0:
            avg_time = np.mean(self.processing_times[-50:])
            self.logger.info(f"Hybrid detection performance: {avg_time:.2f}ms avg, {self.frame_count} frames")
            
            # Log analyzer performance
            contour_stats = self.contour_analyzer.get_performance_stats()
            sift_stats = self.sift_analyzer.get_performance_stats()
            fusion_stats = self.fusion_engine.get_performance_stats()
            
            if contour_stats:
                self.logger.info(f"Contour analyzer: {contour_stats.get('avg_processing_time_ms', 0):.2f}ms avg")
            if sift_stats:
                self.logger.info(f"SIFT analyzer: {sift_stats.get('avg_processing_time_ms', 0):.2f}ms avg")
            if fusion_stats:
                self.logger.info(f"Fusion engine: {fusion_stats.get('avg_processing_time_ms', 0):.2f}ms avg")
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'thread_pool'):
            self.thread_pool.shutdown(wait=True)
        
        self.logger.info("Hybrid detector node cleanup completed")

def main(args=None):
    rclpy.init(args=args)
    
    node = HybridDetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        node.get_logger().error(f"Error in hybrid detector node: {e}")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 