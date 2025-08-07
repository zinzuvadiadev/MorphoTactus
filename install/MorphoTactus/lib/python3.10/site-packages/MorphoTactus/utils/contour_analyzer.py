#!/usr/bin/env python3
"""
Contour Analyzer Module for MorphoTactus
Implements adaptive contour detection and geometric feature extraction
Optimized for Jetson Nano deployment
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class ContourDefect:
    """Data class for contour-based defect information"""
    contour_id: int
    defect_type: str
    location: Tuple[float, float]
    severity_score: float
    bounding_box: Tuple[int, int, int, int]
    area: float
    perimeter: float
    circularity: float
    convexity: float
    description: str

class ContourAnalyzer:
    """
    Advanced contour analyzer with adaptive Canny edge detection
    and geometric feature extraction for defect detection
    """
    
    def __init__(self, config: Dict):
        """Initialize contour analyzer with configuration parameters"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract contour detection parameters
        contour_config = config.get('contour_detection', {})
        self.canny_low = contour_config.get('canny_low_threshold', 50)
        self.canny_high = contour_config.get('canny_high_threshold', 150)
        self.canny_aperture = contour_config.get('canny_aperture_size', 3)
        self.min_area = contour_config.get('min_contour_area', 100)
        self.max_area = contour_config.get('max_contour_area', 50000)
        self.min_perimeter = contour_config.get('min_contour_perimeter', 50)
        self.aspect_ratio_threshold = contour_config.get('aspect_ratio_threshold', 0.1)
        self.circularity_threshold = contour_config.get('circularity_threshold', 0.6)
        self.convexity_threshold = contour_config.get('convexity_threshold', 0.8)
        self.anomaly_threshold = contour_config.get('anomaly_score_threshold', 0.7)
        
        # Performance tracking
        self.processing_times = []
        
    def adaptive_canny_detection(self, image: np.ndarray) -> np.ndarray:
        """
        Adaptive Canny edge detection with automatic threshold calculation
        """
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate adaptive thresholds using image statistics
            mean_intensity = np.mean(blurred)
            std_intensity = np.std(blurred)
            
            # Adaptive threshold calculation
            low_threshold = max(0, mean_intensity - std_intensity)
            high_threshold = min(255, mean_intensity + std_intensity)
            
            # Apply Canny edge detection
            edges = cv2.Canny(blurred, low_threshold, high_threshold, 
                            apertureSize=self.canny_aperture)
            
            return edges
            
        except Exception as e:
            self.logger.error(f"Error in adaptive Canny detection: {e}")
            return np.zeros_like(image)
    
    def extract_geometric_features(self, contour: np.ndarray) -> Dict:
        """
        Extract comprehensive geometric features from contour
        """
        try:
            # Basic features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Circularity (4π * area / perimeter²)
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Moments
            moments = cv2.moments(contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = 0, 0
            
            # Solidity (area / convex hull area)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Extent (contour area / bounding rectangle area)
            extent = area / (w * h) if w * h > 0 else 0
            
            return {
                'area': area,
                'perimeter': perimeter,
                'aspect_ratio': aspect_ratio,
                'circularity': circularity,
                'convexity': convexity,
                'solidity': solidity,
                'extent': extent,
                'centroid': (cx, cy),
                'bounding_rect': (x, y, w, h)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting geometric features: {e}")
            return {}
    
    def calculate_anomaly_score(self, features: Dict, expected_features: Dict = None) -> float:
        """
        Calculate anomaly score based on deviation from expected features
        """
        try:
            if expected_features is None:
                # Default expected features for typical mechanical parts
                expected_features = {
                    'circularity': 0.8,
                    'convexity': 0.9,
                    'solidity': 0.85,
                    'extent': 0.7
                }
            
            # Calculate deviations
            deviations = []
            for key in ['circularity', 'convexity', 'solidity', 'extent']:
                if key in features and key in expected_features:
                    deviation = abs(features[key] - expected_features[key])
                    deviations.append(deviation)
            
            # Normalize and combine deviations
            if deviations:
                anomaly_score = np.mean(deviations)
                return min(1.0, anomaly_score)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating anomaly score: {e}")
            return 0.0
    
    def filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """
        Filter contours based on area, perimeter, and aspect ratio criteria
        """
        filtered_contours = []
        
        for contour in contours:
            features = self.extract_geometric_features(contour)
            
            if not features:
                continue
            
            # Apply filters
            if (self.min_area <= features['area'] <= self.max_area and
                features['perimeter'] >= self.min_perimeter and
                features['aspect_ratio'] >= self.aspect_ratio_threshold):
                filtered_contours.append(contour)
        
        return filtered_contours
    
    def detect_defects(self, image: np.ndarray) -> List[ContourDefect]:
        """
        Main defect detection method using contour analysis
        """
        start_time = time.time()
        defects = []
        
        try:
            # Adaptive Canny edge detection
            edges = self.adaptive_canny_detection(image)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            filtered_contours = self.filter_contours(contours)
            
            # Analyze each contour for defects
            for i, contour in enumerate(filtered_contours):
                features = self.extract_geometric_features(contour)
                
                if not features:
                    continue
                
                # Calculate anomaly score
                anomaly_score = self.calculate_anomaly_score(features)
                
                # Determine defect type and severity
                defect_type = self.classify_defect(features, anomaly_score)
                severity_score = self.calculate_severity(features, anomaly_score)
                
                if severity_score > 0.1:  # Only report significant defects
                    defect = ContourDefect(
                        contour_id=i,
                        defect_type=defect_type,
                        location=features['centroid'],
                        severity_score=severity_score,
                        bounding_box=features['bounding_rect'],
                        area=features['area'],
                        perimeter=features['perimeter'],
                        circularity=features['circularity'],
                        convexity=features['convexity'],
                        description=self.generate_defect_description(features, defect_type)
                    )
                    defects.append(defect)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            self.logger.info(f"Contour analysis completed in {processing_time:.2f}ms, found {len(defects)} defects")
            
        except Exception as e:
            self.logger.error(f"Error in contour defect detection: {e}")
        
        return defects
    
    def classify_defect(self, features: Dict, anomaly_score: float) -> str:
        """
        Classify defect type based on geometric features
        """
        if anomaly_score > self.anomaly_threshold:
            if features['circularity'] < self.circularity_threshold:
                return "geometric_deformity"
            elif features['convexity'] < self.convexity_threshold:
                return "surface_irregularity"
            elif features['solidity'] < 0.7:
                return "structural_defect"
            else:
                return "general_anomaly"
        else:
            return "minor_variation"
    
    def calculate_severity(self, features: Dict, anomaly_score: float) -> float:
        """
        Calculate defect severity score
        """
        # Combine multiple factors for severity calculation
        severity_factors = [
            anomaly_score,
            1.0 - features['circularity'],
            1.0 - features['convexity'],
            1.0 - features['solidity']
        ]
        
        return np.mean(severity_factors)
    
    def generate_defect_description(self, features: Dict, defect_type: str) -> str:
        """
        Generate human-readable defect description
        """
        descriptions = {
            "geometric_deformity": f"Geometric deformity detected (circularity: {features['circularity']:.3f})",
            "surface_irregularity": f"Surface irregularity detected (convexity: {features['convexity']:.3f})",
            "structural_defect": f"Structural defect detected (solidity: {features['solidity']:.3f})",
            "general_anomaly": "General geometric anomaly detected",
            "minor_variation": "Minor geometric variation detected"
        }
        
        return descriptions.get(defect_type, "Unknown defect type")
    
    def get_performance_stats(self) -> Dict:
        """
        Get performance statistics
        """
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'total_analyses': len(self.processing_times)
        } 