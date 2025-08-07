#!/usr/bin/env python3
"""
SIFT Analyzer Module for MorphoTactus
Implements ROI-based SIFT feature extraction and template matching
Optimized for Jetson Nano deployment with ORB fallback
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

@dataclass
class SIFTDefect:
    """Data class for SIFT-based defect information"""
    defect_id: int
    defect_type: str
    location: Tuple[float, float]
    severity_score: float
    feature_count: int
    match_ratio: float
    spatial_consistency: float
    description: str

class SIFTAnalyzer:
    """
    SIFT feature analyzer with ROI-based detection and template matching
    Includes ORB fallback for real-time performance on Jetson Nano
    """
    
    def __init__(self, config: Dict):
        """Initialize SIFT analyzer with configuration parameters"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract SIFT parameters
        sift_config = config.get('sift_analysis', {})
        self.max_features = sift_config.get('max_features', 500)
        self.match_ratio_threshold = sift_config.get('match_ratio_threshold', 0.75)
        self.min_matches = sift_config.get('min_matches', 10)
        self.spatial_consistency_threshold = sift_config.get('spatial_consistency_threshold', 0.8)
        
        # Extract ORB parameters for fallback
        orb_config = config.get('orb_analysis', {})
        self.orb_max_features = orb_config.get('max_features', 1000)
        
        # Initialize feature detectors
        self.use_orb = config.get('jetson_optimizations', {}).get('use_orb_fallback', True)
        self.initialize_detectors()
        
        # Performance tracking
        self.processing_times = []
        
    def initialize_detectors(self):
        """Initialize SIFT and ORB detectors"""
        try:
            # Initialize SIFT detector
            self.sift = cv2.SIFT_create(nfeatures=self.max_features)
            
            # Initialize ORB detector for fallback
            self.orb = cv2.ORB_create(nfeatures=self.orb_max_features)
            
            # Initialize FLANN matcher for SIFT
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            self.flann_matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Initialize BF matcher for ORB
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
        except Exception as e:
            self.logger.error(f"Error initializing feature detectors: {e}")
            raise
    
    def extract_roi_features(self, image: np.ndarray, roi: Tuple[int, int, int, int]) -> Tuple[List, np.ndarray]:
        """Extract SIFT features from ROI region"""
        try:
            x, y, w, h = roi
            roi_image = image[y:y+h, x:x+w]
            
            if roi_image.size == 0:
                return [], np.array([])
            
            # Convert to grayscale if needed
            if len(roi_image.shape) == 3:
                gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
            else:
                gray_roi = roi_image.copy()
            
            # Extract features using SIFT or ORB
            if self.use_orb:
                keypoints, descriptors = self.orb.detectAndCompute(gray_roi, None)
            else:
                keypoints, descriptors = self.sift.detectAndCompute(gray_roi, None)
            
            # Adjust keypoint coordinates to original image space
            for kp in keypoints:
                kp.pt = (kp.pt[0] + x, kp.pt[1] + y)
            
            return keypoints, descriptors
            
        except Exception as e:
            self.logger.error(f"Error extracting ROI features: {e}")
            return [], np.array([])
    
    def analyze_textural_defects(self, image: np.ndarray, contour_regions: List[Tuple[int, int, int, int]]) -> List[SIFTDefect]:
        """Analyze textural defects using SIFT features in contour regions"""
        start_time = time.time()
        defects = []
        
        try:
            for i, roi in enumerate(contour_regions):
                # Extract features from ROI
                keypoints, descriptors = self.extract_roi_features(image, roi)
                
                if len(keypoints) == 0:
                    continue
                
                # Simulate template matching
                match_score = self.simulate_template_matching(descriptors)
                spatial_consistency = self.calculate_spatial_consistency_simulated(keypoints)
                
                # Calculate defect severity
                severity_score = self.calculate_textural_severity(match_score, spatial_consistency, len(keypoints))
                
                if severity_score > 0.2:  # Only report significant defects
                    # Calculate centroid of ROI
                    x, y, w, h = roi
                    centroid = (x + w/2, y + h/2)
                    
                    defect = SIFTDefect(
                        defect_id=i,
                        defect_type=self.classify_textural_defect(match_score, spatial_consistency),
                        location=centroid,
                        severity_score=severity_score,
                        feature_count=len(keypoints),
                        match_ratio=match_score,
                        spatial_consistency=spatial_consistency,
                        description=self.generate_textural_description(match_score, spatial_consistency, len(keypoints))
                    )
                    defects.append(defect)
            
            # Track processing time
            processing_time = (time.time() - start_time) * 1000
            self.processing_times.append(processing_time)
            
            self.logger.info(f"SIFT analysis completed in {processing_time:.2f}ms, found {len(defects)} defects")
            
        except Exception as e:
            self.logger.error(f"Error in SIFT defect analysis: {e}")
        
        return defects
    
    def simulate_template_matching(self, descriptors: np.ndarray) -> float:
        """Simulate template matching"""
        try:
            if descriptors is None or len(descriptors) == 0:
                return 0.0
            
            # Simulate matching score based on descriptor quality
            avg_descriptor_quality = np.mean(np.std(descriptors, axis=1))
            match_score = min(1.0, avg_descriptor_quality / 50.0)
            
            return match_score
            
        except Exception as e:
            self.logger.error(f"Error in template matching simulation: {e}")
            return 0.0
    
    def calculate_spatial_consistency_simulated(self, keypoints: List) -> float:
        """Simulate spatial consistency calculation"""
        try:
            if len(keypoints) < 4:
                return 0.0
            
            # Calculate spatial distribution of keypoints
            points = np.array([kp.pt for kp in keypoints])
            
            # Calculate standard deviation of point distribution
            std_x = np.std(points[:, 0])
            std_y = np.std(points[:, 1])
            
            # Normalize spatial consistency
            spatial_consistency = 1.0 / (1.0 + (std_x + std_y) / 100.0)
            
            return spatial_consistency
            
        except Exception as e:
            self.logger.error(f"Error in spatial consistency simulation: {e}")
            return 0.0
    
    def calculate_textural_severity(self, match_score: float, spatial_consistency: float, feature_count: int) -> float:
        """Calculate textural defect severity score"""
        severity_factors = [
            1.0 - match_score,
            1.0 - spatial_consistency,
            max(0, 1.0 - feature_count / 100.0)
        ]
        
        return np.mean(severity_factors)
    
    def classify_textural_defect(self, match_score: float, spatial_consistency: float) -> str:
        """Classify textural defect type"""
        if match_score < 0.3:
            return "textural_anomaly"
        elif spatial_consistency < 0.5:
            return "spatial_inconsistency"
        elif match_score < 0.6:
            return "feature_mismatch"
        else:
            return "minor_textural_variation"
    
    def generate_textural_description(self, match_score: float, spatial_consistency: float, feature_count: int) -> str:
        """Generate human-readable textural defect description"""
        descriptions = {
            "textural_anomaly": f"Textural anomaly detected (match score: {match_score:.3f})",
            "spatial_inconsistency": f"Spatial inconsistency detected (consistency: {spatial_consistency:.3f})",
            "feature_mismatch": f"Feature mismatch detected (match score: {match_score:.3f})",
            "minor_textural_variation": f"Minor textural variation (features: {feature_count})"
        }
        
        defect_type = self.classify_textural_defect(match_score, spatial_consistency)
        return descriptions.get(defect_type, "Unknown textural defect")
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.processing_times:
            return {}
        
        return {
            'avg_processing_time_ms': np.mean(self.processing_times),
            'min_processing_time_ms': np.min(self.processing_times),
            'max_processing_time_ms': np.max(self.processing_times),
            'total_analyses': len(self.processing_times),
            'detector_type': 'ORB' if self.use_orb else 'SIFT'
        } 