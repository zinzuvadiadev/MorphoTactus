#!/usr/bin/env python3
"""
Fusion Engine Module for MorphoTactus
Combines contour and SIFT analysis results with weighted decision making
Optimized for real-time processing on Jetson Nano
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

@dataclass
class FusionResult:
    """Data class for fusion engine results"""
    overall_confidence: float
    defect_detected: bool
    total_defects: int
    critical_defects: int
    decision_time_ms: float
    fusion_weights: Dict[str, float]
    early_termination: bool

class FusionEngine:
    """
    Fusion engine that combines contour and SIFT analysis results
    with configurable weights and early termination capabilities
    """
    
    def __init__(self, config: Dict):
        """Initialize fusion engine with configuration parameters"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Extract fusion parameters
        fusion_config = config.get('fusion_engine', {})
        self.contour_weight = fusion_config.get('contour_weight', 0.6)
        self.sift_weight = fusion_config.get('sift_weight', 0.4)
        self.overall_confidence_threshold = fusion_config.get('overall_confidence_threshold', 0.75)
        self.critical_defect_threshold = fusion_config.get('critical_defect_threshold', 0.9)
        self.early_termination_enabled = fusion_config.get('early_termination_enabled', True)
        self.max_processing_time_ms = fusion_config.get('max_processing_time_ms', 100)
        
        # Performance tracking
        self.processing_times = []
        self.fusion_decisions = []
        
    def fuse_results(self, contour_defects: List, sift_defects: List, 
                    processing_start_time: float) -> FusionResult:
        """
        Main fusion method that combines contour and SIFT analysis results
        """
        start_time = time.time()
        
        try:
            # Check for early termination
            if self.early_termination_enabled:
                early_termination = self.check_early_termination(contour_defects, sift_defects)
                if early_termination:
                    return self.create_early_termination_result(start_time)
            
            # Calculate individual confidence scores
            contour_confidence = self.calculate_contour_confidence(contour_defects)
            sift_confidence = self.calculate_sift_confidence(sift_defects)
            
            # Weighted fusion
            overall_confidence = (self.contour_weight * contour_confidence + 
                               self.sift_weight * sift_confidence)
            
            # Determine defect status
            defect_detected = overall_confidence > self.overall_confidence_threshold
            
            # Count defects
            total_defects = len(contour_defects) + len(sift_defects)
            critical_defects = self.count_critical_defects(contour_defects, sift_defects)
            
            # Calculate decision time
            decision_time = (time.time() - start_time) * 1000
            
            # Create fusion result
            result = FusionResult(
                overall_confidence=overall_confidence,
                defect_detected=defect_detected,
                total_defects=total_defects,
                critical_defects=critical_defects,
                decision_time_ms=decision_time,
                fusion_weights={'contour': self.contour_weight, 'sift': self.sift_weight},
                early_termination=False
            )
            
            # Track performance
            self.processing_times.append(decision_time)
            self.fusion_decisions.append(result)
            
            self.logger.info(f"Fusion completed in {decision_time:.2f}ms, confidence: {overall_confidence:.3f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in fusion engine: {e}")
            return self.create_error_result(start_time)
    
    def check_early_termination(self, contour_defects: List, sift_defects: List) -> bool:
        """
        Check if early termination should be triggered
        """
        try:
            # Check for critical defects in contour analysis
            for defect in contour_defects:
                if defect.severity_score > self.critical_defect_threshold:
                    self.logger.warning(f"Critical defect detected in contour analysis: {defect.defect_type}")
                    return True
            
            # Check processing time limit
            current_time = time.time()
            if hasattr(self, '_processing_start_time'):
                elapsed_time = (current_time - self._processing_start_time) * 1000
                if elapsed_time > self.max_processing_time_ms:
                    self.logger.warning(f"Processing time limit exceeded: {elapsed_time:.2f}ms")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in early termination check: {e}")
            return False
    
    def calculate_contour_confidence(self, contour_defects: List) -> float:
        """
        Calculate confidence score from contour analysis results
        """
        try:
            if not contour_defects:
                return 0.0
            
            # Calculate confidence based on defect severity and count
            total_severity = sum(defect.severity_score for defect in contour_defects)
            avg_severity = total_severity / len(contour_defects)
            
            # Normalize confidence (higher severity = higher confidence in defect detection)
            confidence = min(1.0, avg_severity * len(contour_defects) / 10.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating contour confidence: {e}")
            return 0.0
    
    def calculate_sift_confidence(self, sift_defects: List) -> float:
        """
        Calculate confidence score from SIFT analysis results
        """
        try:
            if not sift_defects:
                return 0.0
            
            # Calculate confidence based on match ratio and spatial consistency
            total_match_score = sum(1.0 - defect.match_ratio for defect in sift_defects)
            total_spatial_consistency = sum(1.0 - defect.spatial_consistency for defect in sift_defects)
            
            # Combine factors
            avg_match_score = total_match_score / len(sift_defects)
            avg_spatial_consistency = total_spatial_consistency / len(sift_defects)
            
            # Normalize confidence
            confidence = min(1.0, (avg_match_score + avg_spatial_consistency) / 2.0)
            
            return confidence
            
        except Exception as e:
            self.logger.error(f"Error calculating SIFT confidence: {e}")
            return 0.0
    
    def count_critical_defects(self, contour_defects: List, sift_defects: List) -> int:
        """
        Count critical defects based on severity thresholds
        """
        try:
            critical_count = 0
            
            # Count critical contour defects
            for defect in contour_defects:
                if defect.severity_score > self.critical_defect_threshold:
                    critical_count += 1
            
            # Count critical SIFT defects
            for defect in sift_defects:
                if defect.severity_score > self.critical_defect_threshold:
                    critical_count += 1
            
            return critical_count
            
        except Exception as e:
            self.logger.error(f"Error counting critical defects: {e}")
            return 0
    
    def create_early_termination_result(self, start_time: float) -> FusionResult:
        """
        Create result for early termination case
        """
        decision_time = (time.time() - start_time) * 1000
        
        return FusionResult(
            overall_confidence=1.0,  # High confidence for critical defects
            defect_detected=True,
            total_defects=1,
            critical_defects=1,
            decision_time_ms=decision_time,
            fusion_weights={'contour': self.contour_weight, 'sift': self.sift_weight},
            early_termination=True
        )
    
    def create_error_result(self, start_time: float) -> FusionResult:
        """
        Create result for error case
        """
        decision_time = (time.time() - start_time) * 1000
        
        return FusionResult(
            overall_confidence=0.0,
            defect_detected=False,
            total_defects=0,
            critical_defects=0,
            decision_time_ms=decision_time,
            fusion_weights={'contour': self.contour_weight, 'sift': self.sift_weight},
            early_termination=False
        )
    
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
            'total_fusions': len(self.processing_times),
            'early_terminations': sum(1 for result in self.fusion_decisions if result.early_termination),
            'avg_confidence': np.mean([result.overall_confidence for result in self.fusion_decisions])
        }
    
    def update_weights(self, contour_weight: float, sift_weight: float):
        """
        Update fusion weights dynamically
        """
        try:
            # Normalize weights
            total_weight = contour_weight + sift_weight
            if total_weight > 0:
                self.contour_weight = contour_weight / total_weight
                self.sift_weight = sift_weight / total_weight
            else:
                self.contour_weight = 0.5
                self.sift_weight = 0.5
            
            self.logger.info(f"Updated fusion weights - Contour: {self.contour_weight:.3f}, SIFT: {self.sift_weight:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error updating fusion weights: {e}")
    
    def reset_performance_tracking(self):
        """
        Reset performance tracking data
        """
        self.processing_times = []
        self.fusion_decisions = []
        self.logger.info("Performance tracking reset") 