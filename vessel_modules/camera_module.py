"""
Vessel Camera Module - YOLOv5 Detection with Position Tracking
==============================================================

Processes camera feed for plastic detection and calculates relative positions
for autonomous navigation.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
import time


class VesselCamera:
    """
    Camera module for vessel with plastic detection and position tracking
    """
    
    def __init__(self, model, camera_fov_deg=60, camera_range_m=30, img_size=640):
        """
        Initialize vessel camera
        
        Args:
            model: YOLOv5 model for detection
            camera_fov_deg: Camera field of view in degrees
            camera_range_m: Maximum detection range in meters
            img_size: Image size for detection
        """
        self.model = model
        self.camera_fov_deg = camera_fov_deg
        self.camera_range_m = camera_range_m
        self.img_size = img_size
        self.frame_width = None
        self.frame_height = None
        
    def detect_and_track(self, frame: np.ndarray, conf_threshold=0.15) -> Dict:
        """
        Detect plastics and calculate their relative positions
        
        Args:
            frame: Input image frame
            conf_threshold: Confidence threshold for detection
            
        Returns:
            Dictionary with detections and navigation data
        """
        self.frame_height, self.frame_width = frame.shape[:2]
        
        # Run detection
        detections = self._run_detection(frame, conf_threshold)
        
        if not detections:
            return {
                'detections': [],
                'target': None,
                'navigation_command': 'search',
                'frame': frame
            }
        
        # Find closest/best target
        target = self._select_target(detections)
        
        # Calculate navigation command
        nav_command = self._calculate_navigation(target)
        
        # Draw detections on frame
        annotated_frame = self._draw_detections(frame, detections, target)
        
        return {
            'detections': detections,
            'target': target,
            'navigation_command': nav_command,
            'frame': annotated_frame
        }
    
    def _run_detection(self, frame: np.ndarray, conf_threshold: float) -> List[Dict]:
        """Run YOLOv5 detection on frame"""
        # Prepare image
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)
        
        # Inference
        with torch.no_grad():
            pred = self.model(img)[0]
        
        # NMS
        from yolov5.utils.general import non_max_suppression
        pred = non_max_suppression(pred, conf_threshold, 0.40, max_det=20)
        
        detections = []
        for det in pred:
            if det is not None and len(det):
                # Scale coordinates back to original frame
                det[:, :4] = self._scale_coords(
                    (self.img_size, self.img_size),
                    det[:, :4],
                    (self.frame_height, self.frame_width)
                )
                
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Calculate relative position
                    rel_x = (center_x - self.frame_width / 2) / (self.frame_width / 2)
                    rel_y = (center_y - self.frame_height / 2) / (self.frame_height / 2)
                    
                    # Estimate distance (simple approximation based on bbox size)
                    estimated_distance = self._estimate_distance(width, height)
                    
                    # Calculate angle from center
                    angle_deg = rel_x * (self.camera_fov_deg / 2)
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'center': (center_x, center_y),
                        'confidence': float(conf),
                        'class_id': int(cls),
                        'class_name': 'plastic',
                        'relative_x': rel_x,  # -1 (left) to 1 (right)
                        'relative_y': rel_y,  # -1 (top) to 1 (bottom)
                        'angle_deg': angle_deg,
                        'estimated_distance_m': estimated_distance,
                        'size': width * height
                    })
        
        return detections
    
    def _scale_coords(self, img1_shape, coords, img0_shape):
        """Scale coordinates from img1_shape to img0_shape"""
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
        
        coords[:, [0, 2]] -= pad[0]
        coords[:, [1, 3]] -= pad[1]
        coords[:, :4] /= gain
        
        # Clip coordinates
        coords[:, 0].clamp_(0, img0_shape[1])
        coords[:, 1].clamp_(0, img0_shape[0])
        coords[:, 2].clamp_(0, img0_shape[1])
        coords[:, 3].clamp_(0, img0_shape[0])
        
        return coords
    
    def _estimate_distance(self, bbox_width: float, bbox_height: float) -> float:
        """
        Estimate distance to object based on bounding box size
        Assumes average plastic bottle is ~20cm tall
        """
        # Simple inverse relationship: larger bbox = closer object
        # This is a rough approximation - calibrate with real measurements
        avg_bbox_size = (bbox_width + bbox_height) / 2
        max_bbox_size = (self.frame_width + self.frame_height) / 4
        
        # Normalize and invert
        size_ratio = avg_bbox_size / max_bbox_size
        estimated_distance = self.camera_range_m * (1 - size_ratio * 0.9)
        
        return max(1.0, min(self.camera_range_m, estimated_distance))
    
    def _select_target(self, detections: List[Dict]) -> Optional[Dict]:
        """
        Select best target for navigation
        Priority: closest object with high confidence in center of view
        """
        if not detections:
            return None
        
        # Score each detection
        for det in detections:
            # Factors: distance (closer better), confidence, center alignment
            distance_score = 1.0 / (det['estimated_distance_m'] + 1)
            confidence_score = det['confidence']
            center_score = 1.0 - abs(det['relative_x'])  # prefer centered objects
            
            det['target_score'] = (
                distance_score * 0.5 +
                confidence_score * 0.3 +
                center_score * 0.2
            )
        
        # Return highest scoring detection
        return max(detections, key=lambda x: x['target_score'])
    
    def _calculate_navigation(self, target: Dict) -> str:
        """
        Calculate navigation command based on target position
        
        Returns:
            Navigation command: 'forward', 'turn_left', 'turn_right', 'collect', 'search'
        """
        if not target:
            return 'search'
        
        distance = target['estimated_distance_m']
        angle = target['angle_deg']
        
        # If very close, collect
        if distance < 2.0:
            return 'collect'
        
        # If target is centered, move forward
        if abs(angle) < 10:
            return 'forward'
        
        # Turn toward target
        if angle < -10:
            return 'turn_left'
        elif angle > 10:
            return 'turn_right'
        
        return 'forward'
    
    def _draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                        target: Optional[Dict]) -> np.ndarray:
        """Draw bounding boxes and info on frame"""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            is_target = (target and det == target)
            
            # Color: green for target, cyan for others
            color = (0, 255, 0) if is_target else (255, 255, 0)
            thickness = 3 if is_target else 2
            
            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det['class_name']} {det['confidence']:.2f}"
            label += f" | {det['estimated_distance_m']:.1f}m"
            label += f" | {det['angle_deg']:.0f}°"
            
            if is_target:
                label = "🎯 TARGET: " + label
            
            # Background for text
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
            # Draw center point
            cx, cy = map(int, det['center'])
            cv2.circle(annotated, (cx, cy), 5, color, -1)
        
        # Draw center crosshair
        h, w = annotated.shape[:2]
        cv2.line(annotated, (w//2 - 20, h//2), (w//2 + 20, h//2), (0, 255, 255), 2)
        cv2.line(annotated, (w//2, h//2 - 20), (w//2, h//2 + 20), (0, 255, 255), 2)
        
        return annotated
