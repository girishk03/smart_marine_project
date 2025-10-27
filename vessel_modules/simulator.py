"""
Vessel Simulator - Digital Twin for Testing Without Hardware
============================================================

Simulates GPS, camera, sensors, and navigation for safe testing
of the autonomous vessel system.
"""

import numpy as np
import cv2
import random
import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class SimulatedPlastic:
    """Simulated plastic object in the environment"""
    lat: float
    lon: float
    size: float = 1.0  # relative size
    collected: bool = False
    id: int = 0


class VesselSimulator:
    """
    Digital twin simulator for the Smart Marine Vessel
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize vessel simulator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Map center (default: Chennai coast)
        self.map_center_lat = self.config.get('map_center_lat', 13.0827)
        self.map_center_lon = self.config.get('map_center_lon', 80.2707)
        
        # Boat state
        self.boat_lat = self.map_center_lat
        self.boat_lon = self.map_center_lon
        self.boat_heading = 0.0  # degrees
        self.boat_speed = 0.0  # m/s
        
        # Simulation parameters
        self.boat_speed_mps = self.config.get('boat_speed_mps', 1.5)
        self.detection_range_m = self.config.get('detection_range_m', 50)
        self.collection_range_m = self.config.get('collection_range_m', 2)
        
        # Plastic objects
        self.plastics: List[SimulatedPlastic] = []
        self.spawn_plastic_objects(self.config.get('spawn_plastic_count', 10))
        
        # Camera simulation
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov_deg = 60
        
        # Statistics
        self.time_elapsed = 0.0
        self.distance_traveled = 0.0
        
    def spawn_plastic_objects(self, count: int):
        """
        Spawn random plastic objects around the map
        
        Args:
            count: Number of plastic objects to spawn
        """
        self.plastics = []
        
        # Spawn in a radius around map center
        radius_deg = 0.002  # ~200m radius
        
        for i in range(count):
            # Random position in circle
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(0, radius_deg)
            
            lat = self.map_center_lat + distance * math.cos(angle)
            lon = self.map_center_lon + distance * math.sin(angle)
            size = random.uniform(0.5, 2.0)
            
            self.plastics.append(SimulatedPlastic(
                lat=lat,
                lon=lon,
                size=size,
                id=i+1
            ))
    
    def update(self, dt: float, command: str, target_heading: Optional[float] = None):
        """
        Update simulation state
        
        Args:
            dt: Time delta in seconds
            command: Navigation command ('forward', 'turn_left', 'turn_right', 'stop')
            target_heading: Target heading for turning
        """
        self.time_elapsed += dt
        
        # Update heading
        turn_rate = 30.0  # degrees per second
        if command == 'turn_left':
            self.boat_heading -= turn_rate * dt
            self.boat_speed = self.boat_speed_mps * 0.5
        elif command == 'turn_right':
            self.boat_heading += turn_rate * dt
            self.boat_speed = self.boat_speed_mps * 0.5
        elif command == 'forward':
            self.boat_speed = self.boat_speed_mps
            # Gradually adjust to target heading if provided
            if target_heading is not None:
                heading_diff = self._normalize_angle(target_heading - self.boat_heading)
                if abs(heading_diff) > 1:
                    self.boat_heading += np.sign(heading_diff) * min(abs(heading_diff), turn_rate * dt)
        elif command == 'stop':
            self.boat_speed = 0.0
        
        self.boat_heading = self.boat_heading % 360
        
        # Update position based on speed and heading
        if self.boat_speed > 0:
            # Convert speed to lat/lon change
            # Approximate: 1 degree latitude ≈ 111,000 meters
            meters_per_deg_lat = 111000
            meters_per_deg_lon = 111000 * math.cos(math.radians(self.boat_lat))
            
            distance_m = self.boat_speed * dt
            self.distance_traveled += distance_m
            
            # Calculate lat/lon change
            heading_rad = math.radians(self.boat_heading)
            delta_lat = (distance_m * math.cos(heading_rad)) / meters_per_deg_lat
            delta_lon = (distance_m * math.sin(heading_rad)) / meters_per_deg_lon
            
            self.boat_lat += delta_lat
            self.boat_lon += delta_lon
    
    def get_visible_plastics(self) -> List[Dict]:
        """
        Get plastics visible to the boat's camera
        
        Returns:
            List of visible plastic detections
        """
        visible = []
        
        for plastic in self.plastics:
            if plastic.collected:
                continue
            
            # Calculate distance to plastic
            distance = self._calculate_distance(
                self.boat_lat, self.boat_lon,
                plastic.lat, plastic.lon
            )
            
            if distance > self.detection_range_m:
                continue
            
            # Calculate bearing to plastic
            bearing = self._calculate_bearing(
                self.boat_lat, self.boat_lon,
                plastic.lat, plastic.lon
            )
            
            # Check if in camera FOV
            angle_diff = self._normalize_angle(bearing - self.boat_heading)
            if abs(angle_diff) > self.camera_fov_deg / 2:
                continue
            
            # Calculate screen position
            # Center = 0, left = -1, right = 1
            rel_x = angle_diff / (self.camera_fov_deg / 2)
            
            # Size decreases with distance
            apparent_size = plastic.size * (1.0 - distance / self.detection_range_m)
            
            # Confidence decreases with distance
            confidence = 0.9 * (1.0 - distance / self.detection_range_m) * plastic.size
            
            visible.append({
                'id': plastic.id,
                'lat': plastic.lat,
                'lon': plastic.lon,
                'distance': distance,
                'bearing': bearing,
                'angle_deg': angle_diff,
                'relative_x': rel_x,
                'size': apparent_size,
                'confidence': max(0.1, min(0.95, confidence)),
                'estimated_distance_m': distance
            })
        
        return visible
    
    def generate_camera_frame(self, detections: List[Dict]) -> np.ndarray:
        """
        Generate simulated camera frame with detections
        
        Args:
            detections: List of detected plastics
            
        Returns:
            Simulated camera frame
        """
        # Create ocean background
        frame = np.zeros((self.camera_height, self.camera_width, 3), dtype=np.uint8)
        
        # Ocean gradient (dark blue to light blue)
        for y in range(self.camera_height):
            intensity = int(20 + (y / self.camera_height) * 60)
            frame[y, :] = [intensity, intensity // 2, 0]  # Blue gradient
        
        # Add some wave texture
        for _ in range(50):
            x = random.randint(0, self.camera_width)
            y = random.randint(0, self.camera_height)
            cv2.circle(frame, (x, y), random.randint(2, 8), (80, 60, 20), -1)
        
        # Draw detected plastics
        for det in detections:
            # Calculate screen position
            screen_x = int(self.camera_width / 2 + det['relative_x'] * self.camera_width / 2)
            
            # Y position based on distance (closer = lower on screen)
            distance_ratio = det['distance'] / self.detection_range_m
            screen_y = int(self.camera_height * (0.3 + distance_ratio * 0.5))
            
            # Size based on apparent size
            bbox_size = int(det['size'] * 100)
            
            # Draw plastic object (bottle shape)
            x1 = max(0, screen_x - bbox_size // 2)
            y1 = max(0, screen_y - bbox_size)
            x2 = min(self.camera_width, screen_x + bbox_size // 2)
            y2 = min(self.camera_height, screen_y)
            
            # Draw bottle
            color = (0, 255, 255)  # Yellow for plastic
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"Plastic {det['confidence']:.2f} | {det['distance']:.1f}m"
            cv2.putText(frame, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Draw crosshair
        cv2.line(frame, (self.camera_width // 2 - 20, self.camera_height // 2),
                (self.camera_width // 2 + 20, self.camera_height // 2), (0, 255, 255), 1)
        cv2.line(frame, (self.camera_width // 2, self.camera_height // 2 - 20),
                (self.camera_width // 2, self.camera_height // 2 + 20), (0, 255, 255), 1)
        
        # Add HUD info
        cv2.putText(frame, f"Heading: {self.boat_heading:.0f}°", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Speed: {self.boat_speed:.1f} m/s", (10, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def try_collect_plastic(self) -> Optional[SimulatedPlastic]:
        """
        Attempt to collect nearby plastic
        
        Returns:
            Collected plastic object or None
        """
        for plastic in self.plastics:
            if plastic.collected:
                continue
            
            distance = self._calculate_distance(
                self.boat_lat, self.boat_lon,
                plastic.lat, plastic.lon
            )
            
            if distance <= self.collection_range_m:
                plastic.collected = True
                return plastic
        
        return None
    
    def get_state(self) -> Dict:
        """
        Get current simulation state
        
        Returns:
            State dictionary
        """
        uncollected = [p for p in self.plastics if not p.collected]
        collected = [p for p in self.plastics if p.collected]
        
        return {
            'boat_position': {
                'lat': self.boat_lat,
                'lon': self.boat_lon
            },
            'boat_heading': self.boat_heading,
            'boat_speed': self.boat_speed,
            'time_elapsed': self.time_elapsed,
            'distance_traveled': self.distance_traveled,
            'plastics_total': len(self.plastics),
            'plastics_collected': len(collected),
            'plastics_remaining': len(uncollected),
            'collection_progress': len(collected) / len(self.plastics) if self.plastics else 0
        }
    
    def get_plastic_markers(self) -> List[Dict]:
        """
        Get all plastic markers for map display
        
        Returns:
            List of plastic marker data
        """
        markers = []
        for plastic in self.plastics:
            markers.append({
                'lat': plastic.lat,
                'lon': plastic.lon,
                'collected': plastic.collected,
                'id': plastic.id,
                'size': plastic.size
            })
        return markers
    
    def _calculate_distance(self, lat1: float, lon1: float, 
                           lat2: float, lon2: float) -> float:
        """Calculate distance between two GPS coordinates (Haversine)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _calculate_bearing(self, lat1: float, lon1: float,
                          lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon = math.radians(lon2 - lon1)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        
        return bearing_deg
    
    def _normalize_angle(self, angle: float) -> float:
        """Normalize angle to -180 to 180 range"""
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
