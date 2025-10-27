"""
GPS Navigation Module - Autonomous Navigation & Tracking
========================================================

Handles GPS tracking, waypoint navigation, and autopilot control
for the Smart Marine Vessel.
"""

import math
import time
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class GPSCoordinate:
    """GPS coordinate with timestamp"""
    latitude: float
    longitude: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class GPSNavigator:
    """
    GPS-based navigation system for autonomous vessel
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize GPS Navigator
        
        Args:
            config: Configuration dictionary with navigation parameters
        """
        self.config = config or {}
        self.current_position = None
        self.target_position = None
        self.waypoints = []
        self.gps_trail = []
        self.max_trail_points = 100
        
        # Navigation parameters
        self.waypoint_threshold_m = self.config.get('waypoint_threshold_m', 2.0)
        self.heading_tolerance_deg = self.config.get('heading_tolerance_deg', 5.0)
        self.max_speed_mps = self.config.get('max_speed_mps', 2.0)
        self.obstacle_distance_m = self.config.get('obstacle_distance_m', 3.0)
        
        # State
        self.current_heading = 0.0  # degrees
        self.current_speed = 0.0  # m/s
        self.autopilot_active = False
        
    def update_position(self, latitude: float, longitude: float):
        """
        Update current GPS position
        
        Args:
            latitude: Current latitude
            longitude: Current longitude
        """
        self.current_position = GPSCoordinate(latitude, longitude)
        
        # Add to trail
        self.gps_trail.append(self.current_position)
        if len(self.gps_trail) > self.max_trail_points:
            self.gps_trail.pop(0)
    
    def set_target(self, latitude: float, longitude: float):
        """
        Set target destination
        
        Args:
            latitude: Target latitude
            longitude: Target longitude
        """
        self.target_position = GPSCoordinate(latitude, longitude)
    
    def calculate_distance(self, coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """
        Calculate distance between two GPS coordinates using Haversine formula
        
        Args:
            coord1: First coordinate
            coord2: Second coordinate
            
        Returns:
            Distance in meters
        """
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(coord1.latitude)
        lat2_rad = math.radians(coord2.latitude)
        delta_lat = math.radians(coord2.latitude - coord1.latitude)
        delta_lon = math.radians(coord2.longitude - coord1.longitude)
        
        a = (math.sin(delta_lat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        distance = R * c
        return distance
    
    def calculate_bearing(self, coord1: GPSCoordinate, coord2: GPSCoordinate) -> float:
        """
        Calculate bearing from coord1 to coord2
        
        Args:
            coord1: Start coordinate
            coord2: End coordinate
            
        Returns:
            Bearing in degrees (0-360)
        """
        lat1_rad = math.radians(coord1.latitude)
        lat2_rad = math.radians(coord2.latitude)
        delta_lon = math.radians(coord2.longitude - coord1.longitude)
        
        x = math.sin(delta_lon) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) -
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = (math.degrees(bearing_rad) + 360) % 360
        
        return bearing_deg
    
    def get_navigation_command(self, obstacle_distance: Optional[float] = None) -> Dict:
        """
        Calculate navigation command based on current position and target
        
        Args:
            obstacle_distance: Distance to nearest obstacle (meters)
            
        Returns:
            Dictionary with navigation commands
        """
        if not self.current_position or not self.target_position:
            return {
                'command': 'stop',
                'heading': 0,
                'speed': 0,
                'distance_to_target': 0,
                'bearing_to_target': 0,
                'status': 'no_target'
            }
        
        # Calculate distance and bearing to target
        distance = self.calculate_distance(self.current_position, self.target_position)
        bearing = self.calculate_bearing(self.current_position, self.target_position)
        
        # Check if target reached
        if distance < self.waypoint_threshold_m:
            return {
                'command': 'stop',
                'heading': self.current_heading,
                'speed': 0,
                'distance_to_target': distance,
                'bearing_to_target': bearing,
                'status': 'target_reached'
            }
        
        # Check for obstacles
        if obstacle_distance and obstacle_distance < self.obstacle_distance_m:
            return {
                'command': 'stop',
                'heading': self.current_heading,
                'speed': 0,
                'distance_to_target': distance,
                'bearing_to_target': bearing,
                'status': 'obstacle_detected',
                'obstacle_distance': obstacle_distance
            }
        
        # Calculate heading error
        heading_error = self._normalize_angle(bearing - self.current_heading)
        
        # Determine command
        if abs(heading_error) > self.heading_tolerance_deg:
            # Need to turn
            if heading_error > 0:
                command = 'turn_right'
            else:
                command = 'turn_left'
            speed = self.max_speed_mps * 0.5  # Slower when turning
        else:
            # Move forward
            command = 'forward'
            # Speed based on distance (slow down when close)
            if distance < 10:
                speed = self.max_speed_mps * 0.5
            else:
                speed = self.max_speed_mps
        
        return {
            'command': command,
            'heading': bearing,
            'speed': speed,
            'distance_to_target': distance,
            'bearing_to_target': bearing,
            'heading_error': heading_error,
            'status': 'navigating'
        }
    
    def update_heading(self, heading: float):
        """
        Update current heading
        
        Args:
            heading: Current heading in degrees (0-360)
        """
        self.current_heading = heading % 360
    
    def _normalize_angle(self, angle: float) -> float:
        """
        Normalize angle to -180 to 180 range
        
        Args:
            angle: Angle in degrees
            
        Returns:
            Normalized angle
        """
        while angle > 180:
            angle -= 360
        while angle < -180:
            angle += 360
        return angle
    
    def get_trail_coordinates(self) -> List[Tuple[float, float]]:
        """
        Get GPS trail as list of (lat, lon) tuples
        
        Returns:
            List of coordinate tuples
        """
        return [(coord.latitude, coord.longitude) for coord in self.gps_trail]
    
    def calculate_total_distance_traveled(self) -> float:
        """
        Calculate total distance traveled based on GPS trail
        
        Returns:
            Total distance in meters
        """
        if len(self.gps_trail) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(self.gps_trail)):
            total_distance += self.calculate_distance(
                self.gps_trail[i-1],
                self.gps_trail[i]
            )
        
        return total_distance
    
    def get_status(self) -> Dict:
        """
        Get current navigation status
        
        Returns:
            Status dictionary
        """
        status = {
            'autopilot_active': self.autopilot_active,
            'current_position': None,
            'target_position': None,
            'current_heading': self.current_heading,
            'current_speed': self.current_speed,
            'distance_traveled': self.calculate_total_distance_traveled(),
            'trail_points': len(self.gps_trail)
        }
        
        if self.current_position:
            status['current_position'] = {
                'lat': self.current_position.latitude,
                'lon': self.current_position.longitude
            }
        
        if self.target_position:
            status['target_position'] = {
                'lat': self.target_position.latitude,
                'lon': self.target_position.longitude
            }
            
            if self.current_position:
                status['distance_to_target'] = self.calculate_distance(
                    self.current_position,
                    self.target_position
                )
                status['bearing_to_target'] = self.calculate_bearing(
                    self.current_position,
                    self.target_position
                )
        
        return status
