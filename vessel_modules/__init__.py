"""
Smart Marine Vessel - Autonomous Navigation & Collection System
================================================================

This module provides autonomous vessel capabilities for plastic detection,
GPS-based navigation, and automated collection with real-time tracking.

Modules:
    - camera_module: YOLOv5 detection with position tracking
    - gps_navigation: GPS tracking and autopilot navigation
    - object_counter: Collection counter and data logging
    - simulator: Digital twin for testing without hardware
    - net_control: Collection mechanism control (hardware)

Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Smart Marine Project Team"

# Import main components
try:
    from .camera_module import VesselCamera
    from .gps_navigation import GPSNavigator
    from .object_counter import CollectionCounter
    from .simulator import VesselSimulator
    VESSEL_MODULES_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Vessel modules not fully available: {e}")
    VESSEL_MODULES_AVAILABLE = False

__all__ = [
    'VesselCamera',
    'GPSNavigator', 
    'CollectionCounter',
    'VesselSimulator',
    'VESSEL_MODULES_AVAILABLE'
]
