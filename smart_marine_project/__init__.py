"""
Smart Marine Project
===================

A comprehensive plastic waste detection system for marine environments.

This package provides tools for detecting and classifying plastic waste
in marine environments using state-of-the-art YOLOv5 object detection.

Main Components:
- PlasticDetector: Main detection class
- Configuration files for different detection modes
- Utility scripts for easy usage

Example:
    from smart_marine_project.src.plastic_detector import PlasticDetector
    
    detector = PlasticDetector('models/ocean_waste_model_m2/weights/best.pt')
    result = detector.process_image('input.jpg', 'output.jpg')
"""

__version__ = "1.0.0"
__author__ = "Smart Marine Project Team"
__email__ = "contact@smartmarineproject.com"

# Import main classes for easy access
try:
    from .src.plastic_detector import PlasticDetector
    __all__ = ['PlasticDetector']
except ImportError:
    # Handle case where dependencies are not installed
    __all__ = []
