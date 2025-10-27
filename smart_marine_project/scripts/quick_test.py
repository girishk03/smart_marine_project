#!/usr/bin/env python3
"""
Smart Marine Project - Quick Test Script
========================================

Quick test script to verify the installation and functionality.
"""

import os
import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from plastic_detector import PlasticDetector


def test_installation():
    """Test if all dependencies are properly installed"""
    print("🔧 Testing installation...")
    
    try:
        import torch
        import cv2
        import numpy as np
        print("✅ All dependencies imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def test_model_loading():
    """Test if the model can be loaded"""
    print("🤖 Testing model loading...")
    
    model_path = "../models/ocean_waste_model_m2/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    try:
        detector = PlasticDetector(model_path, conf_threshold=0.3)
        print("✅ Model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False


def test_detection():
    """Test detection on a sample image"""
    print("🔍 Testing detection...")
    
    # Check if test images exist
    test_images = [
        "../test/images",
        "../data/test_images",
        "../results/simplified_plastic_detection"
    ]
    
    test_dir = None
    for img_dir in test_images:
        if os.path.exists(img_dir):
            test_dir = img_dir
            break
    
    if not test_dir:
        print("❌ No test images found")
        return False
    
    # Find first image
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    test_image = None
    for ext in image_extensions:
        for file in Path(test_dir).glob(f'*{ext}'):
            test_image = str(file)
            break
        if test_image:
            break
    
    if not test_image:
        print("❌ No test images found in directories")
        return False
    
    try:
        model_path = "../models/ocean_waste_model_m2/weights/best.pt"
        detector = PlasticDetector(model_path, conf_threshold=0.3)
        
        # Create output directory
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Run detection
        start_time = time.time()
        result = detector.process_image(
            test_image, 
            os.path.join(output_dir, "test_result.jpg"),
            line_thickness=1
        )
        processing_time = time.time() - start_time
        
        print(f"✅ Detection completed successfully")
        print(f"   Image: {os.path.basename(test_image)}")
        print(f"   Detections: {result['num_detections']}")
        print(f"   Processing time: {processing_time:.3f}s")
        print(f"   Output: {output_dir}/test_result.jpg")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during detection: {e}")
        return False


def main():
    """Run all tests"""
    print("🌊 Smart Marine Project - Quick Test")
    print("=" * 40)
    
    tests = [
        ("Installation", test_installation),
        ("Model Loading", test_model_loading),
        ("Detection", test_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} Test:")
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Smart Marine Project is ready to use.")
        print("\n🚀 Quick start:")
        print("   python scripts/run_detection.py --input path/to/image.jpg --output result.jpg")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
