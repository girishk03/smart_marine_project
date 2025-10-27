#!/usr/bin/env python3
"""
Test Optimized Plastic Detection Performance
===========================================

Test script to verify improved detection efficiency and accuracy.
"""

import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from plastic_detector import PlasticDetector

def test_performance_comparison():
    """Test detection performance with different confidence levels"""
    print("🚀 Testing Optimized Detection Performance")
    print("=" * 60)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Find test images
    sample_files = []
    for directory in [".", "../"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')) and 'debug' in file.lower():
                    sample_files.append(os.path.join(directory, file))
    
    if not sample_files:
        print("⚠️ No test images found")
        return
    
    # Test with different confidence thresholds
    confidence_levels = [0.1, 0.15, 0.2, 0.25, 0.3]
    
    print(f"📸 Testing on {len(sample_files[:2])} images with {len(confidence_levels)} confidence levels")
    print()
    
    total_detections = {}
    processing_times = {}
    
    for conf in confidence_levels:
        print(f"🎯 Testing with confidence threshold: {conf}")
        print("-" * 40)
        
        detector = PlasticDetector(
            model_path=model_path,
            device='cpu',
            conf_threshold=conf,
            debug_mode=False  # Disable debug for cleaner output
        )
        
        total_detections[conf] = 0
        processing_times[conf] = []
        
        for i, img_path in enumerate(sample_files[:2]):
            try:
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Time the detection
                start_time = time.time()
                detections, detection_info = detector.detect_objects(image)
                processing_time = time.time() - start_time
                
                processing_times[conf].append(processing_time)
                total_detections[conf] += len(detection_info)
                
                print(f"  Image {i+1}: {len(detection_info)} detections ({processing_time:.3f}s)")
                
                # Show detection details
                for j, det in enumerate(detection_info):
                    print(f"    - {det['class_name']}: {det['confidence']:.3f}")
                    
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
        
        avg_time = np.mean(processing_times[conf]) if processing_times[conf] else 0
        print(f"  📊 Total detections: {total_detections[conf]}")
        print(f"  ⏱️ Average processing time: {avg_time:.3f}s")
        print()
    
    # Summary
    print("📈 PERFORMANCE SUMMARY")
    print("=" * 60)
    print(f"{'Confidence':<12} {'Detections':<12} {'Avg Time (s)':<15} {'Efficiency'}")
    print("-" * 60)
    
    for conf in confidence_levels:
        avg_time = np.mean(processing_times[conf]) if processing_times[conf] else 0
        efficiency = total_detections[conf] / max(avg_time, 0.001)  # detections per second
        print(f"{conf:<12} {total_detections[conf]:<12} {avg_time:<15.3f} {efficiency:.1f}")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    
    best_conf = max(confidence_levels, key=lambda x: total_detections[x])
    fastest_conf = min(confidence_levels, key=lambda x: np.mean(processing_times[x]) if processing_times[x] else float('inf'))
    
    print(f"🎯 Best detection rate: {best_conf} ({total_detections[best_conf]} detections)")
    print(f"⚡ Fastest processing: {fastest_conf} ({np.mean(processing_times[fastest_conf]):.3f}s avg)")
    print(f"🏆 Recommended confidence: 0.15-0.2 for balanced performance")

def test_detection_quality():
    """Test detection quality with detailed analysis"""
    print("\n🔍 DETECTION QUALITY ANALYSIS")
    print("=" * 60)
    
    model_path = "models/ocean_waste_model_m2/weights/best.pt"
    detector = PlasticDetector(
        model_path=model_path,
        device='cpu',
        conf_threshold=0.15,  # Optimal threshold
        debug_mode=True  # Enable debug for detailed analysis
    )
    
    # Find one test image for detailed analysis
    test_image = None
    for directory in [".", "../"]:
        if os.path.exists(directory):
            for file in os.listdir(directory):
                if file.lower().endswith('.jpg') and 'debug' in file.lower():
                    test_image = os.path.join(directory, file)
                    break
        if test_image:
            break
    
    if not test_image:
        print("⚠️ No test image found for quality analysis")
        return
    
    print(f"📸 Analyzing: {test_image}")
    print()
    
    try:
        image = cv2.imread(test_image)
        if image is None:
            print("❌ Could not load test image")
            return
        
        # Run detection with debug
        detections, detection_info = detector.detect_objects(image)
        
        print(f"\n📊 QUALITY METRICS")
        print("-" * 30)
        print(f"Raw detections: {detections.shape[0] if detections is not None else 0}")
        print(f"Final detections: {len(detection_info)}")
        print(f"Filter efficiency: {len(detection_info)}/{detections.shape[0] if detections is not None else 0}")
        
        if detection_info:
            confidences = [det['confidence'] for det in detection_info]
            print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
            print(f"Average confidence: {np.mean(confidences):.3f}")
            
            # Class distribution
            classes = {}
            for det in detection_info:
                cls = det['class_name']
                classes[cls] = classes.get(cls, 0) + 1
            
            print(f"Class distribution:")
            for cls, count in classes.items():
                print(f"  - {cls}: {count}")
        
    except Exception as e:
        print(f"❌ Error in quality analysis: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main test function"""
    print("🌊 Smart Marine Project - Optimized Detection Test")
    print("=" * 70)
    
    # Test 1: Performance comparison
    test_performance_comparison()
    
    # Test 2: Quality analysis
    test_detection_quality()
    
    print("\n🏁 Testing Complete!")
    print("\n🎯 KEY IMPROVEMENTS:")
    print("   ✅ Lowered default confidence threshold to 0.2")
    print("   ✅ Improved confidence boosting for plastic bottles")
    print("   ✅ Less aggressive face filtering")
    print("   ✅ More lenient plastic validation")
    print("   ✅ Better size and aspect ratio filtering")

if __name__ == "__main__":
    main()
