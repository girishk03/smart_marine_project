#!/usr/bin/env python3
"""
Smart Marine Project - Simple Detection Script
==============================================

Simple detection script that works with your existing setup.
"""

import os
import sys
import argparse
from pathlib import Path

# Add yolov5 to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../yolov5'))

def main():
    """Simple detection using existing YOLOv5 setup"""
    parser = argparse.ArgumentParser(description='Smart Marine - Simple Detection')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--thickness', type=int, default=1, help='Line thickness')
    
    args = parser.parse_args()
    
    # Use existing YOLOv5 detect.py
    yolov5_path = os.path.join(os.path.dirname(__file__), '../../yolov5')
    detect_script = os.path.join(yolov5_path, 'detect.py')
    model_path = os.path.join(os.path.dirname(__file__), '../models/ocean_waste_model_m2/weights/best.pt')
    
    # Build command
    cmd = f"cd {yolov5_path} && python3 detect.py --weights {model_path} --source {args.input} --conf {args.conf} --line-thickness {args.thickness} --name smart_marine_detection"
    
    print(f"🔍 Running detection...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Confidence: {args.conf}")
    print(f"   Line thickness: {args.thickness}")
    
    # Run detection
    os.system(cmd)
    
    # Move results to desired output location
    results_dir = os.path.join(yolov5_path, 'runs/detect/smart_marine_detection')
    if os.path.exists(results_dir):
        print(f"✅ Detection complete! Results in: {results_dir}")
        print(f"   You can copy results to: {args.output}")
    else:
        print("❌ Detection failed or no results found")

if __name__ == '__main__':
    main()
