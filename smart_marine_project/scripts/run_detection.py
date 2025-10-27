#!/usr/bin/env python3
"""
Smart Marine Project - Easy Detection Runner
============================================

Simple script to run plastic detection with different configurations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

from plastic_detector import PlasticDetector


def main():
    """Main function for easy detection running"""
    parser = argparse.ArgumentParser(description='Smart Marine - Easy Detection Runner')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True, 
                       help='Input image or directory path')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output image or directory path')
    
    # Optional arguments
    parser.add_argument('--model', type=str, 
                       default='../models/ocean_waste_model_m2/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--conf', type=float, default=0.3, 
                       help='Confidence threshold (0.0-1.0)')
    parser.add_argument('--thickness', type=int, default=1, 
                       help='Bounding box line thickness')
    parser.add_argument('--mode', type=str, choices=['fast', 'balanced', 'accurate'], 
                       default='balanced', help='Detection mode')
    parser.add_argument('--save-results', action='store_true', 
                       help='Save detection results to JSON')
    parser.add_argument('--device', type=str, default='cpu',
                       help="Device to use: 'cpu' or CUDA device like '0' or '0,1'")
    
    args = parser.parse_args()
    
    # Set mode-specific parameters
    if args.mode == 'fast':
        args.conf = 0.3
        args.thickness = 1
        print("🚀 Fast mode: Lower confidence, thinner lines")
    elif args.mode == 'accurate':
        args.conf = 0.5
        args.thickness = 2
        print("🎯 Accurate mode: Higher confidence, thicker lines")
    else:  # balanced
        args.conf = 0.3
        args.thickness = 1
        print("⚖️ Balanced mode: Good balance of speed and accuracy")
    
    # Resolve model path relative to this script directory if not absolute
    model_path = args.model
    if not os.path.isabs(model_path):
        script_dir = os.path.dirname(__file__)
        model_path = os.path.abspath(os.path.join(script_dir, model_path))

    # Initialize detector
    print(f"🔧 Initializing detector with model: {model_path}")
    detector = PlasticDetector(model_path, device=args.device, conf_threshold=args.conf)
    
    # Check if input exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input path does not exist: {args.input}")
        return 1
    
    # Run detection
    if input_path.is_file():
        print(f"🔍 Processing single image: {input_path.name}")
        result = detector.process_image(str(input_path), args.output, args.thickness)
        print(f"✅ Found {result['num_detections']} plastic objects")
        
        if args.save_results:
            results_file = Path(args.output).parent / f"{input_path.stem}_results.json"
            detector.save_results(result, str(results_file))
            print(f"📊 Results saved to: {results_file}")
    
    elif input_path.is_dir():
        print(f"🔍 Processing directory: {input_path}")
        results = detector.process_batch(str(input_path), args.output, args.thickness)
        
        # Print summary
        summary = results['summary']
        print(f"\n📊 Processing Summary:")
        print(f"   Images processed: {summary['total_images_processed']}")
        print(f"   Detection rate: {summary['detection_rate']}")
        print(f"   Total plastics found: {summary['total_plastics']}")
        print(f"   Total bottles found: {summary['total_bottles']}")
        print(f"   Average time per image: {summary['average_processing_time']}")
        
        if args.save_results:
            results_file = Path(args.output) / "batch_results.json"
            detector.save_results(results, str(results_file))
            print(f"📊 Results saved to: {results_file}")
    
    print(f"✅ Detection complete! Results saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
