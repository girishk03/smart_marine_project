#!/usr/bin/env python3
"""
Smart Marine Project - Plastic Waste Detection System
====================================================

A comprehensive plastic waste detection system for marine environments.
Detects and classifies plastic waste in images with high accuracy.

Author: Smart Marine Project Team
Version: 1.0.1
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from datetime import datetime

# Try to import from local YOLOv5 repo first, fallback to manual implementations
LOCAL_YOLO = False
attempt_load = None
non_max_suppression = None
letterbox = None
select_device = None
scale_coords = None
Annotator = None
colors = None

try:
    # Add YOLOv5 repo root to path so we can import its local modules
    yolov5_path = os.path.join(os.path.dirname(__file__), '../../yolov5')
    if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
        sys.path.insert(0, yolov5_path)

    try:
        from models.experimental import attempt_load as yolo_attempt_load
        from utils.general import check_img_size, non_max_suppression as yolo_nms
        from utils.augmentations import letterbox as yolo_letterbox
        from utils.torch_utils import select_device as yolo_select_device
        from utils.plots import Annotator as yolo_annotator, colors as yolo_colors
        
        attempt_load = yolo_attempt_load
        non_max_suppression = yolo_nms
        letterbox = yolo_letterbox
        select_device = yolo_select_device
        Annotator = yolo_annotator
        colors = yolo_colors
        
        # Try to import scale_coords (might not exist in newer versions)
        try:
            from utils.general import scale_coords as yolo_scale_coords
            scale_coords = yolo_scale_coords
        except ImportError:
            scale_coords = None  # Will use fallback
        
        print("✅ Successfully imported from local YOLOv5")
        LOCAL_YOLO = True
    except ImportError as e:
        print(f"⚠️ Local YOLOv5 import failed: {e}")
        print("🔄 Using fallback implementations...")
        LOCAL_YOLO = False

except Exception as e:
    print(f"❌ Error setting up YOLOv5 path: {e}")
    LOCAL_YOLO = False

# Fallback implementations if YOLOv5 imports failed
if not LOCAL_YOLO or select_device is None:
    def select_device(device=''):
        """Fallback device selection"""
        if device == 'auto' or device == '':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)

if not LOCAL_YOLO or letterbox is None:
    def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        """Fallback letterbox implementation"""
        shape = img.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, r, (dw, dh)

if not LOCAL_YOLO or non_max_suppression is None:
    def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
        """Fallback NMS implementation"""
        import torchvision
        output = []
        for xi, x in enumerate(prediction):
            x = x[x[:, 4] > conf_thres]
            if not x.shape[0]:
                output.append(torch.zeros((0, 6)))
                continue
            x[:, 5:] *= x[:, 4:5]
            box = x[:, :4]
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            n = x.shape[0]
            if not n:
                output.append(torch.zeros((0, 6)))
                continue
            boxes, scores = x[:, :4], x[:, 4]
            i = torchvision.ops.nms(boxes, scores, iou_thres)
            if i.shape[0] > max_det:
                i = i[:max_det]
            output.append(x[i])
        return output

if not LOCAL_YOLO or Annotator is None:
    class Annotator:
        """Fallback Annotator class"""
        def __init__(self, im, line_width=None, font_size=None):
            self.im = im
            self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)
        
        def box_label(self, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255)):
            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
            if label:
                tf = max(self.lw - 1, 1)
                w, h = cv2.getTextSize(label, 0, fontScale=self.lw / 3, thickness=tf)[0]
                outside = p1[1] - h >= 3
                p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
                cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)
                cv2.putText(self.im, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                           0, self.lw / 3, txt_color, thickness=tf, lineType=cv2.LINE_AA)
        
        def result(self):
            return self.im

if not LOCAL_YOLO or colors is None:
    def colors(i, bgr=False):
        """Fallback colors function"""
        palette = [(255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
                   (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134)]
        c = palette[int(i) % len(palette)]
        return c if bgr else c[::-1]


class PlasticDetector:
    """
    Smart Marine Plastic Detection System
    
    Detects and classifies plastic waste in marine environments with:
    - High accuracy plastic detection
    - Simplified 2-class system (plastic, plastic bottle)
    - Configurable confidence thresholds
    - Batch processing capabilities
    """
    
    def __init__(self, model_path: str, device: str = 'auto', conf_threshold: float = 0.3,
                 iou_threshold: float = 0.45, img_size: int = 640, tta: bool = False):
        """
        Initialize the plastic detector
        
        Args:
            model_path: Path to the trained YOLOv5 model weights
            device: Device to run inference on ('auto', 'cpu', 'cuda')
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = model_path
        self.device = select_device(device)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.tta = tta
        
        # Load model
        self.model = self._load_model()
        
        # Class names for simplified detection
        self.class_names = ['plastic', 'plastic bottle']
        
        # Detection statistics
        self.stats = {
            'total_images': 0,
            'images_with_plastic': 0,
            'total_plastics': 0,
            'total_bottles': 0,
            'processing_time': 0
        }
    
    def _load_model(self):
        """Load the YOLOv5 model"""
        try:
            # Ensure YOLOv5 path is available for model loading
            yolov5_path = os.path.join(os.path.dirname(__file__), 'yolov5')
            if os.path.exists(yolov5_path) and yolov5_path not in sys.path:
                sys.path.insert(0, yolov5_path)
            
            if LOCAL_YOLO and attempt_load is not None:
                # Use local YOLOv5 repo
                print(f"🔧 Loading model with local YOLOv5: {self.model_path}")
                model = attempt_load(self.model_path)
            else:
                # Fallback: Use torch.load directly (deployed mode)
                print(f"🔧 Loading model with torch.load: {self.model_path}")
                
                # Temporarily add yolov5 to path for model loading
                original_path = sys.path.copy()
                try:
                    # Set weights_only=False for YOLOv5 models (trusted source)
                    model = torch.load(self.model_path, map_location=self.device, weights_only=False)
                    if isinstance(model, dict) and 'model' in model:
                        model = model['model']
                    model = model.float()
                finally:
                    # Restore original path
                    sys.path = original_path

            model.to(self.device)
            model.eval()
            print(f"✅ Model loaded successfully from {self.model_path}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def detect_objects(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Detect plastic objects in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (detections, detection_info)
        """
        # Prepare image using YOLOv5's letterbox to preserve aspect ratio
        img_size = int(self.img_size)
        # Handle different model formats
        if hasattr(self.model, 'stride'):
            stride = int(self.model.stride.max())
        else:
            stride = 32  # Default stride for YOLOv5 models
        img0 = image  # original image
        lb_img = letterbox(img0, img_size, stride=stride, auto=True)[0]  # HWC BGR
        # Convert HWC BGR to CHW RGB
        img_chw = lb_img[:, :, ::-1].transpose(2, 0, 1)
        img_chw = np.ascontiguousarray(img_chw)
        img_tensor = torch.from_numpy(img_chw).to(self.device).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            pred = self.model(img_tensor, augment=self.tta)[0]
        
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_threshold, self.iou_threshold, max_det=1000)
        
        # Process detections
        detections = pred[0] if len(pred) > 0 else None

        # Rescale boxes from model img size to original image size
        if detections is not None and len(detections):
            # Try to use YOLOv5's scale_coords if available; otherwise fall back to ratio scaling
            if LOCAL_YOLO and scale_coords is not None:
                try:
                    detections[:, :4] = scale_coords((lb_img.shape[0], lb_img.shape[1]), detections[:, :4], img0.shape).round()
                except Exception as e:
                    print(f"⚠️ Warning: scale_coords failed, using fallback: {e}")
                    # Fallback: simple width/height ratio scaling
                    gain_w = img0.shape[1] / lb_img.shape[1]
                    gain_h = img0.shape[0] / lb_img.shape[0]
                    detections[:, 0] = detections[:, 0] * gain_w
                    detections[:, 2] = detections[:, 2] * gain_w
                    detections[:, 1] = detections[:, 1] * gain_h
                    detections[:, 3] = detections[:, 3] * gain_h
            else:
                # Fallback: simple width/height ratio scaling
                gain_w = img0.shape[1] / lb_img.shape[1]
                gain_h = img0.shape[0] / lb_img.shape[0]
                detections[:, 0] = detections[:, 0] * gain_w
                detections[:, 2] = detections[:, 2] * gain_w
                detections[:, 1] = detections[:, 1] * gain_h
                detections[:, 3] = detections[:, 3] * gain_h

        detection_info = self._process_detections(detections, image.shape)
        
        return detections, detection_info
    
    def _process_detections(self, detections, img_shape) -> List[Dict]:
        """Process raw detections into simplified format"""
        if detections is None:
            return []
        
        detection_info = []
        
        # Original class names from training data (data.yaml)
        # NOTE: Model outputs classes beyond 0-8, suggesting it was trained on different/more data
        original_class_names = ['plastic', 'Bottle cap', 'Can', 'Juice Box', 
                               'Juice box', 'Metal', 'Metal Waste', 'Undefined trash', 'Wood',
                               'Unknown(9)', 'Unknown(10)', 'plastic bottle', 'plastic bag', 
                               'plastic container', 'plastic wrapper', 'Unknown(15)', 'Unknown(16)']
        
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)
            
            # Get original class name for debugging
            original_class = original_class_names[cls] if cls < len(original_class_names) else f"Unknown({cls}))"
            
            # Map classes based on ACTUAL model output (not just data.yaml)
            # ONLY keep plastic and plastic bottle classes, skip everything else
            
            # PLASTIC BOTTLE classes (high confidence detections)
            if cls in [11, 12, 13]:  # plastic bottle, plastic bag, plastic container
                simplified_cls = 1  # "plastic bottle"
                detection_info.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': simplified_cls,
                    'class_name': self.class_names[simplified_cls],
                    'original_class': original_class
                })
            # GENERAL PLASTIC classes
            elif cls in [0, 1, 7, 14]:  # plastic, bottle cap, undefined trash, wrapper
                simplified_cls = 0  # "plastic"
                detection_info.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': simplified_cls,
                    'class_name': self.class_names[simplified_cls],
                    'original_class': original_class
                })
            # CONTAINER classes (juice boxes)
            elif cls in [3, 4]:  # Juice boxes/containers
                simplified_cls = 1  # Map to "plastic bottle"
                detection_info.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': simplified_cls,
                    'class_name': self.class_names[simplified_cls],
                    'original_class': original_class
                })
            # SKIP ALL OTHER CLASSES:
            # - Class 2: Can (metal)
            # - Class 5, 6: Metal/Metal Waste
            # - Class 8: Wood
            # - Class 9, 10, 15, 16: Unknown non-plastic items
            # These will be completely ignored and not shown in results
        
        return detection_info
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict], 
                       line_thickness: int = 1, font_size: int = 12) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            line_thickness: Thickness of bounding box lines
            font_size: Font size for labels
            
        Returns:
            Image with drawn detections
        """
        annotator = Annotator(image, line_width=line_thickness, font_size=font_size)
        
        for det in detections:
            bbox = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            class_id = det['class_id']
            annotator.box_label(bbox, label, color=colors(class_id, True))
        
        return annotator.result()
    
    def process_image(self, image_path: str, output_path: str = None, 
                     line_thickness: int = 1) -> Dict:
        """
        Process a single image
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image (optional)
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Dictionary with detection results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Detect objects
        start_time = datetime.now()
        detections, detection_info = self.detect_objects(image)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Draw detections
        result_image = self.draw_detections(image, detection_info, line_thickness)
        
        # Save result if output path provided
        if output_path:
            cv2.imwrite(output_path, result_image)
        
        # Update statistics
        self.stats['total_images'] += 1
        if detection_info:
            self.stats['images_with_plastic'] += 1
            for det in detection_info:
                self.stats['total_plastics'] += 1
                if det['class_name'] == 'plastic bottle':
                    self.stats['total_bottles'] += 1
        
        self.stats['processing_time'] += processing_time
        
        return {
            'image_path': image_path,
            'detections': detection_info,
            'num_detections': len(detection_info),
            'processing_time': processing_time,
            'output_path': output_path
        }
    
    def process_batch(self, input_dir: str, output_dir: str, 
                     line_thickness: int = 1) -> Dict:
        """
        Process a batch of images
        
        Args:
            input_dir: Directory containing input images
            output_dir: Directory to save output images
            line_thickness: Thickness of bounding box lines
            
        Returns:
            Dictionary with batch processing results
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        print(f"🔍 Found {len(image_files)} images to process")
        
        results = []
        for i, img_file in enumerate(image_files):
            print(f"Processing {i+1}/{len(image_files)}: {img_file.name}")
            
            try:
                output_file = output_path / img_file.name
                result = self.process_image(str(img_file), str(output_file), line_thickness)
                results.append(result)
            except Exception as e:
                print(f"❌ Error processing {img_file.name}: {e}")
                continue
        
        # Generate summary
        summary = self._generate_summary(results)
        
        return {
            'results': results,
            'summary': summary,
            'output_directory': str(output_path)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """Generate processing summary"""
        total_images = len(results)
        images_with_detections = sum(1 for r in results if r['num_detections'] > 0)
        total_detections = sum(r['num_detections'] for r in results)
        total_plastics = sum(1 for r in results for det in r['detections'] 
                           if det['class_name'] == 'plastic')
        total_bottles = sum(1 for r in results for det in r['detections'] 
                          if det['class_name'] == 'plastic bottle')
        avg_processing_time = np.mean([r['processing_time'] for r in results])
        
        return {
            'total_images_processed': total_images,
            'images_with_detections': images_with_detections,
            'detection_rate': f"{(images_with_detections/total_images)*100:.1f}%" if total_images > 0 else "0%",
            'total_detections': total_detections,
            'total_plastics': total_plastics,
            'total_bottles': total_bottles,
            'average_processing_time': f"{avg_processing_time:.3f}s",
            'total_processing_time': f"{sum(r['processing_time'] for r in results):.3f}s"
        }
    
    def save_results(self, results: Dict, output_file: str):
        """Save detection results to JSON file"""
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"📊 Results saved to {output_file}")


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Smart Marine Plastic Detection System')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights')
    parser.add_argument('--input', type=str, required=True, help='Input image or directory')
    parser.add_argument('--output', type=str, required=True, help='Output image or directory')
    parser.add_argument('--conf', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--line-thickness', type=int, default=1, help='Bounding box line thickness')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (auto/cpu/cuda)')
    parser.add_argument('--save-results', type=str, help='Save detection results to JSON file')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = PlasticDetector(args.model, args.device, args.conf)
    
    # Check if input is file or directory
    input_path = Path(args.input)
    
    if input_path.is_file():
        # Single image processing
        print(f"🔍 Processing single image: {input_path.name}")
        result = detector.process_image(str(input_path), args.output, args.line_thickness)
        print(f"✅ Found {result['num_detections']} plastic objects")
        
        if args.save_results:
            detector.save_results(result, args.save_results)
    
    elif input_path.is_dir():
        # Batch processing
        print(f"🔍 Processing directory: {input_path}")
        results = detector.process_batch(str(input_path), args.output, args.line_thickness)
        
        # Print summary
        summary = results['summary']
        print(f"\n📊 Processing Summary:")
        print(f"   Images processed: {summary['total_images_processed']}")
        print(f"   Detection rate: {summary['detection_rate']}")
        print(f"   Total plastics found: {summary['total_plastics']}")
        print(f"   Total bottles found: {summary['total_bottles']}")
        print(f"   Average time per image: {summary['average_processing_time']}")
        
        if args.save_results:
            detector.save_results(results, args.save_results)
    
    else:
        print(f"❌ Input path does not exist: {input_path}")
        return 1
    
    print(f"✅ Processing complete! Results saved to: {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
