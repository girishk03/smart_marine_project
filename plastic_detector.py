#!/usr/bin/env python3
"""
Smart Marine - Ultralytics-based detector for cloud deployment
"""
import os
import sys
import cv2
import numpy as np
from datetime import datetime

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

class PlasticDetector:
    def __init__(self, model_path=None, device="auto", conf_threshold=0.25,
                 iou_threshold=0.45, img_size=640):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.img_size = img_size
        self.model = None
        if model_path is not None and not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found: {model_path}")

        self.model_path = model_path or "yolov5m.pt"
        
        if ULTRALYTICS_AVAILABLE:
            try:
                self.model = YOLO(self.model_path)
                print(f"✅ Model loaded with ultralytics: {self.model_path}")
            except Exception as e:
                raise RuntimeError(f"Model load failed: {self.model_path}") from e
        else:
            print("❌ ultralytics not available")

    def detect(self, image):
        if self.model is None:
            return self._empty_result()
        
        start = datetime.now()
        try:
            if isinstance(image, np.ndarray):
                img = image
            else:
                img = np.array(image)
            
            results = self.model(img, conf=self.conf_threshold, 
                               iou=self.iou_threshold, verbose=False)
            
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    name = r.names.get(cls, "plastic")
                    
                    if any(p in name.lower() for p in 
                           ["bottle", "plastic", "cup", "container"]):
                        detections.append({
                            "class": "plastic",
                            "confidence": conf,
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            elapsed = (datetime.now() - start).total_seconds() * 1000
            return {
                "detections": detections,
                "count": len(detections),
                "processing_time": elapsed,
                "image_size": img.shape[:2]
            }
        except Exception as e:
            print(f"Detection error: {e}")
            return self._empty_result()

    def detect_and_annotate(self, image):
        result = self.detect(image)
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
        
        for det in result["detections"]:
            x1, y1, x2, y2 = det["bbox"]
            conf = det["confidence"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"plastic: {conf:.2f}"
            cv2.putText(img, label, (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        result["annotated_image"] = img
        return result

    def _empty_result(self):
        return {"detections": [], "count": 0, 
                "processing_time": 0, "image_size": (0, 0)}
