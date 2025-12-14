"""
High-accuracy object detector using YOLOv8 ONNX.
YOLOv8 is the latest YOLO model with state-of-the-art accuracy.
"""

import cv2
import numpy as np
from typing import List, Dict
import os
import onnxruntime as ort


class ObjectDetector:
    """Detects real-world objects using YOLOv8 ONNX."""
    
    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    ]
    
    def __init__(self):
        print("Loading YOLOv8 ONNX...")
        model_file = "models/yolov8n.onnx"
        
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"YOLOv8 not found: {model_file}")
        
        print(f"Model size: {os.path.getsize(model_file) / (1024*1024):.1f} MB")
        
        self.session = ort.InferenceSession(model_file, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        input_shape = self.session.get_inputs()[0].shape
        # Handle dynamic or invalid dimensions
        try:
            self.input_h = int(input_shape[2]) if len(input_shape) > 2 and input_shape[2] not in [None, 'batch', 'height'] else 640
            self.input_w = int(input_shape[3]) if len(input_shape) > 3 and input_shape[3] not in [None, 'batch', 'width'] else 640
        except (ValueError, TypeError):
            self.input_h = 640
            self.input_w = 640
        
        print(f"YOLOv8 loaded! Input: {self.input_w}x{self.input_h}")
    
    def detect_objects(self, frame: np.ndarray, config: Dict) -> List[Dict]:
        min_conf = max(0.30, config.get("min_similarity", 30) / 100.0)  # Lowered to 30%
        
        h, w = frame.shape[:2]
        
        # Preprocess
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
        
        # Inference
        outputs = self.session.run(None, {self.input_name: img})[0]
        
        # Post-process (YOLOv8 format: [1, 84, 8400])
        preds = outputs[0].T  # [8400, 84]
        boxes_xywh = preds[:, :4]
        scores = preds[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confs = scores[np.arange(len(class_ids)), class_ids]
        
        mask = confs > min_conf
        boxes_xywh, confs, class_ids = boxes_xywh[mask], confs[mask], class_ids[mask]
        
        # Convert to xyxy and scale
        boxes = []
        for box in boxes_xywh:
            cx, cy, bw, bh = box
            x = max(0, int((cx - bw/2) * w / self.input_w))
            y = max(0, int((cy - bh/2) * h / self.input_h))
            bw = int(bw * w / self.input_w)
            bh = int(bh * h / self.input_h)
            boxes.append([x, y, bw, bh])
        
        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confs.tolist(), min_conf, 0.4) if boxes else []
        
        print(f"YOLOv8: {len(boxes)} detected, {len(indices) if len(indices) > 0 else 0} after NMS")
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, bw, bh = boxes[i]
                conf = float(confs[i])
                cid = int(class_ids[i])
                name = self.COCO_CLASSES[cid] if cid < len(self.COCO_CLASSES) else "unknown"
                
                # NO FILTER - detect ALL objects including person
                
                detections.append({
                    "bbox": [x, y, bw, bh],
                    "centroid": [x + bw//2, y + bh//2],
                    "color": "unknown",
                    "shape": name,
                    "size": "medium",
                    "label": name.title(),
                    "similarity": round(conf * 100, 1),
                    "area": bw * bh,
                    "class_name": name,
                    "confidence": conf
                })
                print(f"  -> {name.title()} ({conf*100:.1f}%)")
        
        return sorted(detections, key=lambda d: d['confidence'], reverse=True)
