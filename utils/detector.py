import os
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path='model/best.pt', labels_path='model/labels.txt'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        # Load labels
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Labels file not found: {labels_path}")
        with open(labels_path, 'r') as f:
            self.labels = [line.strip() for line in f.readlines() if line.strip()]

    def predict(self, image, conf_threshold=0.25):
        """
        Run inference on the given image.
        Returns a list of structured detections.
        """
        # Run YOLO inference
        results = self.model.predict(image, conf=conf_threshold, device=self.device, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                class_name = self.labels[class_id] if class_id < len(self.labels) else f"Class {class_id}"
                
                detections.append({
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name
                })
                
        return detections
