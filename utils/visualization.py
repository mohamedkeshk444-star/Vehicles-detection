import cv2
import numpy as np
from PIL import Image

def draw_boxes(image, detections):
    """
    Draw bounding boxes and labels on the image.
    image: PIL Image or numpy array
    detections: list of dicts with 'box', 'confidence', 'class_name'
    """
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
        
    # Ensure image is in RGB format (if RGBA, convert to RGB)
    if len(img_array.shape) == 3 and img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        conf = det["confidence"]
        class_name = det["class_name"]
        
        # Colors: nice lime green for bounding boxes
        color = (50, 205, 50) # RGB
        
        # Draw rectangle
        thickness = max(2, int(min(img_array.shape[:2]) * 0.005))
        cv2.rectangle(img_array, (x1, y1), (x2, y2), color, thickness)
        
        # Label text
        label = f"{class_name} {conf:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.5, min(img_array.shape[:2]) * 0.001)
        font_thickness = max(1, int(font_scale * 2))
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw background for text
        cv2.rectangle(img_array, (x1, max(0, y1 - text_height - baseline - 5)), (x1 + text_width, max(0, y1)), color, -1)
        
        # Draw text
        cv2.putText(img_array, label, (x1, max(0, y1 - 5)), font, font_scale, (0, 0, 0), font_thickness)
        
    return img_array
