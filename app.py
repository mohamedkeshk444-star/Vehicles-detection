import os
import argparse
from PIL import Image
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

def main():
    parser = argparse.ArgumentParser(description="YOLO Local Testing Script")
    parser.add_argument("--image", type=str, default="assets/demo.png", help="Path to input image")
    parser.add_argument("--output", type=str, default="output.jpg", help="Path to output image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Input image not found at {args.image}")
        return

    print("Loading model...")
    try:
        model = YOLOModel()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print(f"Loading image from {args.image}...")
    try:
        image = Image.open(args.image).convert("RGB")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    print(f"Running inference (threshold: {args.conf})...")
    detections = model.predict(image, conf_threshold=args.conf)

    print(f"Found {len(detections)} objects:")
    for det in detections:
        print(f" - {det['class_name']} ({det['confidence']:.2f}) at {det['box']}")

    print("Drawing bounding boxes...")
    annotated_img_array = draw_boxes(image, detections)
    annotated_img = Image.fromarray(annotated_img_array)

    annotated_img.save(args.output)
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
