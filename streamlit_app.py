import os
import streamlit as st
from PIL import Image

# Import custom modules
from utils.detector import YOLOModel
from utils.visualization import draw_boxes

# --- Page Config ---
st.set_page_config(
    page_title="YOLO Object Detection",
    page_icon="🔍",
    layout="centered"
)

# --- Initialize Model ---
@st.cache_resource
def load_model():
    try:
        return YOLOModel()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- UI ---
st.title("🔍 YOLO Object Detection")
st.markdown("Upload an image or use the demo image to detect objects.")

# Confidence slider
st.sidebar.header("Settings")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

source_img = None

# Sidebar options
st.sidebar.markdown("---")
use_demo = st.sidebar.checkbox("Use Demo Image", value=False)

uploaded_file = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if use_demo:
    demo_path = os.path.join("assets", "demo.png")
    if os.path.exists(demo_path):
        source_img = Image.open(demo_path).convert("RGB")
    else:
        st.sidebar.error("Demo image not found in assets/demo.png")
elif uploaded_file is not None:
    try:
        source_img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Invalid image file: {e}")

if source_img is not None:
    st.subheader("Original Image")
    st.image(source_img, use_container_width=True)

    if st.button("Run Detection", type="primary", use_container_width=True):
        if model is None:
            st.error("Model is not initialized. Please ensure model/best.pt and model/labels.txt exist.")
        else:
            with st.spinner("Running Inference..."):
                try:
                    # Run inference
                    detections = model.predict(source_img, conf_threshold=conf_threshold)
                    
                    # Draw boxes
                    annotated_img_array = draw_boxes(source_img, detections)
                    annotated_img = Image.fromarray(annotated_img_array)
                    
                    st.subheader("Detection Results")
                    st.image(annotated_img, use_container_width=True)
                    
                    with st.expander("Show Detections Details"):
                        if detections:
                            st.json(detections)
                        else:
                            st.info("No objects detected above the confidence threshold.")
                except Exception as e:
                    st.error(f"Error during detection: {e}")
else:
    st.info("Please upload an image or check 'Use Demo Image' from the sidebar to begin.")
