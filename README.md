# YOLO Object Detection - Streamlit Deployment

This is a production-ready YOLO object detection project scaffold optimized for Streamlit Cloud deployment.

## 📂 Project Structure

```text
yolo-terminal-deployment/
│
├── app.py                # Local testing CLI script
├── streamlit_app.py      # Streamlit web application
├── requirements.txt      # Python dependencies
├── packages.txt          # System dependencies (for Streamlit Cloud)
├── README.md             # Project documentation
│
├── model/                # Contains trained weights and classes
│   ├── best.pt           # YOLO weights file
│   └── labels.txt        # Class labels (one per line)
│
├── utils/                # Helper modules
│   ├── detector.py       # YOLO inference logic
│   └── visualization.py  # Bounding box drawing functions
│
└── assets/               # Static files
    └── demo.png          # Demo image
```

## 🚀 How to Run Locally

### 1. Install Dependencies
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run Local CLI Script
You can test the inference locally without a UI:
```bash
python app.py --image assets/demo.png --output output.jpg
```

### 3. Run Streamlit App
To launch the interactive web interface:
```bash
streamlit run streamlit_app.py
```

## ☁️ Streamlit Cloud Deployment Steps

1. Push this entire project directory to a GitHub repository.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click **New app**.
4. Select your GitHub repository, branch, and set the main file path to `streamlit_app.py`.
5. Click **Deploy**.

*Streamlit will automatically read `packages.txt` to install system-level Linux dependencies like `libgl1` required by OpenCV, and `requirements.txt` for Python packages.*
