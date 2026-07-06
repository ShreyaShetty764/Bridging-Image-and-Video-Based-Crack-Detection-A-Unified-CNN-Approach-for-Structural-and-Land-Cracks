# Bridging-Image-and-Video-Based-Crack-Detection-A-Unified-CNN-Approach-for-Structural-and-Land-Cracks
A unified deep learning system for detecting and analyzing cracks in both structural surfaces (via video) and land/soil (via image), built using a custom-trained CNN model. Deployed as a real-time Streamlit web application with automated severity classification, depth estimation, and report generation.

---

# 📌 Project Overview

Manual crack inspection is often time-consuming, labor-intensive, and prone to human error. This project automates the inspection process by combining deep learning and computer vision into a unified system capable of analyzing both structural and land cracks.

The application allows users to upload either a structural inspection video or a land crack image. The trained CNN model detects cracks, estimates crack dimensions, classifies severity, and generates annotated outputs along with downloadable reports.

---

# ✨ Features

- 🎥 Structural crack detection from videos
- 🖼️ Land/soil crack detection from images
- 🧠 Custom CNN-based crack classification
- 📏 Crack width estimation
- 📐 Crack depth estimation
- 🚨 Severity classification (Minor, Moderate, Severe)
- 📊 Confidence score prediction
- 📈 Depth vs Frame visualization
- 📄 Automatic PDF report generation
- 📁 CSV export of crack analysis
- 🌐 Interactive Streamlit web application

---

# 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | CNN Model Development |
| OpenCV | Image & Video Processing |
| Streamlit | Web Application |
| NumPy | Numerical Computing |
| Pandas | Data Processing |
| Matplotlib | Data Visualization |
| ReportLab | PDF Report Generation |

---

# 📁 Project Structure

```text
Bridging-Image-and-Video-Based-Crack-Detection-A-Unified-CNN-Approach-for-Structural-and-Land-Cracks/
│
├── appi.py
├── combinemodel.ipynb
├── datasetcombine.ipynb
├── best_crack_detector.keras
├── requirements.txt
├── README.md
│
├── Test/
│   ├── Images/
│   └── Videos/
│
├── Outputs/
│   ├── Annotated_Images/
│   ├── Annotated_Videos/
│   ├── Reports/
│   └── Graphs/
│
└── Screenshots/
    ├── Home.png
    ├── Structural_Crack.png
    ├── Land_Crack.png
    ├── Dashboard.png
    └── Report.png
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/Bridging-Image-and-Video-Based-Crack-Detection-A-Unified-CNN-Approach-for-Structural-and-Land-Cracks.git
```

Navigate to the project directory

```bash
cd Bridging-Image-and-Video-Based-Crack-Detection-A-Unified-CNN-Approach-for-Structural-and-Land-Cracks
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Application

Ensure the trained CNN model

```
best_crack_detector.keras
```

is available in the project directory.

Launch the Streamlit application

```bash
streamlit run appi.py
```

The application will automatically open in your default web browser.

---

# 🖥️ Application Workflow

## Structural Crack Detection

1. Upload a structural inspection video.
2. Extract frames using OpenCV.
3. Preprocess each frame.
4. Predict crack presence using the CNN model.
5. Detect crack boundaries.
6. Estimate crack width.
7. Estimate crack depth.
8. Classify crack severity.
9. Generate annotated output video.
10. Export PDF and CSV reports.

---

## Land Crack Detection

1. Upload a land/soil crack image.
2. Preprocess the image.
3. Predict crack presence using the CNN model.
4. Detect crack contours.
5. Estimate crack width.
6. Estimate crack depth.
7. Classify crack severity.
8. Generate annotated output image.
9. Export analysis reports.

---

# 🔄 Detection Pipeline

```text
Input Image / Video
        │
        ▼
Image Preprocessing
        │
        ▼
Resize & Normalize
        │
        ▼
CNN Prediction
        │
        ▼
Crack Detection
        │
        ▼
Edge Detection (Canny)
        │
        ▼
Contour Extraction
        │
        ▼
Width Estimation
        │
        ▼
Depth Estimation
        │
        ▼
Severity Classification
        │
        ▼
Annotated Output
        │
        ▼
PDF & CSV Report Generation
```

---

# 📊 Model Performance

| Metric | Value |
|---------|-------|
| Model | Custom CNN |
| Classification Accuracy | **94%** |
| Framework | TensorFlow & Keras |
| Input Types | Images & Videos |
| Deployment | Streamlit |

---

# 📊 Outputs

The application provides:

- ✅ Annotated crack images
- ✅ Annotated crack videos
- ✅ Crack width estimation
- ✅ Crack depth estimation
- ✅ Severity classification
- ✅ Confidence score
- ✅ Depth vs Frame graph
- ✅ CSV report
- ✅ PDF report

---

# 📷 Application Screenshots

Add your screenshots inside the `Screenshots` folder.

Example:

```
Screenshots/
│
├── Home.png
├── Structural_Crack.png
├── Land_Crack.png
├── Dashboard.png
└── Report.png
```

---

# 🚀 Future Enhancements

- Real-time CCTV-based crack monitoring
- Drone-assisted infrastructure inspection
- Crack segmentation using U-Net
- Mobile application deployment
- Cloud deployment using Streamlit Cloud
- Predictive maintenance recommendation system

---

# 💡 Key Highlights

- Unified CNN model for both structural and land crack detection
- Supports both images and videos
- Automated crack width and depth estimation
- Severity classification
- End-to-end Streamlit deployment
- Downloadable PDF and CSV reports
- Research work accepted at IEEE CCIC 2026

---

# 📦 Requirements

```
streamlit
tensorflow
keras
opencv-python
numpy
pandas
matplotlib
Pillow
reportlab
scikit-learn
```

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

# 🧠 Skills Demonstrated

- Deep Learning
- Convolutional Neural Networks (CNN)
- Computer Vision
- TensorFlow
- Keras
- OpenCV
- Image Processing
- Streamlit
- Python
- Data Visualization
- Report Generation

---

# 👩‍💻 Author

**Shreya Shetty**

BE Computer Science Engineering  
Canara Engineering College, Mangalore

## ⭐ If you found this project helpful, consider giving it a Star!
