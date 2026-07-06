# Bridging-Image-and-Video-Based-Crack-Detection-A-Unified-CNN-Approach-for-Structural-and-Land-Cracks
A unified deep learning system for detecting and analyzing cracks in both structural surfaces (via video) and land/soil (via image), built using a custom-trained CNN model. Deployed as a real-time Streamlit web application with automated severity classification, depth estimation, and report generation.

## 📌 Project Overview

Manual crack inspection is often time-consuming, expensive, and prone to human error. This project automates the crack inspection process by combining computer vision and deep learning into a single application capable of analyzing both structural and land cracks.

### Features

- Detect structural cracks from uploaded videos
- Detect land/soil cracks from uploaded images
- Estimate crack width and depth
- Classify crack severity (Minor, Moderate, Severe)
- Generate annotated outputs
- Download PDF and CSV reports
- Real-time Streamlit web application

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Programming Language |
| TensorFlow | Deep Learning Framework |
| Keras | CNN Model |
| OpenCV | Image & Video Processing |
| Streamlit | Web Application |
| NumPy | Numerical Computation |
| Pandas | Data Processing |
| Matplotlib | Visualization |
| ReportLab | PDF Report Generation |

---

## 📁 Project Structure

```text
ML-Based-Structural-and-Land-Crack-Detection/
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
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── image3.png
│   │
│   └── Videos/
│       ├── video1.mp4
│       ├── video2.mp4
│       └── video3.avi
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
    ├── Report.png
    └── Dashboard.png
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/ML-Based-Structural-and-Land-Crack-Detection.git
```

Move into the project folder

```bash
cd ML-Based-Structural-and-Land-Crack-Detection
```

Install the required packages

```bash
pip install -r requirements.txt
```

Or install manually

```bash
pip install streamlit tensorflow keras opencv-python numpy pandas matplotlib reportlab
```

---

## ▶️ Running the Application

Ensure the trained model

```
best_crack_detector.keras
```

is present in the project directory.

Run the application

```bash
streamlit run appi.py
```

The Streamlit application will automatically open in your default browser.

---

## 🖥️ Application Workflow

### Structural Crack Detection (Video)

1. Upload a structural inspection video.
2. Extract frames using OpenCV.
3. Preprocess each frame.
4. Pass every frame through the trained CNN.
5. Detect cracks.
6. Estimate crack width.
7. Estimate crack depth.
8. Classify severity.
9. Generate annotated output video.
10. Export CSV and PDF reports.

---

### Land Crack Detection (Image)

1. Upload an image.
2. Preprocess image.
3. CNN predicts crack/no crack.
4. Detect crack contour.
5. Estimate crack width.
6. Estimate crack depth.
7. Classify severity.
8. Generate annotated output image.
9. Export report.

---

## 🔄 Detection Pipeline

```
Input Image / Video
        │
        ▼
Image Preprocessing
        │
        ▼
Resize (224 × 224)
        │
        ▼
Normalization
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
CSV & PDF Report Generation
```

---

## 📊 Outputs

The application generates

- Annotated crack image
- Annotated crack video
- Crack width
- Estimated crack depth
- Confidence score
- Severity classification
- Depth vs Frame graph
- CSV report
- PDF report

---

## 📈 Model Performance

| Metric | Value |
|---------|-------|
| Model | Custom CNN |
| Accuracy | **94%** |
| Framework | TensorFlow & Keras |
| Supported Inputs | Images & Videos |
| Deployment | Streamlit |

---

## 📊 Crack Severity Classification

| Crack Width | Severity |
|-------------|----------|
| < 1 mm | Minor |
| 1 – 3 mm | Moderate |
| > 3 mm | Severe |

---

## 📷 Screenshots

Add your screenshots inside the **Screenshots** folder.

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

## 🚀 Future Enhancements

- Real-time CCTV crack monitoring
- Drone-based crack inspection
- Crack segmentation using U-Net
- Mobile application
- Cloud deployment
- Automatic maintenance recommendation system

---

## 💡 Key Features

- Unified CNN model for structural and land crack detection
- Image and video processing support
- Automatic crack width estimation
- Automatic crack depth estimation
- Severity classification
- Annotated outputs
- Downloadable PDF reports
- CSV export
- Interactive Streamlit interface
- IEEE Conference Publication

---

## 📦 Requirements

```
streamlit
tensorflow
keras
opencv-python
numpy
pandas
matplotlib
reportlab
```

Install using

```bash
pip install -r requirements.txt
```

---

## 🧠 Skills Demonstrated

- Deep Learning
- Convolutional Neural Networks (CNN)
- Computer Vision
- Image Processing
- OpenCV
- TensorFlow
- Keras
- Streamlit
- Python
- Data Visualization
- Report Generation

---

## 👩‍💻 Author

**Shreya Shetty**

BE Computer Science Engineering  
Canara Engineering College, Mangalore

**GitHub:** https://github.com/yourusername

**LinkedIn:** https://linkedin.com/in/your-profile

**Email:** your-email@example.com

---

## ⭐ If you found this project useful, don't forget to give it a Star!
