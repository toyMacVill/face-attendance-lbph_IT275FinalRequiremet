# face-attendance-lbph_IT275FinalRequiremet

# Face Recognition Attendance System

A real-time biometric attendance monitoring system using OpenCV, Haar Cascade Classifier, and Local Binary Pattern Histogram (LBPH). The system performs facial detection, facial recognition, webcam-based attendance monitoring, and automatic attendance logging through a Streamlit web application.

---

# Features

* Real-time webcam face recognition
* Facial image registration and training
* Dynamic LBPH model retraining
* Image-based facial recognition
* Automated attendance logging
* CSV attendance export
* Streamlit web interface
* Lightweight and CPU-friendly implementation

---

# Technologies Used

| Component            | Technology    |
| -------------------- | ------------- |
| Programming Language | Python        |
| Computer Vision      | OpenCV        |
| Face Detection       | Haar Cascade  |
| Face Recognition     | LBPH          |
| Web Application      | Streamlit     |
| Data Handling        | Pandas, NumPy |

---

# System Requirements

## Hardware Requirements

| Component | Minimum Requirement    |
| --------- | ---------------------- |
| Processor | Intel i5 or equivalent |
| RAM       | 8 GB                   |
| Webcam    | 720p Webcam            |
| Storage   | 2 GB Free Space        |

---

## Software Requirements

| Software  | Version        |
| --------- | -------------- |
| Python    | 3.10 or higher |
| OpenCV    | Latest         |
| Streamlit | Latest         |
| NumPy     | Latest         |
| Pandas    | Latest         |
| Pillow    | Latest         |

---

# Project Structure

```text
FaceRecognitionAttendance/
│
├── app.py
├── requirements.txt
├── haarcascade_frontalface_default.xml
│
├── dataset/
│   ├── Person1/
│   ├── Person2/
│
├── trainer/
│   ├── trainer.yml
│   └── labels.npy
│
└── attendance/
    └── attendance.csv
```

---

# Installation Guide

## 1. Clone the Repository

```bash
git clone [https://github.com/your-username/FaceRecognitionAttendance](https://github.com/toyMacVill/face-attendance-lbph_IT275FinalRequiremet).git
```

---

## 2. Navigate to the Project Directory

```bash
cd FaceRecognitionAttendance
```

---

## 3. Install Required Libraries

```bash
pip install opencv-contrib-python streamlit numpy pandas pillow
```

---

# Download Haar Cascade File

Download the Haar Cascade XML file from:

[https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml](https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml)

Place the file inside the project root directory.

---

# Dataset Preparation

Create folders inside the `dataset` directory for each registered user.

Example:

```text
dataset/
    Elizor/
    Jerick/
```

Add 6–10 facial images for each user.

## Image Guidelines

* Use clear frontal facial images
* Ensure proper lighting
* Avoid blurry images
* Include slight facial angle variations

---

# Running the System

## Start the Streamlit Application

```bash
streamlit run app.py
```

---

# Access the Web Application

After execution, Streamlit generates a local server address such as:

```text
http://localhost:8501
```

Open the URL in your web browser.

---

# System Modules

## Train New Face

* Enter user name
* Upload facial images
* Automatically train the LBPH model

---

## Recognize from Image

* Upload image
* Perform facial recognition
* Display identified users

---

## Live Webcam Recognition

* Start webcam
* Perform real-time facial recognition
* Automatically log attendance

---

## Attendance Records

* View attendance logs
* Download CSV attendance records

---

# Attendance Storage

Attendance records are automatically saved in:

```text
attendance/attendance.csv
```

Example:

```csv
Name,DateTime
Elizor,2026-05-09 14:20:11
Jerick,2026-05-09 14:21:55
```

---

# Model Retraining

Whenever new users are added:

1. Upload new facial images
2. Retrain the model through the web interface

The system automatically updates:

* `trainer.yml`
* `labels.npy`

---

# Stopping the Application

Press:

```text
CTRL + C
```

inside the terminal to stop the Streamlit server.

---

# Future Improvements

* Deep learning-based facial recognition
* YOLOv8-face integration
* FaceNet embeddings
* Anti-spoofing mechanism
* MySQL/Firebase integration
* Mobile application deployment
* Multi-camera recognition

---

# Researchers

This project was developed for academic and research purposes in the field of:

* Computer Vision
* Biometrics
* Artificial Intelligence
* Attendance Automation Systems

---

# License

This project is intended for educational and research use only.
