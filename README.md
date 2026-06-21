# 😷 Face & Mask Compliance Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge)

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Face_&_Mask_System-00f5d4?style=for-the-badge)](https://bkiran27-face-detection-system.streamlit.app)

**An interactive Computer Vision application for face detection and mask compliance analysis. Powered by OpenCV Haar Cascades and a lightweight Deep Learning SSD (Single Shot Detector) model.**

### 🌐 [**Live Demo → bkiran27-face-detection-system.streamlit.app**](https://bkiran27-face-detection-system.streamlit.app)

[Features](#-features) • [Deep Learning Models](#-deep-learning-models) • [Installation](#-installation) • [CLI Usage](#-cli-usage) • [Streamlit Web App](#-streamlit-web-app) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure)

</div>

---

> [!NOTE]  
> **First of all, we hope the people in the world defeat COVID-2019 as soon as possible. Stay strong, all the countries in the world.**  
> *我们做出了这套口罩检测模型并开源了对应的推理代码，支持五大主流深度学习框架（PyTorch、TensorFlow、Keras、MXNet和Caffe）及PaddlePaddle和OpenCV DNN。*

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 👁️ **Face Detection (Haar)** | Traditional high-speed OpenCV Haar Cascade face detection |
| 😷 **Mask Detection (DL)** | Deep Learning Single Shot Detector (SSD) face mask compliance classification |
| 📸 **Image Detection** | Upload images and detect faces/masks with color-coded bounding boxes |
| 📷 **Webcam Detection** | Capture snapshots from webcam for instant analysis |
| 🎥 **Video Detection** | Run real-time frame-by-frame analysis on uploaded video files |
| 📊 **Compliance Metrics** | Dynamic KPIs showing Total faces, wearing mask count, no mask count, and compliance % |
| ✂️ **Cropped Analysis** | Auto-crops and displays faces labeled with classification status & confidence |
| 📥 **Download Results** | Export annotated frames with bounding boxes and compliance badges |
| ⚙️ **Tunable Parameters** | Tweak thresholds (Scale Factor, Min Neighbors, Confidence, NMS IoU) in the sidebar |
| 🎨 **Premium UI/UX** | Modern dark glassmorphic interface with micro-animations and glowing indicators |

---

## 🧠 Deep Learning Models

The deep learning face mask compliance detection system utilizes a customized, highly optimized **Single Shot Detector (SSD)** network:
*   **Lightweight Backbone:** Designed with only 8 convolutional layers, containing about **1.01M parameters** and 24 total layers. It is optimized to achieve high frames per second (FPS) on CPU, edge devices, and web browsers.
*   **Dataset:** Trained on a verified dataset of **7,971 images** from WIDER Face and MAFA datasets, with manual verification of annotations to maintain high accuracy and exclude synthetic/digitally altered mask images.
*   **Target input size:** 260x260 pixels.

### Supported Frameworks & Weight Files
We provide pre-trained model weights and loading scripts across all popular deep learning frameworks. They are located inside the `models/` directory:

*   **PyTorch:** `face_mask_detection.pth` & `model360.pth`
*   **TensorFlow:** `face_mask_detection.pb` (Frozen Graph) & `face_mask_detection.tflite` (Lite)
*   **Keras:** `face_mask_detection.json` & `face_mask_detection.hdf5`
*   **MXNet:** `face_mask_detection.params`
*   **Caffe:** `face_mask_detection.prototxt` & `face_mask_detection.caffemodel`
*   **PaddlePaddle:** `paddle/` directory (`__model__`, `__params__`)
*   **OpenCV DNN:** Loads the Caffe or TensorFlow models directly (default engine for the Streamlit app)

---

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/BKiran27/face-detection-system.git
    cd face-detection-system
    ```

2.  **Install core requirements (needed for Streamlit web app):**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Install deep learning frameworks (optional, only needed for framework CLI scripts):**
    ```bash
    pip install -r frameworks_requirements.txt
    ```

---

## 💻 CLI Usage

You can run command-line inference on individual images or live video feeds using your favorite framework.

### 1. PyTorch
```bash
python pytorch_infer.py --img-mode 1 --img-path sample_images/person.png
# For webcam video mode:
python pytorch_infer.py --img-mode 0 --video-path 0
```

### 2. TensorFlow
```bash
python tensorflow_infer.py --img-mode 1 --img-path sample_images/person.png
# For webcam video mode:
python tensorflow_infer.py --img-mode 0 --video-path 0
```

### 3. Keras
```bash
python keras_infer.py --img-mode 1 --img-path sample_images/person.png
```

### 4. MXNet
```bash
python mxnet_infer.py --img-mode 1 --img-path sample_images/person.png
```

### 5. Caffe
```bash
python caffe_infer.py --img-mode 1 --img-path sample_images/person.png
```

### 6. PaddlePaddle
```bash
python paddle_infer.py --model_dir models/paddle
```

### 7. OpenCV DNN (Primary Inference Engine)
OpenCV DNN runs inference using Caffe models without needing TensorFlow or PyTorch installed:
```bash
python opencv_dnn_infer.py --img-mode 1 --img-path sample_images/person.png
# For webcam video mode:
python opencv_dnn_infer.py --img-mode 0 --video-path 0
```

---

## 🚀 Streamlit Web App

To launch the interactive, glassmorphic Streamlit web application:
```bash
streamlit run app.py
```
This will open the application in your default web browser at `http://localhost:8501`.

*Note: The Streamlit app uses **OpenCV DNN** under the hood. This provides fast, hardware-agnostic inference on CPUs and allows deployment to cloud environments (like Streamlit Cloud) without exceeding memory and slug size constraints.*

---

## 📂 Project Structure

```
face-detection-system/
│
├── app.py                      # Streamlit web application
├── detect.py                   # Core Haar Cascade face detection module
├── mask_detect.py              # Core OpenCV DNN face mask detection module
├── requirements.txt            # Core Streamlit app dependencies
├── frameworks_requirements.txt # Standalone DL frameworks dependencies
├── README.md                   # Project documentation
│
├── models/                     # Model weights and definitions
│   ├── MainModel.py            # Custom backbone structure definition
│   ├── face_mask_detection.pth # PyTorch model weights
│   ├── face_mask_detection.pb  # TensorFlow model graph
│   ├── face_mask_detection.tflite # TF Lite model
│   ├── face_mask_detection.hdf5 # Keras model weights
│   ├── face_mask_detection.caffemodel & .prototxt # Caffe model definition & weights
│   └── paddle/                 # PaddlePaddle model files
│
├── load_model/                 # Python modules to load framework-specific weights
│   ├── pytorch_loader.py
│   ├── tensorflow_loader.py
│   └── ...
│
├── utils/                      # Bounding box decoder & NMS algorithms
│   ├── anchor_generator.py
│   ├── anchor_decode.py
│   └── nms.py
│
├── sample_images/              # Sample test images
└── screenshots/                # Application UI screenshots
```

---

## 🔮 Future Enhancements

*   [ ] 🆔 Face Recognition using local embedding databases
*   [ ] 📋 Attendance management reporting
*   [ ] 😊 Emotion classification (happy, surprised, neutral, etc.)
*   [ ] 📊 Real-time analytics dashboard with compliance logs
*   [ ] 🎥 WebRTC stream support for seamless mobile browser capture
