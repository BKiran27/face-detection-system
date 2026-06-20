# 👁️ Face Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.13-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C853?style=for-the-badge)

**A real-time Face Detection System powered by OpenCV's Haar Cascade Classifier with an interactive Streamlit web interface.**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [Project Structure](#-project-structure)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📸 **Image Detection** | Upload images and detect faces with bounding boxes |
| 📷 **Webcam Detection** | Capture snapshots from webcam for instant detection |
| 🎥 **Video Detection** | Process video files frame-by-frame for face detection |
| 🔢 **Face Counting** | Automatically counts all detected faces |
| ✂️ **Face Cropping** | Extract and display individual face crops |
| 📥 **Download Results** | Save processed images with detections |
| ⚙️ **Tunable Parameters** | Adjust Scale Factor, Min Neighbors, Min Face Size |
| 🎨 **Custom Box Colors** | Choose your preferred bounding box color |
| 🖥️ **Premium Dark UI** | Glassmorphism design with gradient accents |

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BKiran27/face-detection-system.git
   cd face-detection-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser:**
   The app will automatically open at `http://localhost:8501`

---

## 📖 Usage

### 📸 Image Detection
1. Navigate to the **"📸 Image Detection"** tab
2. Upload an image (JPG, PNG, BMP, WebP)
3. View detected faces with bounding boxes
4. Adjust detection parameters in the sidebar
5. Download the result image

### 📷 Webcam Detection
1. Navigate to the **"📷 Webcam Detection"** tab
2. Allow camera access when prompted
3. Take a snapshot
4. View detection results instantly

### 🎥 Video Detection
1. Navigate to the **"🎥 Video Detection"** tab
2. Upload a video file (MP4, AVI, MOV, MKV)
3. Configure frame skip and max frames
4. Click **"▶️ Start Detection"**
5. Watch frame-by-frame detection in real-time

### ⚙️ Sidebar Settings
- **Scale Factor** (1.01–1.50): Controls image pyramid scaling
- **Min Neighbors** (1–15): Filters false positives
- **Min Face Size** (10–200px): Minimum detectable face size
- **Box Color**: Customize bounding box color
- **Show Labels**: Toggle face number labels
- **Show Cropped Faces**: Toggle cropped face gallery

---

## 🧠 How It Works

### Haar Cascade Classifier

This project uses OpenCV's pre-trained **Haar Cascade Classifier** for face detection:

```
Input Image → Grayscale → Image Pyramid → Sliding Window → Haar Features → Cascade Stages → Detected Faces
```

1. **Grayscale Conversion**: Reduces computational complexity
2. **Image Pyramid**: Creates multi-scale representations
3. **Sliding Window**: Scans the image at each scale
4. **Haar Features**: Evaluates edge, line, and rectangle patterns
5. **Cascade Stages**: Multiple classifier stages quickly reject non-face regions
6. **Non-Maximum Suppression**: Merges overlapping detections

### Face Detection vs. Face Recognition

| Aspect | Detection | Recognition |
|--------|-----------|-------------|
| **Question** | "Is there a face?" | "Whose face is it?" |
| **Output** | Bounding boxes | Identity labels |
| **Complexity** | Moderate | High |
| **This Project** | ✅ Yes | ❌ Future work |

---

## 📂 Project Structure

```
face-detection-system/
│
├── app.py                  # Streamlit web application
├── detect.py               # Core face detection module
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── .gitignore              # Git ignore rules
│
├── sample_images/          # Sample test images
├── data/                   # Data directory (extensible)
├── models/                 # Models directory (extensible)
└── screenshots/            # App screenshots
```

### Key Files

| File | Purpose |
|------|---------|
| `detect.py` | Reusable face detection functions (detect, draw, crop, count) |
| `app.py` | Streamlit UI with 4 tabs: Image, Webcam, Video, About |
| `requirements.txt` | All Python dependencies |

---

## 🛠️ Tech Stack

- **Python 3.14** — Core programming language
- **OpenCV 4.13** — Computer Vision & Haar Cascade
- **Streamlit** — Interactive web framework
- **NumPy** — Numerical array operations
- **Pillow** — Image format handling

---

## 🔮 Future Enhancements

- [ ] 🆔 Face Recognition using known face databases
- [ ] 📋 Attendance tracking system
- [ ] 😊 Emotion detection (happy, sad, angry, etc.)
- [ ] 😷 Face mask detection
- [ ] 📊 Real-time analytics dashboard
- [ ] 🎥 WebRTC live video streaming
- [ ] 🧠 Deep learning-based detection (DNN/MTCNN)

---

## 💼 Resume Description

> **Face Detection System** — Developed a real-time Face Detection System using Python and OpenCV. Implemented image preprocessing, Haar Cascade-based face detection, webcam integration, and an interactive Streamlit application capable of detecting multiple faces in images, video files, and live webcam feeds. Features include adjustable detection parameters, face counting, face cropping, and a premium dark-themed UI with glassmorphism design.

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

**Built with ❤️ by [BKiran27](https://github.com/BKiran27)**

Python • OpenCV • Streamlit

</div>
