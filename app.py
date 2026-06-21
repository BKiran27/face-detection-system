"""
Face Detection & Mask Compliance System — Streamlit Application
================================================================
An interactive web application for detecting faces and determining face mask compliance
in images, webcam snapshots, and video files. Powered by OpenCV Haar Cascades
and a lightweight Deep Learning SSD (Single Shot Detector) model.

Author: BKiran27
Run:    streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import io

from detect import (
    load_face_cascade,
    detect_faces,
    draw_bounding_boxes,
    draw_face_count_badge,
    crop_faces,
    get_face_count,
    process_video_frame,
)

from mask_detect import (
    load_mask_net,
    detect_masks,
    draw_mask_bounding_boxes,
    draw_mask_count_badge,
    crop_mask_faces,
)

# ─── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Face & Mask Detection System",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Global Styles ── */
    * {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1b2a 30%, #1b1b3a 60%, #0a0a1a 100%);
    }

    /* ── Header Styling ── */
    .main-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
        margin-bottom: 1.5rem;
    }

    .main-header h1 {
        background: linear-gradient(135deg, #00f5d4, #7b61ff, #f72585);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0.3rem;
    }

    .main-header p {
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* ── Glassmorphism Cards ── */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        border-color: rgba(0, 245, 212, 0.3);
        box-shadow: 0 8px 32px rgba(0, 245, 212, 0.1);
        transform: translateY(-2px);
    }

    /* ── KPI Metric Cards ── */
    .metric-container {
        display: flex;
        gap: 1rem;
        justify-content: center;
        flex-wrap: wrap;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.1), rgba(123, 97, 255, 0.1));
        border: 1px solid rgba(0, 245, 212, 0.2);
        border-radius: 16px;
        padding: 1.2rem 2rem;
        text-align: center;
        min-width: 160px;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 30px rgba(0, 245, 212, 0.15);
    }

    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f5d4, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .metric-label {
        color: #8892b0;
        font-size: 0.85rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }

    /* ── Sidebar Styling ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b1b3a 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #00f5d4 !important;
    }

    /* ── Tab Styling ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 10px 20px;
        color: #8892b0;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.15), rgba(123, 97, 255, 0.15)) !important;
        color: #00f5d4 !important;
    }

    /* ── Upload Area ── */
    .stFileUploader > div {
        border: 2px dashed rgba(0, 245, 212, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(0, 245, 212, 0.02) !important;
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        border-color: rgba(0, 245, 212, 0.6) !important;
        background: rgba(0, 245, 212, 0.05) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, #00f5d4, #7b61ff) !important;
        color: #0a0a1a !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        padding: 0.6rem 2rem !important;
        transition: all 0.3s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(0, 245, 212, 0.3) !important;
    }

    /* ── Slider ── */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00f5d4, #7b61ff) !important;
    }

    /* ── Divider ── */
    .gradient-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, #00f5d4, #7b61ff, transparent);
        margin: 1.5rem 0;
        border: none;
    }

    /* ── Feature Badge ── */
    .feature-badge {
        display: inline-block;
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.15), rgba(123, 97, 255, 0.15));
        border: 1px solid rgba(0, 245, 212, 0.3);
        border-radius: 20px;
        padding: 4px 14px;
        font-size: 0.78rem;
        color: #00f5d4;
        font-weight: 600;
        margin: 2px;
    }

    /* ── About Section Icons ── */
    .tech-stack-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 0;
        color: #ccd6f6;
        font-size: 0.95rem;
    }

    /* ── Image Container ── */
    .image-container {
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* ── Animated Pulse ── */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }

    /* ── Success/Info Banners ── */
    .success-banner {
        background: linear-gradient(135deg, rgba(0, 245, 212, 0.1), rgba(0, 200, 100, 0.1));
        border: 1px solid rgba(0, 245, 212, 0.3);
        border-radius: 12px;
        padding: 1rem 1.5rem;
        color: #00f5d4;
        font-weight: 500;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Face Cascade & DNN Net (cached) ─────────────────────────────────

@st.cache_resource
def get_face_cascade():
    """Cache the face cascade classifier for performance."""
    return load_face_cascade()


@st.cache_resource
def get_mask_net():
    """Cache the deep learning mask detection network."""
    return load_mask_net()


# ─── Helper Functions ────────────────────────────────────────────────────────────

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV BGR format."""
    rgb_array = np.array(pil_image)
    if len(rgb_array.shape) == 2:
        # Grayscale
        return cv2.cvtColor(rgb_array, cv2.COLOR_GRAY2BGR)
    elif rgb_array.shape[2] == 4:
        # RGBA
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGBA2BGR)
    else:
        # RGB
        return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convert OpenCV BGR image to PIL RGB format."""
    rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_image)


def get_image_download_bytes(cv2_image: np.ndarray) -> bytes:
    """Convert OpenCV image to downloadable PNG bytes."""
    pil_image = cv2_to_pil(cv2_image)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def render_metrics(face_count: int, image_shape: tuple, task: str = "face", mask_count: int = 0, nomask_count: int = 0):
    """Render metric cards based on selected task."""
    if task == "face":
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-value">{face_count}</div>
                <div class="metric-label">Faces Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{image_shape[1]}×{image_shape[0]}</div>
                <div class="metric-label">Resolution</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{"✅" if face_count > 0 else "❌"}</div>
                <div class="metric-label">Detection Status</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        compliance = (mask_count / face_count * 100) if face_count > 0 else 100.0
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-card">
                <div class="metric-value">{face_count}</div>
                <div class="metric-label">Total Faces</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="background: linear-gradient(135deg, #00f5d4, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{mask_count}</div>
                <div class="metric-label">Wearing Mask</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="background: linear-gradient(135deg, #f72585, #ff4d4d); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{nomask_count}</div>
                <div class="metric-label">No Mask</div>
            </div>
            <div class="metric-card">
                <div class="metric-value" style="background: linear-gradient(135deg, #7b61ff, #00f5d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;">{compliance:.0f}%</div>
                <div class="metric-label">Compliance Rate</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ─── Header ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>😷 Face Detection & Mask Compliance</h1>
    <p>Detect faces and classify mask compliance using OpenCV Cascades & Deep Learning Models</p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🎯 System Mode")
    task_mode = st.radio(
        "Select Task Mode:",
        options=["👁️ Face Detection (Haar)", "😷 Face Mask Detection (DL)"],
        index=0,
        help="Switch between Haar Cascade Face Detection or Deep Learning-based Face Mask Detection."
    )
    
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    if task_mode == "👁️ Face Detection (Haar)":
        st.markdown("## ⚙️ Haar Cascade Settings")
        
        scale_factor = st.slider(
            "🔍 Scale Factor",
            min_value=1.01,
            max_value=1.50,
            value=1.10,
            step=0.01,
            help="How much the image size is reduced at each scale. Lower = more thorough but slower.",
        )

        min_neighbors = st.slider(
            "🎯 Min Neighbors",
            min_value=1,
            max_value=15,
            value=5,
            step=1,
            help="Higher values reduce false positives but may miss some faces.",
        )

        min_face_size = st.slider(
            "📏 Min Face Size (px)",
            min_value=10,
            max_value=200,
            value=30,
            step=10,
            help="Minimum face size in pixels. Increase to ignore small faces.",
        )
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 🎨 Display Options")
        
        show_labels = st.checkbox("Show face labels", value=True)
        show_cropped = st.checkbox("Show cropped faces", value=True)
        box_color_hex = st.color_picker("Box color", "#00FF80")

        # Convert hex to BGR
        box_color_rgb = tuple(int(box_color_hex[i:i+2], 16) for i in (1, 3, 5))
        box_color_bgr = (box_color_rgb[2], box_color_rgb[1], box_color_rgb[0])
    
    else:
        st.markdown("## ⚙️ Deep Learning Settings")
        
        mask_conf_thresh = st.slider(
            "🧠 Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum classification score to accept a mask/nomask detection.",
        )

        mask_iou_thresh = st.slider(
            "⚡ NMS IoU Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.4,
            step=0.05,
            help="Intersection Over Union threshold used by Non-Max Suppression to clean overlapping boxes.",
        )
        
        st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
        st.markdown("## 🎨 Display Options")
        
        show_labels = st.checkbox("Show classification labels", value=True)
        show_cropped = st.checkbox("Show cropped faces", value=True)
        # BGR box color is automatically selected (Green for Mask, Red for NoMask)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card" style="text-align:center;">
        <p style="color:#8892b0; font-size:0.8rem; margin:0;">
            Built with ❤️ by <strong style="color:#00f5d4;">BKiran27</strong><br>
            OpenCV • Streamlit • Python
        </p>
    </div>
    """, unsafe_allow_html=True)


# ─── Main Tabs ───────────────────────────────────────────────────────────────────

tab_image, tab_webcam, tab_video, tab_about = st.tabs([
    "📸 Image Detection",
    "📷 Webcam Detection",
    "🎥 Video Detection",
    "ℹ️ About",
])

# Preload resources
cascade = get_face_cascade()
mask_net = None
if task_mode == "😷 Face Mask Detection (DL)":
    try:
        mask_net = get_mask_net()
    except Exception as e:
        st.error(f"❌ Error loading Deep Learning Face Mask Detection model: {e}")
        st.info("Please make sure models/face_mask_detection.prototxt and face_mask_detection.caffemodel exist.")


# ─── Tab 1: Image Detection ─────────────────────────────────────────────────────

with tab_image:
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">📸 Upload an Image</h3>
        <p style="color:#8892b0;">Upload a photo and the system will run {"Face Mask Detection" if task_mode != "👁️ Face Detection (Haar)" else "Face Detection"} on it.</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="image_uploader",
        label_visibility="collapsed",
    )

    # Load sample images
    sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")
    sample_images = []
    if os.path.exists(sample_dir):
        sample_images = [
            f for f in os.listdir(sample_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
        ]

    use_sample = False
    selected_sample = None

    if sample_images and not uploaded_file:
        st.markdown("**Or try a sample image:**")
        cols = st.columns(len(sample_images))
        for idx, sample_name in enumerate(sample_images):
            sample_path = os.path.join(sample_dir, sample_name)
            with cols[idx]:
                sample_img = Image.open(sample_path)
                st.image(sample_img, caption=sample_name, use_container_width=True)
                if st.button(f"Use {sample_name}", key=f"sample_{idx}"):
                    use_sample = True
                    selected_sample = sample_path

    # Process image
    image_to_process = None

    if uploaded_file is not None:
        pil_img = Image.open(uploaded_file)
        image_to_process = pil_to_cv2(pil_img)
    elif use_sample and selected_sample:
        image_to_process = cv2.imread(selected_sample)

    if image_to_process is not None:
        with st.spinner("🔍 Running detection..."):
            
            # --- TASK 1: Face Detection (Haar Cascade) ---
            if task_mode == "👁️ Face Detection (Haar)":
                faces = detect_faces(
                    image_to_process,
                    cascade,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=(min_face_size, min_face_size),
                )
                face_count = get_face_count(faces)

                result_image = draw_bounding_boxes(
                    image_to_process, faces, color=box_color_bgr, show_label=show_labels
                )
                result_image = draw_face_count_badge(result_image, face_count)
                
                # Metrics & Outputs
                render_metrics(face_count, image_to_process.shape, task="face")

            # --- TASK 2: Face Mask Detection (DL SSD) ---
            else:
                if mask_net is not None:
                    detections = detect_masks(
                        image_to_process,
                        mask_net,
                        conf_thresh=mask_conf_thresh,
                        iou_thresh=mask_iou_thresh
                    )
                    face_count = len(detections)
                    mask_count = sum(1 for d in detections if d['class_id'] == 0)
                    nomask_count = sum(1 for d in detections if d['class_id'] == 1)

                    result_image = draw_mask_bounding_boxes(
                        image_to_process, detections, show_label=show_labels
                    )
                    result_image = draw_mask_count_badge(result_image, mask_count, nomask_count)
                    
                    # Metrics & Outputs
                    render_metrics(face_count, image_to_process.shape, task="mask", mask_count=mask_count, nomask_count=nomask_count)
                else:
                    result_image = image_to_process.copy()
                    face_count = 0

        # Results Display
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Original")
            st.image(cv2_to_pil(image_to_process), use_container_width=True)
        with col2:
            st.markdown("#### 🎯 Detected")
            st.image(cv2_to_pil(result_image), use_container_width=True)

        # Download button
        download_bytes = get_image_download_bytes(result_image)
        st.download_button(
            label="📥 Download Result",
            data=download_bytes,
            file_name="mask_compliance_result.png" if task_mode != "👁️ Face Detection (Haar)" else "face_detection_result.png",
            mime="image/png",
        )

        # Cropped faces analysis
        if show_cropped and face_count > 0:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### ✂️ Cropped Face Analysis")
            
            if task_mode == "👁️ Face Detection (Haar)":
                cropped = crop_faces(image_to_process, faces)
                cols = st.columns(min(face_count, 5))
                for idx, face_img in enumerate(cropped):
                    with cols[idx % len(cols)]:
                        st.image(
                            cv2_to_pil(face_img),
                            caption=f"Face #{idx + 1}",
                            use_container_width=True,
                        )
            else:
                cropped_details = crop_mask_faces(image_to_process, detections)
                cols = st.columns(min(face_count, 5))
                for idx, (face_img, det) in enumerate(cropped_details):
                    with cols[idx % len(cols)]:
                        conf_percentage = f"{det['confidence']:.0%}"
                        label_class = det['class_name']
                        badge = f"🟢 {label_class} ({conf_percentage})" if det['class_id'] == 0 else f"🔴 {label_class} ({conf_percentage})"
                        st.image(
                            cv2_to_pil(face_img),
                            caption=badge,
                            use_container_width=True,
                        )

        if face_count == 0:
            if task_mode == "👁️ Face Detection (Haar)":
                st.markdown("""
                <div class="success-banner" style="border-color: rgba(247, 37, 133, 0.3); color: #f72585;">
                    ⚠️ No faces detected. Try adjusting the <strong>Scale Factor</strong> or <strong>Min Neighbors</strong> in the sidebar.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="success-banner" style="border-color: rgba(247, 37, 133, 0.3); color: #f72585;">
                    ⚠️ No faces or masks detected. Try lowering the <strong>Confidence Threshold</strong> in the sidebar.
                </div>
                """, unsafe_allow_html=True)


# ─── Tab 2: Webcam Detection ────────────────────────────────────────────────────

with tab_webcam:
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">📷 Webcam Snapshot</h3>
        <p style="color:#8892b0;">Take a photo using your webcam and analyze mask compliance instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    camera_image = st.camera_input(
        "Take a photo",
        key="webcam_capture",
        label_visibility="collapsed",
    )

    if camera_image is not None:
        pil_img = Image.open(camera_image)
        cv2_img = pil_to_cv2(pil_img)

        with st.spinner("🔍 Running webcam snapshot detection..."):
            if task_mode == "👁️ Face Detection (Haar)":
                faces = detect_faces(
                    cv2_img,
                    cascade,
                    scale_factor=scale_factor,
                    min_neighbors=min_neighbors,
                    min_size=(min_face_size, min_face_size),
                )
                face_count = get_face_count(faces)

                result_image = draw_bounding_boxes(
                    cv2_img, faces, color=box_color_bgr, show_label=show_labels
                )
                result_image = draw_face_count_badge(result_image, face_count)
                
                render_metrics(face_count, cv2_img.shape, task="face")
            
            else:
                if mask_net is not None:
                    detections = detect_masks(
                        cv2_img,
                        mask_net,
                        conf_thresh=mask_conf_thresh,
                        iou_thresh=mask_iou_thresh
                    )
                    face_count = len(detections)
                    mask_count = sum(1 for d in detections if d['class_id'] == 0)
                    nomask_count = sum(1 for d in detections if d['class_id'] == 1)

                    result_image = draw_mask_bounding_boxes(
                        cv2_img, detections, show_label=show_labels
                    )
                    result_image = draw_mask_count_badge(result_image, mask_count, nomask_count)
                    
                    render_metrics(face_count, cv2_img.shape, task="mask", mask_count=mask_count, nomask_count=nomask_count)
                else:
                    result_image = cv2_img.copy()
                    face_count = 0

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📷 Captured")
            st.image(cv2_to_pil(cv2_img), use_container_width=True)
        with col2:
            st.markdown("#### 🎯 Detected")
            st.image(cv2_to_pil(result_image), use_container_width=True)

        download_bytes = get_image_download_bytes(result_image)
        st.download_button(
            label="📥 Download Result",
            data=download_bytes,
            file_name="webcam_mask_result.png" if task_mode != "👁️ Face Detection (Haar)" else "webcam_face_result.png",
            mime="image/png",
            key="webcam_download",
        )

        if show_cropped and face_count > 0:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### ✂️ Cropped Face Analysis")
            
            if task_mode == "👁️ Face Detection (Haar)":
                cropped = crop_faces(cv2_img, faces)
                cols = st.columns(min(face_count, 5))
                for idx, face_img in enumerate(cropped):
                    with cols[idx % len(cols)]:
                        st.image(
                            cv2_to_pil(face_img),
                            caption=f"Face #{idx + 1}",
                            use_container_width=True,
                        )
            else:
                cropped_details = crop_mask_faces(cv2_img, detections)
                cols = st.columns(min(face_count, 5))
                for idx, (face_img, det) in enumerate(cropped_details):
                    with cols[idx % len(cols)]:
                        conf_percentage = f"{det['confidence']:.0%}"
                        label_class = det['class_name']
                        badge = f"🟢 {label_class} ({conf_percentage})" if det['class_id'] == 0 else f"🔴 {label_class} ({conf_percentage})"
                        st.image(
                            cv2_to_pil(face_img),
                            caption=badge,
                            use_container_width=True,
                        )


# ─── Tab 3: Video Detection ─────────────────────────────────────────────────────

with tab_video:
    st.markdown(f"""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">🎥 Video Processing</h3>
        <p style="color:#8892b0;">Upload a video file to run frame-by-frame analysis.</p>
    </div>
    """, unsafe_allow_html=True)

    video_file = st.file_uploader(
        "Choose a video file",
        type=["mp4", "avi", "mov", "mkv"],
        key="video_uploader",
        label_visibility="collapsed",
    )

    if video_file is not None:
        # Save to temp file for OpenCV to read
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(video_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)

        if not cap.isOpened():
            st.error("❌ Could not open video file.")
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-value">{total_frames}</div>
                    <div class="metric-label">Total Frames</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{fps:.0f}</div>
                    <div class="metric-label">FPS</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{width}×{height}</div>
                    <div class="metric-label">Resolution</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Process controls
            skip_frames = st.slider(
                "⏭️ Process every Nth frame",
                min_value=1,
                max_value=30,
                value=3,
                help="Skip frames for faster processing. 1 = every frame.",
            )

            max_frames = st.slider(
                "🔢 Max frames to process",
                min_value=10,
                max_value=min(total_frames, 500),
                value=min(100, total_frames),
                help="Limit the number of frames to process.",
            )

            if st.button("▶️ Start Detection", key="start_video"):
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()

                frame_idx = 0
                processed = 0
                total_faces_found = 0
                total_masks_found = 0
                total_nomasks_found = 0

                while cap.isOpened() and processed < max_frames:
                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_idx % skip_frames == 0:
                        
                        # Haar face detection
                        if task_mode == "👁️ Face Detection (Haar)":
                            result, faces = process_video_frame(
                                frame, cascade, scale_factor, min_neighbors
                            )
                            face_count = get_face_count(faces)
                            total_faces_found += face_count
                            caption_txt = f"Frame {frame_idx} — {face_count} face(s)"
                        
                        # Deep learning face mask detection
                        else:
                            if mask_net is not None:
                                detections = detect_masks(
                                    frame,
                                    mask_net,
                                    conf_thresh=mask_conf_thresh,
                                    iou_thresh=mask_iou_thresh
                                )
                                face_count = len(detections)
                                mask_count = sum(1 for d in detections if d['class_id'] == 0)
                                nomask_count = sum(1 for d in detections if d['class_id'] == 1)
                                
                                total_faces_found += face_count
                                total_masks_found += mask_count
                                total_nomasks_found += nomask_count
                                
                                result = draw_mask_bounding_boxes(frame, detections, show_label=show_labels)
                                result = draw_mask_count_badge(result, mask_count, nomask_count)
                                compliance_txt = f"({mask_count}/{face_count} masked)" if face_count > 0 else ""
                                caption_txt = f"Frame {frame_idx} — {face_count} face(s) {compliance_txt}"
                            else:
                                result = frame.copy()
                                face_count = 0
                                caption_txt = f"Frame {frame_idx}"

                        frame_placeholder.image(
                            cv2_to_pil(result),
                            caption=caption_txt,
                            use_container_width=True,
                        )

                        processed += 1
                        progress = processed / max_frames
                        progress_bar.progress(progress)
                        
                        status_str = f"Processing frame **{frame_idx}** / {total_frames} | Processed: **{processed}**"
                        if task_mode == "👁️ Face Detection (Haar)":
                            status_str += f" | Faces in frame: **{face_count}**"
                        else:
                            status_str += f" | Faces: **{face_count}** (😷 Masked: **{mask_count}** | 🔴 Unmasked: **{nomask_count}**)"
                            
                        status_text.markdown(status_str)

                    frame_idx += 1

                cap.release()
                progress_bar.progress(1.0)

                if task_mode == "👁️ Face Detection (Haar)":
                    st.markdown(f"""
                    <div class="success-banner">
                        ✅ Video processing complete! Processed <strong>{processed}</strong> frames. 
                        Total face detections: <strong>{total_faces_found}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    overall_compliance = (total_masks_found / total_faces_found * 100) if total_faces_found > 0 else 100.0
                    st.markdown(f"""
                    <div class="success-banner">
                        ✅ Video processing complete! Processed <strong>{processed}</strong> frames.<br>
                        • Total detections: <strong>{total_faces_found}</strong><br>
                        • Wearing mask: <strong>{total_masks_found}</strong><br>
                        • Without mask: <strong>{total_nomasks_found}</strong><br>
                        • Overall Mask Compliance: <strong>{overall_compliance:.1f}%</strong>
                    </div>
                    """, unsafe_allow_html=True)

        # Cleanup temp file
        try:
            os.unlink(tfile.name)
        except Exception:
            pass


# ─── Tab 4: About ───────────────────────────────────────────────────────────────

with tab_about:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">ℹ️ About This Project</h3>
        <p style="color:#8892b0;">
            An interactive Computer Vision demonstration application built using Python, OpenCV, and Streamlit.
            It offers two primary modes of operation: traditional face detection using Haar Cascade classifiers 
            and modern deep learning face mask compliance detection using Single Shot MultiBox Detector (SSD) models.
        </p>
        <p style="color:#00f5d4; font-weight: 600; font-size:1.1rem; margin-top: 1rem;">
            🌍 "First of all, we hope the people in the world defeat COVID-2019 as soon as possible. Stay strong, all the countries in the world."
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">😷 DL SSD Model Details</h4>
            <p style="color:#ccd6f6; line-height:1.7;">
                The Face Mask Detection system utilizes a lightweight <strong>Single Shot Detector (SSD)</strong> architecture:
            </p>
            <ul style="color:#ccd6f6; line-height:1.6; padding-left:20px;">
                <li><strong>Efficiency:</strong> Customized backbone with only 8 convolutional layers, totaling <strong>1.01M parameters</strong> and 24 total layers. Fully optimized for high FPS inference on standard CPU and mobile devices.</li>
                <li><strong>Dataset:</strong> Trained on a verification dataset of approximately <strong>7,971 images</strong> sourced from public datasets (WIDER Face & MAFA), with annotations double-verified manually for high classification accuracy.</li>
                <li><strong>Resolution:</strong> The network processes frames at a target shape of <strong>260×260</strong> pixels.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">⚙️ Open Source Framework Support</h4>
            <p style="color:#ccd6f6; line-height:1.7;">
                We have open-sourced the models and corresponding inference codes across all the following mainstream deep learning frameworks in this project directory:
            </p>
            <div style="color:#ccd6f6; display:grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div>🔥 <strong>PyTorch</strong> (.pth)</div>
                <div>⚡ <strong>TensorFlow</strong> (.pb, .tflite)</div>
                <div>🧠 <strong>Keras</strong> (.json, .hdf5)</div>
                <div>🌀 <strong>MXNet</strong> (.params)</div>
                <div>☕ <strong>Caffe</strong> (.prototxt, .caffemodel)</div>
                <div>🛶 <strong>PaddlePaddle</strong> (paddle/)</div>
                <div>👁️ <strong>OpenCV DNN</strong> (Inference)</div>
            </div>
            <p style="color:#8892b0; font-size:0.85rem; margin-top: 15px;">
                Look at the root level script files (e.g. <code>pytorch_infer.py</code>, <code>tensorflow_infer.py</code>) to run CLI inference using your favorite framework locally.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">🛠️ Tech Stack</h4>
            <div style="color:#ccd6f6;">
                <div class="tech-stack-item">🐍 <strong>Python 3.12</strong> — Core language</div>
                <div class="tech-stack-item">👁️ <strong>OpenCV 4.13</strong> — Computer Vision Engine</div>
                <div class="tech-stack-item">🚀 <strong>Streamlit</strong> — Glassmorphic Web App UI</div>
                <div class="tech-stack-item">🔢 <strong>NumPy</strong> — Tensor & Anchor decoding</div>
                <div class="tech-stack-item">🖼️ <strong>Pillow</strong> — Image processing functions</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">✨ Main Capabilities</h4>
            <div>
                <span class="feature-badge">👁️ Face Detection</span>
                <span class="feature-badge">😷 Face Mask Detection</span>
                <span class="feature-badge">📸 Image File Upload</span>
                <span class="feature-badge">📷 Webcam Snapshot</span>
                <span class="feature-badge">🎥 Video Frame Processing</span>
                <span class="feature-badge">📊 Compliance Metrics</span>
                <span class="feature-badge">✂️ Cropped Face Grids</span>
                <span class="feature-badge">📥 Result Downloads</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#00f5d4; margin-top:0;">🚀 Future Extensions</h4>
        <div>
            <span class="feature-badge">🆔 Face Recognition</span>
            <span class="feature-badge">📋 Digital Attendance</span>
            <span class="feature-badge">😊 Emotion Analysis</span>
            <span class="feature-badge">📊 Live Analytics Dashboard</span>
            <span class="feature-badge">🎥 Real-time WebRTC streams</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem; color:#4a5568; font-size:0.8rem;">
    <p>😷 Face & Mask Compliance System v1.1 &nbsp;|&nbsp; Built with OpenCV & Streamlit &nbsp;|&nbsp; 
    <a href="https://github.com/BKiran27" style="color:#00f5d4; text-decoration:none;">BKiran27</a></p>
</div>
""", unsafe_allow_html=True)
