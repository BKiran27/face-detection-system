"""
Face Detection System — Streamlit Application
==============================================
An interactive web application for detecting faces in images,
webcam snapshots, and video files using OpenCV Haar Cascade.

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

# ─── Page Configuration ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Face Detection System",
    page_icon="👁️",
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


# ─── Initialize Face Cascade (cached) ───────────────────────────────────────────

@st.cache_resource
def get_face_cascade():
    """Cache the face cascade classifier for performance."""
    return load_face_cascade()


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


def render_metrics(face_count: int, image_shape: tuple):
    """Render metric cards."""
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


# ─── Header ──────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>👁️ Face Detection System</h1>
    <p>Real-time face detection powered by OpenCV & Haar Cascade Classifier</p>
</div>
<div class="gradient-divider"></div>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

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

cascade = get_face_cascade()


# ─── Tab 1: Image Detection ─────────────────────────────────────────────────────

with tab_image:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">📸 Upload an Image</h3>
        <p style="color:#8892b0;">Upload a photo and the system will detect all faces in it.</p>
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
        with st.spinner("🔍 Detecting faces..."):
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

        # Metrics
        render_metrics(face_count, image_to_process.shape)

        # Results
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
            file_name="face_detection_result.png",
            mime="image/png",
        )

        # Cropped faces
        if show_cropped and face_count > 0:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### ✂️ Cropped Faces")
            cropped = crop_faces(image_to_process, faces)
            cols = st.columns(min(face_count, 5))
            for idx, face_img in enumerate(cropped):
                with cols[idx % len(cols)]:
                    st.image(
                        cv2_to_pil(face_img),
                        caption=f"Face #{idx + 1}",
                        use_container_width=True,
                    )

        if face_count == 0:
            st.markdown("""
            <div class="success-banner" style="border-color: rgba(247, 37, 133, 0.3); color: #f72585;">
                ⚠️ No faces detected. Try adjusting the <strong>Scale Factor</strong> or <strong>Min Neighbors</strong> in the sidebar.
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 2: Webcam Detection ────────────────────────────────────────────────────

with tab_webcam:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">📷 Webcam Snapshot</h3>
        <p style="color:#8892b0;">Take a photo using your webcam and detect faces instantly.</p>
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

        with st.spinner("🔍 Detecting faces in webcam snapshot..."):
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

        render_metrics(face_count, cv2_img.shape)

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
            file_name="webcam_face_detection.png",
            mime="image/png",
            key="webcam_download",
        )

        if show_cropped and face_count > 0:
            st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
            st.markdown("#### ✂️ Cropped Faces")
            cropped = crop_faces(cv2_img, faces)
            cols = st.columns(min(face_count, 5))
            for idx, face_img in enumerate(cropped):
                with cols[idx % len(cols)]:
                    st.image(
                        cv2_to_pil(face_img),
                        caption=f"Face #{idx + 1}",
                        use_container_width=True,
                    )


# ─── Tab 3: Video Detection ─────────────────────────────────────────────────────

with tab_video:
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#ccd6f6; margin-top:0;">🎥 Video Face Detection</h3>
        <p style="color:#8892b0;">Upload a video file to detect faces frame by frame.</p>
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

                while cap.isOpened() and processed < max_frames:
                    success, frame = cap.read()
                    if not success:
                        break

                    if frame_idx % skip_frames == 0:
                        result, faces = process_video_frame(
                            frame, cascade, scale_factor, min_neighbors
                        )
                        face_count = get_face_count(faces)
                        total_faces_found += face_count

                        frame_placeholder.image(
                            cv2_to_pil(result),
                            caption=f"Frame {frame_idx} — {face_count} face(s)",
                            use_container_width=True,
                        )

                        processed += 1
                        progress = processed / max_frames
                        progress_bar.progress(progress)
                        status_text.markdown(
                            f"Processing frame **{frame_idx}** / {total_frames} "
                            f"| Processed: **{processed}** | Faces in frame: **{face_count}**"
                        )

                    frame_idx += 1

                cap.release()
                progress_bar.progress(1.0)

                st.markdown(f"""
                <div class="success-banner">
                    ✅ Video processing complete! Processed <strong>{processed}</strong> frames. 
                    Total face detections across all frames: <strong>{total_faces_found}</strong>
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
            A real-time Face Detection System built with Python, OpenCV, and Streamlit.
            This project demonstrates Computer Vision fundamentals using the 
            Haar Cascade Classifier algorithm.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">🔍 Face Detection vs Recognition</h4>
            <table style="width:100%; color:#ccd6f6; border-collapse:collapse;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.1);">
                    <th style="padding:12px 8px; text-align:left; color:#8892b0;">Aspect</th>
                    <th style="padding:12px 8px; text-align:left; color:#00f5d4;">Detection</th>
                    <th style="padding:12px 8px; text-align:left; color:#7b61ff;">Recognition</th>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 8px;">Question</td>
                    <td style="padding:10px 8px;">Is there a face?</td>
                    <td style="padding:10px 8px;">Whose face is it?</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 8px;">Output</td>
                    <td style="padding:10px 8px;">Bounding boxes</td>
                    <td style="padding:10px 8px;">Identity labels</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding:10px 8px;">Complexity</td>
                    <td style="padding:10px 8px;">Moderate</td>
                    <td style="padding:10px 8px;">High</td>
                </tr>
                <tr>
                    <td style="padding:10px 8px;">Training Data</td>
                    <td style="padding:10px 8px;">Pre-trained</td>
                    <td style="padding:10px 8px;">Custom dataset</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">🧠 How Haar Cascade Works</h4>
            <p style="color:#ccd6f6; line-height:1.8;">
                <strong>1. Image Pyramid:</strong> The image is scaled down progressively to detect faces at different sizes.<br><br>
                <strong>2. Sliding Window:</strong> A detection window slides across each scaled image.<br><br>
                <strong>3. Haar Features:</strong> The classifier evaluates edge, line, and rectangle features within each window.<br><br>
                <strong>4. Cascade Stages:</strong> Multiple classifier stages quickly reject non-face regions, focusing computation on likely face areas.<br><br>
                <strong>5. Detection:</strong> Windows passing all stages are marked as detected faces.
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
                <div class="tech-stack-item">🐍 <strong>Python 3.14</strong> — Core language</div>
                <div class="tech-stack-item">👁️ <strong>OpenCV 4.13</strong> — Computer Vision</div>
                <div class="tech-stack-item">🚀 <strong>Streamlit</strong> — Web framework</div>
                <div class="tech-stack-item">🔢 <strong>NumPy</strong> — Array operations</div>
                <div class="tech-stack-item">🖼️ <strong>Pillow</strong> — Image processing</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color:#00f5d4; margin-top:0;">✨ Features</h4>
            <div>
                <span class="feature-badge">📸 Image Detection</span>
                <span class="feature-badge">📷 Webcam Capture</span>
                <span class="feature-badge">🎥 Video Processing</span>
                <span class="feature-badge">🔢 Face Counting</span>
                <span class="feature-badge">✂️ Face Cropping</span>
                <span class="feature-badge">📥 Download Results</span>
                <span class="feature-badge">⚙️ Tunable Parameters</span>
                <span class="feature-badge">🎨 Custom Box Colors</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h4 style="color:#00f5d4; margin-top:0;">🚀 Future Enhancements</h4>
        <div>
            <span class="feature-badge">🆔 Face Recognition</span>
            <span class="feature-badge">📋 Attendance System</span>
            <span class="feature-badge">😊 Emotion Detection</span>
            <span class="feature-badge">😷 Mask Detection</span>
            <span class="feature-badge">📊 Analytics Dashboard</span>
            <span class="feature-badge">🎥 Real-time WebRTC</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; padding:1rem; color:#4a5568; font-size:0.8rem;">
    <p>👁️ Face Detection System v1.0 &nbsp;|&nbsp; Built with OpenCV & Streamlit &nbsp;|&nbsp; 
    <a href="https://github.com/BKiran27" style="color:#00f5d4; text-decoration:none;">BKiran27</a></p>
</div>
""", unsafe_allow_html=True)
