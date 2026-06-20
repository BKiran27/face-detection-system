"""
Face Detection Module
=====================
Core face detection logic using OpenCV's Haar Cascade Classifier.
Provides functions for detecting, drawing, cropping, and counting faces
in images and video frames.

Author: BKiran27
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


def load_face_cascade() -> cv2.CascadeClassifier:
    """
    Load OpenCV's pre-trained Haar Cascade model for frontal face detection.
    
    Returns:
        cv2.CascadeClassifier: The loaded face cascade classifier.
    
    Raises:
        IOError: If the cascade file cannot be loaded.
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    if face_cascade.empty():
        raise IOError(f"Failed to load Haar Cascade from: {cascade_path}")
    
    return face_cascade


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert a BGR image to grayscale for face detection.
    
    Grayscale conversion improves:
    - Processing speed
    - Memory usage
    - Detection accuracy
    
    Args:
        image: Input BGR image as numpy array.
    
    Returns:
        Grayscale version of the input image.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def detect_faces(
    image: np.ndarray,
    face_cascade: Optional[cv2.CascadeClassifier] = None,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: Tuple[int, int] = (30, 30)
) -> np.ndarray:
    """
    Detect faces in an image using Haar Cascade classifier.
    
    Args:
        image: Input image (BGR or grayscale).
        face_cascade: Pre-loaded cascade classifier. Loads default if None.
        scale_factor: How much the image size is reduced at each image scale.
                      Values closer to 1.0 are more thorough but slower.
        min_neighbors: How many neighbors each candidate rectangle should have
                       to retain it. Higher values = fewer detections but higher quality.
        min_size: Minimum possible face size. Faces smaller than this are ignored.
    
    Returns:
        numpy array of face rectangles, each as (x, y, w, h).
    """
    if face_cascade is None:
        face_cascade = load_face_cascade()
    
    # Convert to grayscale if the image is in color
    if len(image.shape) == 3:
        gray = preprocess_image(image)
    else:
        gray = image
    
    # Equalize histogram for better contrast
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    
    return faces if len(faces) > 0 else np.array([])


def draw_bounding_boxes(
    image: np.ndarray,
    faces: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 128),
    thickness: int = 2,
    show_label: bool = True,
    show_confidence: bool = False
) -> np.ndarray:
    """
    Draw bounding boxes around detected faces with optional labels.
    
    Args:
        image: Input BGR image.
        faces: Array of face rectangles (x, y, w, h).
        color: BGR color for the bounding box.
        thickness: Line thickness in pixels.
        show_label: Whether to show "Face #N" label above each box.
        show_confidence: Whether to show face area as a proxy metric.
    
    Returns:
        Copy of the image with bounding boxes drawn.
    """
    result = image.copy()
    
    for idx, (x, y, w, h) in enumerate(faces):
        # Draw the main rectangle
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        # Draw corner accents for a modern look
        corner_length = min(w, h) // 4
        accent_color = (0, 255, 255)  # Cyan accents
        accent_thickness = thickness + 1
        
        # Top-left corner
        cv2.line(result, (x, y), (x + corner_length, y), accent_color, accent_thickness)
        cv2.line(result, (x, y), (x, y + corner_length), accent_color, accent_thickness)
        
        # Top-right corner
        cv2.line(result, (x + w, y), (x + w - corner_length, y), accent_color, accent_thickness)
        cv2.line(result, (x + w, y), (x + w, y + corner_length), accent_color, accent_thickness)
        
        # Bottom-left corner
        cv2.line(result, (x, y + h), (x + corner_length, y + h), accent_color, accent_thickness)
        cv2.line(result, (x, y + h), (x, y + h - corner_length), accent_color, accent_thickness)
        
        # Bottom-right corner
        cv2.line(result, (x + w, y + h), (x + w - corner_length, y + h), accent_color, accent_thickness)
        cv2.line(result, (x + w, y + h), (x + w, y + h - corner_length), accent_color, accent_thickness)
        
        if show_label:
            label = f"Face #{idx + 1}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            label_thickness = 1
            
            # Get text size for background rectangle
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, label_thickness
            )
            
            # Draw label background
            cv2.rectangle(
                result,
                (x, y - text_h - 10),
                (x + text_w + 8, y),
                color,
                -1  # Filled rectangle
            )
            
            # Draw label text
            cv2.putText(
                result,
                label,
                (x + 4, y - 6),
                font,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA
            )
    
    return result


def draw_face_count_badge(
    image: np.ndarray,
    count: int,
    position: str = "top-right"
) -> np.ndarray:
    """
    Draw a face count badge on the image.
    
    Args:
        image: Input BGR image.
        count: Number of faces detected.
        position: Badge position ("top-right", "top-left", "bottom-right", "bottom-left").
    
    Returns:
        Image with the face count badge.
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    text = f"{count} face{'s' if count != 1 else ''} detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    padding = 10
    badge_w = text_w + 2 * padding
    badge_h = text_h + 2 * padding + baseline
    
    # Calculate position
    if position == "top-right":
        bx = w - badge_w - 10
        by = 10
    elif position == "top-left":
        bx = 10
        by = 10
    elif position == "bottom-right":
        bx = w - badge_w - 10
        by = h - badge_h - 10
    else:  # bottom-left
        bx = 10
        by = h - badge_h - 10
    
    # Draw semi-transparent background
    overlay = result.copy()
    badge_color = (0, 200, 100) if count > 0 else (0, 0, 200)
    cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), badge_color, -1)
    cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)
    
    # Draw border
    cv2.rectangle(result, (bx, by), (bx + badge_w, by + badge_h), badge_color, 2)
    
    # Draw text
    cv2.putText(
        result,
        text,
        (bx + padding, by + padding + text_h),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )
    
    return result


def crop_faces(
    image: np.ndarray,
    faces: np.ndarray,
    padding: int = 20
) -> List[np.ndarray]:
    """
    Crop individual face regions from the image.
    
    Args:
        image: Input BGR image.
        faces: Array of face rectangles (x, y, w, h).
        padding: Extra pixels around each face crop.
    
    Returns:
        List of cropped face images.
    """
    h, w = image.shape[:2]
    cropped_faces = []
    
    for (x, y, fw, fh) in faces:
        # Apply padding with boundary checks
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + fw + padding)
        y2 = min(h, y + fh + padding)
        
        face_crop = image[y1:y2, x1:x2]
        cropped_faces.append(face_crop)
    
    return cropped_faces


def get_face_count(faces: np.ndarray) -> int:
    """
    Return the number of detected faces.
    
    Args:
        faces: Array of face rectangles from detect_faces().
    
    Returns:
        Integer count of detected faces.
    """
    return len(faces) if len(faces) > 0 else 0


def process_video_frame(
    frame: np.ndarray,
    face_cascade: cv2.CascadeClassifier,
    scale_factor: float = 1.1,
    min_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single video frame for face detection.
    
    Args:
        frame: Input video frame (BGR).
        face_cascade: Pre-loaded cascade classifier.
        scale_factor: Scale factor for detection.
        min_neighbors: Minimum neighbors for detection.
    
    Returns:
        Tuple of (processed_frame_with_boxes, detected_faces_array).
    """
    faces = detect_faces(frame, face_cascade, scale_factor, min_neighbors)
    result = draw_bounding_boxes(frame, faces)
    result = draw_face_count_badge(result, get_face_count(faces))
    
    return result, faces


# ─── Standalone CLI usage ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python detect.py <image_path>")
        print("       python detect.py --webcam")
        sys.exit(1)
    
    cascade = load_face_cascade()
    
    if sys.argv[1] == "--webcam":
        print("Starting webcam face detection... Press ESC to quit.")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            sys.exit(1)
        
        while True:
            success, frame = cap.read()
            if not success:
                break
            
            result, faces = process_video_frame(frame, cascade)
            cv2.imshow("Face Detection — Press ESC to quit", result)
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        cap.release()
        cv2.destroyAllWindows()
    else:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Error: Could not read image: {image_path}")
            sys.exit(1)
        
        faces = detect_faces(image, cascade)
        count = get_face_count(faces)
        print(f"Detected {count} face(s) in: {image_path}")
        
        result = draw_bounding_boxes(image, faces)
        result = draw_face_count_badge(result, count)
        
        cv2.imshow("Face Detection — Press any key to close", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
