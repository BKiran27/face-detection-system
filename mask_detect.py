"""
Face Mask Detection Module
============================
Core deep learning face mask detection using OpenCV's DNN module.
Loads the Caffe-based Face Mask Detection model and performs fast inference
without requiring heavy deep learning frameworks like TensorFlow or PyTorch.

Author: BKiran27
"""

import cv2
import numpy as np
import os
import streamlit as st
from typing import List, Tuple, Dict, Any

from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression

# ─── Model Configurations ────────────────────────────────────────────────────────

# Target input shape for the default SSD face mask detection model
TARGET_SHAPE = (260, 260)

# Anchor configurations matching the 260x260 input model
FEATURE_MAP_SIZES = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
ANCHOR_SIZES = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
ANCHOR_RATIOS = [[1, 0.62, 0.42]] * 5

# Generate anchors for decoding bounding boxes
anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SIZES, ANCHOR_RATIOS)
anchors_exp = np.expand_dims(anchors, axis=0)

# Class mapping: 0 -> Mask, 1 -> NoMask
id2class = {0: 'Mask', 1: 'NoMask'}
colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # BGR: 0: Green (Mask), 1: Red (NoMask)


@st.cache_resource
def load_mask_net() -> cv2.dnn.Net:
    """
    Load the pre-trained Caffe face mask detection model using OpenCV DNN.
    Cached for high performance in Streamlit.
    """
    proto_path = os.path.join("models", "face_mask_detection.prototxt")
    model_path = os.path.join("models", "face_mask_detection.caffemodel")
    
    if not os.path.exists(proto_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Required model files not found: \n- {proto_path}\n- {model_path}\n"
            "Please ensure you copied the models correctly."
        )
    
    net = cv2.dnn.readNet(model_path, proto_path)
    return net


def get_outputs_names(net: cv2.dnn.Net) -> List[str]:
    """
    Helper to query all unconnected output layers of the neural network
    compatible with multiple OpenCV versions.
    """
    layers_names = net.getLayerNames()
    unconnected_layers = net.getUnconnectedOutLayers()
    
    # Handle older vs newer OpenCV versions
    try:
        # In newer OpenCV versions, getUnconnectedOutLayers returns a flat list of layer indices
        return [layers_names[i - 1] for i in unconnected_layers]
    except (TypeError, IndexError):
        # In older versions, it might return a nested list
        return [layers_names[i[0] - 1] for i in unconnected_layers]


def detect_masks(
    image: np.ndarray,
    net: cv2.dnn.Net,
    conf_thresh: float = 0.5,
    iou_thresh: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Perform deep learning-based face mask detection on an image using OpenCV DNN.
    
    Args:
        image: Input BGR image.
        net: The loaded OpenCV DNN net.
        conf_thresh: Classification confidence threshold.
        iou_thresh: Intersection Over Union (IoU) threshold for Non-Max Suppresson.
        
    Returns:
        List of dictionaries containing detection details:
        [{
            'box': [xmin, ymin, xmax, ymax],
            'class_id': 0 or 1,
            'class_name': 'Mask' or 'NoMask',
            'confidence': float
        }]
    """
    height, width, _ = image.shape
    
    # 1. Preprocess: Resize, scale pixel values to [0, 1] range, convert to NCHW blob
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0/255.0, size=TARGET_SHAPE)
    net.setInput(blob)
    
    # 2. Run Forward Pass
    y_bboxes_output, y_cls_output = net.forward(get_outputs_names(net))
    
    # Remove the batch dimension (batch size is always 1 for inference)
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    
    # Get highest scoring class and score for each box
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    
    # 3. Apply Single-Class Non-Max Suppression (NMS)
    keep_idxs = single_class_non_max_suppression(
        y_bboxes,
        bbox_max_scores,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh
    )
    
    # 4. Formulate Detections List
    detections = []
    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = int(bbox_max_score_classes[idx])
        bbox = y_bboxes[idx]
        
        # Scale bounding box coordinate ratios back to image dimensions
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)
        
        detections.append({
            'box': [xmin, ymin, xmax, ymax],
            'class_id': class_id,
            'class_name': id2class[class_id],
            'confidence': conf
        })
        
    return detections


def draw_mask_bounding_boxes(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    show_label: bool = True
) -> np.ndarray:
    """
    Draw bounding boxes and class labels (Mask/NoMask) on a BGR image.
    
    Args:
        image: Input BGR image.
        detections: List of detections returned by detect_masks().
        show_label: Whether to print class name and confidence above the box.
        
    Returns:
        Annotated image copy.
    """
    result = image.copy()
    height, width = result.shape[:2]
    
    # Line thickness scaled by image dimensions
    tl = max(2, round(0.0025 * (height + width) * 0.5))
    
    for idx, det in enumerate(detections):
        xmin, ymin, xmax, ymax = det['box']
        class_id = det['class_id']
        conf = det['confidence']
        class_name = det['class_name']
        
        color = colors.get(class_id, (255, 255, 255))
        
        # Draw bounding box
        cv2.rectangle(result, (xmin, ymin), (xmax, ymax), color, tl)
        
        # Draw aesthetic corner accents
        w_box = xmax - xmin
        h_box = ymax - ymin
        corner_length = min(w_box, h_box) // 4
        accent_thickness = tl + 1
        
        # Accent color (Cyan for Mask, Yellow/Orange for NoMask)
        accent_color = (255, 255, 0) if class_id == 0 else (0, 255, 255)
        
        # Top-left corner
        cv2.line(result, (xmin, ymin), (xmin + corner_length, ymin), accent_color, accent_thickness)
        cv2.line(result, (xmin, ymin), (xmin, ymin + corner_length), accent_color, accent_thickness)
        # Top-right corner
        cv2.line(result, (xmax, ymin), (xmax - corner_length, ymin), accent_color, accent_thickness)
        cv2.line(result, (xmax, ymin), (xmax, ymin + corner_length), accent_color, accent_thickness)
        # Bottom-left corner
        cv2.line(result, (xmin, ymax), (xmin + corner_length, ymax), accent_color, accent_thickness)
        cv2.line(result, (xmin, ymax), (xmin, ymax - corner_length), accent_color, accent_thickness)
        # Bottom-right corner
        cv2.line(result, (xmax, ymax), (xmax - corner_length, ymax), accent_color, accent_thickness)
        cv2.line(result, (xmax, ymax), (xmax, ymax - corner_length), accent_color, accent_thickness)
        
        if show_label:
            label = f"{class_name}: {conf:.0%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.4, min(1.0, 0.001 * (w_box + h_box)))
            label_thickness = max(1, tl - 1)
            
            # Text size for drawing background
            (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, label_thickness)
            
            # Background rectangle for text label
            cv2.rectangle(
                result,
                (xmin, ymin - text_h - 10),
                (xmin + text_w + 8, ymin),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                result,
                label,
                (xmin + 4, ymin - 6),
                font,
                font_scale,
                (255, 255, 255),
                label_thickness,
                cv2.LINE_AA
            )
            
    return result


def draw_mask_count_badge(
    image: np.ndarray,
    mask_count: int,
    nomask_count: int,
    position: str = "top-right"
) -> np.ndarray:
    """
    Draw a summary mask-compliance badge on the image.
    """
    result = image.copy()
    h, w = result.shape[:2]
    
    total = mask_count + nomask_count
    compliance = (mask_count / total * 100) if total > 0 else 100.0
    
    text = f"Masks: {mask_count} | No Masks: {nomask_count} ({compliance:.0f}% compliant)"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    padding = 8
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
    
    # Color badge based on compliance rate
    if total == 0:
        badge_color = (128, 128, 128)  # Grey if none detected
    elif compliance >= 80:
        badge_color = (0, 200, 100)    # Green
    elif compliance >= 50:
        badge_color = (0, 165, 255)    # Orange
    else:
        badge_color = (0, 0, 200)      # Red
        
    cv2.rectangle(overlay, (bx, by), (bx + badge_w, by + badge_h), badge_color, -1)
    cv2.addWeighted(overlay, 0.75, result, 0.25, 0, result)
    
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


def crop_mask_faces(
    image: np.ndarray,
    detections: List[Dict[str, Any]],
    padding: int = 15
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Crop face regions from the image based on face mask detections.
    
    Returns:
        List of tuples containing (cropped_face_image, detection_dict)
    """
    h, w = image.shape[:2]
    cropped_faces = []
    
    for det in detections:
        xmin, ymin, xmax, ymax = det['box']
        
        # Apply padding with boundaries check
        x1 = max(0, xmin - padding)
        y1 = max(0, ymin - padding)
        x2 = min(w, xmax + padding)
        y2 = min(h, ymax + padding)
        
        # Check for valid crops
        if x2 > x1 and y2 > y1:
            face_crop = image[y1:y2, x1:x2]
            cropped_faces.append((face_crop, det))
            
    return cropped_faces
