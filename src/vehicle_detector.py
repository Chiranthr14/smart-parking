"""
Vehicle Detector for SmartPark
Uses lightweight, local YOLO11n ONNX inference via ONNX Runtime / OpenCV DNN.
Detects two-wheelers (motorcycles and scooters, COCO class 3: motorcycle).
Strictly excludes pedestrians, cars, bicycles, buses, and trucks.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO 80 Class Names
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Two-wheeler target class in COCO (motorcycle index is 3)
TARGET_CLASS_ID = 3  # motorcycle & scooter
TARGET_CLASS_NAME = "motorcycle"


class VehicleDetector:
    """YOLO11n ONNX vehicle detector for two-wheeler parking occupancy."""

    def __init__(
        self,
        model_name: str = "yolo11n.onnx",
        model_dir: Optional[Path] = None,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        target_classes: Optional[List[int]] = None,
        input_size: Tuple[int, int] = (640, 640)
    ):
        self.model_name = model_name
        self.model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent.parent / "models"
        self.model_path = self.model_dir / self.model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.target_classes = target_classes if target_classes is not None else [TARGET_CLASS_ID]
        self.input_size = input_size
        self.session = None
        self.is_loaded = False
        self.backend = "unknown"
        self.device = "CPU"

    def load_model(self) -> bool:
        """Load the ONNX model into ONNX Runtime session."""
        if self.is_loaded and self.session is not None:
            return True

        if not self.model_path.exists():
            # Check for alternative in current directory
            alt = Path(__file__).resolve().parent.parent / self.model_name
            if alt.exists():
                self.model_path = alt
            else:
                logger.error(f"YOLO ONNX model not found at: {self.model_path}")
                return False

        try:
            import onnxruntime as ort
            providers = ort.get_available_providers()
            # Prefer CUDA if available, fallback to CPUExecutionProvider
            selected_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
            
            self.session = ort.InferenceSession(str(self.model_path), providers=selected_providers)
            self.backend = "ONNXRuntime"
            self.device = "GPU (CUDA)" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"
            self.is_loaded = True
            logger.info(f"Loaded {self.model_name} via {self.backend} on {self.device}")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize onnxruntime: {e}. Trying cv2.dnn fallback...")
            try:
                self.session = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.backend = "OpenCV_DNN"
                self.device = "CPU"
                self.is_loaded = True
                logger.info(f"Loaded {self.model_name} via OpenCV DNN fallback.")
                return True
            except Exception as e2:
                logger.error(f"Failed to load ONNX model via cv2.dnn: {e2}")
                self.is_loaded = False
                return False

    def _letterbox(self, img: np.ndarray, target_shape=(640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """Letterbox resize image to 640x640 while maintaining aspect ratio."""
        h, w = img.shape[:2]
        tw, th = target_shape
        scale = min(th / h, tw / w)
        nw, nh = int(round(w * scale)), int(round(h * scale))

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
        
        pad_x = (tw - nw) // 2
        pad_y = (th - nh) // 2
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        return canvas, scale, (pad_x, pad_y)

    def detect(self, frame: np.ndarray, conf_thresh: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run inference on frame.
        Filters specifically for motorcycles/scooters (COCO id: 3).
        Excludes other classes.
        """
        if not self.is_loaded or self.session is None:
            if not self.load_model():
                return []

        conf = conf_thresh if conf_thresh is not None else self.confidence_threshold
        h, w = frame.shape[:2]

        # Preprocessing: letterbox & normalize
        canvas, scale, (pad_x, pad_y) = self._letterbox(frame, self.input_size)
        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        boxes, scores, class_ids = [], [], []

        try:
            if self.backend == "ONNXRuntime":
                input_name = self.session.get_inputs()[0].name
                output_name = self.session.get_outputs()[0].name
                outputs = self.session.run([output_name], {input_name: blob})[0]
            else:
                # OpenCV DNN
                self.session.setInput(blob)
                outputs = self.session.forward()

            # Output shape: (1, 84, 8400) -> transpose to (8400, 84)
            preds = outputs[0].T

            for row in preds:
                class_scores = row[4:]
                cls_id = int(np.argmax(class_scores))
                score = float(class_scores[cls_id])

                # Filter target classes (class 3 = motorcycle / scooter)
                if score >= conf and cls_id in self.target_classes:
                    cx, cy, bw, bh = row[0:4]
                    # Undo letterbox to original frame coordinates
                    x1 = (cx - bw / 2.0 - pad_x) / scale
                    y1 = (cy - bh / 2.0 - pad_y) / scale
                    x2 = (cx + bw / 2.0 - pad_x) / scale
                    y2 = (cy + bh / 2.0 - pad_y) / scale

                    # Clip to frame boundary
                    x1 = max(0, min(w, x1))
                    y1 = max(0, min(h, y1))
                    x2 = max(0, min(w, x2))
                    y2 = max(0, min(h, y2))

                    boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
                    scores.append(score)
                    class_ids.append(cls_id)

        except Exception as e:
            logger.error(f"Inference error in VehicleDetector: {e}")
            return []

        if len(boxes) == 0:
            return []

        # Apply Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(boxes, scores, conf, self.nms_threshold)
        detections: List[Dict[str, Any]] = []

        if len(indices) > 0:
            for idx in indices:
                # Handle single-scalar or nested array output from cv2.dnn.NMSBoxes
                i = idx[0] if isinstance(idx, (list, tuple, np.ndarray)) else idx
                bx, by, bw, bh = boxes[i]
                c_score = scores[i]
                c_id = class_ids[i]
                c_name = COCO_CLASSES[c_id] if 0 <= c_id < len(COCO_CLASSES) else TARGET_CLASS_NAME

                detections.append({
                    "box": [bx, by, bx + bw, by + bh],  # [x1, y1, x2, y2]
                    "confidence": round(float(c_score), 3),
                    "class_id": c_id,
                    "class_name": c_name
                })

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        color: Tuple[int, int, int] = (0, 165, 255),  # Orange for two-wheelers
        draw_labels: bool = True
    ) -> np.ndarray:
        """Draw bounding boxes and confidence tags on the frame."""
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = map(int, det["box"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            if draw_labels:
                label = f"{det['class_name']} {det['confidence']:.2f}"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - lh - 4)), (x1 + lw, y1), color, -1)
                cv2.putText(
                    annotated,
                    label,
                    (x1, max(lh, y1 - 2)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )
        return annotated
