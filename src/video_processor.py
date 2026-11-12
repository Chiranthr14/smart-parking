"""
Video Processor for SmartPark
Handles video loading, frame extraction, streaming analysis, and rendering.
Optimized for CPU inference with configurable frame skipping.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Generator, List
import cv2
import numpy as np
from PIL import Image

from src.vehicle_detector import VehicleDetector
from src.parking_slot_manager import ParkingSlotManager, ParkingSlot
from src.parking_detector import ParkingDetector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Manages OpenCV video capture and processing pipelines."""

    def __init__(self, frame_skip: int = 2):
        self.frame_skip = max(1, frame_skip)

    @staticmethod
    def get_video_metadata(video_path: str) -> Dict[str, Any]:
        """Extract resolution, fps, total frames, and duration from video."""
        if not video_path or not os.path.exists(video_path):
            return {
                "valid": False,
                "error": f"File does not exist: {video_path}",
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "total_frames": 0,
                "duration_sec": 0.0
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "valid": False,
                "error": f"Failed to open video capture for: {video_path}",
                "width": 0,
                "height": 0,
                "fps": 0.0,
                "total_frames": 0,
                "duration_sec": 0.0
            }

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(total_frames / fps, 2) if fps > 0 else 0.0
        cap.release()

        return {
            "valid": True,
            "error": None,
            "width": width,
            "height": height,
            "fps": round(fps, 2),
            "total_frames": total_frames,
            "duration_sec": duration_sec
        }

    @staticmethod
    def extract_representative_frame(
        video_path: str,
        frame_index: Optional[int] = None,
        as_rgb: bool = True
    ) -> Optional[np.ndarray]:
        """Extract a single frame from the video (e.g. for polygon configuration)."""
        if not video_path or not os.path.exists(video_path):
            return None

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_index is None:
            # Default to 15% into the video or frame 0
            frame_index = min(max(0, int(total_frames * 0.15)), max(0, total_frames - 1))

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            return None

        if as_rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def analyze_zone_snapshot(
        self,
        video_path: str,
        slots: List[ParkingSlot],
        detector: VehicleDetector,
        parking_detector: ParkingDetector,
        sample_frames: int = 15
    ) -> Dict[str, Any]:
        """
        Analyze a short sample sequence of frames to compute robust,
        temporally-smoothed occupancy counts without needing to play the entire video.
        """
        if not video_path or not os.path.exists(video_path):
            return {
                "status": "missing_video",
                "error": "Video file not found",
                "total_spaces": len(slots),
                "vacant_spaces": 0,
                "occupied_spaces": 0,
                "uncertain_spaces": 0,
                "availability_percentage": 0.0
            }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {
                "status": "error",
                "error": "Unable to open video file",
                "total_spaces": len(slots),
                "vacant_spaces": 0,
                "occupied_spaces": 0,
                "uncertain_spaces": 0,
                "availability_percentage": 0.0
            }

        parking_detector.reset_smoothing()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // (sample_frames + 2))
        
        last_eval = None
        annotated_frame = None

        for i in range(sample_frames):
            frame_idx = min(total_frames - 1, i * step)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            h, w = frame.shape[:2]
            detections = detector.detect(frame)
            last_eval = parking_detector.evaluate_frame(slots, detections, w, h)
            
            # Annotate the middle sample frame for snapshot preview
            if i == sample_frames // 2 or annotated_frame is None:
                annotated_bgr = parking_detector.annotate_frame(frame, last_eval)
                annotated_bgr = detector.draw_detections(annotated_bgr, detections)
                annotated_frame = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        cap.release()

        if last_eval is None:
            return {
                "status": "error",
                "error": "Failed to read frames from video",
                "total_spaces": len(slots),
                "vacant_spaces": 0,
                "occupied_spaces": 0,
                "uncertain_spaces": 0,
                "availability_percentage": 0.0
            }

        return {
            "status": "analyzed",
            "total_spaces": last_eval["total_spaces"],
            "vacant_spaces": last_eval["vacant_spaces"],
            "occupied_spaces": last_eval["occupied_spaces"],
            "uncertain_spaces": last_eval["uncertain_spaces"],
            "availability_percentage": last_eval["availability_percentage"],
            "slots": last_eval["slots"],
            "preview_frame": annotated_frame,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
