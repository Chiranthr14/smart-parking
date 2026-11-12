"""
Parking Occupancy Detector for SmartPark
Evaluates intersection between detected two-wheelers and parking slot polygons.
Includes temporal smoothing, uncertainty calculation, and visual annotation.
"""

import logging
from collections import deque
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from shapely.geometry import Polygon, box

from src.parking_slot_manager import ParkingSlot

logger = logging.getLogger(__name__)

# Standard Status Constants
STATUS_OCCUPIED = "occupied"
STATUS_VACANT = "vacant"
STATUS_UNCERTAIN = "uncertain"

# BGR Colors for OpenCV rendering
COLOR_OCCUPIED = (0, 0, 220)       # Red
COLOR_VACANT = (40, 200, 40)       # Green
COLOR_UNCERTAIN = (128, 128, 128)  # Gray


class TemporalSmoother:
    """Maintains a rolling window of occupancy classifications to reduce flicker."""

    def __init__(self, window_size: int = 7, occupied_ratio: float = 0.60, vacant_ratio: float = 0.60):
        self.window_size = max(1, window_size)
        self.occupied_ratio = occupied_ratio
        self.vacant_ratio = vacant_ratio
        # Map slot_id -> deque of classifications
        self.history: Dict[str, deque] = {}

    def update(self, slot_id: str, raw_status: str) -> str:
        if slot_id not in self.history:
            self.history[slot_id] = deque(maxlen=self.window_size)
        
        self.history[slot_id].append(raw_status)
        window = list(self.history[slot_id])
        n = len(window)

        if n == 0:
            return STATUS_UNCERTAIN

        occ_count = window.count(STATUS_OCCUPIED)
        vac_count = window.count(STATUS_VACANT)

        if (occ_count / n) >= self.occupied_ratio:
            return STATUS_OCCUPIED
        elif (vac_count / n) >= self.vacant_ratio:
            return STATUS_VACANT
        else:
            # Cannot be confidently determined -> strictly UNCERTAIN (never assume vacant!)
            return STATUS_UNCERTAIN

    def reset(self):
        self.history.clear()


class ParkingDetector:
    """Determines parking occupancy for a zone's slots based on vehicle detections."""

    def __init__(
        self,
        occupied_threshold: float = 0.25,
        uncertain_threshold: float = 0.10,
        enable_smoothing: bool = True,
        smoothing_window: int = 7
    ):
        self.occupied_threshold = occupied_threshold
        self.uncertain_threshold = uncertain_threshold
        self.enable_smoothing = enable_smoothing
        self.smoother = TemporalSmoother(window_size=smoothing_window)

    def reset_smoothing(self):
        """Reset historical window (e.g. when switching zones or restarting video)."""
        self.smoother.reset()

    def evaluate_frame(
        self,
        slots: List[ParkingSlot],
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int
    ) -> Dict[str, Any]:
        """
        Evaluate occupancy of each slot in the frame.
        Returns slot statuses, counts, and metrics.
        """
        slot_results = []
        occupied_count = 0
        vacant_count = 0
        uncertain_count = 0

        # Pre-build shapely boxes for all vehicle detections
        veh_boxes = []
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            try:
                b = box(x1, y1, x2, y2)
                veh_boxes.append((b, det))
            except Exception:
                continue

        for slot in slots:
            pixel_pts = slot.to_pixel_points(frame_width, frame_height)
            if len(pixel_pts) < 3:
                continue

            try:
                poly = Polygon(pixel_pts)
                if not poly.is_valid:
                    poly = poly.buffer(0)
            except Exception:
                continue

            slot_area = poly.area
            if slot_area <= 0:
                continue

            max_overlap = 0.0
            matched_vehicle = None

            # Calculate intersection over slot area
            for b_geom, det in veh_boxes:
                try:
                    if poly.intersects(b_geom):
                        inter_area = poly.intersection(b_geom).area
                        overlap_ratio = inter_area / slot_area
                        if overlap_ratio > max_overlap:
                            max_overlap = overlap_ratio
                            matched_vehicle = det
                except Exception:
                    continue

            # Determine instantaneous raw classification
            if max_overlap >= self.occupied_threshold:
                raw_status = STATUS_OCCUPIED
            elif max_overlap >= self.uncertain_threshold:
                raw_status = STATUS_UNCERTAIN
            else:
                raw_status = STATUS_VACANT

            # Apply temporal smoothing if enabled
            if self.enable_smoothing:
                final_status = self.smoother.update(slot.slot_id, raw_status)
            else:
                final_status = raw_status

            # Count statuses
            if final_status == STATUS_OCCUPIED:
                occupied_count += 1
            elif final_status == STATUS_UNCERTAIN:
                uncertain_count += 1
            else:
                # Vacant
                pass

            slot_results.append({
                "slot_id": slot.slot_id,
                "name": slot.name,
                "status": final_status,
                "raw_status": raw_status,
                "overlap_ratio": round(max_overlap, 3),
                "pixel_points": pixel_pts,
                "centroid": slot.get_centroid(frame_width, frame_height),
                "matched_vehicle": matched_vehicle
            })

        total_spaces = len(slots)
        # Consistent formula: Vacant = Total - Occupied - Uncertain
        vacant_count = max(0, total_spaces - occupied_count - uncertain_count)
        availability_pct = round((vacant_count / total_spaces * 100.0), 1) if total_spaces > 0 else 0.0

        return {
            "total_spaces": total_spaces,
            "occupied_spaces": occupied_count,
            "vacant_spaces": vacant_count,
            "uncertain_spaces": uncertain_count,
            "availability_percentage": availability_pct,
            "slots": slot_results
        }

    def annotate_frame(
        self,
        frame: np.ndarray,
        evaluation: Dict[str, Any],
        draw_vehicle_boxes: bool = True,
        draw_slot_polygons: bool = True
    ) -> np.ndarray:
        """Draw parking slots and detection statuses on the video frame."""
        annotated = frame.copy()
        overlay = frame.copy()
        h, w = frame.shape[:2]

        for s in evaluation.get("slots", []):
            pts = s["pixel_points"]
            status = s["status"]

            if status == STATUS_OCCUPIED:
                color = COLOR_OCCUPIED
            elif status == STATUS_UNCERTAIN:
                color = COLOR_UNCERTAIN
            else:
                color = COLOR_VACANT

            if draw_slot_polygons:
                # Semi-transparent polygon fill
                cv2.fillPoly(overlay, [pts], color)
                # Crisp outline border
                cv2.polylines(annotated, [pts], isClosed=True, color=color, thickness=2)

                # Centroid badge
                cx, cy = s["centroid"]
                badge_text = f"{s['slot_id']}"
                (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (cx - 4, cy - th - 4), (cx + tw + 4, cy + 4), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    badge_text,
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        # Blend semi-transparent fill with outline frame
        if draw_slot_polygons:
            alpha = 0.35
            cv2.addWeighted(overlay, alpha, annotated, 1 - alpha, 0, annotated)

        return annotated
