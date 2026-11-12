"""
Parking Slot Manager for SmartPark
Handles creation, normalization, scaling, and geometric properties of parking slot polygons.
"""

from typing import List, Dict, Any, Tuple, Optional
import numpy as np


class ParkingSlot:
    """Represents an individual two-wheeler parking space."""

    def __init__(
        self,
        slot_id: str,
        points_normalized: List[List[float]],
        name: Optional[str] = None
    ):
        self.slot_id = str(slot_id)
        self.name = name or f"Slot {slot_id}"
        # Clamp normalized points to [0.0, 1.0]
        self.points_normalized = [
            [max(0.0, min(1.0, float(p[0]))), max(0.0, min(1.0, float(p[1])))]
            for p in points_normalized
        ]

    def to_pixel_points(self, width: int, height: int) -> np.ndarray:
        """Scale normalized points to pixel coordinates for a given frame dimension."""
        pixel_pts = [
            [int(round(p[0] * width)), int(round(p[1] * height))]
            for p in self.points_normalized
        ]
        return np.array(pixel_pts, dtype=np.int32)

    def get_centroid(self, width: int, height: int) -> Tuple[int, int]:
        """Compute the centroid (x, y) in pixel coordinates for drawing text labels."""
        pts = self.to_pixel_points(width, height)
        if len(pts) == 0:
            return (0, 0)
        cx = int(np.mean(pts[:, 0]))
        cy = int(np.mean(pts[:, 1]))
        return (cx, cy)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON storage."""
        return {
            "slot_id": self.slot_id,
            "name": self.name,
            "points_normalized": self.points_normalized
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParkingSlot":
        return cls(
            slot_id=data.get("slot_id", "1"),
            points_normalized=data.get("points_normalized", []),
            name=data.get("name")
        )

    @classmethod
    def from_pixel_points(
        cls,
        slot_id: str,
        pixel_points: List[Tuple[float, float]],
        ref_width: int,
        ref_height: int,
        name: Optional[str] = None
    ) -> "ParkingSlot":
        """Factory method to construct a ParkingSlot from raw pixel points."""
        w = max(1.0, float(ref_width))
        h = max(1.0, float(ref_height))
        norm_pts = [
            [float(p[0]) / w, float(p[1]) / h]
            for p in pixel_points
        ]
        return cls(slot_id=slot_id, points_normalized=norm_pts, name=name)


class ParkingSlotManager:
    """Manages collections of parking slots for a zone."""

    def __init__(self, slots_data: Optional[List[Dict[str, Any]]] = None):
        self.slots: List[ParkingSlot] = []
        if slots_data:
            self.load_from_data(slots_data)

    def load_from_data(self, slots_data: List[Dict[str, Any]]):
        self.slots = []
        for s in slots_data:
            if "points_normalized" in s and len(s["points_normalized"]) >= 3:
                self.slots.append(ParkingSlot.from_dict(s))
            elif "points" in s and len(s["points"]) >= 3:
                # Backwards-compatibility if raw points were saved
                self.slots.append(ParkingSlot(
                    slot_id=s.get("slot_id", "1"),
                    points_normalized=s.get("points"),
                    name=s.get("name")
                ))

    def add_slot(self, slot: ParkingSlot):
        self.slots.append(slot)

    def remove_slot(self, slot_id: str) -> bool:
        initial_len = len(self.slots)
        self.slots = [s for s in self.slots if s.slot_id != str(slot_id)]
        return len(self.slots) < initial_len

    def clear(self):
        self.slots = []

    def count(self) -> int:
        return len(self.slots)

    def to_list(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self.slots]
