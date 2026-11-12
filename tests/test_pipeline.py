"""
Pipeline Verification Test for SmartPark
Tests ZoneManager, VehicleDetector, ParkingDetector, and RecommendationEngine.
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zone_manager import ZoneManager
from src.parking_slot_manager import ParkingSlotManager, ParkingSlot
from src.vehicle_detector import VehicleDetector
from src.parking_detector import ParkingDetector, STATUS_OCCUPIED, STATUS_VACANT, STATUS_UNCERTAIN
from src.recommendation import RecommendationEngine
from src.video_processor import VideoProcessor


def test_zone_manager():
    print("\n--- Testing ZoneManager ---")
    zm = ZoneManager()
    zones = zm.get_all_zones()
    assert len(zones) == 5, f"Expected 5 zones, got {len(zones)}"
    for zid, z in zones.items():
        print(f"Zone {zid}: {z['name']} | Video: {z['video_filename']} | Found: {z['video_found']}")
        assert z["video_found"], f"Video file not found for Zone {zid}"
    print("ZoneManager test PASSED.")


def test_parking_slot_and_detector():
    print("\n--- Testing ParkingSlot & ParkingDetector ---")
    # Simulate a slot covering bottom of screen
    slot1 = ParkingSlot(
        slot_id="A-1",
        points_normalized=[[0.1, 0.6], [0.9, 0.6], [0.9, 0.95], [0.1, 0.95]],
        name="Slot 1"
    )
    # Simulate an empty slot
    slot2 = ParkingSlot(
        slot_id="A-2",
        points_normalized=[[0.1, 0.1], [0.4, 0.1], [0.4, 0.3], [0.1, 0.3]],
        name="Slot 2"
    )

    detector = ParkingDetector(occupied_threshold=0.25, uncertain_threshold=0.10)
    
    # Fake vehicle detection overlapping slot 1
    fake_detections = [
        {"box": [100, 600, 400, 850], "confidence": 0.85, "class_id": 3, "class_name": "motorcycle"}
    ]
    eval_res = detector.evaluate_frame([slot1, slot2], fake_detections, frame_width=478, frame_height=850)

    print(f"Evaluated slots: Total={eval_res['total_spaces']}, Occ={eval_res['occupied_spaces']}, Vac={eval_res['vacant_spaces']}, Unc={eval_res['uncertain_spaces']}")
    assert eval_res["total_spaces"] == 2
    assert eval_res["occupied_spaces"] >= 1
    assert eval_res["vacant_spaces"] >= 1
    print("ParkingSlot & ParkingDetector test PASSED.")


def test_recommendation_engine():
    print("\n--- Testing RecommendationEngine ---")
    engine = RecommendationEngine()
    test_metrics = {
        "A": {"status": "analyzed", "total_spaces": 5, "vacant_spaces": 2, "occupied_spaces": 3, "uncertain_spaces": 0, "availability_percentage": 40.0},
        "B": {"status": "analyzed", "total_spaces": 8, "vacant_spaces": 5, "occupied_spaces": 3, "uncertain_spaces": 0, "availability_percentage": 62.5},
        "C": {"status": "analyzed", "total_spaces": 4, "vacant_spaces": 1, "occupied_spaces": 3, "uncertain_spaces": 0, "availability_percentage": 25.0},
        "D": {"status": "analyzed", "total_spaces": 6, "vacant_spaces": 3, "occupied_spaces": 2, "uncertain_spaces": 1, "availability_percentage": 50.0},
        "E": {"status": "analyzed", "total_spaces": 5, "vacant_spaces": 0, "occupied_spaces": 5, "uncertain_spaces": 0, "availability_percentage": 0.0}
    }
    rec = engine.evaluate_zones(test_metrics)
    print("Recommendation output:", rec["title"])
    print("Explanation:", rec["explanation"])
    assert rec["recommended_zones"][0]["zone_id"] == "B", f"Expected B, got {rec['recommended_zones'][0]['zone_id']}"
    print("RecommendationEngine test PASSED.")


if __name__ == "__main__":
    test_zone_manager()
    test_parking_slot_and_detector()
    test_recommendation_engine()
    print("\n ALL UNIT TESTS PASSED!")
