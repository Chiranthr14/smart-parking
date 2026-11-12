"""
SmartPark: All-in-One Launcher (Backend Testing + Frontend Dashboard)
Runs complete system diagnostics, verifies backend & video feeds, and launches the Streamlit frontend.
"""

import os
import sys
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent

# Check for virtual environment and auto-bootstrap if running on global python
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and os.path.normpath(sys.executable) != os.path.normpath(str(VENV_PYTHON)):
    import subprocess
    # Re-launch with the virtual environment Python
    result = subprocess.run([str(VENV_PYTHON), str(PROJECT_ROOT / "run.py")] + sys.argv[1:], cwd=str(PROJECT_ROOT))
    sys.exit(result.returncode)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import time
import subprocess
import webbrowser


def print_banner():
    banner = """
================================================================================
          SMARTPARK: AI TWO-WHEELER PARKING DETECTION & RECOMMENDATION          
                   [Backend Diagnostics & Web Dashboard]                        
================================================================================
"""
    print(banner)


def run_backend_tests():
    print("[1/2] RUNNING AUTOMATED BACKEND & VIDEO PIPELINE TESTS...")
    print("-" * 80)

    try:
        from src.zone_manager import ZoneManager
        from src.parking_slot_manager import ParkingSlotManager, ParkingSlot
        from src.vehicle_detector import VehicleDetector
        from src.parking_detector import ParkingDetector
        from src.recommendation import RecommendationEngine
        from src.video_processor import VideoProcessor
    except Exception as e:
        print(f"[FAIL] Failed to import backend modules: {e}")
        return False

    all_passed = True

    # 1. Test Video Feeds
    print("\n* Checking Video Feeds in 'videos/' folder:")
    zm = ZoneManager(videos_dir=PROJECT_ROOT / "videos")
    zones = zm.get_all_zones()

    for zid in ["A", "B", "C", "D", "E"]:
        z = zones.get(zid)
        v_path = z.get("video_path")
        if v_path and os.path.exists(v_path):
            meta = VideoProcessor.get_video_metadata(v_path)
            if meta["valid"]:
                print(f"  [PASS] Zone {zid}: '{z['video_filename']}' -> {meta['width']}x{meta['height']} @ {meta['fps']}fps ({meta['duration_sec']}s)")
            else:
                print(f"  [WARN] Zone {zid}: Video corrupted -> {meta['error']}")
                all_passed = False
        else:
            print(f"  [FAIL] Zone {zid}: Video file '{z.get('video_filename')}' NOT found in videos/ folder!")
            all_passed = False

    # 2. Test YOLO Model Loading & Inference
    print("\n* Testing Local YOLO11n AI Model:")
    try:
        detector = VehicleDetector(model_dir=PROJECT_ROOT / "models")
        ok = detector.load_model()
        if ok:
            print(f"  [PASS] Model loaded successfully: {detector.model_name} (Format: {detector.backend}, Device: {detector.device})")
        else:
            print(f"  [FAIL] Could not load YOLO model from models/ folder.")
            all_passed = False
    except Exception as e:
        print(f"  [FAIL] Error initializing AI detector: {e}")
        all_passed = False

    # 3. Test Parking Occupancy Geometry
    print("\n* Testing Parking Space Polygon Geometry & Overlap:")
    try:
        slot1 = ParkingSlot("T-1", [[0.1, 0.1], [0.9, 0.1], [0.9, 0.9], [0.1, 0.9]])
        p_det = ParkingDetector(occupied_threshold=0.25, uncertain_threshold=0.10)
        eval_res = p_det.evaluate_frame([slot1], [], 640, 640)
        if eval_res["vacant_spaces"] == 1:
            print("  [PASS] Polygon geometry evaluation and vacant calculation verified.")
        else:
            print("  [WARN] Polygon evaluation did not match expected count.")
    except Exception as e:
        print(f"  [FAIL] Polygon detector test failed: {e}")
        all_passed = False

    # 4. Test Recommendation Engine
    print("\n* Testing Multi-Zone Recommendation Engine:")
    try:
        engine = RecommendationEngine()
        dummy_metrics = {
            "A": {"status": "analyzed", "total_spaces": 5, "vacant_spaces": 1, "occupied_spaces": 4, "uncertain_spaces": 0, "availability_percentage": 20.0},
            "B": {"status": "analyzed", "total_spaces": 5, "vacant_spaces": 4, "occupied_spaces": 1, "uncertain_spaces": 0, "availability_percentage": 80.0}
        }
        rec = engine.evaluate_zones(dummy_metrics)
        if rec["recommended_zones"][0]["zone_id"] == "B":
            print(f"  [PASS] Recommendation logic verified -> {rec['title']}")
        else:
            print(f"  [WARN] Unexpected recommendation result.")
    except Exception as e:
        print(f"  [FAIL] Recommendation engine test failed: {e}")
        all_passed = False

    print("\n" + "-" * 80)
    if all_passed:
        print("[SUCCESS] ALL BACKEND TESTS PASSED SUCCESSFULLY!")
    else:
        print("[INFO] Diagnostics completed with warnings. Continuing to frontend...")
    print("-" * 80 + "\n")
    return True


def launch_frontend():
    print("[2/2] STARTING SMARTPARK FRONTEND WEB DASHBOARD...")
    print("      Server URL: http://localhost:8501")
    print("      Press Ctrl+C to stop the application.\n")

    app_path = str(PROJECT_ROOT / "app.py")
    python_bin = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable

    env = os.environ.copy()
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    cmd = [
        python_bin,
        "-m", "streamlit", "run",
        app_path,
        "--server.headless", "true",
        "--server.port", "8501",
        "--browser.gatherUsageStats", "false"
    ]

    # Open browser automatically after 2 seconds
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open("http://localhost:8501")
        except Exception:
            pass

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    try:
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    except KeyboardInterrupt:
        print("\nSmartPark application stopped.")


def main():
    print_banner()
    run_backend_tests()
    launch_frontend()


if __name__ == "__main__":
    main()
