# 🏍️ SmartPark: AI-Based College Two-Wheeler Parking Detection & Recommendation System

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![AI Model](https://img.shields.io/badge/AI%20Model-YOLO11n%20(ONNX)-green.svg)](https://github.com/ultralytics/ultralytics)
[![UI Framework](https://img.shields.io/badge/Dashboard-Streamlit-red.svg)](https://streamlit.io/)
[![Computer Vision](https://img.shields.io/badge/Vision-OpenCV-brightgreen.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An intelligent computer vision system tailored for college campus two-wheeler parking management. SmartPark processes angled CCTV camera footage across **five parking zones (A–E)**, detects motorcycles and scooters using a lightweight local **YOLO11n** model, calculates individual parking bay occupancy using geometric polygon intersections, and delivers real-time parking recommendations with a single command.

---

## 📌 Key Highlights

- **100% Local & Offline AI:** Runs on consumer CPU/GPU via ONNX Runtime without cloud API subscriptions or fees.
- **Dedicated Two-Wheeler Filtering:** Exclusively classifies motorcycles and scooters (COCO class 3) while ignoring pedestrians, bicycles, cars, and buses.
- **Manual Polygon Configuration:** Interactive polygon marking tool with normalized coordinates (`[0.0, 1.0]`) for resolution-independent scaling.
- **Temporal Smoothing:** Multi-frame rolling window to eliminate flicker caused by rider dismounting or temporary occlusion.
- **Clear Status Color Coding:**
  - 🟢 **Green:** Confidently Vacant
  - 🔴 **Red:** Confidently Occupied
  - ⚪ **Gray:** Uncertain (never assumed vacant)
- **Mathematical Consistency:**
  $$\text{Vacant Spaces} = \text{Total Spaces} - \text{Occupied Spaces} - \text{Uncertain Spaces}$$
- **All-in-One Runner:** Single command executes backend diagnostics, pipeline tests, and launches the interactive dashboard.

---

## 🏗️ Project Architecture

```text
smart-parking/
├── app.py                      # Main Streamlit Web Application Dashboard
├── run.py                      # Master All-in-One Launcher (Diagnostics + App)
├── run.bat                     # Windows 1-click batch launcher
├── run.ps1                     # Windows PowerShell launcher script
├── requirements.txt            # Project dependencies
├── README.md                   # Complete documentation & usage manual
├── .gitignore                  # Git exclusion rules
├── config/
│   ├── zones.json              # Parking bay polygon definitions (Zones A-E)
│   └── settings.json           # Model parameters & occupancy thresholds
├── models/
│   └── yolo11n.onnx            # Lightweight YOLO11n weights (~10.4 MB)
├── videos/                     # Prerecorded CCTV parking footage
│   ├── v1.mp4                  # Zone A
│   ├── v2.mp4                  # Zone B
│   ├── v3.mp4                  # Zone C
│   ├── v4.mp4                  # Zone D
│   └── v5.mp4                  # Zone E
├── src/
│   ├── __init__.py
│   ├── zone_manager.py         # Zone registry and video resolver
│   ├── vehicle_detector.py     # YOLO11n two-wheeler detector
│   ├── parking_slot_manager.py # Coordinate normalization and polygon geometry
│   ├── parking_detector.py     # Shapely polygon intersection and temporal smoother
│   ├── video_processor.py      # OpenCV video decoding and frame annotation
│   └── recommendation.py       # Best zone selection and tie-breaker engine
└── tests/
    ├── test_pipeline.py        # Automated unit test suite
    └── test_ui_tables.py       # UI table validation test
```

---

## 📹 Parking Zone Video Mapping

The system supports five distinct parking zones matching the campus CCTV locations:

| Zone | Video File | Campus Location | Configured Slots |
|---|---|---|---|
| **Zone A** | `v1.mp4` | Engineering Block Stand | 4 Slots |
| **Zone B** | `v2.mp4` | Library Annex Stand | 5 Slots |
| **Zone C** | `v3.mp4` | Science Block Stand | 4 Slots |
| **Zone D** | `v4.mp4` *(or `v 4.mp4`)* | Sports Complex Stand | 5 Slots |
| **Zone E** | `v5.mp4` | Auditorium & Canteen Stand | 4 Slots |

> **Note:** The video file resolver automatically handles filename variations (such as spaces in `v 4.mp4` or different container formats like `.avi` / `.mkv`).

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/mahanthprasad/smart-parking.git
cd smart-parking
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
Run the installation command:
```bash
pip install -r requirements.txt
```

---

## ⚡ How to Run (Single Command)

To run both backend automated tests and launch the frontend dashboard:

```bash
python run.py
```

*(On Windows, you can also double-click `run.bat` or run `.\run.ps1`)*

### What Happens Automatically:
1. **Automated Backend Testing:**
   - Validates all 5 videos in `videos/`.
   - Verifies the YOLO11n AI model and loads it onto CPU/GPU.
   - Tests parking space polygon evaluation.
   - Tests the recommendation engine.
2. **Interactive Web Dashboard Launch:**
   - Starts the Streamlit dashboard on `http://localhost:8501`.
   - Automatically opens your default web browser to the dashboard.

---

## 🖥️ Web Dashboard Features

1. **Main Dashboard:** 
   - Zone overview cards (A–E) showing Total, Vacant, Occupied, and Uncertain counts.
   - Real-time availability progress bars.
   - Dynamic recommendation banner highlighting the best parking zone with transparent reasoning.
2. **Video Analysis Inspector:** 
   - Select any zone to play and analyze recorded CCTV video.
   - Displays bounding boxes with confidence scores on detected two-wheelers.
   - Displays color-coded parking slots (Green = Vacant, Red = Occupied, Gray = Uncertain).
3. **Parking Space Configuration Tool:** 
   - Click to configure or adjust parking bay polygons on representative frames.
   - Fully normalized coordinates allow seamless scaling to any display resolution.
4. **System Status:** 
   - Displays AI model metadata, camera specifications, and device status.

---

## 🧠 Model & Dataset Details

- **Model:** Pretrained YOLO11n exported to ONNX format (size: 10.4 MB).
- **Target Class:** COCO Class ID 3 (`motorcycle`).
- **Class Note:** Standard COCO datasets classify mopeds, scooters, and motorcycles under the `motorcycle` label. SmartPark embraces this to cover all college two-wheelers while filtering out non-target objects (cars, buses, pedestrians).

---

## 🔮 Future Enhancements

- [ ] Live RTSP CCTV camera streaming integration.
- [ ] Automatic camera reconnection & network failover.
- [ ] Student mobile companion app / LED display board integration.

---

## 📄 License
This project is licensed under the MIT License.
