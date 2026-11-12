"""
SmartPark: AI-Based College Two-Wheeler Parking Detection and Recommendation System
Modern Streamlit Dashboard
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
from PIL import Image
import streamlit as st

def render_markdown_table(rows: List[Dict[str, Any]]) -> str:
    """Renders a list of dictionaries as a GitHub-flavored Markdown table without PyArrow."""
    if not rows:
        return "_No records configured._"
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(h, "")) for h in headers) + " |")
    return "\n".join(lines)


# Configure page
st.set_page_config(
    page_title="SmartPark - Two-Wheeler Parking AI",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Project root path setup
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.zone_manager import ZoneManager, load_settings
from src.parking_slot_manager import ParkingSlotManager, ParkingSlot
from src.vehicle_detector import VehicleDetector
from src.parking_detector import ParkingDetector, STATUS_OCCUPIED, STATUS_VACANT, STATUS_UNCERTAIN
from src.recommendation import RecommendationEngine
from src.video_processor import VideoProcessor

# Try to import streamlit-drawable-canvas for interactive polygon drawing
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except Exception:
    CANVAS_AVAILABLE = False


# Custom CSS for Modern Clean Dashboard
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 16px;
        border-left: 5px solid #007bff;
        box-shadow: 0 2px 4px rgba(0,0,0,0.06);
        margin-bottom: 12px;
    }
    .status-badge-vacant {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .status-badge-occupied {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .status-badge-uncertain {
        background-color: #6c757d;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85em;
    }
    .rec-box {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
        border: 2px solid #4caf50;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 20px;
    }
    .rec-title {
        color: #1b5e20;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
    .disclaimer-banner {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #856404;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
@st.cache_resource
def get_zone_manager():
    return ZoneManager()

@st.cache_resource
def get_vehicle_detector():
    vd = VehicleDetector()
    vd.load_model()
    return vd

@st.cache_resource
def get_recommendation_engine():
    return RecommendationEngine()

zone_mgr = get_zone_manager()
detector = get_vehicle_detector()
rec_engine = get_recommendation_engine()
settings = load_settings()

if "zone_analysis_cache" not in st.session_state:
    st.session_state.zone_analysis_cache = {}

if "analysis_running" not in st.session_state:
    st.session_state.analysis_running = False


# Sidebar Configuration & Controls
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/motorcycle.png", width=64)
    st.title("SmartPark AI")
    st.caption("College Two-Wheeler Parking System")
    st.divider()

    st.subheader("⚙️ Detection Parameters")
    conf_thresh = st.slider("Vehicle Confidence", 0.15, 0.90, 0.35, 0.05,
                            help="Minimum YOLO detection confidence for two-wheelers")
    occ_thresh = st.slider("Occupancy Overlap Ratio", 0.10, 0.70, 0.25, 0.05,
                           help="Slot overlap ratio above which space is classified as Occupied")
    unc_thresh = st.slider("Uncertainty Lower Threshold", 0.05, 0.30, 0.10, 0.02,
                           help="Overlap ratio below which space is safely classified as Vacant")
    frame_skip = st.slider("Frame Skip (Speed)", 1, 6, 2, 1,
                           help="Process every Nth frame for smooth CPU performance")
    enable_smoothing = st.checkbox("Temporal Smoothing", value=True,
                                  help="Multi-frame history to prevent flickering")

    st.divider()
    st.info(f"**AI Engine:** {detector.model_name}\n\n**Device:** {detector.device}\n\n**Backend:** {detector.backend}")
    st.caption("Class 3: Motorcycle & Scooter (COCO)")


# Main Navigation Tabs
tab_dash, tab_video, tab_config, tab_status = st.tabs([
    "📊 Main Dashboard & Recommendations",
    "📹 Video Analysis",
    "📐 Parking Space Configuration",
    "⚙️ System Status"
])


# ==============================================================================
# TAB 1: MAIN DASHBOARD & RECOMMENDATION PANEL
# ==============================================================================
with tab_dash:
    st.markdown("### 🏍️ College Two-Wheeler Parking Dashboard")
    
    st.markdown("""
    <div class="disclaimer-banner">
        ⚠️ <strong>Prerecorded CCTV Analysis Mode:</strong> Availability metrics are evaluated locally from recorded camera footage. Live RTSP streaming is reserved for future deployment.
    </div>
    """, unsafe_allow_html=True)

    zones = zone_mgr.get_all_zones()

    # Quick Multi-Zone Analyzer button
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        run_all = st.button("⚡ Run Multi-Zone Analysis", type="primary", use_container_width=True)
    with col_info:
        st.caption("Analyzes footage across all 5 parking zones (A, B, C, D, E) to produce immediate occupancy figures and parking recommendations.")

    # Execute analysis across all 5 zones if triggered or if cache is empty
    if run_all or not st.session_state.zone_analysis_cache:
        with st.spinner("Analyzing camera footage across Zone A, B, C, D, E..."):
            p_detector = ParkingDetector(
                occupied_threshold=occ_thresh,
                uncertain_threshold=unc_thresh,
                enable_smoothing=enable_smoothing
            )
            v_processor = VideoProcessor(frame_skip=frame_skip)

            results = {}
            for zid, zdata in zones.items():
                slot_objs = [ParkingSlot.from_dict(s) for s in zdata.get("slots", [])]
                v_path = zdata.get("video_path")
                if not v_path or not os.path.exists(v_path):
                    results[zid] = {
                        "name": zdata["name"],
                        "status": "missing_video",
                        "total_spaces": len(slot_objs),
                        "vacant_spaces": 0,
                        "occupied_spaces": 0,
                        "uncertain_spaces": 0,
                        "availability_percentage": 0.0,
                        "error_message": f"Video file {zdata['video_filename']} not found"
                    }
                elif len(slot_objs) == 0:
                    results[zid] = {
                        "name": zdata["name"],
                        "status": "unconfigured",
                        "total_spaces": 0,
                        "vacant_spaces": 0,
                        "occupied_spaces": 0,
                        "uncertain_spaces": 0,
                        "availability_percentage": 0.0,
                        "error_message": "No parking slots configured"
                    }
                else:
                    analysis = v_processor.analyze_zone_snapshot(
                        video_path=v_path,
                        slots=slot_objs,
                        detector=detector,
                        parking_detector=p_detector,
                        sample_frames=12
                    )
                    results[zid] = {
                        "name": zdata["name"],
                        "status": analysis["status"],
                        "total_spaces": analysis["total_spaces"],
                        "vacant_spaces": analysis["vacant_spaces"],
                        "occupied_spaces": analysis["occupied_spaces"],
                        "uncertain_spaces": analysis["uncertain_spaces"],
                        "availability_percentage": analysis["availability_percentage"],
                        "preview_frame": analysis.get("preview_frame"),
                        "timestamp": analysis.get("timestamp", time.strftime("%H:%M:%S"))
                    }

            st.session_state.zone_analysis_cache = results

    cache = st.session_state.zone_analysis_cache

    # Recommendation Panel
    rec_result = rec_engine.evaluate_zones(cache)

    if rec_result["status"] == "available":
        st.markdown(f"""
        <div class="rec-box">
            <div class="rec-title">🎯 {rec_result['title']}</div>
            <p style="font-size: 1.1rem; color: #2e7d32; margin-bottom: 6px;"><strong>{rec_result['explanation']}</strong></p>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">{rec_result['disclaimer']}</p>
        </div>
        """, unsafe_allow_html=True)
    elif rec_result["status"] == "full":
        st.error(f"🛑 **{rec_result['title']}** — {rec_result['explanation']}")
    else:
        st.warning(f"⚠️ **{rec_result['title']}** — {rec_result['explanation']}")

    st.markdown("#### 🏢 Parking Zone Availability Overview")

    # 5 Zone Cards
    zone_cols = st.columns(5)
    for idx, zid in enumerate(["A", "B", "C", "D", "E"]):
        zdata = zones.get(zid, {})
        zmetric = cache.get(zid, {})
        col = zone_cols[idx]

        with col:
            total = zmetric.get("total_spaces", len(zdata.get("slots", [])))
            vacant = zmetric.get("vacant_spaces", 0)
            occupied = zmetric.get("occupied_spaces", 0)
            uncertain = zmetric.get("uncertain_spaces", 0)
            avail_pct = zmetric.get("availability_percentage", 0.0)

            # Zone Card Container
            with st.container(border=True):
                st.subheader(f"Zone {zid}")
                st.caption(zdata.get("description", "Parking Stand"))

                # Big Vacant Metric
                st.metric(
                    label="Vacant Spaces",
                    value=f"{vacant} / {total}",
                    delta=f"{avail_pct}% free" if total > 0 else "0%"
                )

                # Progress bar
                st.progress(float(avail_pct) / 100.0 if total > 0 else 0.0)

                # Breakdown
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<span class='status-badge-occupied'>Occ: {occupied}</span>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<span class='status-badge-uncertain'>Unc: {uncertain}</span>", unsafe_allow_html=True)

                st.write("")
                # Show snapshot preview if available
                if "preview_frame" in zmetric and zmetric["preview_frame"] is not None:
                    st.image(zmetric["preview_frame"], caption=f"Zone {zid} Snapshot", use_container_width=True)
                else:
                    st.caption(f"Video: {zdata.get('video_filename')}")


# ==============================================================================
# TAB 2: VIDEO ANALYSIS & INSPECTOR
# ==============================================================================
with tab_video:
    st.markdown("### 📹 CCTV Parking Video Inspector & Playback")

    v_col1, v_col2 = st.columns([1, 2])

    with v_col1:
        selected_zone_vid = st.selectbox(
            "Select Parking Zone for Inspection:",
            ["A", "B", "C", "D", "E"],
            format_func=lambda x: f"Zone {x} ({zones[x]['name']})",
            key="vid_zone_select"
        )
        current_zone_data = zones[selected_zone_vid]
        st.write(f"**Location:** {current_zone_data['description']}")
        st.write(f"**Video File:** `{current_zone_data['video_filename']}`")

        # Video metadata
        v_meta = VideoProcessor.get_video_metadata(current_zone_data.get("video_path"))
        if v_meta["valid"]:
            st.write(f"**Resolution:** {v_meta['width']} × {v_meta['height']}")
            st.write(f"**FPS:** {v_meta['fps']} | **Duration:** {v_meta['duration_sec']}s ({v_meta['total_frames']} frames)")
        else:
            st.error(f"Video error: {v_meta['error']}")

        slot_count = len(current_zone_data.get("slots", []))
        st.write(f"**Configured Slots:** {slot_count}")

        st.divider()
        analyze_btn = st.button("▶️ Start Video Detection Analysis", type="primary", use_container_width=True)

    with v_col2:
        video_display = st.empty()
        stats_display = st.empty()

        if analyze_btn:
            v_path = current_zone_data.get("video_path")
            if not v_path or not os.path.exists(v_path):
                st.error("Video file not found.")
            elif slot_count == 0:
                st.warning("Please configure parking slots first in the Configuration tab.")
            else:
                cap = cv2.VideoCapture(v_path)
                p_detector = ParkingDetector(
                    occupied_threshold=occ_thresh,
                    uncertain_threshold=unc_thresh,
                    enable_smoothing=enable_smoothing
                )
                slot_objs = [ParkingSlot.from_dict(s) for s in current_zone_data["slots"]]

                frame_idx = 0
                st.caption("Processing video stream frame by frame with AI inference...")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    frame_idx += 1
                    if frame_idx % frame_skip != 0:
                        continue

                    h, w = frame.shape[:2]

                    # 1. Detect two-wheelers
                    detections = detector.detect(frame, conf_thresh=conf_thresh)

                    # 2. Evaluate parking slot occupancy
                    eval_res = p_detector.evaluate_frame(slot_objs, detections, w, h)

                    # 3. Draw annotations
                    annotated = p_detector.annotate_frame(frame, eval_res)
                    annotated = detector.draw_detections(annotated, detections)

                    # Convert to RGB for Streamlit
                    frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_display.image(frame_rgb, caption=f"Zone {selected_zone_vid} Analysis (Frame {frame_idx})", use_container_width=True)

                    # Update statistics live
                    stats_display.markdown(f"""
                    **Frame Status:** Total: `{eval_res['total_spaces']}` | 
                    🟢 Vacant: `{eval_res['vacant_spaces']}` | 
                    🔴 Occupied: `{eval_res['occupied_spaces']}` | 
                    ⚪ Uncertain: `{eval_res['uncertain_spaces']}` | 
                    Availability: `{eval_res['availability_percentage']}%`
                    """)

                    # Brief yield for UI responsiveness
                    time.sleep(0.02)

                cap.release()
                st.success(f"Video analysis completed for Zone {selected_zone_vid}!")
        else:
            # Show initial frame preview
            v_path = current_zone_data.get("video_path")
            if v_path and os.path.exists(v_path):
                rep_frame = VideoProcessor.extract_representative_frame(v_path, as_rgb=True)
                if rep_frame is not None:
                    # Draw current configured slots on preview
                    slot_objs = [ParkingSlot.from_dict(s) for s in current_zone_data["slots"]]
                    p_det = ParkingDetector(occupied_threshold=occ_thresh, uncertain_threshold=unc_thresh)
                    eval_dummy = p_det.evaluate_frame(slot_objs, [], rep_frame.shape[1], rep_frame.shape[0])
                    bgr_preview = cv2.cvtColor(rep_frame, cv2.COLOR_RGB2BGR)
                    annotated_preview = p_det.annotate_frame(bgr_preview, eval_dummy)
                    video_display.image(cv2.cvtColor(annotated_preview, cv2.COLOR_BGR2RGB),
                                        caption=f"Zone {selected_zone_vid} - Configured Slots Preview",
                                        use_container_width=True)


# ==============================================================================
# TAB 3: PARKING SPACE CONFIGURATION
# ==============================================================================
with tab_config:
    st.markdown("### 📐 Interactive Parking Space Configuration Tool")
    st.write("Mark individual parking slots on CCTV footage for each zone. Slot coordinates are normalized to support any display or native resolution.")

    cfg_zone_id = st.selectbox(
        "Select Zone to Configure:",
        ["A", "B", "C", "D", "E"],
        format_func=lambda x: f"Zone {x} ({zones[x]['name']})",
        key="cfg_zone_select"
    )

    target_zone = zone_mgr.get_zone(cfg_zone_id)
    target_vpath = target_zone.get("video_path")

    if not target_vpath or not os.path.exists(target_vpath):
        st.error(f"Cannot configure Zone {cfg_zone_id}: Video file `{target_zone.get('video_filename')}` not found.")
    else:
        # Load representative frame
        rep_frame = VideoProcessor.extract_representative_frame(target_vpath, as_rgb=True)
        if rep_frame is None:
            st.error("Failed to extract representative video frame.")
        else:
            fh, fw = rep_frame.shape[:2]
            slot_mgr = ParkingSlotManager(target_zone.get("slots", []))

            col_canvas, col_mgmt = st.columns([3, 2])

            with col_canvas:
                st.subheader(f"Camera View: Zone {cfg_zone_id} ({fw} × {fh} px)")

                # If streamlit-drawable-canvas is available, provide interactive polygon drawing
                if CANVAS_AVAILABLE:
                    st.caption("👉 Use the Polygon tool below to click corners around a parking space. Double-click or close the polygon to finish.")
                    
                    # Resize frame for comfortable canvas display
                    canvas_w = min(fw, 640)
                    canvas_h = int(fh * (canvas_w / fw))
                    display_pil = Image.fromarray(rep_frame).resize((canvas_w, canvas_h))

                    canvas_result = st_canvas(
                        fill_color="rgba(40, 167, 69, 0.3)",
                        stroke_width=2,
                        stroke_color="#00ff00",
                        background_image=display_pil,
                        update_streamlit=True,
                        height=canvas_h,
                        width=canvas_w,
                        drawing_mode="polygon",
                        key=f"canvas_zone_{cfg_zone_id}"
                    )

                    if canvas_result.json_data is not None:
                        objects = canvas_result.json_data.get("objects", [])
                        if objects:
                            st.info(f"Drawn Polygons on Canvas: {len(objects)}")
                            if st.button("➕ Convert Canvas Polygons to Zone Slots", key=f"btn_conv_{cfg_zone_id}"):
                                for i, obj in enumerate(objects):
                                    if obj.get("type") == "path":
                                        path = obj.get("path", [])
                                        pts = []
                                        for p_cmd in path:
                                            if len(p_cmd) >= 3 and p_cmd[0] in ['M', 'L']:
                                                # Convert canvas coordinates to normalized [0.0, 1.0]
                                                nx = float(p_cmd[1]) / float(canvas_w)
                                                ny = float(p_cmd[2]) / float(canvas_h)
                                                pts.append([max(0.0, min(1.0, nx)), max(0.0, min(1.0, ny))])
                                        if len(pts) >= 3:
                                            new_id = f"{cfg_zone_id}-{slot_mgr.count() + 1}"
                                            slot_mgr.add_slot(ParkingSlot(new_id, pts, name=f"Slot {new_id}"))
                                
                                zone_mgr.update_zone_slots(cfg_zone_id, slot_mgr.to_list(), reference_resolution=(fw, fh))
                                st.success(f"Saved {slot_mgr.count()} slots to Zone {cfg_zone_id} configuration!")
                                st.rerun()

                else:
                    # Fallback preview if canvas component not installed
                    bgr_prev = cv2.cvtColor(rep_frame, cv2.COLOR_RGB2BGR)
                    p_det = ParkingDetector()
                    eval_prev = p_det.evaluate_frame(slot_mgr.slots, [], fw, fh)
                    annotated_prev = p_det.annotate_frame(bgr_prev, eval_prev)
                    st.image(cv2.cvtColor(annotated_prev, cv2.COLOR_BGR2RGB), use_container_width=True)

            with col_mgmt:
                st.subheader(f"Configured Slots ({slot_mgr.count()})")

                # Display table of currently saved slots
                if slot_mgr.count() > 0:
                    slot_rows = []
                    for s in slot_mgr.slots:
                        slot_rows.append({
                            "Slot ID": s.slot_id,
                            "Points": len(s.points_normalized),
                            "Coords": str([[round(p[0], 2), round(p[1], 2)] for p in s.points_normalized[:2]]) + "..."
                        })
                    st.markdown(render_markdown_table(slot_rows))

                    # Delete slot
                    slot_to_del = st.selectbox("Select slot to delete:", [s.slot_id for s in slot_mgr.slots], key="del_slot_sel")
                    if st.button("🗑️ Delete Selected Slot", key="btn_del_slot"):
                        slot_mgr.remove_slot(slot_to_del)
                        zone_mgr.update_zone_slots(cfg_zone_id, slot_mgr.to_list())
                        st.success(f"Deleted slot {slot_to_del}.")
                        st.rerun()

                    if st.button("⚠️ Clear All Slots for this Zone", key="btn_clear_slots"):
                        slot_mgr.clear()
                        zone_mgr.update_zone_slots(cfg_zone_id, [])
                        st.warning("Cleared all slots.")
                        st.rerun()

                else:
                    st.info("No parking slots configured for this zone yet.")

                st.divider()
                st.markdown("#### ➕ Manual Normalized Coordinates Adder")
                with st.expander("Add Custom Slot Manually"):
                    m_slot_id = st.text_input("Slot ID", value=f"{cfg_zone_id}-{slot_mgr.count() + 1}")
                    st.caption("Normalized bounds (0.0 to 1.0):")
                    c_x1, c_x2 = st.columns(2)
                    with c_x1:
                        m_x1 = st.slider("X Min", 0.0, 1.0, 0.20, 0.02)
                        m_y1 = st.slider("Y Min", 0.0, 1.0, 0.50, 0.02)
                    with c_x2:
                        m_x2 = st.slider("X Max", 0.0, 1.0, 0.80, 0.02)
                        m_y2 = st.slider("Y Max", 0.0, 1.0, 0.75, 0.02)

                    if st.button("Save Manual Slot", key="btn_save_manual"):
                        pts = [
                            [m_x1, m_y1],
                            [m_x2, m_y1],
                            [m_x2, m_y2],
                            [m_x1, m_y2]
                        ]
                        slot_mgr.add_slot(ParkingSlot(m_slot_id, pts))
                        zone_mgr.update_zone_slots(cfg_zone_id, slot_mgr.to_list(), reference_resolution=(fw, fh))
                        st.success(f"Saved slot {m_slot_id}!")
                        st.rerun()


# ==============================================================================
# TAB 4: SYSTEM STATUS & ARCHITECTURE
# ==============================================================================
with tab_status:
    st.markdown("### ⚙️ System Status & Hardware Architecture")

    s1, s2 = st.columns(2)

    with s1:
        st.markdown("#### 🧠 AI Model Status")
        st.write(f"- **Model Architecture:** YOLO11n (Nano)")
        st.write(f"- **Model Format:** ONNX Runtime / OpenCV DNN")
        st.write(f"- **Local Model File:** `models/yolo11n.onnx` (10.4 MB)")
        st.write(f"- **Inference Execution Device:** {detector.device}")
        st.write(f"- **Backend Driver:** {detector.backend}")
        st.write(f"- **Target Vehicle Class:** COCO Class 3 (`motorcycle`)")
        st.write(f"- **Excluded Classes:** Cars (2), Buses (5), Trucks (7), Bicycles (1), Persons (0)")

        st.info("""
        **Two-Wheeler Class Note:**
        Standard YOLO COCO pretrained weights recognize scooters, mopeds, and motorcycles collectively under the `motorcycle` class (id: 3). The system correctly includes all two-wheelers while filtering out non-target objects.
        """)

    with s2:
        st.markdown("#### 📹 Video Feeds Inspection")
        vid_rows = []
        for zid in ["A", "B", "C", "D", "E"]:
            z = zones[zid]
            vp = z.get("video_path")
            meta = VideoProcessor.get_video_metadata(vp)
            vid_rows.append({
                "Zone": f"Zone {zid}",
                "Filename": z.get("video_filename"),
                "Resolution": f"{meta['width']}x{meta['height']}" if meta['valid'] else "N/A",
                "FPS": meta['fps'] if meta['valid'] else "N/A",
                "Duration": f"{meta['duration_sec']}s" if meta['valid'] else "N/A",
                "Status": "✅ Ready" if meta['valid'] else "❌ Missing"
            })
        st.markdown(render_markdown_table(vid_rows))

        st.caption(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
