import sys
from pathlib import Path
sys.path.insert(0, r"C:\chiranth\SmartPark")

from src.zone_manager import ZoneManager
from src.parking_slot_manager import ParkingSlotManager
from src.video_processor import VideoProcessor
from app import render_markdown_table

zm = ZoneManager()
zones = zm.get_all_zones()

# 1. Test slot table rendering
target_zone = zm.get_zone('A')
slot_mgr = ParkingSlotManager(target_zone.get('slots', []))
slot_rows = []
for s in slot_mgr.slots:
    slot_rows.append({
        'Slot ID': s.slot_id,
        'Points': len(s.points_normalized),
        'Coords': str([[round(p[0], 2), round(p[1], 2)] for p in s.points_normalized[:2]]) + '...'
    })
md_slots = render_markdown_table(slot_rows)
print("--- Slot Table Render Test ---")
print(md_slots)

# 2. Test video table rendering
vid_rows = []
for zid in ['A', 'B', 'C', 'D', 'E']:
    z = zones[zid]
    meta = VideoProcessor.get_video_metadata(z.get('video_path'))
    vid_rows.append({
        'Zone': f"Zone {zid}",
        'Filename': z.get('video_filename'),
        'Resolution': f"{meta['width']}x{meta['height']}",
        'FPS': meta['fps'],
        'Duration': f"{meta['duration_sec']}s",
        'Status': 'Ready'
    })
md_vids = render_markdown_table(vid_rows)
print("\n--- Video Feeds Table Render Test ---")
print(md_vids)
print("\nAll UI table rendering verified without PyArrow!")
