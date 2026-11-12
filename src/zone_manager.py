"""
Zone Manager for SmartPark
Manages parking zones (A-E), video mappings, configurations, and metadata.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Base directory of the SmartPark project
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
VIDEOS_DIR = PROJECT_ROOT / "videos"
ZONES_CONFIG_PATH = CONFIG_DIR / "zones.json"
SETTINGS_CONFIG_PATH = CONFIG_DIR / "settings.json"


# Default zone identifiers and associated video patterns
DEFAULT_ZONE_SPECS = {
    "A": {
        "name": "Zone A",
        "description": "Block A Two-Wheeler Lot",
        "video_aliases": ["v1.mp4", "v1.avi", "v1.mkv", "v1"]
    },
    "B": {
        "name": "Zone B",
        "description": "Block B Two-Wheeler Lot",
        "video_aliases": ["v2.mp4", "v2.avi", "v2.mkv", "v2"]
    },
    "C": {
        "name": "Zone C",
        "description": "Block C Two-Wheeler Lot",
        "video_aliases": ["v3.mp4", "v3.avi", "v3.mkv", "v3"]
    },
    "D": {
        "name": "Zone D",
        "description": "Block D Two-Wheeler Lot",
        "video_aliases": ["v 4.mp4", "v4.mp4", "v 4.avi", "v4.avi", "v4"]
    },
    "E": {
        "name": "Zone E",
        "description": "Block E Two-Wheeler Lot",
        "video_aliases": ["v5.mp4", "v5.avi", "v5.mkv", "v5"]
    }
}


def load_settings() -> Dict[str, Any]:
    """Load settings.json or return defaults."""
    if SETTINGS_CONFIG_PATH.exists():
        try:
            with open(SETTINGS_CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load settings.json: {e}")
    return {}


def find_video_file_for_zone(zone_id: str, custom_videos_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Search for the video file corresponding to a given zone.
    Handles filename variations like 'v 4.mp4' or 'v4.mp4'.
    """
    search_dir = Path(custom_videos_dir) if custom_videos_dir else VIDEOS_DIR
    if not search_dir.exists():
        logger.warning(f"Videos directory {search_dir} does not exist.")
        return None

    spec = DEFAULT_ZONE_SPECS.get(zone_id.upper())
    if not spec:
        return None

    all_files = list(search_dir.iterdir())
    
    # 1. Check exact alias matches first
    for alias in spec["video_aliases"]:
        for file in all_files:
            if file.is_file() and file.name.lower() == alias.lower():
                return file

    # 2. Check relaxed pattern match (e.g. contains 'v1' or 'v 4')
    zone_index = str(list(DEFAULT_ZONE_SPECS.keys()).index(zone_id.upper()) + 1)
    for file in all_files:
        if file.is_file():
            name_lower = file.name.lower()
            if f"v{zone_index}" in name_lower or f"v {zone_index}" in name_lower:
                return file

    return None


class ZoneManager:
    """Manages zone definitions, video paths, and slot configurations."""

    def __init__(self, config_path: Optional[Path] = None, videos_dir: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else ZONES_CONFIG_PATH
        self.videos_dir = Path(videos_dir) if videos_dir else VIDEOS_DIR
        self.zones: Dict[str, Dict[str, Any]] = {}
        self.load_zones()

    def load_zones(self) -> Dict[str, Dict[str, Any]]:
        """Load zone configurations from file or initialize with defaults."""
        data = {}
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f).get("zones", {})
            except Exception as e:
                logger.error(f"Error reading {self.config_path}: {e}")

        # Ensure all zones A through E are represented
        self.zones = {}
        for zone_id, spec in DEFAULT_ZONE_SPECS.items():
            zone_data = data.get(zone_id, {})
            video_file = find_video_file_for_zone(zone_id, self.videos_dir)
            video_name = video_file.name if video_file else zone_data.get("video_filename", f"v{zone_id}.mp4")

            self.zones[zone_id] = {
                "id": zone_id,
                "name": zone_data.get("name", spec["name"]),
                "description": zone_data.get("description", spec["description"]),
                "video_filename": video_name,
                "video_path": str(video_file.resolve()) if video_file else None,
                "video_found": video_file is not None,
                "reference_resolution": zone_data.get("reference_resolution", None),
                "slots": zone_data.get("slots", [])
            }

        return self.zones

    def save_zones(self) -> bool:
        """Persist current zone configurations to disk."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            export_data = {"zones": {}}
            for zone_id, z in self.zones.items():
                export_data["zones"][zone_id] = {
                    "id": z["id"],
                    "name": z["name"],
                    "description": z["description"],
                    "video_filename": z["video_filename"],
                    "reference_resolution": z.get("reference_resolution"),
                    "slots": z.get("slots", [])
                }
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
            logger.info(f"Saved zone configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save zone config: {e}")
            return False

    def get_zone(self, zone_id: str) -> Optional[Dict[str, Any]]:
        return self.zones.get(zone_id.upper())

    def update_zone_slots(self, zone_id: str, slots: List[Dict[str, Any]], reference_resolution: Optional[Tuple[int, int]] = None) -> bool:
        """Update slots for a specific zone and save."""
        zone_id = zone_id.upper()
        if zone_id not in self.zones:
            return False
        self.zones[zone_id]["slots"] = slots
        if reference_resolution:
            self.zones[zone_id]["reference_resolution"] = list(reference_resolution)
        return self.save_zones()

    def get_all_zones(self) -> Dict[str, Dict[str, Any]]:
        return self.zones
