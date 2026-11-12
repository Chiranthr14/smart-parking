"""
Zone Recommendation Engine for SmartPark
Compares parking zones and identifies optimal parking locations based on
verified vacant two-wheeler spaces.
"""

from typing import Dict, Any, List, Optional


class RecommendationEngine:
    """Evaluates all five parking zones and produces actionable recommendations."""

    def __init__(self):
        self.disclaimer = (
            "NOTICE: Recommendations are computed in real time from recorded CCTV video footage. "
            "Continuous live CCTV feeds (RTSP) are reserved for future phases."
        )

    def evaluate_zones(self, zone_metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Input format:
        zone_metrics = {
            "A": {
                "name": "Zone A",
                "status": "ready" | "analyzed" | "unconfigured" | "error",
                "total_spaces": int,
                "vacant_spaces": int,
                "occupied_spaces": int,
                "uncertain_spaces": int,
                "availability_percentage": float,
                "error_message": Optional[str]
            },
            ...
        }
        """
        valid_zones = []
        unavailable_zones = []
        unconfigured_zones = []

        for zid, data in zone_metrics.items():
            status = data.get("status", "error")
            total = data.get("total_spaces", 0)

            if status in ["error", "missing_video"]:
                unavailable_zones.append(zid)
            elif total == 0:
                unconfigured_zones.append(zid)
            else:
                # Confidently verified metrics
                valid_zones.append({
                    "zone_id": zid,
                    "name": data.get("name", f"Zone {zid}"),
                    "vacant_spaces": data.get("vacant_spaces", 0),
                    "total_spaces": total,
                    "occupied_spaces": data.get("occupied_spaces", 0),
                    "uncertain_spaces": data.get("uncertain_spaces", 0),
                    "availability_percentage": data.get("availability_percentage", 0.0)
                })

        # Case 1: No zones have valid analysis data
        if not valid_zones:
            if unconfigured_zones and not unavailable_zones:
                return {
                    "status": "unconfigured",
                    "recommended_zones": [],
                    "title": "Parking Spaces Not Configured",
                    "explanation": "Please mark and configure parking spaces for the zones in the configuration tool.",
                    "details": {},
                    "disclaimer": self.disclaimer
                }
            return {
                "status": "unavailable",
                "recommended_zones": [],
                "title": "Data Unavailable",
                "explanation": "Unable to evaluate parking availability. Check video files and configurations.",
                "details": {"unavailable_zones": unavailable_zones},
                "disclaimer": self.disclaimer
            }

        # Sort valid zones by vacant spaces descending, then availability percentage descending
        sorted_zones = sorted(
            valid_zones,
            key=lambda z: (z["vacant_spaces"], z["availability_percentage"]),
            reverse=True
        )

        max_vacant = sorted_zones[0]["vacant_spaces"]

        # Case 2: All configured zones are full (0 vacant spaces)
        if max_vacant <= 0:
            return {
                "status": "full",
                "recommended_zones": [],
                "title": "No Parking Spaces Available",
                "explanation": "All analyzed parking zones are currently fully occupied or uncertain.",
                "ranked_zones": sorted_zones,
                "disclaimer": self.disclaimer
            }

        # Case 3: Find top zone(s) - handle ties
        top_zones = [z for z in sorted_zones if z["vacant_spaces"] == max_vacant]

        if len(top_zones) == 1:
            best = top_zones[0]
            title = f"Recommended: {best['name']}"
            explanation = (
                f"{best['name']} has the highest number of verified vacant spaces "
                f"({best['vacant_spaces']} available out of {best['total_spaces']}, "
                f"{best['availability_percentage']}% availability)."
            )
        else:
            names = " & ".join([z["name"] for z in top_zones])
            title = f"Tied Recommendation: {names}"
            explanation = (
                f"{names} are tied with {max_vacant} verified vacant spaces each. "
                f"Choose either zone for two-wheeler parking."
            )

        return {
            "status": "available",
            "recommended_zones": top_zones,
            "title": title,
            "explanation": explanation,
            "ranked_zones": sorted_zones,
            "unavailable_zones": unavailable_zones,
            "unconfigured_zones": unconfigured_zones,
            "disclaimer": self.disclaimer
        }
