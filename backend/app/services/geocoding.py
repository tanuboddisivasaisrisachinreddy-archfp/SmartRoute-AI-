"""Location search: curated Hyderabad landmarks first, then OSM Nominatim
restricted to the Hyderabad bounding box."""
from __future__ import annotations

import logging
import time
from difflib import SequenceMatcher

import requests

from reinforcement_learning.environment.config import HYDERABAD_BBOX
from reinforcement_learning.environment.graph import haversine_m

log = logging.getLogger("geocoding")


class Geocoder:
    def __init__(self, landmarks: list[dict], enabled: bool = True,
                 base_url: str = "https://nominatim.openstreetmap.org", contact: str = ""):
        self.landmarks = landmarks
        self.enabled = enabled
        self.base_url = base_url.rstrip("/")
        self.ua = "SmartRouteAI/1.0 (academic project" + (f"; {contact}" if contact else "") + ")"
        self.cache: dict[str, tuple[float, list]] = {}
        self._last_call = 0.0

    def _landmark_hits(self, q: str) -> list[dict]:
        ql = q.lower().strip()
        scored = []
        for lm in self.landmarks:
            name = lm["name"].lower()
            score = 1.0 if ql in name or ql in lm["id"] else SequenceMatcher(None, ql, name).ratio()
            if score >= 0.55:
                scored.append((score, lm))
        scored.sort(key=lambda x: -x[0])
        return [{"name": lm["name"], "lat": lm["lat"], "lon": lm["lon"], "source": "landmark",
                 "category": lm.get("category"), "id": lm["id"]} for _, lm in scored]

    def _throttle(self) -> None:
        wait = 1.05 - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def _nominatim(self, q: str, limit: int) -> list[dict]:
        key = q.lower().strip()
        if key in self.cache and time.time() - self.cache[key][0] < 86400:
            return self.cache[key][1]
        b = HYDERABAD_BBOX
        self._throttle()
        r = requests.get(f"{self.base_url}/search", params={
            "q": q if "hyderabad" in key else f"{q}, Hyderabad", "format": "jsonv2", "limit": limit,
            "countrycodes": "in", "viewbox": f"{b['west']},{b['north']},{b['east']},{b['south']}", "bounded": 1},
            headers={"User-Agent": self.ua}, timeout=8)
        r.raise_for_status()
        hits = [{"name": h.get("name") or h["display_name"].split(",")[0], "display_name": h["display_name"],
                 "lat": float(h["lat"]), "lon": float(h["lon"]), "source": "nominatim",
                 "category": h.get("type")} for h in r.json()]
        self.cache[key] = (time.time(), hits)
        return hits

    def search(self, q: str, limit: int = 8) -> dict:
        results = self._landmark_hits(q)
        warning = None
        if self.enabled and len(q.strip()) >= 3 and len(results) < limit:
            try:
                for h in self._nominatim(q, limit):
                    if all(haversine_m(h["lat"], h["lon"], r["lat"], r["lon"]) > 150 for r in results):
                        results.append(h)
            except Exception as exc:
                warning = f"OpenStreetMap geocoder unavailable ({exc.__class__.__name__}); showing built-in places."
                log.warning(warning)
        return {"query": q, "results": results[:limit], "warning": warning}

    def reverse(self, lat: float, lon: float) -> dict:
        best = min(self.landmarks, key=lambda lm: haversine_m(lat, lon, lm["lat"], lm["lon"]))
        d = float(haversine_m(lat, lon, best["lat"], best["lon"]))
        if d < 600:
            return {"name": best["name"], "lat": lat, "lon": lon, "source": "landmark", "distance_m": round(d)}
        if self.enabled:
            try:
                self._throttle()
                r = requests.get(f"{self.base_url}/reverse", params={"lat": lat, "lon": lon, "format": "jsonv2",
                                                                    "zoom": 16}, headers={"User-Agent": self.ua},
                                 timeout=8)
                r.raise_for_status()
                j = r.json()
                addr = j.get("address", {})
                name = addr.get("road") or addr.get("suburb") or addr.get("neighbourhood") or j.get("name")
                area = addr.get("suburb") or addr.get("city_district")
                if name:
                    return {"name": f"{name}{', ' + area if area and area != name else ''}", "lat": lat, "lon": lon,
                            "source": "nominatim"}
            except Exception as exc:
                log.warning("reverse geocoding failed: %s", exc.__class__.__name__)
        return {"name": f"Pin near {best['name']} ({d / 1000:.1f} km)", "lat": lat, "lon": lon, "source": "pin"}
