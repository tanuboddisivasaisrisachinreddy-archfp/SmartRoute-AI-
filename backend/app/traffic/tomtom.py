"""Optional live traffic from the TomTom Traffic Flow API.

Endpoint: Flow Segment Data
    GET https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point=LAT,LON&key=KEY

For a sample of arterial segments spread over Hyderabad the current and
free-flow speeds are fetched; congestion = 1 - currentSpeed / freeFlowSpeed is
assigned to graph edges within 400 m of each sampled point.  All other edges
keep the simulated value, and the UI labels the mode as
"Live traffic (TomTom) blended with simulation".  Without TOMTOM_API_KEY this
module is inactive and no live data is ever claimed.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import requests

from reinforcement_learning.environment.graph import RoadGraph
from reinforcement_learning.environment.traffic import TrafficModel

log = logging.getLogger("tomtom")
URL = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"


class TomTomProvider:
    def __init__(self, api_key: str, n_points: int = 30, min_refresh_s: int = 300):
        self.api_key = api_key
        self.n_points = n_points
        self.min_refresh_s = min_refresh_s
        self.last_refresh = 0.0
        self.last_result: dict | None = None

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def sample_edges(self, g: RoadGraph) -> np.ndarray:
        major = np.flatnonzero(g.road_class <= 2)  # motorway / trunk / primary
        if len(major) == 0:
            major = np.arange(g.n_edges)
        # spread points on a spatial grid so the whole city is covered
        lat, lon = g.edge_mid_lat[major], g.edge_mid_lon[major]
        k = int(np.ceil(np.sqrt(self.n_points)))
        li = np.clip(((lat - lat.min()) / (np.ptp(lat) + 1e-9) * k).astype(int), 0, k - 1)
        lo = np.clip(((lon - lon.min()) / (np.ptp(lon) + 1e-9) * k).astype(int), 0, k - 1)
        cells: dict[tuple, int] = {}
        for e, a, b in zip(major, li, lo):
            cells.setdefault((a, b), int(e))
        return np.array(list(cells.values())[: self.n_points])

    def refresh(self, g: RoadGraph, traffic: TrafficModel, force: bool = False) -> dict:
        if not self.configured:
            raise RuntimeError("TOMTOM_API_KEY is not configured")
        if not force and time.time() - self.last_refresh < self.min_refresh_s and self.last_result:
            return {**self.last_result, "cached": True}
        obs: dict[int, float] = {}
        samples, errors = [], 0
        for e in self.sample_edges(g):
            la, lo = float(g.edge_mid_lat[e]), float(g.edge_mid_lon[e])
            try:
                r = requests.get(URL, params={"point": f"{la},{lo}", "key": self.api_key, "unit": "KMPH"}, timeout=10)
                r.raise_for_status()
                f = r.json()["flowSegmentData"]
                cur, free = float(f["currentSpeed"]), float(f["freeFlowSpeed"])
                c = 0.95 if f.get("roadClosure") else float(np.clip(1 - cur / max(free, 1), 0, 0.95))
                samples.append({"lat": la, "lon": lo, "current_kph": cur, "free_flow_kph": free, "congestion": c,
                                "confidence": f.get("confidence")})
                for ne in g.edges_within(la, lo, 400):
                    obs[int(ne)] = max(obs.get(int(ne), 0.0), c)
            except Exception as exc:
                errors += 1
                log.warning("TomTom request failed: %s", exc.__class__.__name__)
        if not samples:
            raise RuntimeError("TomTom returned no usable observations (check key / quota)")
        traffic.live_obs = obs
        traffic.live_meta = {"provider": "TomTom Traffic Flow API", "sampled_points": len(samples),
                             "covered_edges": len(obs), "coverage_share": round(len(obs) / g.n_edges, 4),
                             "errors": errors, "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "note": "Edges without a nearby observation use the simulation model."}
        traffic._cache_t = None
        self.last_refresh = time.time()
        self.last_result = {**traffic.live_meta, "samples": samples}
        return self.last_result
