"""Time-dependent traffic model for the Hyderabad road graph.

IMPORTANT: unless a live provider (TomTom) has supplied observations, every
number produced here is SIMULATED.  The API labels it
"Simulated Traffic – Not Live Traffic".

Model (documented in docs/traffic_model.md)
-------------------------------------------
For edge e at simulation minute t:

    c_e(t) = clip( L(t) * k_class(e) * h_e  +  0.10 * z_e(t)  +  I_e(t),  0, 0.95 )

* ``L(t)``   network-wide level = scenario base level x time-of-day profile
             (+ a decaying "surge" for sudden increases).
* ``k_class`` sensitivity of the road class (arterials congest more than the ORR).
* ``h_e``    static spatial factor: Gaussian bumps around assumed congestion
             hotspots (Ameerpet, HITEC City, ...) times seeded per-edge noise.
* ``z_e``    Ornstein-Uhlenbeck noise (mean-reverting random fluctuation).
* ``I_e``    incidents / random congestion zones with onset, duration and
             linear recovery.

Speed and travel time:

    v_e(t) = v_free(e) * (1 - 0.85 * c_e(t))        (floored at 4 km/h)
    tt_e(t) = length_e / v_e(t)
"""
from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from .graph import RoadGraph, haversine_m

# Calibrated on the Hyderabad arterial graph so that mean congestion is about
# 0.12 (low), 0.35 (moderate), 0.5 (heavy) and 0.6 (peak) at evening rush hour.
SCENARIO_LEVELS = {
    "low": 0.12,
    "moderate": 0.30,
    "heavy": 0.45,
    "peak": 0.55,
    "random": 0.35,
}
SCENARIO_LABELS = {
    "low": "Low traffic",
    "moderate": "Moderate traffic",
    "heavy": "Heavy traffic",
    "peak": "Peak-hour traffic",
    "random": "Random congestion",
}
# motorway, trunk, primary, secondary, tertiary, other
CLASS_SENSITIVITY = np.array([0.55, 1.0, 1.1, 1.0, 0.9, 0.8])
SPEED_DROP = 0.85
MIN_SPEED_KPH = 4.0
NOISE_SCALE = 0.10
OU_TAU_MIN = 30.0

# Assumed congestion hotspots (modelling assumption, NOT measured data).
HOTSPOTS = [
    ("Ameerpet", 17.4375, 78.4482, 1.35),
    ("HITEC City", 17.4504, 78.3810, 1.35),
    ("Kukatpally", 17.4948, 78.3996, 1.25),
    ("Mehdipatnam", 17.3950, 78.4400, 1.25),
    ("Secunderabad", 17.4337, 78.5016, 1.25),
    ("LB Nagar", 17.3457, 78.5522, 1.25),
    ("Charminar", 17.3616, 78.4747, 1.3),
    ("Panjagutta", 17.4260, 78.4510, 1.3),
    ("Gachibowli", 17.4401, 78.3489, 1.2),
    ("Uppal", 17.4058, 78.5591, 1.2),
]
HOTSPOT_RADIUS_M = 2500.0
STATIC_SEED = 2024
# network-level thresholds of the 4 traffic buckets used in the Q-learning state
BUCKET_EDGES = [0.15, 0.30, 0.45]


def time_of_day_profile(minute: float) -> float:
    """Synthetic weekday profile in [~0.3, 1.0]: peaks ~10:00 and ~18:45.

    The shape is an assumption for Indian metro traffic; see the Datasets page
    for a comparison with profiles computed from METR-LA / PEMS-BAY.
    """
    h = (minute % 1440) / 60.0
    morning = 0.62 * math.exp(-(((h - 10.0) / 1.5) ** 2))
    evening = 0.72 * math.exp(-(((h - 18.75) / 1.9) ** 2))
    midday = 0.22 * math.exp(-(((h - 14.0) / 2.5) ** 2))
    return min(1.0, 0.28 + morning + evening + midday)


@dataclass
class Incident:
    id: int
    lat: float
    lon: float
    radius_m: float
    severity: float
    start_min: float
    duration_min: float
    recovery_min: float
    kind: str = "incident"  # incident | random_congestion | scenario_event
    label: str = ""
    edge_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    edge_w: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def intensity(self, t: float) -> float:
        if t < self.start_min:
            return 0.0
        end = self.start_min + self.duration_min
        if t <= end:
            return self.severity
        if self.recovery_min > 0 and t <= end + self.recovery_min:
            return self.severity * (1 - (t - end) / self.recovery_min)
        return 0.0

    def expired(self, t: float) -> bool:
        return t > self.start_min + self.duration_min + self.recovery_min

    def to_dict(self, t: float) -> dict:
        return {
            "id": self.id,
            "lat": self.lat,
            "lon": self.lon,
            "radius_m": self.radius_m,
            "severity": round(self.severity, 3),
            "current_intensity": round(self.intensity(t), 3),
            "start_min": self.start_min,
            "duration_min": self.duration_min,
            "recovery_min": self.recovery_min,
            "kind": self.kind,
            "label": self.label,
            "affected_edges": int(len(self.edge_ids)),
            "status": "recovering" if t > self.start_min + self.duration_min else (
                "pending" if t < self.start_min else "active"),
        }


class TrafficModel:
    def __init__(self, graph: RoadGraph, scenario: str = "moderate", clock_min: float = 9 * 60 + 30,
                 seed: int = 42):
        if scenario not in SCENARIO_LEVELS:
            raise ValueError(f"unknown scenario {scenario}")
        self.graph = graph
        self.scenario = scenario
        self.clock_min = float(clock_min)
        self.start_clock = float(clock_min)
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.intensity_scale = 1.0  # user-controlled multiplier from the control panel
        self.surge_mag = 0.0
        self.surge_start = 0.0
        self.surge_decay_min = 30.0
        self.incidents: list[Incident] = []
        self._ids = itertools.count(1)
        self.live_obs: dict[int, float] = {}  # edge id -> observed congestion (TomTom)
        self.live_meta: dict = {}
        self._build_static()
        self.z = self.rng.standard_normal(graph.n_edges) * 0.5
        self._inc_map: dict[int, list[tuple[Incident, float]]] = {}
        self.history: list[dict] = []
        self._cache_t: float | None = None
        self._cache: np.ndarray | None = None

    # ----------------------------------------------------------- static parts
    def _build_static(self) -> None:
        g = self.graph
        h = np.ones(g.n_edges)
        for _, la, lo, peak in HOTSPOTS:
            d = haversine_m(g.edge_mid_lat, g.edge_mid_lon, la, lo)
            h += (peak - 1.0) * np.exp(-((d / HOTSPOT_RADIUS_M) ** 2))
        # The static spatial pattern is a property of the (simulated) city, so it
        # uses a fixed seed: training, evaluation and the live API all see the
        # same hotspots.  Only the dynamic parts (noise, events) use ``seed``.
        noise = np.random.default_rng(STATIC_SEED).uniform(0.85, 1.15, g.n_edges)
        self.spatial = h * noise
        self.class_sens = CLASS_SENSITIVITY[g.road_class.astype(int)]
        self.static_factor = self.spatial * self.class_sens

    # ------------------------------------------------------------ dynamics
    def level(self, t: float | None = None) -> float:
        t = self.clock_min if t is None else t
        base = SCENARIO_LEVELS[self.scenario] * self.intensity_scale
        prof = 1.0 if self.scenario == "peak" else time_of_day_profile(t)
        surge = 0.0
        if self.surge_mag > 0 and t >= self.surge_start:
            surge = self.surge_mag * math.exp(-(t - self.surge_start) / self.surge_decay_min)
        return float(min(1.2, base * prof + surge))

    def level_bucket(self, t: float | None = None) -> int:
        """Discretised network traffic level used in the Q-learning state (0..3)."""
        lv = self.level(t)
        return int(np.searchsorted(BUCKET_EDGES, lv, side="right"))

    def congestion_array(self, t: float | None = None) -> np.ndarray:
        t = self.clock_min if t is None else t
        if self._cache_t == t and self._cache is not None:
            return self._cache
        c = self.level(t) * self.static_factor + NOISE_SCALE * self.z
        for inc in self.incidents:
            k = inc.intensity(t)
            if k > 0 and len(inc.edge_ids):
                c[inc.edge_ids] += k * inc.edge_w
        if self.live_obs:
            ids = np.fromiter(self.live_obs.keys(), dtype=np.int64)
            c[ids] = np.fromiter(self.live_obs.values(), dtype=float)
        c = np.clip(c, 0.0, 0.95)
        self._cache_t, self._cache = t, c
        return c

    def edge_congestion(self, e: int, t: float | None = None) -> float:
        t = self.clock_min if t is None else t
        if e in self.live_obs:
            return self.live_obs[e]
        c = self.level(t) * self.static_factor[e] + NOISE_SCALE * self.z[e]
        for inc, w in self._inc_map.get(int(e), ()):
            c += inc.intensity(t) * w
        return min(0.95, max(0.0, c))

    def edge_speed_kph(self, e: int, t: float | None = None) -> float:
        return max(MIN_SPEED_KPH, self.graph.speed_kph[e] * (1 - SPEED_DROP * self.edge_congestion(e, t)))

    def edge_time_s(self, e: int, t: float | None = None) -> float:
        return float(self.graph.length_m[e] / (self.edge_speed_kph(e, t) / 3.6))

    def travel_time_array(self, t: float | None = None) -> np.ndarray:
        c = self.congestion_array(t)
        speed = np.maximum(MIN_SPEED_KPH, self.graph.speed_kph * (1 - SPEED_DROP * c))
        return self.graph.length_m / (speed / 3.6)

    def _invalidate(self) -> None:
        self._cache_t = None
        self._inc_map = {}
        for inc in self.incidents:
            for e, w in zip(inc.edge_ids.tolist(), inc.edge_w.tolist()):
                self._inc_map.setdefault(e, []).append((inc, w))

    def advance(self, minutes: float = 5.0) -> dict:
        """Advance the simulation clock, evolving noise and incidents."""
        minutes = float(minutes)
        steps = max(1, int(math.ceil(minutes / 5.0)))
        dt = minutes / steps
        for _ in range(steps):
            a = math.exp(-dt / OU_TAU_MIN)
            self.z = self.z * a + math.sqrt(1 - a * a) * self.rng.standard_normal(self.graph.n_edges)
            self.clock_min += dt
            if self.scenario == "random" and self.rng.random() < min(1.0, dt / 15.0):
                self._spawn_random_zone()
        before = len(self.incidents)
        self.incidents = [i for i in self.incidents if not i.expired(self.clock_min)]
        self._invalidate()
        snap = self.summary()
        self.history.append({k: snap[k] for k in ("clock_min", "level", "mean_congestion", "p90_congestion",
                                                    "active_incidents")})
        self.history = self.history[-288:]
        return {"advanced_min": minutes, "expired_incidents": before - len(self.incidents)}

    def _spawn_random_zone(self) -> None:
        e = int(self.rng.integers(self.graph.n_edges))
        self.add_incident(float(self.graph.edge_mid_lat[e]), float(self.graph.edge_mid_lon[e]),
                          radius_m=float(self.rng.uniform(500, 1200)), severity=float(self.rng.uniform(0.2, 0.45)),
                          duration_min=float(self.rng.uniform(15, 45)), recovery_min=15, kind="random_congestion",
                          label="Random congestion zone")

    # ------------------------------------------------------------ controls
    def set_scenario(self, scenario: str) -> None:
        if scenario not in SCENARIO_LEVELS:
            raise ValueError(f"unknown scenario {scenario}")
        self.scenario = scenario
        self._cache_t = None

    def set_intensity(self, scale: float) -> None:
        self.intensity_scale = float(np.clip(scale, 0.0, 2.0))
        self._cache_t = None

    def add_incident(self, lat: float, lon: float, radius_m: float = 700, severity: float = 0.6,
                     duration_min: float = 45, recovery_min: float = 20, start_min: float | None = None,
                     kind: str = "incident", label: str = "Road incident",
                     edge_ids: np.ndarray | None = None) -> Incident:
        g = self.graph
        if edge_ids is None:
            d = haversine_m(g.edge_mid_lat, g.edge_mid_lon, lat, lon)
            ids = np.nonzero(d <= radius_m)[0]
            w = 1.0 - 0.6 * (d[ids] / max(radius_m, 1.0))  # strongest at the centre
        else:
            ids = np.asarray(edge_ids, dtype=np.int64)
            w = np.ones(len(ids))
        inc = Incident(id=next(self._ids), lat=lat, lon=lon, radius_m=radius_m,
                       severity=float(np.clip(severity, 0.05, 0.95)),
                       start_min=self.clock_min if start_min is None else start_min,
                       duration_min=duration_min, recovery_min=recovery_min, kind=kind, label=label,
                       edge_ids=ids, edge_w=w)
        self.incidents.append(inc)
        self._invalidate()
        return inc

    def clear_incident(self, incident_id: int | None = None) -> int:
        before = len(self.incidents)
        if incident_id is None:
            self.incidents = []
        else:
            self.incidents = [i for i in self.incidents if i.id != incident_id]
        self._invalidate()
        return before - len(self.incidents)

    def trigger_surge(self, magnitude: float = 0.3, decay_min: float = 30.0) -> None:
        """Sudden network-wide traffic increase that recovers exponentially."""
        self.surge_mag = float(magnitude)
        self.surge_start = self.clock_min
        self.surge_decay_min = float(decay_min)
        self._cache_t = None

    def reset(self, scenario: str | None = None, clock_min: float | None = None) -> None:
        self.__init__(self.graph, scenario or self.scenario,
                      self.start_clock if clock_min is None else clock_min, self.seed)

    def randomize(self, rng: np.random.Generator) -> None:
        """Used during RL training: new scenario, clock and noise each episode."""
        self.scenario = str(rng.choice(list(SCENARIO_LEVELS)))
        self.clock_min = float(rng.uniform(6 * 60, 23 * 60))
        self.intensity_scale = 1.0
        self.z = rng.standard_normal(self.graph.n_edges) * 0.7
        self.incidents = []
        self.surge_mag = 0.0
        self.live_obs = {}
        if rng.random() < 0.35:
            for _ in range(int(rng.integers(1, 4))):
                e = int(rng.integers(self.graph.n_edges))
                self.add_incident(float(self.graph.edge_mid_lat[e]), float(self.graph.edge_mid_lon[e]),
                                  radius_m=float(rng.uniform(400, 1200)), severity=float(rng.uniform(0.3, 0.8)),
                                  duration_min=float(rng.uniform(20, 90)))
        self._invalidate()

    def clone(self) -> "TrafficModel":
        other = copy.copy(self)
        other.z = self.z.copy()
        other.incidents = [copy.copy(i) for i in self.incidents]
        other.live_obs = dict(self.live_obs)
        other.history = []
        other.rng = np.random.default_rng(self.rng.integers(1 << 31))
        other._ids = itertools.count(max([i.id for i in self.incidents], default=0) + 1000)
        other._invalidate()
        return other

    # ------------------------------------------------------------- reporting
    @property
    def mode(self) -> str:
        return "live" if self.live_obs else "simulation"

    def summary(self) -> dict:
        c = self.congestion_array()
        h = int(self.clock_min // 60) % 24
        m = int(self.clock_min % 60)
        return {
            "mode": self.mode,
            "label": ("Live traffic (TomTom) blended with simulation"
                      if self.live_obs else "Simulated Traffic – Not Live Traffic"),
            "scenario": self.scenario,
            "scenario_label": SCENARIO_LABELS[self.scenario],
            "intensity_scale": self.intensity_scale,
            "clock_min": round(self.clock_min, 2),
            "clock": f"{h:02d}:{m:02d}",
            "elapsed_min": round(self.clock_min - self.start_clock, 2),
            "level": round(self.level(), 4),
            "time_of_day_factor": round(time_of_day_profile(self.clock_min), 4),
            "mean_congestion": round(float(c.mean()), 4),
            "p90_congestion": round(float(np.percentile(c, 90)), 4),
            "heavy_share": round(float((c > 0.6).mean()), 4),
            "active_incidents": sum(1 for i in self.incidents if i.intensity(self.clock_min) > 0),
            "incidents": [i.to_dict(self.clock_min) for i in self.incidents],
            "surge": round(self.surge_mag * math.exp(-(self.clock_min - self.surge_start) / self.surge_decay_min)
                           if self.surge_mag else 0.0, 4),
            "live": self.live_meta or None,
        }
