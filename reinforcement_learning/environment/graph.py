"""Road-network representation used by every algorithm.

The processed Hyderabad graph (built by ``scripts/preprocess_data.py`` from
OpenStreetMap via OSMnx) is loaded into:

* compact NumPy arrays (fast access inside RL training loops),
* a CSR adjacency structure ``out_ptr / out_idx`` (outgoing edge ids per node),
* a ``networkx.DiGraph`` (used by the Dijkstra baseline),
* a haversine ``BallTree`` for nearest-intersection snapping.

Nodes are road intersections, edges are directed road segments.  Parallel
edges were collapsed during preprocessing, so an edge is uniquely identified
by ``(u, v)`` and by its integer id.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np

from .config import GRAPH_FILE, ROAD_CLASSES

EARTH_RADIUS_M = 6_371_008.8


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres (works with scalars or NumPy arrays)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(a, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


@dataclass
class RoadGraph:
    osmid: np.ndarray
    lat: np.ndarray
    lon: np.ndarray
    eu: np.ndarray
    ev: np.ndarray
    length_m: np.ndarray
    road_class: np.ndarray  # int index into ROAD_CLASSES
    speed_kph: np.ndarray  # free-flow speed
    oneway: np.ndarray
    lanes: np.ndarray
    names: list[str]
    geometry: list[list[list[float]]]  # per edge: [[lat, lon], ...]
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.n_nodes = int(len(self.lat))
        self.n_edges = int(len(self.eu))
        order = np.argsort(self.eu, kind="stable")
        self.out_idx = order.astype(np.int32)
        counts = np.bincount(self.eu, minlength=self.n_nodes)
        self.out_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        in_counts = np.bincount(self.ev, minlength=self.n_nodes)
        self.out_degree = counts.astype(np.int32)
        self.in_degree = in_counts.astype(np.int32)
        self.edge_lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.eu, self.ev))}
        self.free_time_s = self.length_m / (self.speed_kph / 3.6)
        # midpoint of each edge (for incident / traffic sampling)
        self.edge_mid_lat = (self.lat[self.eu] + self.lat[self.ev]) / 2
        self.edge_mid_lon = (self.lon[self.eu] + self.lon[self.ev]) / 2
        self._nx: nx.DiGraph | None = None
        self._tree = None

    # ------------------------------------------------------------------ access
    def out_edges(self, node: int) -> np.ndarray:
        return self.out_idx[self.out_ptr[node] : self.out_ptr[node + 1]]

    def edge_id(self, u: int, v: int) -> int:
        return self.edge_lookup[(u, v)]

    def road_class_name(self, eid: int) -> str:
        return ROAD_CLASSES[int(self.road_class[eid])]

    def distance_m(self, a: int, b: int) -> float:
        return float(haversine_m(self.lat[a], self.lon[a], self.lat[b], self.lon[b]))

    def distances_to(self, target: int) -> np.ndarray:
        """Straight-line distance (m) from every node to ``target``."""
        return haversine_m(self.lat, self.lon, self.lat[target], self.lon[target])

    @property
    def nx_graph(self) -> nx.DiGraph:
        if self._nx is None:
            g = nx.DiGraph()
            g.add_nodes_from(range(self.n_nodes))
            for i in range(self.n_edges):
                g.add_edge(int(self.eu[i]), int(self.ev[i]), eid=i, length=float(self.length_m[i]))
            self._nx = g
        return self._nx

    # ---------------------------------------------------------------- snapping
    def nearest_node(self, lat: float, lon: float) -> tuple[int, float]:
        """Return (node index, distance in metres) of the closest intersection."""
        if self._tree is None:
            from sklearn.neighbors import BallTree

            self._tree = BallTree(np.radians(np.c_[self.lat, self.lon]), metric="haversine")
        dist, idx = self._tree.query(np.radians([[lat, lon]]), k=1)
        return int(idx[0][0]), float(dist[0][0] * EARTH_RADIUS_M)

    def nodes_within(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        d = haversine_m(self.lat, self.lon, lat, lon)
        return np.nonzero(d <= radius_m)[0]

    def edges_within(self, lat: float, lon: float, radius_m: float) -> np.ndarray:
        d = haversine_m(self.edge_mid_lat, self.edge_mid_lon, lat, lon)
        return np.nonzero(d <= radius_m)[0]

    # -------------------------------------------------------------- geometry
    def path_geometry(self, edge_ids: list[int]) -> list[list[float]]:
        coords: list[list[float]] = []
        for eid in edge_ids:
            seg = self.geometry[eid]
            if coords and seg and coords[-1] == seg[0]:
                coords.extend(seg[1:])
            else:
                coords.extend(seg)
        return coords

    def nodes_to_edges(self, nodes: list[int]) -> list[int]:
        return [self.edge_lookup[(a, b)] for a, b in zip(nodes[:-1], nodes[1:])]

    def bounds(self) -> dict:
        return {
            "south": float(self.lat.min()),
            "north": float(self.lat.max()),
            "west": float(self.lon.min()),
            "east": float(self.lon.max()),
        }

    # ---------------------------------------------------------------- I/O
    def to_dict(self) -> dict:
        return {
            "meta": self.meta,
            "nodes": {
                "osmid": [int(x) for x in self.osmid],
                "lat": [round(float(x), 7) for x in self.lat],
                "lon": [round(float(x), 7) for x in self.lon],
            },
            "edges": {
                "u": [int(x) for x in self.eu],
                "v": [int(x) for x in self.ev],
                "length": [round(float(x), 2) for x in self.length_m],
                "road_class": [int(x) for x in self.road_class],
                "speed_kph": [round(float(x), 1) for x in self.speed_kph],
                "oneway": [bool(x) for x in self.oneway],
                "lanes": [int(x) for x in self.lanes],
                "name": self.names,
                "geometry": self.geometry,
            },
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "RoadGraph":
        n, e = d["nodes"], d["edges"]
        return cls(
            osmid=np.asarray(n["osmid"], dtype=np.int64),
            lat=np.asarray(n["lat"], dtype=np.float64),
            lon=np.asarray(n["lon"], dtype=np.float64),
            eu=np.asarray(e["u"], dtype=np.int32),
            ev=np.asarray(e["v"], dtype=np.int32),
            length_m=np.asarray(e["length"], dtype=np.float64),
            road_class=np.asarray(e["road_class"], dtype=np.int8),
            speed_kph=np.asarray(e["speed_kph"], dtype=np.float64),
            oneway=np.asarray(e["oneway"], dtype=bool),
            lanes=np.asarray(e["lanes"], dtype=np.int8),
            names=list(e["name"]),
            geometry=e["geometry"],
            meta=d.get("meta", {}),
        )

    @classmethod
    def load(cls, path: Path | str | None = None) -> "RoadGraph":
        path = Path(path or GRAPH_FILE)
        if not path.exists():
            raise FileNotFoundError(
                f"Processed road graph not found at {path}. Run `python scripts/download_data.py "
                f"--osm` and `python scripts/preprocess_data.py` first (see README)."
            )
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def build_synthetic_grid(rows: int = 12, cols: int = 12, spacing_m: float = 400.0,
                         origin=(17.43, 78.40), seed: int = 0) -> RoadGraph:
    """Small grid graph used ONLY by unit tests when the real graph is absent.

    It is labelled ``synthetic`` in its metadata and never used by the API.
    """
    rng = np.random.default_rng(seed)
    dlat = spacing_m / 111_320
    dlon = spacing_m / (111_320 * math.cos(math.radians(origin[0])))
    lat, lon, osmid = [], [], []
    for r in range(rows):
        for c in range(cols):
            lat.append(origin[0] + r * dlat)
            lon.append(origin[1] + c * dlon)
            osmid.append(r * cols + c)
    eu, ev, cls_, spd = [], [], [], []

    def add(a, b, k):
        eu.append(a)
        ev.append(b)
        cls_.append(k)

    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            k = 2 if (r % 4 == 0 or c % 4 == 0) else 4  # arterials every 4 blocks
            if c + 1 < cols:
                add(i, i + 1, k)
                add(i + 1, i, k)
            if r + 1 < rows:
                add(i, i + cols, k)
                add(i + cols, i, k)
    lat_a, lon_a = np.array(lat), np.array(lon)
    eu_a, ev_a = np.array(eu), np.array(ev)
    length = haversine_m(lat_a[eu_a], lon_a[eu_a], lat_a[ev_a], lon_a[ev_a]) * rng.uniform(1.0, 1.15, len(eu))
    from .config import DEFAULT_SPEEDS_KPH

    spd = [DEFAULT_SPEEDS_KPH[ROAD_CLASSES[k]] for k in cls_]
    geometry = [[[float(lat_a[a]), float(lon_a[a])], [float(lat_a[b]), float(lon_a[b])]] for a, b in zip(eu, ev)]
    return RoadGraph(
        osmid=np.array(osmid), lat=lat_a, lon=lon_a, eu=eu_a.astype(np.int32), ev=ev_a.astype(np.int32),
        length_m=length, road_class=np.array(cls_, dtype=np.int8), speed_kph=np.array(spd, dtype=float),
        oneway=np.zeros(len(eu), bool), lanes=np.full(len(eu), 2, np.int8),
        names=[f"Synthetic road {i}" for i in range(len(eu))], geometry=geometry,
        meta={"source": "synthetic-test-grid", "synthetic": True},
    )
