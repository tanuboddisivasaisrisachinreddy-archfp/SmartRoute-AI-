#!/usr/bin/env python3
"""Turn downloaded raw data into the compact files the app uses.

    python scripts/preprocess_data.py            # road graph + landmarks (+ traffic profiles if present)
    python scripts/preprocess_data.py --traffic-only

Outputs (data/processed/):
    hyderabad_graph.json.gz   directed road graph (nodes = intersections, edges = segments)
    graph_stats.json          size / road-class statistics
    landmarks.json            Hyderabad locations snapped to graph intersections
    traffic_profiles.json     time-of-day congestion profiles from METR-LA / PEMS-BAY
                              (LOS ANGELES / SF BAY AREA data - benchmarking only)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
import numpy as np
import pandas as pd

from reinforcement_learning.environment.config import (DEFAULT_SPEEDS_KPH, EXTERNAL_DIR, GRAPH_FILE, LANDMARKS_FILE,
                                                        PROCESSED_DIR, RAW_DIR, ROAD_CLASSES, road_class_of)
from reinforcement_learning.environment.graph import RoadGraph
from reinforcement_learning.environment.landmarks import seed_landmarks

log = logging.getLogger("preprocess")


def _first(x):
    return x[0] if isinstance(x, (list, tuple)) and x else x


def parse_speed(value) -> float | None:
    value = _first(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    m = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if not m:
        return None
    v = float(m.group(1))
    if "mph" in str(value):
        v *= 1.609
    return v if 5 <= v <= 130 else None


def parse_lanes(value) -> int:
    value = _first(value)
    try:
        return int(max(1, min(8, float(str(value).split(";")[0]))))
    except (TypeError, ValueError):
        return 2


def build_graph(graphml: Path) -> RoadGraph:
    import osmnx as ox

    log.info("Loading %s", graphml)
    G = ox.load_graphml(graphml)
    log.info("Raw graph: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    try:  # OSMnx >= 2.0
        G = ox.truncate.largest_component(G, strongly=True)
    except AttributeError:  # OSMnx 1.x
        G = ox.utils_graph.get_largest_component(G, strongly=True)
    log.info("Largest strongly connected component: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    best: dict[tuple, dict] = {}
    for u, v, _, d in G.edges(keys=True, data=True):
        if u == v:
            continue
        if (u, v) not in best or d.get("length", 1e9) < best[(u, v)].get("length", 1e9):
            best[(u, v)] = d
    nodes = list(G.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    lat = np.array([G.nodes[n]["y"] for n in nodes], dtype=float)
    lon = np.array([G.nodes[n]["x"] for n in nodes], dtype=float)
    eu, ev, length, rc, speed, oneway, lanes, names, geom = [], [], [], [], [], [], [], [], []
    for (u, v), d in best.items():
        cls = road_class_of(d.get("highway"))
        ms = parse_speed(d.get("maxspeed"))
        default = DEFAULT_SPEEDS_KPH[cls]
        # posted limits are rarely achievable in city traffic: use 80% of the limit, bounded around the default
        spd = float(np.clip(0.8 * ms, 0.6 * default, 1.3 * default)) if ms else default
        g = d.get("geometry")
        if g is not None:
            coords = [[round(y, 6), round(x, 6)] for x, y in g.coords]
        else:
            coords = [[round(lat[index[u]], 6), round(lon[index[u]], 6)], [round(lat[index[v]], 6), round(lon[index[v]], 6)]]
        name = d.get("name") or d.get("ref") or ""
        if isinstance(name, (list, tuple)):
            name = " / ".join(str(n) for n in name)
        eu.append(index[u])
        ev.append(index[v])
        length.append(float(d.get("length", 0.0)) or 1.0)
        rc.append(ROAD_CLASSES.index(cls))
        speed.append(spd)
        oneway.append(bool(d.get("oneway", False)))
        lanes.append(parse_lanes(d.get("lanes")))
        names.append(str(name))
        geom.append(coords)
    raw_meta = {}
    meta_file = graphml.with_suffix(".meta.json")
    if meta_file.exists():
        raw_meta = json.loads(meta_file.read_text())
    sig = hashlib.sha1(np.asarray(eu, np.int32).tobytes() + np.asarray(ev, np.int32).tobytes()
                       + np.round(np.asarray(length), 1).tobytes()).hexdigest()[:12]
    graph = RoadGraph(
        osmid=np.array(nodes, dtype=np.int64), lat=lat, lon=lon, eu=np.array(eu, np.int32), ev=np.array(ev, np.int32),
        length_m=np.array(length), road_class=np.array(rc, np.int8), speed_kph=np.array(speed),
        oneway=np.array(oneway), lanes=np.array(lanes, np.int8), names=names, geometry=geom,
        meta={"source": "OpenStreetMap (ODbL 1.0) via OSMnx", "city": "Hyderabad, Telangana, India",
              "signature": sig, "synthetic": False, "raw": raw_meta,
              "processing": ["largest strongly connected component", "parallel edges collapsed (shortest kept)",
                             "self-loops removed", "road classes mapped (links -> parent class)",
                             "free-flow speed = class default or 0.8 x OSM maxspeed"]},
    )
    return graph


def graph_stats(g: RoadGraph) -> dict:
    cls = Counter(ROAD_CLASSES[int(c)] for c in g.road_class)
    km = {c: round(float(g.length_m[g.road_class == i].sum() / 1000), 1) for i, c in enumerate(ROAD_CLASSES)}
    return {"nodes": g.n_nodes, "edges": g.n_edges, "total_km": round(float(g.length_m.sum() / 1000), 1),
            "edges_by_class": dict(cls), "km_by_class": km, "oneway_share": round(float(g.oneway.mean()), 3),
            "mean_out_degree": round(float(g.out_degree.mean()), 3), "max_out_degree": int(g.out_degree.max()),
            "bounds": g.bounds(), "signature": g.meta.get("signature")}


def build_landmarks(g: RoadGraph) -> list[dict]:
    src = RAW_DIR / "landmarks_geocoded.json"
    items = json.loads(src.read_text()) if src.exists() else seed_landmarks()
    out = []
    for lm in items:
        node, dist = g.nearest_node(lm["lat"], lm["lon"])
        lm.update(node=node, node_lat=float(g.lat[node]), node_lon=float(g.lon[node]), snap_distance_m=round(dist, 1))
        out.append(lm)
    return out


# --------------------------------------------------------------- traffic
def _is_aux(name: str) -> bool:
    n = name.lower()
    return any(k in n for k in ("meta", "location", "distance", "adj"))


def _numeric_speed_table(df: pd.DataFrame) -> pd.DataFrame | None:
    """Accept only a time-indexed table of numeric sensor speeds."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return None
    num = df.apply(pd.to_numeric, errors="coerce")
    num = num.loc[:, num.notna().mean() > 0.5]
    return num if num.shape[1] >= 10 and num.shape[0] >= 1000 else None


def _read_h5_raw(path: Path) -> pd.DataFrame:
    """Read a pandas 'fixed' format HDF5 frame (as in DCRNN's pems-bay.h5) with
    PyTables directly; used when pandas cannot decode the legacy string metadata."""
    import tables

    with tables.open_file(str(path), "r") as h:
        for grp in h.walk_groups():
            names = set(grp._v_children)
            if {"block0_values", "axis0", "axis1"} <= names:
                vals = grp.block0_values.read()
                cols = [c.decode() if isinstance(c, bytes) else str(c) for c in grp.axis0.read()]
                idx = pd.to_datetime(grp.axis1.read())
                if vals.shape == (len(cols), len(idx)):
                    vals = vals.T
                return pd.DataFrame(vals, index=idx, columns=cols)
    raise ValueError("no frame found in HDF5 file")


def _load_speed_frame(folder: Path) -> pd.DataFrame | None:
    # largest file first: the speed table is far bigger than metadata files
    for f in sorted(folder.glob("*.h5"), key=lambda p: -p.stat().st_size):
        if _is_aux(f.name):
            continue
        for reader in (pd.read_hdf, _read_h5_raw):
            try:
                df = _numeric_speed_table(reader(f))
                if df is not None:
                    return df
                log.warning("%s is not a speed table - skipped", f.name)
                break
            except Exception as exc:  # e.g. old pandas HDF strings under pandas 3
                log.warning("Could not read %s with %s (%s)", f.name, reader.__name__, exc)
    for f in sorted(folder.glob("*.csv"), key=lambda p: -p.stat().st_size):
        if _is_aux(f.name):
            continue
        df = pd.read_csv(f)
        first = df.columns[0]
        try:
            idx = pd.to_datetime(df[first])
        except (ValueError, TypeError):
            continue
        df = df.drop(columns=[first])
        df.index = pd.DatetimeIndex(idx)
        df = _numeric_speed_table(df)
        if df is not None:
            return df
    return None


def traffic_profile(df: pd.DataFrame, name: str, origin: str) -> dict:
    speeds = df.replace(0, np.nan)
    free = speeds.quantile(0.95)
    cong = (1 - speeds / free).clip(lower=0, upper=1)
    weekday = cong[cong.index.day_of_week < 5]
    by_hour = weekday.groupby(weekday.index.hour).mean().mean(axis=1)
    mean_speed = speeds[speeds.index.day_of_week < 5].groupby(speeds.index[speeds.index.day_of_week < 5].hour).mean().mean(axis=1)
    return {
        "dataset": name, "geographic_origin": origin, "is_hyderabad_data": False,
        "sensors": int(df.shape[1]), "timesteps": int(df.shape[0]),
        "start": str(df.index.min()), "end": str(df.index.max()),
        "missing_rate": round(float(speeds.isna().mean().mean()), 4),
        "hourly_congestion_weekday": [round(float(by_hour.get(h, np.nan)), 4) for h in range(24)],
        "hourly_mean_speed_mph_weekday": [round(float(mean_speed.get(h, np.nan)), 2) for h in range(24)],
        "method": "congestion = 1 - speed / p95(speed) per sensor; weekday mean by hour",
    }


def build_traffic_profiles() -> dict:
    from reinforcement_learning.environment.traffic import time_of_day_profile

    profiles = {"simulated_hyderabad_profile": [round(time_of_day_profile(h * 60 + 30), 4) for h in range(24)],
                "datasets": []}
    for folder, name, origin in [(EXTERNAL_DIR / "metr-la", "METR-LA", "Los Angeles County, USA"),
                                 (EXTERNAL_DIR / "pems-bay", "PEMS-BAY", "San Francisco Bay Area, USA")]:
        if not folder.exists():
            log.info("%s not downloaded - skipping (run download_data.py --traffic)", name)
            continue
        df = _load_speed_frame(folder)
        if df is None:
            log.warning("No readable speed table in %s", folder)
            continue
        log.info("%s: %s", name, df.shape)
        profiles["datasets"].append(traffic_profile(df, name, origin))
    return profiles


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--graphml", default=str(RAW_DIR / "hyderabad_drive.graphml"))
    p.add_argument("--traffic-only", action="store_true")
    a = p.parse_args()
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not a.traffic_only:
        g = build_graph(Path(a.graphml))
        g.save(GRAPH_FILE)
        stats = graph_stats(g)
        (PROCESSED_DIR / "graph_stats.json").write_text(json.dumps(stats, indent=2))
        log.info("Saved %s: %s", GRAPH_FILE, stats)
        lms = build_landmarks(g)
        LANDMARKS_FILE.write_text(json.dumps(lms, indent=2))
        log.info("Saved %d landmarks", len(lms))
    profiles = build_traffic_profiles()
    (PROCESSED_DIR / "traffic_profiles.json").write_text(json.dumps(profiles, indent=2))
    log.info("Saved traffic profiles (%d datasets)", len(profiles["datasets"]))


if __name__ == "__main__":
    main()
