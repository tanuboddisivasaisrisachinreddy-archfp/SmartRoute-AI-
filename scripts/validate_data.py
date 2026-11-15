#!/usr/bin/env python3
"""Validate processed data before training / deployment.

    python scripts/validate_data.py        # exits non-zero if a check fails
"""
from __future__ import annotations

import json
import sys

import _common  # noqa: F401
import networkx as nx
import numpy as np

from reinforcement_learning.environment.config import GRAPH_FILE, HYDERABAD_BBOX, LANDMARKS_FILE, PROCESSED_DIR
from reinforcement_learning.environment.graph import RoadGraph


def main() -> int:
    checks = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "passed": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + (f" - {detail}" if detail else ""))

    if not GRAPH_FILE.exists():
        check("graph file exists", False, str(GRAPH_FILE))
        return 1
    g = RoadGraph.load()
    check("graph is real OSM data (not synthetic)", not g.meta.get("synthetic", False), g.meta.get("source", ""))
    check("graph has > 1000 intersections", g.n_nodes > 1000, f"{g.n_nodes} nodes, {g.n_edges} edges")
    b = HYDERABAD_BBOX
    inside = ((g.lat >= b["south"] - 0.02) & (g.lat <= b["north"] + 0.02) &
              (g.lon >= b["west"] - 0.02) & (g.lon <= b["east"] + 0.02)).mean()
    check("all nodes inside Hyderabad bbox", inside > 0.999, f"{inside:.4f}")
    check("no NaN coordinates", not (np.isnan(g.lat).any() or np.isnan(g.lon).any()))
    check("positive edge lengths", (g.length_m > 0).all(), f"min {g.length_m.min():.2f} m")
    check("positive speeds", (g.speed_kph > 0).all())
    check("geometry for every edge", all(len(x) >= 2 for x in g.geometry))
    check("strongly connected", nx.is_strongly_connected(g.nx_graph))
    if LANDMARKS_FILE.exists():
        lms = json.loads(LANDMARKS_FILE.read_text())
        worst = max(lms, key=lambda x: x["snap_distance_m"])
        check("landmarks snap within 1.5 km of a road intersection", worst["snap_distance_m"] < 1500,
              f"worst: {worst['name']} {worst['snap_distance_m']} m")
        check("19 required landmarks present", len(lms) >= 19, str(len(lms)))
    else:
        check("landmarks file exists", False)
    prof = PROCESSED_DIR / "traffic_profiles.json"
    if prof.exists():
        p = json.loads(prof.read_text())
        for d in p["datasets"]:
            check(f"{d['dataset']} labelled as non-Hyderabad", d["is_hyderabad_data"] is False, d["geographic_origin"])
    report = PROCESSED_DIR / "validation_report.json"
    report.write_text(json.dumps(checks, indent=2))
    failed = [c for c in checks if not c["passed"]]
    print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed - report: {report}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
