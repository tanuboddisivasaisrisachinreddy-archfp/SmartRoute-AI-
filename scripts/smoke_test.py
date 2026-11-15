#!/usr/bin/env python3
"""End-to-end check of a running backend (local or deployed on Render).

    python scripts/smoke_test.py http://localhost:8000
    python scripts/smoke_test.py https://smartroute-api.onrender.com --origin https://smartroute-ai.onrender.com

Exits non-zero if any check fails.  Use it before claiming a deployment works.
"""
from __future__ import annotations

import argparse
import sys
import time

import requests


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base", help="backend base URL")
    p.add_argument("--origin", help="frontend origin to test CORS for")
    a = p.parse_args()
    base = a.base.rstrip("/")
    ok = True

    def check(name, cond, detail=""):
        nonlocal ok
        ok &= bool(cond)
        print(("PASS " if cond else "FAIL ") + name + (f"  ({detail})" if detail else ""))

    t = time.time()
    h = requests.get(f"{base}/api/health", timeout=120).json()  # long timeout: Render cold start
    check("health reachable", h.get("status") in ("ok", "degraded"), f"{time.time() - t:.1f}s")
    check("road graph loaded", h.get("graph_loaded"), str(h.get("graph")))
    check("graph is real (not synthetic)", h.get("graph") and not h["graph"].get("synthetic"))
    check("Q-learning model loaded", h.get("models", {}).get("q_learning", {}).get("loaded"))
    check("DQN model loaded", h.get("models", {}).get("dqn", {}).get("loaded"))
    print("     database:", h.get("database"))

    s = requests.get(f"{base}/api/locations/search", params={"q": "Charminar"}, timeout=30).json()
    check("search finds Charminar", s["results"] and s["results"][0]["name"] == "Charminar")

    body = {"source": {"lat": 17.4504, "lon": 78.3810, "name": "Hitech City"},
            "destination": {"lat": 17.3616, "lon": 78.4747, "name": "Charminar"},
            "preference": "fastest", "scenario": "peak", "event": "incident_on_route"}
    t = time.time()
    r = requests.post(f"{base}/api/routes/compare", json=body, timeout=120)
    check("compare returns 200", r.status_code == 200, f"{time.time() - t:.1f}s")
    if r.status_code == 200:
        j = r.json()
        for route in j["routes"]:
            m = route["metrics"] or {}
            check(f"{route['label']} reached destination", route["success"],
                  f"{m.get('travel_time_min')} min, {m.get('distance_km')} km, {route['computation_ms']} ms")
            check(f"{route['label']} has road geometry", len(route["geometry"]) > 10)
        check("traffic labelled", "Simulated" in j["traffic_label"] or j["traffic_mode"] == "live", j["traffic_label"])
    if a.origin:
        pre = requests.options(f"{base}/api/routes/compare", headers={
            "Origin": a.origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type"}, timeout=30)
        check("CORS allows frontend origin", pre.headers.get("access-control-allow-origin") in (a.origin, "*"),
              pre.headers.get("access-control-allow-origin", "none"))
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
