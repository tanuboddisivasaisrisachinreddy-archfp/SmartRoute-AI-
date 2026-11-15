"""Batch comparison of Dijkstra, Q-Learning and DQN under identical conditions."""
from __future__ import annotations

import itertools
import os
import json
import time
from pathlib import Path

import numpy as np

from reinforcement_learning.environment.config import PROCESSED_DIR

from .engine import ALGO_LABELS, ALGORITHMS, MAX_SNAP_M, RoutingEngine, RoutingError

EXPERIMENT_DIR = Path(os.getenv("SMARTROUTE_EXPERIMENT_DIR", PROCESSED_DIR / "experiments"))


def landmark_pairs(engine: RoutingEngine, n: int, seed: int = 0) -> list[tuple[dict, dict]]:
    lms = [lm for lm in engine.landmarks if lm.get("snap_distance_m", 0) <= MAX_SNAP_M]
    pairs = [(a, b) for a, b in itertools.permutations(lms, 2) if a["node"] != b["node"]]
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=min(n, len(pairs)), replace=False)
    return [pairs[i] for i in idx]


def run_benchmark(engine: RoutingEngine, n_pairs: int = 10, scenarios=("moderate", "peak"),
                  preferences=("fastest",), events=("none",), seed: int = 0) -> dict:
    t_start = time.time()
    rows = []
    for (a, b), scen, pref, ev in itertools.product(landmark_pairs(engine, n_pairs, seed), scenarios, preferences,
                                                     events):
        try:
            res = engine.compare((float(engine.g.lat[a["node"]]), float(engine.g.lon[a["node"]])),
                                 (float(engine.g.lat[b["node"]]), float(engine.g.lon[b["node"]])),
                                 pref, scen, ev, include_alternatives=False)
        except RoutingError:
            continue
        for r in res["routes"]:
            m = r["metrics"] or {}
            rows.append({
                "source": a["name"], "destination": b["name"], "scenario": scen, "preference": pref, "event": ev,
                "algorithm": r["algorithm"], "success": r["success"], "computation_ms": r["computation_ms"],
                "route_score": r["route_score"], "distance_km": m.get("distance_km"),
                "travel_time_min": m.get("travel_time_min"), "congestion_exposure": m.get("congestion_exposure"),
                "total_reward": m.get("total_reward"), "generalized_cost": m.get("generalized_cost"),
                "best": res["best_algorithm"] == r["algorithm"],
                "training_mode": r["diagnostics"].get("training_mode"),
            })
    return {"created_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "duration_s": round(time.time() - t_start, 1),
            "config": {"n_pairs": n_pairs, "scenarios": list(scenarios), "preferences": list(preferences),
                       "events": list(events), "seed": seed},
            "graph": {"nodes": engine.g.n_nodes, "edges": engine.g.n_edges,
                      "signature": engine.g.meta.get("signature"), "synthetic": engine.g.meta.get("synthetic", False)},
            "models": engine.model_status, "summary": summarize(rows), "rows": rows}


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    for algo in ALGORITHMS:
        rs = [r for r in rows if r["algorithm"] == algo]
        if not rs:
            continue
        ok = [r for r in rs if r["success"]]

        def mean(key, data=ok):
            vals = [r[key] for r in data if r[key] is not None]
            return round(float(np.mean(vals)), 3) if vals else None

        # paired comparison: only cases where Dijkstra and this algorithm both succeeded
        paired = []
        for r in ok:
            d = next((x for x in rows if x["algorithm"] == "dijkstra" and x["success"] and
                      all(x[k] == r[k] for k in ("source", "destination", "scenario", "preference", "event"))), None)
            if d and d["travel_time_min"]:
                paired.append(r["travel_time_min"] / d["travel_time_min"])
        out.append({
            "algorithm": algo, "label": ALGO_LABELS[algo], "runs": len(rs), "success_rate": round(len(ok) / len(rs), 3),
            "avg_distance_km": mean("distance_km"), "avg_travel_time_min": mean("travel_time_min"),
            "avg_congestion_exposure": mean("congestion_exposure"), "avg_total_reward": mean("total_reward"),
            "avg_computation_ms": mean("computation_ms", rs), "avg_route_score": mean("route_score", rs),
            "wins": sum(r["best"] for r in rs),
            "time_ratio_vs_dijkstra": round(float(np.mean(paired)), 3) if paired else None,
        })
    return out


def save_experiment(result: dict) -> Path:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPERIMENT_DIR / f"evaluation_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(result))
    (EXPERIMENT_DIR / "latest.json").write_text(json.dumps(result))
    return path


def latest_experiment() -> dict | None:
    p = EXPERIMENT_DIR / "latest.json"
    return json.loads(p.read_text()) if p.exists() else None
