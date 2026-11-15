"""Aggregate statistics computed from stored comparisons (never hard-coded)."""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from .engine import ALGO_LABELS, ALGORITHMS


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return round(float(np.mean(vals)), 3) if vals else None


def summarize_comparisons(docs: list[dict]) -> dict:
    per_algo = defaultdict(list)
    wins = Counter()
    pairs = Counter()
    scenarios = Counter()
    prefs = Counter()
    timeline = []
    for d in docs:
        wins[d.get("best_algorithm")] += 1
        pairs[f"{d.get('source_name')} → {d.get('destination_name')}"] += 1
        scenarios[d.get("scenario")] += 1
        prefs[d.get("preference")] += 1
        best = None
        for r in d.get("routes", []):
            per_algo[r["algorithm"]].append(r)
            if r["algorithm"] == d.get("best_algorithm"):
                best = r
        if best and best.get("travel_time_min") is not None:
            timeline.append({"created_at": d.get("created_at"), "travel_time_min": best["travel_time_min"],
                             "distance_km": best["distance_km"], "congestion": best["congestion_exposure"],
                             "best": d.get("best_algorithm")})
    algos = []
    for a in ALGORITHMS:
        rs = per_algo.get(a, [])
        ok = [r for r in rs if r.get("success")]
        algos.append({
            "algorithm": a, "label": ALGO_LABELS[a], "runs": len(rs),
            "success_rate": round(len(ok) / len(rs), 3) if rs else None,
            "avg_travel_time_min": _mean([r.get("travel_time_min") for r in ok]),
            "avg_distance_km": _mean([r.get("distance_km") for r in ok]),
            "avg_congestion": _mean([r.get("congestion_exposure") for r in ok]),
            "avg_reward": _mean([r.get("total_reward") for r in ok]),
            "avg_computation_ms": _mean([r.get("computation_ms") for r in rs]),
            "avg_score": _mean([r.get("route_score") for r in rs]),
            "wins": wins.get(a, 0),
        })
    all_ok = [r for rs in per_algo.values() for r in rs if r.get("success")]
    return {
        "total_comparisons": len(docs),
        "total_routes_evaluated": sum(len(d.get("routes", [])) for d in docs),
        "avg_travel_time_min": _mean([r.get("travel_time_min") for r in all_ok]),
        "avg_distance_km": _mean([r.get("distance_km") for r in all_ok]),
        "avg_congestion": _mean([r.get("congestion_exposure") for r in all_ok]),
        "algorithms": algos,
        "popular_pairs": [{"pair": k, "count": v} for k, v in pairs.most_common(8)],
        "scenarios": [{"scenario": k, "count": v} for k, v in scenarios.most_common()],
        "preferences": [{"preference": k, "count": v} for k, v in prefs.most_common()],
        "timeline": list(reversed(timeline[:60])),
    }


def compact_comparison(res: dict) -> dict:
    """Small document stored per comparison (no geometry)."""
    return {
        "source_name": res["source"].get("landmark") or res["source"].get("name") or
        f"{res['source']['lat']:.4f},{res['source']['lon']:.4f}",
        "destination_name": res["destination"].get("landmark") or res["destination"].get("name") or
        f"{res['destination']['lat']:.4f},{res['destination']['lon']:.4f}",
        "source": {k: res["source"][k] for k in ("lat", "lon", "node")},
        "destination": {k: res["destination"][k] for k in ("lat", "lon", "node")},
        "preference": res["preference"], "scenario": res["scenario"], "traffic_mode": res["traffic_mode"],
        "event": (res.get("event") or {}).get("type"), "best_algorithm": res["best_algorithm"],
        "routes": [{"algorithm": r["algorithm"], "success": r["success"], "computation_ms": r["computation_ms"],
                    "route_score": r["route_score"], **({k: r["metrics"][k] for k in (
                        "distance_km", "travel_time_min", "congestion_exposure", "total_reward", "segments",
                        "estimated_cost_inr")} if r["metrics"] else {})} for r in res["routes"]],
    }
