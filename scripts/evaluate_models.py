#!/usr/bin/env python3
"""Evaluate Dijkstra, Q-Learning and DQN on identical Hyderabad trips.

    python scripts/evaluate_models.py --pairs 20 --scenarios low,moderate,heavy,peak
    python scripts/evaluate_models.py --pairs 10 --events none,incident_on_route

Results go to data/processed/experiments/ (and MongoDB when MONGODB_URI is set).
"""
from __future__ import annotations

import argparse
import logging

import _common  # noqa: F401

from backend.app.services.benchmark import run_benchmark, save_experiment
from backend.app.services.engine import RoutingEngine
from reinforcement_learning.environment.graph import RoadGraph

log = logging.getLogger("evaluate")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pairs", type=int, default=15)
    p.add_argument("--scenarios", default="low,moderate,heavy,peak")
    p.add_argument("--preferences", default="fastest")
    p.add_argument("--events", default="none,incident_on_route")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    engine = RoutingEngine(RoadGraph.load())
    res = run_benchmark(engine, a.pairs, a.scenarios.split(","), a.preferences.split(","), a.events.split(","),
                        a.seed)
    path = save_experiment(res)
    try:
        from backend.app.database.store import get_store

        get_store().insert("experiments", {k: v for k, v in res.items() if k != "rows"} | {"rows": res["rows"][:2000]})
    except Exception as exc:  # database optional
        log.warning("Not stored in database: %s", exc)
    print(f"\nSaved {path}\n")
    print(f"{'Algorithm':<12}{'success':>9}{'time(min)':>11}{'dist(km)':>10}{'exposure':>10}{'reward':>9}"
          f"{'ms':>9}{'wins':>6}{'t/Dijkstra':>12}")
    for s in res["summary"]:
        print(f"{s['label']:<12}{s['success_rate']:>9.2f}{s['avg_travel_time_min'] or 0:>11.2f}"
              f"{s['avg_distance_km'] or 0:>10.2f}{s['avg_congestion_exposure'] or 0:>10.3f}"
              f"{s['avg_total_reward'] or 0:>9.2f}{s['avg_computation_ms'] or 0:>9.1f}{s['wins']:>6}"
              f"{s['time_ratio_vs_dijkstra'] or 0:>12.3f}")


if __name__ == "__main__":
    main()
