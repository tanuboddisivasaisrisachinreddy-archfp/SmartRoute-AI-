"""Routing engine: loads the Hyderabad graph + trained models once and runs
Dijkstra, Q-Learning and DQN under identical conditions.

This module has no FastAPI dependency so it is reused by
``scripts/evaluate_models.py``.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from backend.app.routing.dijkstra import alternative_routes, dijkstra_route
from reinforcement_learning.dqn.agent import DQNAgent
from reinforcement_learning.environment.config import CHECKPOINT_DIR, LANDMARKS_FILE, PROCESSED_DIR
from reinforcement_learning.environment.graph import RoadGraph
from reinforcement_learning.environment.landmarks import seed_landmarks
from reinforcement_learning.environment.route_env import PREFERENCES
from reinforcement_learning.environment.traffic import SCENARIO_LEVELS, TrafficModel
from reinforcement_learning.evaluation.metrics import cost_model_doc, evaluate_path
from reinforcement_learning.q_learning.agent import QLearningAgent

log = logging.getLogger("engine")

ALGORITHMS = ("dijkstra", "q_learning", "dqn")
ALGO_LABELS = {"dijkstra": "Dijkstra", "q_learning": "Q-Learning", "dqn": "DQN"}
ALGO_COLORS = {"dijkstra": "#2563eb", "q_learning": "#16a34a", "dqn": "#9333ea"}
MAX_SNAP_M = 2500.0


class RoutingError(ValueError):
    pass


class RoutingEngine:
    def __init__(self, graph: RoadGraph, checkpoint_dir: Path = CHECKPOINT_DIR, seed: int = 42,
                 ql_ondemand_episodes: int = 600, ql_ondemand_budget_s: float = 8.0, dqn_use_torch: bool = False):
        self.g = graph
        self.checkpoint_dir = Path(checkpoint_dir)
        self.seed = seed
        self.lock = threading.RLock()
        self.traffic = TrafficModel(graph, "moderate", seed=seed)
        self.landmarks = self._load_landmarks()
        self.ql: QLearningAgent | None = None
        self.dqn: DQNAgent | None = None
        self.model_status: dict = {}
        self.ql_ondemand_episodes = ql_ondemand_episodes
        self.ql_ondemand_budget_s = ql_ondemand_budget_s
        self.dqn_use_torch = dqn_use_torch
        self._ondemand: OrderedDict = OrderedDict()
        self.load_models()

    # ------------------------------------------------------------- loading
    def _load_landmarks(self) -> list[dict]:
        if LANDMARKS_FILE.exists():
            lms = json.loads(LANDMARKS_FILE.read_text())
        else:
            lms = seed_landmarks()
        for lm in lms:
            if "node" not in lm or lm["node"] >= self.g.n_nodes:
                node, d = self.g.nearest_node(lm["lat"], lm["lon"])
                lm.update(node=node, snap_distance_m=round(d, 1))
        return lms

    def load_models(self) -> dict:
        status = {}
        ql_path = self.checkpoint_dir / "q_learning.npz"
        try:
            self.ql = QLearningAgent.load(self.g, ql_path)
            status["q_learning"] = {"loaded": True, "path": str(ql_path), "tables": len(self.ql.tables),
                                    "trained_at": self.ql.meta.get("trained_at")}
        except FileNotFoundError:
            self.ql = QLearningAgent(self.g)
            status["q_learning"] = {"loaded": False, "reason": "checkpoint not found - on-demand training only",
                                    "path": str(ql_path)}
        except Exception as exc:
            self.ql = QLearningAgent(self.g)
            status["q_learning"] = {"loaded": False, "reason": str(exc), "path": str(ql_path)}
        dqn_path = self.checkpoint_dir / "dqn.npz"
        try:
            self.dqn = DQNAgent.load(self.g, dqn_path, use_torch=self.dqn_use_torch)
            status["dqn"] = {"loaded": True, "path": str(dqn_path), "episodes": self.dqn.meta.get("episodes_trained"),
                             "rolling_success": self.dqn.meta.get("rolling_success"),
                             "trained_at": self.dqn.meta.get("saved_at"),
                             "inference_backend": "torch" if self.dqn.torch_net is not None else "numpy"}
        except FileNotFoundError:
            self.dqn = None
            status["dqn"] = {"loaded": False, "reason": "checkpoint not found - run scripts/train_dqn.py",
                             "path": str(dqn_path)}
        except Exception as exc:
            self.dqn = None
            status["dqn"] = {"loaded": False, "reason": str(exc), "path": str(dqn_path)}
        self._ondemand.clear()
        self.model_status = status
        log.info("Model status: %s", status)
        return status

    # ------------------------------------------------------------- helpers
    def snap(self, lat: float, lon: float) -> tuple[int, float]:
        node, dist = self.g.nearest_node(lat, lon)
        if dist > MAX_SNAP_M:
            raise RoutingError(f"Location ({lat:.4f}, {lon:.4f}) is {dist / 1000:.1f} km from the nearest road in the "
                               "Hyderabad network - choose a point inside the covered area.")
        return node, dist

    def landmark_for_node(self, node: int) -> dict | None:
        for lm in self.landmarks:
            if lm["node"] == node:
                return lm
        return None

    def scenario_traffic(self, scenario: str) -> TrafficModel:
        with self.lock:
            t = self.traffic.clone()
        if scenario not in ("current", "live"):
            if scenario not in SCENARIO_LEVELS:
                raise RoutingError(f"unknown traffic scenario '{scenario}'")
            t.set_scenario(scenario)
            t.live_obs = {}
        return t

    # ------------------------------------------------------------ algorithms
    def _run_dijkstra(self, traffic, s, d, pref, t0):
        r = dijkstra_route(self.g, traffic, s, d, pref, t0)
        r["notes"] = "Planned once on the traffic snapshot at departure (static plan)."
        return r

    def _ensure_ql_table(self, s: int, d: int, pref: str) -> dict:
        if self.ql.has_table(pref, d):
            key = (pref, d)
            info = self._ondemand.get(key)
            if info:
                self._ondemand.move_to_end(key)
                return {"training_mode": "on-demand (cached)", **info}
            return {"training_mode": "pretrained", "training_episodes": self.ql.tables[key].episodes,
                    "training_time_ms": 0}
        t = time.perf_counter()
        train_traffic = TrafficModel(self.g, "moderate", seed=self.seed + d)
        hist = self.ql.train_goal(train_traffic, d, pref, episodes=self.ql_ondemand_episodes, focus=[s],
                                  time_budget_s=self.ql_ondemand_budget_s,
                                  rng=np.random.default_rng(self.seed + d + s))
        info = {"training_episodes": len(hist), "training_time_ms": round((time.perf_counter() - t) * 1000, 1),
                "training_success_rate": round(float(np.mean([h["reached"] for h in hist[-50:]])), 3)}
        self._ondemand[(pref, d)] = info
        while len(self._ondemand) > 40:
            old, _ = self._ondemand.popitem(last=False)
            self.ql.tables.pop(old, None)
        return {"training_mode": "on-demand", **info}

    def _run_ql(self, traffic, s, d, pref, t0):
        info = self._ensure_ql_table(s, d, pref)
        r = self.ql.route(traffic, s, d, pref, t0)
        r.update(info)
        r["notes"] = (f"Greedy policy from Q-table ({info['training_mode']}); "
                      f"{r['fallback_steps']} heuristic fallback steps at unexplored intersections.")
        return r

    def _run_dqn(self, traffic, s, d, pref, t0):
        if self.dqn is None:
            return {"edges": [], "nodes": [s], "reached": False, "unavailable": True,
                    "notes": "DQN checkpoint not loaded. Train it with `python scripts/train_dqn.py`."}
        r = self.dqn.route(traffic, s, d, pref, t0)
        r["notes"] = "Greedy policy of the trained Q-network, re-evaluated at every intersection."
        return r

    # ------------------------------------------------------------- compare
    def compare(self, source: tuple[float, float], destination: tuple[float, float], preference: str = "fastest",
                scenario: str = "current", event: str = "none", algorithms=ALGORITHMS,
                include_alternatives: bool = True) -> dict:
        if preference not in PREFERENCES:
            raise RoutingError(f"unknown preference '{preference}'")
        s, s_snap = self.snap(*source)
        d, d_snap = self.snap(*destination)
        if s == d:
            raise RoutingError("Source and destination snap to the same intersection - pick points further apart.")
        traffic = self.scenario_traffic(scenario)
        t0 = traffic.clock_min
        cfg = PREFERENCES[preference]
        event_info = None
        if event == "incident_on_route":
            event_info = self._inject_incident(traffic, s, d, preference, t0)
        elif event == "surge":
            traffic.trigger_surge(0.35, 40)
            event_info = {"type": "surge", "description": "Sudden network-wide traffic increase at departure, "
                                                          "recovering with a 40-minute time constant."}
        results = []
        runners = {"dijkstra": self._run_dijkstra, "q_learning": self._run_ql, "dqn": self._run_dqn}
        for algo in algorithms:
            t = time.perf_counter()
            try:
                with self.lock if algo == "q_learning" else _nullctx():
                    r = runners[algo](traffic, s, d, preference, t0)
                error = None
            except Exception as exc:  # report algorithm failure instead of crashing the comparison
                log.exception("%s failed", algo)
                r, error = {"edges": [], "nodes": [s], "reached": False}, str(exc)
            elapsed = (time.perf_counter() - t) * 1000
            metrics = evaluate_path(self.g, traffic, r["edges"], s, d, t0, cfg) if r["edges"] else None
            results.append(self._package(algo, r, metrics, elapsed, error))
        self._score(results)
        best = max((x for x in results if x["success"]), key=lambda x: x["route_score"], default=None)
        alternatives = []
        if include_alternatives:
            for i, alt in enumerate(alternative_routes(self.g, traffic, s, d, preference, t0, k=2)):
                m = evaluate_path(self.g, traffic, alt["edges"], s, d, t0, cfg)
                alternatives.append({"id": f"alt-{i + 1}", "label": f"Dijkstra alternative {i + 1}",
                                     "metrics": m, "geometry": self.g.path_geometry(alt["edges"])})
        return {
            "source": self._endpoint(s, s_snap, source), "destination": self._endpoint(d, d_snap, destination),
            "preference": preference, "scenario": scenario if scenario != "current" else traffic.scenario,
            "departure_clock": traffic.summary()["clock"], "traffic_mode": traffic.mode,
            "traffic_label": traffic.summary()["label"], "event": event_info, "routes": results,
            "best_algorithm": best["algorithm"] if best else None,
            "explanation": self._explain(results, best, preference, event_info),
            "alternatives": alternatives, "cost_model": cost_model_doc(),
            "reward_config": cfg.to_dict(),
        }

    def _inject_incident(self, traffic: TrafficModel, s, d, pref, t0) -> dict:
        plan = dijkstra_route(self.g, traffic, s, d, pref, t0)
        if not plan["edges"]:
            return {"type": "incident_on_route", "description": "No route to place an incident on."}
        edges = plan["edges"]
        k = max(0, int(len(edges) * 0.45))
        e = edges[k]
        planned_min = float(traffic.travel_time_array(t0)[edges[:k]].sum() / 60) if k else 0.0
        start = t0 + 0.4 * planned_min
        inc = traffic.add_incident(float(self.g.edge_mid_lat[e]), float(self.g.edge_mid_lon[e]), radius_m=600,
                                   severity=0.8, duration_min=60, recovery_min=20, start_min=start,
                                   kind="scenario_event", label="Accident (appears after departure)")
        return {"type": "incident_on_route", "incident": inc.to_dict(t0), "appears_at_min_after_departure":
                round(start - t0, 1), "description": (
                    "A road accident appears on the departure-time fastest corridor "
                    f"{round(start - t0, 1)} min after departure. Static plans cannot react; agents that "
                    "re-evaluate traffic at each intersection can.")}

    def _endpoint(self, node, snap_m, original) -> dict:
        lm = self.landmark_for_node(node)
        return {"node": int(node), "osmid": int(self.g.osmid[node]), "lat": float(self.g.lat[node]),
                "lon": float(self.g.lon[node]), "requested": {"lat": original[0], "lon": original[1]},
                "snap_distance_m": round(snap_m, 1), "landmark": lm["name"] if lm else None}

    def _package(self, algo, r, metrics, elapsed_ms, error) -> dict:
        success = bool(r.get("reached") and metrics and metrics["reached_destination"])
        diag = {k: v for k, v in r.items() if k not in ("edges", "nodes")}
        return {
            "algorithm": algo, "label": ALGO_LABELS[algo], "color": ALGO_COLORS[algo], "success": success,
            "error": error, "computation_ms": round(elapsed_ms, 2), "metrics": metrics,
            "geometry": self.g.path_geometry(r["edges"]) if r["edges"] else [],
            "edge_ids": [int(e) for e in r["edges"]], "node_count": len(r["nodes"]),
            "road_names": _road_names(self.g, r["edges"]), "diagnostics": diag,
        }

    @staticmethod
    def _score(results: list[dict]) -> None:
        costs = [x["metrics"]["generalized_cost"] for x in results if x["success"]]
        best = min(costs) if costs else None
        for x in results:
            x["route_score"] = round(100 * best / x["metrics"]["generalized_cost"], 1) if (
                x["success"] and best and x["metrics"]["generalized_cost"] > 0) else 0.0

    @staticmethod
    def _explain(results, best, pref, event) -> str:
        if not best:
            return "No algorithm reached the destination under these conditions."
        m = best["metrics"]
        parts = [f"{best['label']} gives the best {pref} route (score {best['route_score']}): "
                 f"{m['travel_time_min']} min, {m['distance_km']} km, congestion exposure "
                 f"{m['congestion_exposure']:.2f} ({m['congestion_level']})."]
        for x in results:
            if x is best:
                continue
            if not x["success"]:
                why = x["diagnostics"].get("notes") or x.get("error") or "did not reach the destination"
                parts.append(f"{x['label']} did not produce a valid route ({why}).")
                continue
            o = x["metrics"]
            dt = o["travel_time_min"] - m["travel_time_min"]
            dd = o["distance_km"] - m["distance_km"]
            parts.append(f"{x['label']}: {abs(dt):.1f} min {'slower' if dt >= 0 else 'faster'}, "
                         f"{abs(dd):.2f} km {'longer' if dd >= 0 else 'shorter'}, exposure {o['congestion_exposure']:.2f}"
                         f" (score {x['route_score']}).")
        if event and event.get("type") == "incident_on_route":
            parts.append("Dijkstra planned on the departure-time snapshot, so it could not anticipate the accident "
                         "that appeared after departure; RL agents observe congestion at every intersection.")
        if best["algorithm"] == "dijkstra":
            parts.append("Under these conditions the classical optimal planner was not beaten by the learned "
                         "policies - this is expected when traffic changes little during the trip.")
        return " ".join(parts)

    # ------------------------------------------------------------- network
    def network_geojson(self) -> dict:
        """Undirected display segments (two directions merged) for map overlays."""
        g = self.g
        seen = {}
        segments = []
        for e in range(g.n_edges):
            u, v = int(g.eu[e]), int(g.ev[e])
            key = (min(u, v), max(u, v))
            if key in seen:
                segments[seen[key]]["ids"].append(e)
                continue
            seen[key] = len(segments)
            geom = g.geometry[e]
            if len(geom) > 6:  # light decimation for display only
                step = int(np.ceil(len(geom) / 6))
                geom = geom[::step] + [geom[-1]]
            segments.append({"ids": [e], "c": [[round(p[0], 5), round(p[1], 5)] for p in geom],
                             "k": int(g.road_class[e])})
        return {"segments": segments, "bounds": g.bounds(), "nodes": g.n_nodes, "edges": g.n_edges,
                "meta": {k: v for k, v in g.meta.items() if k != "raw"}}


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _road_names(g: RoadGraph, edges: list[int], limit: int = 8) -> list[str]:
    names = []
    for e in edges:
        n = g.names[e]
        if n and (not names or names[-1] != n) and n not in names:
            names.append(n)
        if len(names) >= limit:
            break
    return names


def load_processed_json(name: str) -> dict | list | None:
    p = PROCESSED_DIR / name
    return json.loads(p.read_text()) if p.exists() else None
