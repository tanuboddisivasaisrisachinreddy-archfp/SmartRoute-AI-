"""HTTP API.  Interactive documentation: /docs (Swagger) and /redoc."""
from __future__ import annotations

import importlib.util
import json
import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Response

from backend.app.core.settings import settings
from backend.app.core.state import state
from backend.app.schemas.api import (ClearIncidentRequest, CompareRequest, ExperimentRequest, IncidentRequest,
                                     OptimizeRequest, ResetRequest, SimulateRequest, TrainingRequest)
from backend.app.schemas.responses import (CompareResponse, HealthResponse, SearchResponse, TrafficSummary,
                                           TrainingStatus)
from backend.app.services.analytics import compact_comparison, summarize_comparisons
from backend.app.services.benchmark import latest_experiment, run_benchmark, save_experiment
from backend.app.services.datasets import CATALOG, dataset_status
from backend.app.services.engine import RoutingError
from reinforcement_learning.dqn.agent import DQNAgent
from reinforcement_learning.environment.config import CHECKPOINT_DIR
from reinforcement_learning.environment.route_env import PREFERENCES
from reinforcement_learning.environment.traffic import SCENARIO_LABELS

log = logging.getLogger("api")
router = APIRouter(prefix="/api")

_network_cache: bytes | None = None


def engine():
    if state.engine is None:
        raise HTTPException(503, detail=f"Road network not loaded: {state.startup_error}. Run "
                                        "scripts/download_data.py and scripts/preprocess_data.py (see README).")
    return state.engine


def _parse_clock(s: str) -> float:
    h, m = s.split(":")
    if not (0 <= int(h) < 24 and 0 <= int(m) < 60):
        raise HTTPException(422, "clock must be HH:MM")
    return int(h) * 60 + int(m)


def _safe_store(col: str, doc: dict) -> str | None:
    try:
        return state.store.insert(col, doc)
    except Exception as exc:
        log.warning("store insert failed: %s", exc)
        return None


# ----------------------------------------------------------------- health
@router.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    eng = state.engine
    return {
        "status": "ok" if eng else "degraded",
        "version": settings.version,
        "uptime_s": round(time.time() - state.started_at, 1),
        "graph_loaded": eng is not None,
        "graph": ({"nodes": eng.g.n_nodes, "edges": eng.g.n_edges, "signature": eng.g.meta.get("signature"),
                   "source": eng.g.meta.get("source"), "synthetic": eng.g.meta.get("synthetic", False)}
                  if eng else None),
        "startup_error": state.startup_error,
        "traffic_mode": eng.traffic.mode if eng else None,
        "tomtom_configured": state.tomtom.configured,
        "database": state.store.status(),
        "models": eng.model_status if eng else {},
        "training": state.trainer.status()["state"],
    }


@router.get("/models", tags=["system"])
def models():
    eng = engine()
    return {"status": eng.model_status, "dqn_architecture": DQNAgent.describe(),
            "reward_preferences": {k: v.to_dict() for k, v in PREFERENCES.items()},
            "q_learning": eng.ql.meta if eng.ql else None, "dqn": eng.dqn.meta if eng.dqn else None}


# ----------------------------------------------------------------- locations
@router.get("/locations/search", response_model=SearchResponse, tags=["locations"])
def search_locations(q: str = Query(..., min_length=1, max_length=120), limit: int = Query(8, ge=1, le=20)):
    engine()
    return state.geocoder.search(q, limit)


@router.get("/locations/landmarks", tags=["locations"])
def landmarks():
    eng = engine()
    return {"landmarks": [{k: lm.get(k) for k in ("id", "name", "lat", "lon", "category", "source", "node",
                                                  "snap_distance_m")} for lm in eng.landmarks]}


@router.get("/locations/reverse", tags=["locations"])
def reverse(lat: float = Query(..., ge=17.0, le=17.8), lon: float = Query(..., ge=78.0, le=78.9)):
    eng = engine()
    res = state.geocoder.reverse(lat, lon)
    node, dist = eng.g.nearest_node(lat, lon)
    res.update(node=node, snap_distance_m=round(dist, 1), routable=dist <= 2500)
    return res


# ----------------------------------------------------------------- map
@router.get("/map/network", tags=["map"])
def network():
    global _network_cache
    eng = engine()
    if _network_cache is None:
        _network_cache = json.dumps(eng.network_geojson(), separators=(",", ":")).encode()
    return Response(_network_cache, media_type="application/json",
                    headers={"Cache-Control": "public, max-age=3600"})


# ----------------------------------------------------------------- routes
@router.post("/routes/compare", response_model=CompareResponse, tags=["routes"])
def compare(req: CompareRequest):
    eng = engine()
    if req.scenario == "live":
        if not state.tomtom.configured:
            raise HTTPException(400, "Live traffic requires TOMTOM_API_KEY on the backend. Use a simulated scenario.")
        try:
            with eng.lock:
                state.tomtom.refresh(eng.g, eng.traffic)
        except Exception as exc:
            raise HTTPException(502, f"TomTom live traffic unavailable: {exc}")
    try:
        res = eng.compare((req.source.lat, req.source.lon), (req.destination.lat, req.destination.lon),
                          req.preference, req.scenario, req.event, tuple(req.algorithms), req.include_alternatives)
    except RoutingError as exc:
        raise HTTPException(422, str(exc))
    if req.source.name:
        res["source"]["name"] = req.source.name
    if req.destination.name:
        res["destination"]["name"] = req.destination.name
    doc = compact_comparison(res)
    res["id"] = _safe_store("route_comparisons", doc)
    _safe_store("searches", {"source": doc["source_name"], "destination": doc["destination_name"],
                             "preference": req.preference, "scenario": req.scenario})
    return res


@router.post("/routes/optimize", tags=["routes"])
def optimize(req: OptimizeRequest):
    eng = engine()
    algos = ("dijkstra", "q_learning", "dqn") if req.algorithm == "best" else (req.algorithm,)
    try:
        res = eng.compare((req.source.lat, req.source.lon), (req.destination.lat, req.destination.lon),
                          req.preference, req.scenario, "none", algos, include_alternatives=False)
    except RoutingError as exc:
        raise HTTPException(422, str(exc))
    chosen = next((r for r in res["routes"] if r["algorithm"] == res["best_algorithm"]), None)
    if chosen is None:
        raise HTTPException(404, "No algorithm reached the destination: " + res["explanation"])
    return {"route": chosen, "explanation": res["explanation"], "traffic_label": res["traffic_label"],
            "source": res["source"], "destination": res["destination"]}


@router.get("/routes/recent", tags=["routes"])
def recent(limit: int = Query(20, ge=1, le=200)):
    return {"items": state.store.find("route_comparisons", limit)}


# ----------------------------------------------------------------- traffic
def _traffic_payload(include_edges: bool) -> dict:
    eng = engine()
    with eng.lock:
        s = eng.traffic.summary()
        if include_edges:
            s["edge_congestion"] = np.round(eng.traffic.congestion_array(), 2).tolist()
        s["history"] = eng.traffic.history[-96:]
    s["scenarios"] = SCENARIO_LABELS
    s["tomtom_configured"] = state.tomtom.configured
    return s


@router.get("/traffic/current", response_model=TrafficSummary, tags=["traffic"])
def traffic_current(include_edges: bool = False):
    return _traffic_payload(include_edges)


@router.post("/traffic/simulate", response_model=TrafficSummary, tags=["traffic"])
def traffic_simulate(req: SimulateRequest):
    eng = engine()
    with eng.lock:
        t = eng.traffic
        if req.scenario:
            t.set_scenario(req.scenario)
        if req.intensity is not None:
            t.set_intensity(req.intensity)
        if req.clock:
            t.clock_min = _parse_clock(req.clock)
            t._cache_t = None
        if req.surge:
            t.trigger_surge(req.surge)
        if req.advance_min:
            t.advance(req.advance_min)
    return _traffic_payload(True)


@router.post("/traffic/advance", response_model=TrafficSummary, tags=["traffic"])
def traffic_advance(minutes: float = Query(15, gt=0, le=240)):
    eng = engine()
    with eng.lock:
        eng.traffic.advance(minutes)
    return _traffic_payload(True)


@router.post("/traffic/incident", tags=["traffic"])
def traffic_incident(req: IncidentRequest):
    eng = engine()
    with eng.lock:
        lat, lon = req.lat, req.lon
        if req.random or lat is None or lon is None:
            major = np.flatnonzero(eng.g.road_class <= 3)
            e = int(np.random.default_rng().choice(major))
            lat, lon = float(eng.g.edge_mid_lat[e]), float(eng.g.edge_mid_lon[e])
        node, dist = eng.g.nearest_node(lat, lon)
        if dist > 3000:
            raise HTTPException(422, "Incident location is outside the Hyderabad road network")
        inc = eng.traffic.add_incident(lat, lon, req.radius_m, req.severity, req.duration_min, req.recovery_min,
                                       label=req.label)
    return {"incident": inc.to_dict(eng.traffic.clock_min), "traffic": _traffic_payload(True)}


@router.post("/traffic/incident/clear", tags=["traffic"])
def traffic_clear(req: ClearIncidentRequest):
    eng = engine()
    with eng.lock:
        n = eng.traffic.clear_incident(req.incident_id)
    if req.incident_id is not None and n == 0:
        raise HTTPException(404, f"incident {req.incident_id} not found")
    return {"cleared": n, "traffic": _traffic_payload(True)}


@router.post("/traffic/reset", response_model=TrafficSummary, tags=["traffic"])
def traffic_reset(req: ResetRequest | None = None):
    eng = engine()
    req = req or ResetRequest()
    with eng.lock:
        eng.traffic.reset(req.scenario, _parse_clock(req.clock))
        eng.traffic.start_clock = _parse_clock(req.clock)
    return _traffic_payload(True)


@router.post("/traffic/live/refresh", tags=["traffic"])
def traffic_live_refresh(force: bool = False):
    eng = engine()
    if not state.tomtom.configured:
        raise HTTPException(400, "TOMTOM_API_KEY is not configured on the backend - live mode unavailable.")
    try:
        with eng.lock:
            res = state.tomtom.refresh(eng.g, eng.traffic, force)
    except Exception as exc:
        raise HTTPException(502, f"TomTom request failed: {exc}")
    return {"live": res, "traffic": _traffic_payload(True)}


@router.post("/traffic/live/disable", tags=["traffic"])
def traffic_live_disable():
    eng = engine()
    with eng.lock:
        eng.traffic.live_obs = {}
        eng.traffic.live_meta = {}
        eng.traffic._cache_t = None
    return _traffic_payload(True)


# ----------------------------------------------------------------- training
@router.post("/training/start", response_model=TrainingStatus, tags=["training"])
def training_start(req: TrainingRequest):
    engine()
    if not settings.enable_server_training:
        raise HTTPException(403, "Server-side training is disabled (ENABLE_SERVER_TRAINING=false). "
                                 "Train offline with the scripts and upload checkpoints.")
    if req.algorithm == "dqn" and importlib.util.find_spec("torch") is None:
        raise HTTPException(503, "PyTorch is not installed on this server, so DQN training is unavailable here. "
                                 "Train on your computer with `python scripts/train_dqn.py` and deploy the checkpoint.")
    limit = settings.max_ql_episodes if req.algorithm == "q_learning" else settings.max_dqn_episodes
    if req.episodes > limit:
        raise HTTPException(422, f"episodes limited to {limit} on this server (MAX_TRAINING_EPISODES_*).")
    params = req.model_dump()
    if params["learning_rate"] is None:
        params["learning_rate"] = 0.3 if req.algorithm == "q_learning" else 5e-4
    if params["discount_factor"] is None:
        params["discount_factor"] = 1.0 if req.algorithm == "q_learning" else 0.995
    try:
        return state.trainer.start(req.algorithm, params)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc))


@router.post("/training/stop", response_model=TrainingStatus, tags=["training"])
def training_stop():
    return state.trainer.stop()


@router.get("/training/status", response_model=TrainingStatus, tags=["training"])
def training_status():
    return state.trainer.status()


@router.get("/training/history", tags=["training"])
def training_history():
    out = {}
    for algo, fname in (("q_learning", "q_learning_history.json"), ("dqn", "dqn_history.json")):
        p = CHECKPOINT_DIR / fname
        if p.exists():
            try:
                d = json.loads(p.read_text())
                meta = d.get("meta", {})
                hist = d.get("history", [])
                curve = meta.get("curve") or []
                out[algo] = {"curve": curve, "summary": meta.get("summary"),
                             "config": {k: v for k, v in meta.items() if k not in ("curve", "summary", "final")},
                             "episodes": len(hist), "final": meta.get("final")}
            except Exception as exc:
                out[algo] = {"error": str(exc)}
        else:
            out[algo] = None
    out["runs"] = [{k: v for k, v in r.items() if k != "curve"} for r in state.store.find("training_runs", 20)]
    return out


# ----------------------------------------------------------------- experiments
@router.post("/experiments/run", tags=["experiments"])
def experiments_run(req: ExperimentRequest):
    eng = engine()
    n_runs = req.pairs * len(req.scenarios) * len(req.preferences) * len(req.events)
    if n_runs > 60:
        raise HTTPException(422, f"{n_runs} comparisons requested; maximum per request is 60. "
                                 "Use scripts/evaluate_models.py for larger experiments.")
    res = run_benchmark(eng, req.pairs, req.scenarios, req.preferences, req.events, req.seed)
    if not res["rows"]:
        raise HTTPException(422, "No routable landmark pairs found in the loaded road network.")
    save_experiment(res)
    _safe_store("experiments", res)
    return res


@router.get("/experiments/latest", tags=["experiments"])
def experiments_latest():
    res = latest_experiment()
    if res is None:
        items = state.store.find("experiments", 1)
        res = items[0] if items else None
    if res is None:
        raise HTTPException(404, "No experiment has been run yet.")
    return res


# ----------------------------------------------------------------- analytics
@router.get("/analytics/summary", tags=["analytics"])
def analytics_summary(limit: int = Query(500, ge=1, le=5000)):
    docs = state.store.find("route_comparisons", limit)
    out = summarize_comparisons(docs)
    out["total_searches"] = state.store.count("searches")
    out["database"] = state.store.status()["type"]
    exp = latest_experiment()
    out["latest_experiment"] = {"created_at": exp["created_at"], "summary": exp["summary"],
                                "config": exp["config"]} if exp else None
    eng = state.engine
    out["traffic"] = eng.traffic.summary() if eng else None
    if out["traffic"]:
        out["traffic"].pop("incidents", None)
        out["traffic"]["history"] = eng.traffic.history[-96:]
    out["models"] = eng.model_status if eng else {}
    out["training"] = state.trainer.status()["state"]
    return out


# ----------------------------------------------------------------- datasets
@router.get("/datasets", tags=["datasets"])
def datasets():
    return {"catalog": CATALOG, "status": dataset_status()}
