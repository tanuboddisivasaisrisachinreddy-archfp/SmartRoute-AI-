"""FastAPI endpoints (TestClient) on the synthetic graph."""
import time

import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")


@pytest.fixture(scope="module")
def client(grid):
    from fastapi.testclient import TestClient

    from backend.app.core.state import state
    from backend.app.main import app

    state.load(grid)
    state.engine.landmarks = [
        {"id": f"n{i}", "name": f"Grid node {i}", "lat": float(grid.lat[i]), "lon": float(grid.lon[i]), "node": i,
         "snap_distance_m": 0.0, "category": "test", "source": "test"} for i in (0, 11, 70, 143)]
    state.geocoder.landmarks = state.engine.landmarks
    with TestClient(app) as c:
        yield c


def pt(g, i):
    return {"lat": float(g.lat[i]), "lon": float(g.lon[i])}


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    j = r.json()
    assert j["graph_loaded"] and j["traffic_mode"] == "simulation"
    assert j["database"]["type"] == "local-file"


def test_compare_endpoint_returns_geometry_and_metrics(client, grid):
    r = client.post("/api/routes/compare", json={"source": pt(grid, 0), "destination": pt(grid, 143),
                                                 "preference": "fastest", "scenario": "peak",
                                                 "event": "incident_on_route"})
    assert r.status_code == 200, r.text
    j = r.json()
    algos = {x["algorithm"]: x for x in j["routes"]}
    assert set(algos) == {"dijkstra", "q_learning", "dqn"}
    d = algos["dijkstra"]
    assert d["success"] and len(d["geometry"]) > 2 and d["metrics"]["distance_km"] > 0
    assert algos["q_learning"]["success"]
    assert algos["dqn"]["success"] is False  # no checkpoint in the test environment -> reported, not faked
    assert j["traffic_label"] == "Simulated Traffic – Not Live Traffic"
    assert j["best_algorithm"] in ("dijkstra", "q_learning")
    assert j["event"]["type"] == "incident_on_route"


def test_compare_validation(client, grid):
    r = client.post("/api/routes/compare", json={"source": {"lat": 40.7, "lon": -74.0}, "destination": pt(grid, 5)})
    assert r.status_code == 422  # New York is rejected
    r = client.post("/api/routes/compare", json={"source": pt(grid, 3), "destination": pt(grid, 3)})
    assert r.status_code == 422


def test_optimize(client, grid):
    r = client.post("/api/routes/optimize", json={"source": pt(grid, 11), "destination": pt(grid, 70),
                                                  "algorithm": "dijkstra"})
    assert r.status_code == 200 and r.json()["route"]["algorithm"] == "dijkstra"


def test_traffic_endpoints(client, grid):
    assert client.post("/api/traffic/reset", json={"scenario": "low", "clock": "08:00"}).json()["clock"] == "08:00"
    s = client.post("/api/traffic/simulate", json={"scenario": "heavy", "advance_min": 30}).json()
    assert s["scenario"] == "heavy" and s["clock"] == "08:30" and len(s["edge_congestion"]) == grid.n_edges
    inc = client.post("/api/traffic/incident", json={**pt(grid, 40), "severity": 0.8}).json()
    assert inc["incident"]["affected_edges"] > 0
    cur = client.get("/api/traffic/current").json()
    assert cur["active_incidents"] == 1
    assert client.post("/api/traffic/incident/clear", json={"incident_id": inc["incident"]["id"]}).json()["cleared"] == 1
    assert client.post("/api/traffic/incident/clear", json={"incident_id": 9999}).status_code == 404
    assert client.post("/api/traffic/live/refresh").status_code == 400  # no TomTom key -> no fake live data


def test_locations_network_datasets_analytics(client):
    assert client.get("/api/locations/search", params={"q": "Grid node 7"}).status_code == 200
    net = client.get("/api/map/network").json()
    assert net["segments"] and net["edges"] > 0
    ds = client.get("/api/datasets").json()
    assert {c["id"] for c in ds["catalog"]} >= {"osm-hyderabad", "metr-la", "pems-bay"}
    assert next(c for c in ds["catalog"] if c["id"] == "metr-la")["is_hyderabad_data"] is False
    a = client.get("/api/analytics/summary").json()
    assert a["total_comparisons"] >= 1  # recorded by the compare test


def test_experiment_endpoint(client):
    r = client.post("/api/experiments/run", json={"pairs": 2, "scenarios": ["moderate"], "events": ["none"]})
    assert r.status_code == 200, r.text
    assert {s["algorithm"] for s in r.json()["summary"]} == {"dijkstra", "q_learning", "dqn"}
    assert client.get("/api/experiments/latest").status_code == 200


def test_training_job_lifecycle(client):
    r = client.post("/api/training/start", json={"algorithm": "q_learning", "episodes": 20, "goals": "hitech-city"})
    # landmarks of the real list are not on the synthetic graph; the job runs on the real graph only,
    # so here we only check the job is accepted and reported - not its result.
    assert r.status_code in (200, 409), r.text
    for _ in range(50):
        s = client.get("/api/training/status").json()
        if s["state"] not in ("running", "starting"):
            break
        time.sleep(0.2)
    assert s["state"] in ("completed", "failed", "running", "stopped")
    assert client.get("/api/training/history").status_code == 200
