# API reference (summary)

Base URL: `http://localhost:8000` locally. Full schemas: `/docs`.

## POST /api/routes/compare
```json
{
  "source": {"lat": 17.4504, "lon": 78.3810, "name": "Hitech City"},
  "destination": {"lat": 17.3616, "lon": 78.4747, "name": "Charminar"},
  "preference": "fastest",            // fastest | shortest | balanced
  "scenario": "peak",                 // current | low | moderate | heavy | peak | random | live
  "event": "incident_on_route",       // none | incident_on_route | surge
  "algorithms": ["dijkstra", "q_learning", "dqn"],
  "include_alternatives": true
}
```
Coordinates outside the Hyderabad coverage box are rejected with 422.

Response (abridged):
```json
{
  "source": {"node": 812, "lat": 17.4501, "lon": 78.3807, "snap_distance_m": 38.2, "landmark": "Hitech City"},
  "traffic_mode": "simulation",
  "traffic_label": "Simulated Traffic – Not Live Traffic",
  "routes": [{
    "algorithm": "q_learning", "label": "Q-Learning", "color": "#16a34a", "success": true,
    "computation_ms": 12.4, "route_score": 100.0,
    "metrics": {"distance_km": 16.2, "travel_time_min": 41.3, "congestion_exposure": 0.47,
                "congestion_level": "Heavy", "segments": 87, "total_reward": 3.1,
                "generalized_cost": 52.0, "estimated_cost_inr": 139.6},
    "geometry": [[17.4501, 78.3807], "..."],
    "diagnostics": {"training_mode": "pretrained", "fallback_steps": 0, "notes": "..."}
  }],
  "best_algorithm": "q_learning",
  "explanation": "Q-Learning gives the best fastest route ...",
  "alternatives": [{"id": "alt-1", "metrics": {}, "geometry": []}]
}
```
`route_score` = 100 × (lowest generalized cost among successful routes) / (this route's cost).

## Traffic
* `GET /api/traffic/current?include_edges=true` returns the scenario, clock, level, mean/p90 congestion, incidents, history, and `edge_congestion[]` indexed by edge id.
* `POST /api/traffic/simulate` `{"scenario":"heavy","intensity":1.2,"advance_min":15,"surge":0.3,"clock":"18:30"}` (all fields optional).
* `POST /api/traffic/advance?minutes=15`
* `POST /api/traffic/incident` `{"lat":17.4375,"lon":78.4482,"radius_m":800,"severity":0.7,"duration_min":45}` or `{"random":true}`
* `POST /api/traffic/incident/clear` `{"incident_id": 3}` (or `null` to clear all)
* `POST /api/traffic/reset` `{"scenario":"moderate","clock":"09:30"}`
* `POST /api/traffic/live/refresh` returns 400 when `TOMTOM_API_KEY` is not set.

## Training
* `POST /api/training/start` `{"algorithm":"q_learning","episodes":300,"learning_rate":0.5,"discount_factor":0.99,"epsilon":1.0,"seed":42,"preferences":["fastest"],"goals":"hitech-city,charminar"}`
* `POST /api/training/start` `{"algorithm":"dqn","episodes":300,"learning_rate":0.0005,"batch_size":128,"buffer_size":40000}` returns 503 if PyTorch is missing on the server.
* `GET /api/training/status` returns the state (idle/running/completed/failed/stopped) and progress: current episode, total, average reward, episode reward, epsilon, loss, completion rate and curve.
* `GET /api/training/history` returns the saved curves from the checkpoint folder plus the stored runs.

## Experiments / analytics
* `POST /api/experiments/run` `{"pairs":6,"scenarios":["moderate","peak"],"preferences":["fastest"],"events":["none","incident_on_route"]}` (≤ 60 trips)
* `GET /api/experiments/latest`, `GET /api/analytics/summary`, `GET /api/datasets`, `GET /api/models`
