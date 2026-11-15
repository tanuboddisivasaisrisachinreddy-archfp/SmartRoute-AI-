"""Response models (used for validation and the OpenAPI docs at /docs)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class _Open(BaseModel):
    model_config = ConfigDict(extra="allow")


class RouteMetrics(_Open):
    reached_destination: bool
    distance_km: float
    travel_time_min: float
    congestion_exposure: float
    congestion_level: str
    congested_minutes: float
    segments: int
    total_reward: float
    generalized_cost: float
    fuel_litres: float
    estimated_cost_inr: float
    revisits: int
    avg_speed_kph: float


class RouteResult(_Open):
    algorithm: str
    label: str
    color: str
    success: bool
    error: str | None
    computation_ms: float
    route_score: float
    metrics: RouteMetrics | None
    geometry: list[list[float]]
    edge_ids: list[int]
    node_count: int
    road_names: list[str]
    diagnostics: dict[str, Any]


class Endpoint(_Open):
    node: int
    osmid: int
    lat: float
    lon: float
    snap_distance_m: float
    landmark: str | None


class CompareResponse(_Open):
    id: str | None = None
    source: Endpoint
    destination: Endpoint
    preference: str
    scenario: str
    departure_clock: str
    traffic_mode: str
    traffic_label: str
    event: dict[str, Any] | None
    routes: list[RouteResult]
    best_algorithm: str | None
    explanation: str
    alternatives: list[dict[str, Any]]
    cost_model: dict[str, Any]
    reward_config: dict[str, Any]


class HealthResponse(_Open):
    status: str
    graph_loaded: bool
    traffic_mode: str | None
    database: dict[str, Any]
    models: dict[str, Any]


class LocationResult(_Open):
    name: str
    lat: float
    lon: float
    source: str


class SearchResponse(_Open):
    query: str
    results: list[LocationResult]
    warning: str | None


class TrafficSummary(_Open):
    mode: str
    label: str
    scenario: str
    clock: str
    level: float
    mean_congestion: float
    incidents: list[dict[str, Any]]


class TrainingStatus(_Open):
    state: str
    job: dict[str, Any] | None
    progress: dict[str, Any] | None
