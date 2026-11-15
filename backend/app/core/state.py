"""Process-wide application state, created once at startup."""
from __future__ import annotations

import logging
import time

from backend.app.core.settings import settings
from backend.app.database.store import get_store
from backend.app.services.engine import RoutingEngine
from backend.app.services.geocoding import Geocoder
from backend.app.services.training_jobs import TrainingJobManager
from backend.app.traffic.tomtom import TomTomProvider
from reinforcement_learning.environment.graph import RoadGraph

log = logging.getLogger("state")


class AppState:
    def __init__(self):
        self.started_at = time.time()
        self.engine: RoutingEngine | None = None
        self.startup_error: str | None = None
        self.store = get_store()
        self.tomtom = TomTomProvider(settings.tomtom_api_key, settings.tomtom_sample_points,
                                     settings.tomtom_min_refresh_s)
        self.trainer = TrainingJobManager(on_complete=self.reload_models, store=self.store)
        self.geocoder: Geocoder | None = None
        self.last_experiment: dict | None = None

    def load(self, graph: RoadGraph | None = None) -> None:
        try:
            graph = graph or RoadGraph.load()
            self.engine = RoutingEngine(graph, ql_ondemand_episodes=settings.ql_ondemand_episodes,
                                        ql_ondemand_budget_s=settings.ql_ondemand_budget_s,
                                        dqn_use_torch=settings.dqn_inference_backend == "torch")
            self.geocoder = Geocoder(self.engine.landmarks, settings.nominatim_enabled, settings.nominatim_url,
                                     settings.contact_email)
            log.info("Graph loaded: %d nodes / %d edges", graph.n_nodes, graph.n_edges)
        except Exception as exc:
            self.startup_error = str(exc)
            log.error("Routing engine not available: %s", exc)

    def reload_models(self) -> dict:
        if self.engine is None:
            raise RuntimeError("engine not loaded")
        with self.engine.lock:
            return self.engine.load_models()


state = AppState()
