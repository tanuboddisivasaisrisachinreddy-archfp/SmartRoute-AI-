"""Dijkstra baseline (NetworkX) on the Hyderabad road graph.

Edge weights are computed from a *snapshot* of traffic at departure time
(this is what a classical planner knows when the trip starts):

    w_e = w_time * tt_e(t0)[min] + w_dist * len_e[km] + w_cong * c_e(t0) * tt_e(t0)

The weights per preference are the same ones used by the RL reward, so all
three algorithms optimise the same objective.  Alternatives are produced with
the penalty method (edges of previous routes get x1.4 weight).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from reinforcement_learning.environment.graph import RoadGraph
from reinforcement_learning.environment.route_env import PREFERENCES
from reinforcement_learning.environment.traffic import TrafficModel


def edge_weights(graph: RoadGraph, traffic: TrafficModel, preference: str, t0: float) -> np.ndarray:
    cfg = PREFERENCES[preference]
    tt_min = traffic.travel_time_array(t0) / 60.0
    cong = traffic.congestion_array(t0)
    return cfg.w_time * tt_min + cfg.w_dist * graph.length_m / 1000.0 + cfg.w_cong * cong * tt_min


def dijkstra_route(graph: RoadGraph, traffic: TrafficModel, source: int, goal: int,
                   preference: str = "fastest", t0: float | None = None,
                   weights: np.ndarray | None = None) -> dict:
    t0 = traffic.clock_min if t0 is None else t0
    w = edge_weights(graph, traffic, preference, t0) if weights is None else weights
    G = graph.nx_graph
    try:
        nodes = nx.dijkstra_path(G, source, goal, weight=lambda u, v, d: float(w[d["eid"]]))
    except nx.NetworkXNoPath:
        return {"edges": [], "nodes": [source], "reached": False, "planned_cost": None}
    edges = graph.nodes_to_edges(nodes)
    return {"edges": edges, "nodes": nodes, "reached": nodes[-1] == goal,
            "planned_cost": float(w[edges].sum()) if edges else 0.0}


def alternative_routes(graph: RoadGraph, traffic: TrafficModel, source: int, goal: int,
                       preference: str = "fastest", t0: float | None = None, k: int = 2,
                       penalty: float = 1.4) -> list[dict]:
    t0 = traffic.clock_min if t0 is None else t0
    w = edge_weights(graph, traffic, preference, t0).copy()
    routes = []
    seen = set()
    for _ in range(k + 1):
        r = dijkstra_route(graph, traffic, source, goal, preference, t0, weights=w)
        if not r["edges"]:
            break
        key = tuple(r["edges"])
        if key not in seen:
            seen.add(key)
            routes.append(r)
        w[r["edges"]] *= penalty
    return routes[1:]  # first one equals the main Dijkstra route
