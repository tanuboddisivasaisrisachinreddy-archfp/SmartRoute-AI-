"""Helpers to load the graph and landmark goals for training scripts."""
from __future__ import annotations

import json

from ..environment.config import LANDMARKS_FILE
from ..environment.graph import RoadGraph, build_synthetic_grid
from ..environment.landmarks import seed_landmarks


def load_graph(synthetic: bool = False) -> RoadGraph:
    if synthetic:
        return build_synthetic_grid(14, 14)
    return RoadGraph.load()


def landmark_goals(graph: RoadGraph, synthetic: bool = False) -> list[tuple[str, int]]:
    if synthetic:
        n = graph.n_nodes
        return [("grid-corner", n - 1), ("grid-centre", n // 2 + 7), ("grid-left", 7 * 14)]
    if LANDMARKS_FILE.exists():
        lms = json.loads(LANDMARKS_FILE.read_text())
    else:
        lms = seed_landmarks()
        for lm in lms:
            lm["node"] = graph.nearest_node(lm["lat"], lm["lon"])[0]
    seen, goals = set(), []
    for lm in lms:
        if lm["node"] not in seen:
            seen.add(lm["node"])
            goals.append((lm["id"], int(lm["node"])))
    return goals
