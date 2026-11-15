"""Shared configuration for the Hyderabad routing environment.

All paths can be overridden with environment variables so the same code runs
locally (macOS), in tests, and on Render.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("SMARTROUTE_DATA_DIR", REPO_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
RUNTIME_DIR = Path(os.getenv("SMARTROUTE_RUNTIME_DIR", DATA_DIR / "runtime"))

CHECKPOINT_DIR = Path(
    os.getenv("MODEL_CHECKPOINT_DIR", REPO_ROOT / "reinforcement_learning" / "checkpoints")
)

GRAPH_FILE = Path(os.getenv("GRAPH_FILE", PROCESSED_DIR / "hyderabad_graph.json.gz"))
LANDMARKS_FILE = PROCESSED_DIR / "landmarks.json"

# Bounding box that covers every location the project must support, from
# Miyapur / Kukatpally in the north-west to RGIA airport in the south and
# Uppal / LB Nagar in the east.  (north, south, east, west) in degrees.
HYDERABAD_BBOX = {
    "north": float(os.getenv("BBOX_NORTH", 17.535)),
    "south": float(os.getenv("BBOX_SOUTH", 17.215)),
    "east": float(os.getenv("BBOX_EAST", 78.585)),
    "west": float(os.getenv("BBOX_WEST", 78.305)),
}

# Road classes kept in the routing graph (arterial network).  Tertiary and
# residential streets are excluded by default
# on purpose: they would multiply the graph size ~10x, which makes tabular
# Q-learning infeasible and exceeds Render's free-tier memory.
ROAD_FILTER = os.getenv(
    "OSM_ROAD_FILTER",
    '["highway"~"motorway|motorway_link|trunk|trunk_link|primary|primary_link|'
    'secondary|secondary_link"]',
)

ROAD_CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "other"]

# Assumed free-flow urban speeds (km/h) used when OSM has no usable maxspeed.
# These are modelling assumptions for Hyderabad city driving, documented in
# docs/traffic_model.md - not measurements.
DEFAULT_SPEEDS_KPH = {
    "motorway": 80.0,
    "trunk": 50.0,
    "primary": 40.0,
    "secondary": 32.0,
    "tertiary": 26.0,
    "other": 20.0,
}


def road_class_of(highway: str | list | None) -> str:
    """Map an OSM highway tag (possibly a list) to one of ROAD_CLASSES."""
    if isinstance(highway, (list, tuple)):
        # choose the most important class present
        classes = [road_class_of(h) for h in highway]
        return min(classes, key=ROAD_CLASSES.index)
    if not highway:
        return "other"
    base = str(highway).replace("_link", "")
    return base if base in ROAD_CLASSES else "other"
