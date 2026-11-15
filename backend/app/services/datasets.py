"""Dataset catalogue (documentation) + live status of files on this server."""
from __future__ import annotations

import json

from reinforcement_learning.environment.config import (EXTERNAL_DIR, GRAPH_FILE, LANDMARKS_FILE, PROCESSED_DIR,
                                                        RAW_DIR, ROAD_FILTER)

CATALOG = [
    {
        "id": "osm-hyderabad",
        "name": "OpenStreetMap – Hyderabad road network",
        "source_urls": ["https://www.openstreetmap.org/", "https://download.geofabrik.de/asia/india.html"],
        "geographic_coverage": "Hyderabad, Telangana, India (bbox incl. Miyapur, Uppal, LB Nagar, RGIA airport)",
        "is_hyderabad_data": True,
        "features": ["intersections (nodes)", "road segments (edges)", "length", "highway class", "one-way",
                     "lanes", "maxspeed", "road names", "geometry"],
        "file_format": "OSM XML via Overpass → GraphML (raw) → gzipped JSON (processed)",
        "license": "Open Database License (ODbL) 1.0 – © OpenStreetMap contributors",
        "preprocessing": ["OSMnx graph_from_bbox with arterial road filter", "largest strongly connected component",
                          "parallel edges collapsed, self-loops removed", "free-flow speed assignment",
                          "landmark snapping"],
        "intended_use": "Primary routing environment for Dijkstra, Q-Learning and DQN.",
        "filter": ROAD_FILTER,
    },
    {
        "id": "metr-la",
        "name": "METR-LA traffic speed dataset",
        "source_urls": ["https://zenodo.org/records/5146275", "https://github.com/liyaguang/DCRNN"],
        "geographic_coverage": "Los Angeles County highways, USA (207 loop detectors, Mar–Jun 2012)",
        "is_hyderabad_data": False,
        "features": ["speed (mph) per sensor", "5-minute timestamps", "sensor graph"],
        "file_format": "CSV / HDF5",
        "license": "As published on Zenodo (see record); cite Li et al., ICLR 2018 (DCRNN)",
        "preprocessing": ["zeros treated as missing", "free-flow = p95 speed per sensor",
                          "congestion = 1 − speed/free-flow", "weekday hourly mean profile"],
        "intended_use": "Benchmarking and traffic-pattern research only – NOT Hyderabad traffic.",
        "citation": "Li, Y., Yu, R., Shahabi, C., Liu, Y. Diffusion Convolutional Recurrent Neural Network: "
                    "Data-Driven Traffic Forecasting. ICLR 2018.",
    },
    {
        "id": "pems-bay",
        "name": "PEMS-BAY traffic speed dataset",
        "source_urls": ["https://zenodo.org/records/4263971", "https://zenodo.org/records/4264445"],
        "geographic_coverage": "San Francisco Bay Area, USA (325 Caltrans PeMS sensors, Jan–Jun 2017)",
        "is_hyderabad_data": False,
        "features": ["speed (mph) per sensor", "5-minute timestamps", "sensor locations"],
        "file_format": "HDF5 (pems-bay.h5) + CSV metadata",
        "license": "As published on Zenodo (see record); data originally from Caltrans PeMS",
        "preprocessing": ["same pipeline as METR-LA"],
        "intended_use": "Time-series experiments and benchmarking only – NOT Hyderabad traffic.",
        "citation": "Li et al., ICLR 2018 (DCRNN).",
    },
    {
        "id": "india-geodata",
        "name": "India Geodata – urban roads (evaluated, not integrated)",
        "source_urls": ["https://yashveeeeeeer.github.io/india-geodata/",
                        "https://github.com/yashveeeeeeer/india-geodata"],
        "geographic_coverage": "India-wide collection; Hyderabad urban-road coverage must be verified per file",
        "is_hyderabad_data": None,
        "features": ["road line geometries (varies by source)"],
        "file_format": "Parquet / PMTiles / GeoJSONL / Shapefile / GeoJSON",
        "license": "Repository: CC BY 4.0; individual sources CC0 / ODbL / GODL – check metadata.json",
        "preprocessing": ["inventory only: scripts/download_data.py --geodata"],
        "intended_use": "Candidate supplementary source. Not used for routing because the OSM graph already "
                        "provides topology (one-way, connectivity) that line-only datasets lack.",
    },
    {
        "id": "tomtom",
        "name": "TomTom Traffic Flow API (optional live traffic)",
        "source_urls": ["https://developer.tomtom.com/"],
        "geographic_coverage": "Global incl. Hyderabad (sampled arterial points)",
        "is_hyderabad_data": True,
        "features": ["currentSpeed", "freeFlowSpeed", "confidence", "roadClosure"],
        "file_format": "JSON over HTTPS",
        "license": "TomTom developer terms – API key required, stored only in backend env vars",
        "preprocessing": ["congestion = 1 − current/free-flow", "assigned to edges within 400 m"],
        "intended_use": "Optional live mode. Without a key the app uses the labelled simulation.",
    },
]


def _size(p):
    return p.stat().st_size if p.exists() else None


def dataset_status() -> dict:
    stats = PROCESSED_DIR / "graph_stats.json"
    prof = PROCESSED_DIR / "traffic_profiles.json"
    inv = EXTERNAL_DIR / "india_geodata_inventory.json"
    val = PROCESSED_DIR / "validation_report.json"
    return {
        "raw_graphml_present": (RAW_DIR / "hyderabad_drive.graphml").exists(),
        "processed_graph": {"path": str(GRAPH_FILE), "bytes": _size(GRAPH_FILE)},
        "graph_stats": json.loads(stats.read_text()) if stats.exists() else None,
        "landmarks_present": LANDMARKS_FILE.exists(),
        "metr_la_present": (EXTERNAL_DIR / "metr-la").exists(),
        "pems_bay_present": (EXTERNAL_DIR / "pems-bay").exists(),
        "traffic_profiles": json.loads(prof.read_text()) if prof.exists() else None,
        "india_geodata_inventory": json.loads(inv.read_text()) if inv.exists() else None,
        "validation": json.loads(val.read_text()) if val.exists() else None,
    }
