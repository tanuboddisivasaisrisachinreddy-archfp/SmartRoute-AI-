# Data

Large raw files are **not committed**. Everything here is reproducible with the scripts in `scripts/`.

```
data/
├── raw/         downloaded as-is (OSM GraphML, geocoding results)          – git-ignored
├── external/    third-party benchmark datasets (METR-LA, PEMS-BAY, ...)    – git-ignored
├── processed/   compact files the app loads (graph, landmarks, profiles)   – git-ignored, shipped via ARTIFACTS_URL
└── runtime/     local JSON-lines store used when MongoDB is not configured – git-ignored
```

```bash
python scripts/download_data.py --osm --landmarks   # Hyderabad road network + geocoded landmarks
python scripts/download_data.py --traffic           # METR-LA + PEMS-BAY (~215 MB, optional)
python scripts/download_data.py --geodata           # india-geodata inventory (optional)
python scripts/preprocess_data.py                   # build data/processed/*
python scripts/validate_data.py                     # sanity checks -> validation_report.json
```

---

## 1. OpenStreetMap – Hyderabad road network (primary routing environment)

| | |
|---|---|
| **Name** | OpenStreetMap drive network, Hyderabad |
| **Source URL** | https://www.openstreetmap.org/ (via Overpass API, downloaded with OSMnx). Regional extract for reference: https://download.geofabrik.de/asia/india.html (not downloaded – 1.6 GB) |
| **Geographic coverage** | Bounding box N 17.535, S 17.215, E 78.585, W 78.305 – Miyapur/Kukatpally to RGIA airport, Financial District to Uppal/LB Nagar. Configurable with `BBOX_*` env vars. |
| **Features** | intersections (lat/lon, OSM id), directed road segments, length (m), `highway` class, one-way, lanes, maxspeed, name/ref, LineString geometry |
| **File format** | Overpass JSON → `raw/hyderabad_drive.graphml` → `processed/hyderabad_graph.json.gz` |
| **License** | Open Database License (ODbL) 1.0 – © OpenStreetMap contributors. Attribution is shown on every map. |
| **Preprocessing** | road filter `motorway…secondary` (+ links) · largest strongly connected component · self-loops removed · parallel edges collapsed (shortest kept) · `_link` mapped to parent class · free-flow speed = class default or 0.8 × OSM maxspeed (bounded) · landmarks snapped to the nearest intersection |
| **Intended use** | The graph on which Dijkstra, Q-Learning and DQN operate. |

Why only arterial roads? Including tertiary/residential streets multiplies the graph roughly ten-fold; tabular Q-learning then needs far more episodes and the API exceeds Render free-tier memory. Change `OSM_ROAD_FILTER` to include `tertiary` if you have the compute.

### Landmarks
19 required Hyderabad locations are defined in `reinforcement_learning/environment/landmarks.py` with approximate seed coordinates. `download_data.py --landmarks` re-geocodes each with **Nominatim** (bounded to the Hyderabad box, 1 request/s per the OSM usage policy). A result is accepted only if it lies within 3 km of the seed; otherwise the seed is kept and the rejection is logged in `raw/landmarks_geocoded.json`.

## 2. METR-LA (benchmark only – Los Angeles, USA)

| | |
|---|---|
| **Source URL** | https://zenodo.org/records/5146275 · original: https://github.com/liyaguang/DCRNN |
| **Geographic coverage** | 207 loop detectors on Los Angeles County highways, March–June 2012. **Not Hyderabad.** |
| **Features** | traffic speed (mph) per sensor at 5-minute intervals |
| **File format** | CSV (Zenodo record), HDF5 in the original release |
| **License** | As stated on the Zenodo record; cite Li et al., *DCRNN*, ICLR 2018 |
| **Preprocessing** | zeros → missing · free-flow = 95th percentile speed per sensor · congestion = 1 − speed / free-flow · weekday mean by hour |
| **Intended use** | Benchmarking the shape of daily congestion and testing the traffic-feature pipeline. Shown on the Datasets page next to the simulator's assumed Hyderabad profile, always labelled with its origin. |

## 3. PEMS-BAY (benchmark only – San Francisco Bay Area, USA)

| | |
|---|---|
| **Source URL** | https://zenodo.org/records/4263971 (also https://zenodo.org/records/4264445) |
| **Geographic coverage** | 325 Caltrans PeMS sensors, Bay Area, January–June 2017. **Not Hyderabad.** |
| **Features** | traffic speed (mph) per sensor, 5-minute intervals; sensor locations |
| **File format** | HDF5 `pems-bay.h5` (needs `tables`) + CSV metadata |
| **License** | As stated on the Zenodo record; data from Caltrans PeMS; cite Li et al., ICLR 2018 |
| **Preprocessing** | identical to METR-LA |
| **Intended use** | Time-series experiments and benchmarking only. |

If the Zenodo API is unreachable, download the files manually from the record pages and place them in `data/external/metr-la/` and `data/external/pems-bay/`, then run `python scripts/preprocess_data.py --traffic-only`.

## 4. India Geodata – urban roads (evaluated, not integrated)

| | |
|---|---|
| **Source URL** | https://yashveeeeeeer.github.io/india-geodata/ · https://github.com/yashveeeeeeer/india-geodata |
| **Geographic coverage** | India-wide catalogue. The README lists "Urban roads" under `data/infrastructure/`, sourced from ramSeraph/indian_transport; per-city coverage (Hyderabad) is not stated on the landing page. |
| **Features** | road line geometries (attributes depend on the source) |
| **File format** | Parquet, PMTiles, GeoJSONL, Shapefile, GeoJSON |
| **License** | Repository CC BY 4.0; underlying sources CC0 / ODbL / GODL – check each `metadata.json` |
| **Preprocessing** | `download_data.py --geodata` writes `external/india_geodata_inventory.json` listing the infrastructure folder and any files mentioning Hyderabad/Telangana |
| **Intended use** | Candidate supplementary layer. **Not used for routing**: line datasets lack the turn/one-way/connectivity topology that the OSM graph provides, and coverage/date must be verified first. |

## 5. TomTom Traffic Flow API (optional live traffic)

| | |
|---|---|
| **Source URL** | https://developer.tomtom.com/ – Flow Segment Data endpoint |
| **Coverage** | Hyderabad arterial roads (sampled at `TOMTOM_SAMPLE_POINTS` grid points, default 30) |
| **Features** | currentSpeed, freeFlowSpeed, confidence, roadClosure |
| **License** | TomTom developer terms; key kept in the backend env var `TOMTOM_API_KEY` only |
| **Preprocessing** | congestion = 1 − current / free-flow, assigned to edges within 400 m of a sample |
| **Intended use** | Optional live mode, labelled "Live traffic (TomTom) blended with simulation". Without a key the app never claims live data. |

## Simulated traffic

When no live provider is configured, all congestion values come from the documented simulator in `reinforcement_learning/environment/traffic.py` (see `docs/traffic_model.md`) and the UI shows **"Simulated Traffic – Not Live Traffic"**. Hotspot locations and the daily profile are modelling assumptions, not measurements.
