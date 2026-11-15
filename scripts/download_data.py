#!/usr/bin/env python3
"""Download the public datasets used by SmartRoute AI.

    python scripts/download_data.py --osm          # Hyderabad road network (OSMnx / Overpass)
    python scripts/download_data.py --landmarks    # geocode Hyderabad landmarks (Nominatim)
    python scripts/download_data.py --traffic      # METR-LA + PEMS-BAY (Zenodo)  ~230 MB
    python scripts/download_data.py --geodata      # inventory of india-geodata urban roads
    python scripts/download_data.py --all

Only the Hyderabad bounding box is downloaded from OpenStreetMap - never the
1.6 GB Geofabrik India extract.
"""
from __future__ import annotations

import argparse
import json
import logging
import time

import _common  # noqa: F401  (sets sys.path)
import requests

from reinforcement_learning.environment.config import (EXTERNAL_DIR, HYDERABAD_BBOX, PROCESSED_DIR, RAW_DIR,
                                                        ROAD_FILTER)
from reinforcement_learning.environment.graph import haversine_m
from reinforcement_learning.environment.landmarks import seed_landmarks

log = logging.getLogger("download")
USER_AGENT = _common.USER_AGENT

ZENODO_RECORDS = {
    # record id -> (dataset, filename filter)
    "4263971": ("PEMS-BAY", lambda k: True),                 # pems-bay.h5 + sensor metadata (~139 MB)
    "5146275": ("METR-LA", lambda k: "metr" in k.lower()),   # METR-LA csv files only (~75 MB)
}


def download_osm() -> None:
    import osmnx as ox

    ox.settings.use_cache = True
    ox.settings.cache_folder = str(RAW_DIR / "osmnx_cache")
    ox.settings.log_console = True
    ox.settings.requests_timeout = 300
    b = HYDERABAD_BBOX
    log.info("Downloading Hyderabad drive network for bbox %s", b)
    log.info("Road filter: %s", ROAD_FILTER)
    bbox = (b["west"], b["south"], b["east"], b["north"])  # OSMnx >= 2.0: (left, bottom, right, top)
    G = ox.graph_from_bbox(bbox, custom_filter=ROAD_FILTER, simplify=True, retain_all=False,
                           truncate_by_edge=True)
    out = RAW_DIR / "hyderabad_drive.graphml"
    out.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(G, out)
    meta = {"downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "bbox": b, "filter": ROAD_FILTER,
            "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "osmnx_version": ox.__version__,
            "source": "OpenStreetMap contributors via Overpass API (ODbL 1.0)"}
    (RAW_DIR / "hyderabad_drive.meta.json").write_text(json.dumps(meta, indent=2))
    log.info("Saved %s (%d nodes, %d edges)", out, meta["nodes"], meta["edges"])


def geocode_landmarks() -> None:
    url = "https://nominatim.openstreetmap.org/search"
    b = HYDERABAD_BBOX
    viewbox = f"{b['west']},{b['north']},{b['east']},{b['south']}"
    results = []
    for lm in seed_landmarks():
        try:
            r = requests.get(url, params={"q": lm["query"], "format": "jsonv2", "limit": 1, "countrycodes": "in",
                                          "viewbox": viewbox, "bounded": 1},
                             headers={"User-Agent": USER_AGENT}, timeout=30)
            r.raise_for_status()
            hits = r.json()
        except Exception as exc:  # network problems -> keep seed
            log.warning("Geocoding %s failed: %s", lm["name"], exc)
            hits = []
        if hits:
            la, lo = float(hits[0]["lat"]), float(hits[0]["lon"])
            dist = float(haversine_m(la, lo, lm["lat"], lm["lon"]))
            if dist <= 3000:
                lm.update(lat=la, lon=lo, source="nominatim", osm_display_name=hits[0].get("display_name"),
                          seed_offset_m=round(dist))
            else:
                lm["geocode_rejected"] = {"lat": la, "lon": lo, "offset_m": round(dist)}
                log.warning("Rejected geocode for %s (%.0f m from seed)", lm["name"], dist)
        results.append(lm)
        time.sleep(1.1)  # Nominatim usage policy: max 1 request / second
    out = RAW_DIR / "landmarks_geocoded.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    log.info("Saved %s (%d via Nominatim)", out, sum(r["source"] == "nominatim" for r in results))


def download_zenodo(record: str, dataset: str, keep) -> None:
    meta = requests.get(f"https://zenodo.org/api/records/{record}", timeout=60).json()
    target = EXTERNAL_DIR / dataset.lower()
    target.mkdir(parents=True, exist_ok=True)
    manifest = {"record": record, "title": meta.get("metadata", {}).get("title"),
                "license": (meta.get("metadata", {}).get("license") or {}).get("id"),
                "doi": meta.get("doi"), "files": []}
    for f in meta.get("files", []):
        key = f["key"]
        if not keep(key):
            continue
        dest = target / key
        url = f["links"]["self"]
        if dest.exists() and dest.stat().st_size == f.get("size"):
            log.info("%s already downloaded", dest)
        else:
            log.info("Downloading %s (%.1f MB)", key, f.get("size", 0) / 1e6)
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        manifest["files"].append({"name": key, "size": f.get("size"), "checksum": f.get("checksum")})
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2))


def download_traffic() -> None:
    for record, (name, keep) in ZENODO_RECORDS.items():
        try:
            download_zenodo(record, name, keep)
        except Exception as exc:
            log.error("Could not download %s automatically (%s).", name, exc)
            log.error("Manual download: open https://zenodo.org/records/%s, download the files and place "
                      "them in %s", record, EXTERNAL_DIR / name.lower())


def inventory_geodata() -> None:
    """Record what the india-geodata repository offers for urban roads.

    The dataset is NOT integrated into the routing graph automatically: its
    coverage, date and licence must be checked first (see data/README.md)."""
    api = "https://api.github.com/repos/yashveeeeeeer/india-geodata/contents/data/infrastructure"
    out = EXTERNAL_DIR / "india_geodata_inventory.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        items = requests.get(api, headers={"User-Agent": USER_AGENT}, timeout=60).json()
        listing = [{"name": i.get("name"), "type": i.get("type"), "path": i.get("path")} for i in items]
        urban = [i for i in listing if "urban" in (i["name"] or "").lower()]
        report = {"checked_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "infrastructure": listing,
                  "urban_road_entries": urban,
                  "note": "Check metadata.json in the urban roads folder for Hyderabad coverage and licence "
                          "before integrating."}
        for u in urban:
            if u["type"] == "dir":
                sub = requests.get(f"https://api.github.com/repos/yashveeeeeeer/india-geodata/contents/{u['path']}",
                                   headers={"User-Agent": USER_AGENT}, timeout=60).json()
                u["files"] = [s.get("name") for s in sub if isinstance(s, dict)]
                hyd = [n for n in u["files"] if n and ("hyderabad" in n.lower() or "telangana" in n.lower())]
                u["hyderabad_related_files"] = hyd
        out.write_text(json.dumps(report, indent=2))
        log.info("Saved %s", out)
    except Exception as exc:
        log.error("Could not query GitHub API (%s). Browse https://yashveeeeeeer.github.io/india-geodata/ "
                  "manually.", exc)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--osm", action="store_true")
    p.add_argument("--landmarks", action="store_true")
    p.add_argument("--traffic", action="store_true")
    p.add_argument("--geodata", action="store_true")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()
    if not any([a.osm, a.landmarks, a.traffic, a.geodata, a.all]):
        a.osm = a.landmarks = True
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if a.osm or a.all:
        download_osm()
    if a.landmarks or a.all:
        geocode_landmarks()
    if a.traffic or a.all:
        download_traffic()
    if a.geodata or a.all:
        inventory_geodata()


if __name__ == "__main__":
    main()
