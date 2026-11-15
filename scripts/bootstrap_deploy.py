#!/usr/bin/env python3
"""Prepare data + models on a fresh server (Render build step).

Render's filesystem starts empty on each deploy, so the build must create
the road graph and checkpoints.  Strategy, in order:

1. ARTIFACTS_URL set  -> download the archive made by package_artifacts.py
                         (recommended: real trained models, fast build).
2. Graph already present (e.g. committed processed file) -> keep it.
3. BOOTSTRAP_FROM_OSM=true -> install osmnx, download Hyderabad from OSM,
                         preprocess, and train a quick Q-learning model.
                         (DQN needs PyTorch and is not trained here.)

Never fails the build silently: missing pieces are printed clearly and the
API reports them on /api/health.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

import _common  # noqa: F401

from reinforcement_learning.environment.config import CHECKPOINT_DIR, GRAPH_FILE, REPO_ROOT


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def from_archive(url: str) -> None:
    print(f"Downloading artifacts from {url}")
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "a.tar.gz")
        urllib.request.urlretrieve(url, path)
        with tarfile.open(path) as tar:
            for m in tar.getmembers():
                if m.name.startswith(("/", "..")) or ".." in m.name.split("/"):
                    raise SystemExit(f"unsafe path in archive: {m.name}")
                if not m.name.startswith(("data/processed/", "reinforcement_learning/checkpoints/")):
                    raise SystemExit(f"unexpected file in archive: {m.name}")
            tar.extractall(REPO_ROOT)
    print("Artifacts extracted.")


def main() -> None:
    url = os.getenv("ARTIFACTS_URL", "").strip()
    if url:
        from_archive(url)
    elif GRAPH_FILE.exists():
        print(f"Using existing graph {GRAPH_FILE}")
    elif os.getenv("BOOTSTRAP_FROM_OSM", "false").lower() == "true":
        run([sys.executable, "-m", "pip", "install", "osmnx>=2.0"])
        run([sys.executable, "scripts/download_data.py", "--osm", "--landmarks"])
        run([sys.executable, "scripts/preprocess_data.py"])
        run([sys.executable, "scripts/validate_data.py"])
        episodes = os.getenv("BOOTSTRAP_QL_EPISODES", "600")
        run([sys.executable, "scripts/train_q_learning.py", "--episodes", episodes, "--preferences", "fastest"])
    else:
        print("WARNING: no road graph available. Set ARTIFACTS_URL (recommended) or BOOTSTRAP_FROM_OSM=true.")
        return
    for name in ("q_learning.npz", "dqn.npz"):
        p = CHECKPOINT_DIR / name
        print(f"{name}: {'present' if p.exists() else 'MISSING'}")
    print(f"graph: {'present' if GRAPH_FILE.exists() else 'MISSING'}")


if __name__ == "__main__":
    main()
