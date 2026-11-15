#!/usr/bin/env python3
"""Bundle the processed Hyderabad graph + trained checkpoints for deployment.

    python scripts/package_artifacts.py            # -> dist/smartroute-artifacts.tar.gz

Upload the archive as a GitHub Release asset and set ARTIFACTS_URL on Render
to its download URL.  Large binaries never need to be committed to git.
"""
from __future__ import annotations

import hashlib
import tarfile

import _common  # noqa: F401

from reinforcement_learning.environment.config import CHECKPOINT_DIR, PROCESSED_DIR, REPO_ROOT

FILES = [
    PROCESSED_DIR / "hyderabad_graph.json.gz",
    PROCESSED_DIR / "graph_stats.json",
    PROCESSED_DIR / "landmarks.json",
    PROCESSED_DIR / "traffic_profiles.json",
    PROCESSED_DIR / "validation_report.json",
    PROCESSED_DIR / "experiments" / "latest.json",
    CHECKPOINT_DIR / "q_learning.npz",
    CHECKPOINT_DIR / "q_learning_history.json",
    CHECKPOINT_DIR / "dqn.npz",
    CHECKPOINT_DIR / "dqn.pt",
    CHECKPOINT_DIR / "dqn_history.json",
]


def main() -> None:
    out = REPO_ROOT / "dist" / "smartroute-artifacts.tar.gz"
    missing = [str(f.relative_to(REPO_ROOT)) for f in FILES[:3] if not f.exists()]
    if missing:
        raise SystemExit(f"Required files missing: {missing}. Run download/preprocess first.")
    out.parent.mkdir(exist_ok=True)
    with tarfile.open(out, "w:gz") as tar:
        for f in FILES:
            if f.exists():
                tar.add(f, arcname=str(f.relative_to(REPO_ROOT)))
                print("added", f.relative_to(REPO_ROOT))
            else:
                print("skipped (not found)", f.relative_to(REPO_ROOT))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"\n{out}  {out.stat().st_size / 1e6:.1f} MB  sha256={sha}")


if __name__ == "__main__":
    main()
