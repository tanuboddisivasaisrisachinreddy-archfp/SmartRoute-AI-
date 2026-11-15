"""Shared fixtures.

Tests never write into data/processed or the real checkpoint folder: runtime,
experiment and checkpoint directories are redirected to a temp folder before
any project module is imported.
"""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_TMP = tempfile.mkdtemp(prefix="smartroute-tests-")
os.environ["SMARTROUTE_RUNTIME_DIR"] = os.path.join(_TMP, "runtime")
os.environ["SMARTROUTE_EXPERIMENT_DIR"] = os.path.join(_TMP, "experiments")
os.environ.setdefault("MODEL_CHECKPOINT_DIR", os.path.join(_TMP, "checkpoints"))
os.environ["NOMINATIM_ENABLED"] = "false"
os.environ.pop("MONGODB_URI", None)
os.environ.pop("TOMTOM_API_KEY", None)

import pytest  # noqa: E402

from reinforcement_learning.environment.config import GRAPH_FILE  # noqa: E402
from reinforcement_learning.environment.graph import RoadGraph, build_synthetic_grid  # noqa: E402
from reinforcement_learning.environment.traffic import TrafficModel  # noqa: E402


@pytest.fixture(scope="session")
def grid():
    return build_synthetic_grid(12, 12)


@pytest.fixture()
def traffic(grid):
    return TrafficModel(grid, "moderate", seed=7)


@pytest.fixture(scope="session")
def real_graph():
    if not GRAPH_FILE.exists():
        pytest.skip("real Hyderabad graph not built yet (run download_data.py + preprocess_data.py)")
    return RoadGraph.load()


@pytest.fixture(scope="session")
def tmp_root():
    return Path(_TMP)
