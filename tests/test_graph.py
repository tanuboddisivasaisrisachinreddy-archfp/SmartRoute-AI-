"""Road graph loading (real Hyderabad graph when available, synthetic otherwise)."""
import json

import networkx as nx
import numpy as np

from reinforcement_learning.environment.config import HYDERABAD_BBOX, LANDMARKS_FILE
from reinforcement_learning.environment.graph import RoadGraph, haversine_m


def test_synthetic_graph_structure(grid):
    assert grid.meta["synthetic"] is True
    assert grid.n_nodes == 144
    for n in range(grid.n_nodes):
        out = grid.out_edges(n)
        assert all(grid.eu[e] == n for e in out)
    assert nx.is_strongly_connected(grid.nx_graph)


def test_save_and_load_roundtrip(grid, tmp_path):
    p = tmp_path / "g.json.gz"
    grid.save(p)
    g2 = RoadGraph.load(p)
    assert g2.n_nodes == grid.n_nodes and g2.n_edges == grid.n_edges
    assert np.allclose(g2.length_m, grid.length_m, atol=0.01)
    assert g2.geometry[5] == grid.geometry[5]


def test_nearest_node(grid):
    n, d = grid.nearest_node(float(grid.lat[20]) + 1e-5, float(grid.lon[20]))
    assert n == 20 and d < 5


def test_haversine_known_distance():
    # Charminar -> Hitech City is ~14 km in a straight line
    d = haversine_m(17.3616, 78.4747, 17.4504, 78.3810)
    assert 13_000 < d < 15_000


# ---------------------------------------------------------------- real data
def test_real_graph_is_hyderabad(real_graph):
    g = real_graph
    assert not g.meta.get("synthetic", False)
    assert g.n_nodes > 1000
    b = HYDERABAD_BBOX
    assert g.lat.min() >= b["south"] - 0.02 and g.lat.max() <= b["north"] + 0.02
    assert g.lon.min() >= b["west"] - 0.02 and g.lon.max() <= b["east"] + 0.02
    assert (g.length_m > 0).all()
    assert all(len(geom) >= 2 for geom in g.geometry)


def test_real_graph_strongly_connected(real_graph):
    assert nx.is_strongly_connected(real_graph.nx_graph)


def test_real_landmarks_snap_close(real_graph):
    assert LANDMARKS_FILE.exists()
    lms = json.loads(LANDMARKS_FILE.read_text())
    names = {lm["name"] for lm in lms}
    for required in ("Hitech City", "Charminar", "Hyderabad Airport (RGIA)", "Secunderabad", "Gachibowli"):
        assert required in names
    for lm in lms:
        assert lm["snap_distance_m"] < 1500, lm
