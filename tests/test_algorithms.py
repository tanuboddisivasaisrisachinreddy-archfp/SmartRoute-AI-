"""Dijkstra, Q-Learning and DQN produce valid, connected paths that reach the goal."""
import json
import subprocess
import sys

import networkx as nx
import numpy as np
import pytest

from backend.app.routing.dijkstra import alternative_routes, dijkstra_route, edge_weights
from reinforcement_learning.dqn.agent import DQNAgent
from reinforcement_learning.dqn.network import HIDDEN, NumpyQNetwork
from reinforcement_learning.environment.route_env import N_FEATURES
from reinforcement_learning.environment.traffic import TrafficModel
from reinforcement_learning.evaluation.metrics import evaluate_path
from reinforcement_learning.q_learning.agent import QLearningAgent, QLearningConfig

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


def assert_valid_path(g, edges, source, goal):
    assert edges, "empty path"
    assert int(g.eu[edges[0]]) == source
    assert int(g.ev[edges[-1]]) == goal
    for a, b in zip(edges[:-1], edges[1:]):
        assert g.ev[a] == g.eu[b], "path is not connected"


def test_dijkstra_matches_networkx_optimum(grid, traffic):
    r = dijkstra_route(grid, traffic, 0, grid.n_nodes - 1, "shortest")
    assert_valid_path(grid, r["edges"], 0, grid.n_nodes - 1)
    w = edge_weights(grid, traffic, "shortest", traffic.clock_min)
    G = grid.nx_graph
    best = nx.dijkstra_path_length(G, 0, grid.n_nodes - 1, weight=lambda u, v, d: float(w[d["eid"]]))
    assert r["planned_cost"] == pytest.approx(best)


def test_dijkstra_alternatives_differ(grid, traffic):
    main = dijkstra_route(grid, traffic, 0, grid.n_nodes - 1)
    alts = alternative_routes(grid, traffic, 0, grid.n_nodes - 1, k=2)
    assert alts
    for a in alts:
        assert_valid_path(grid, a["edges"], 0, grid.n_nodes - 1)
        assert a["edges"] != main["edges"]


def test_q_learning_training_and_inference(grid):
    goal = grid.n_nodes - 1
    agent = QLearningAgent(grid, QLearningConfig(episodes_per_goal=400, seed=1))
    hist = agent.train_goal(TrafficModel(grid, "moderate", seed=1), goal, "fastest")
    assert len(hist) == 400
    assert np.mean([h["reached"] for h in hist[-50:]]) > 0.9
    # learning: later episodes earn more reward than early ones
    assert np.mean([h["reward"] for h in hist[-100:]]) > np.mean([h["reward"] for h in hist[:100]])
    tr = TrafficModel(grid, "peak", seed=2)
    r = agent.route(tr, 0, goal)
    assert r["reached"]
    assert_valid_path(grid, r["edges"], 0, goal)
    m = evaluate_path(grid, tr, r["edges"], 0, goal, tr.clock_min)
    assert m["reached_destination"] and m["distance_km"] > 0


def test_q_learning_save_load_roundtrip(grid, tmp_path):
    agent = QLearningAgent(grid, QLearningConfig(episodes_per_goal=50))
    agent.train_goal(TrafficModel(grid, seed=0), 30, "fastest")
    p = tmp_path / "q.npz"
    agent.save(p)
    a2 = QLearningAgent.load(grid, p)
    assert a2.has_table("fastest", 30)
    assert np.allclose(a2.tables[("fastest", 30)].q, agent.tables[("fastest", 30)].q, atol=0.05)


def test_q_learning_training_is_reproducible(grid):
    def run():
        a = QLearningAgent(grid, QLearningConfig(episodes_per_goal=60, seed=5))
        h = a.train_goal(TrafficModel(grid, seed=5), 77, "balanced", rng=np.random.default_rng(9))
        return [x["reward"] for x in h]

    assert run() == run()


def test_q_learning_rejects_missing_table(grid, traffic):
    with pytest.raises(KeyError):
        QLearningAgent(grid).route(traffic, 0, 5)


def _random_dqn(grid, seed=0):
    rng = np.random.default_rng(seed)
    sizes = [N_FEATURES, *HIDDEN, 1]
    ws = [(rng.normal(0, 0.3, (b, a)).astype(np.float32), np.zeros(b, np.float32)) for a, b in zip(sizes[:-1], sizes[1:])]
    agent = DQNAgent(grid)
    agent.np_net = NumpyQNetwork(ws)
    return agent


def test_dqn_inference_returns_connected_path_and_masks(grid, traffic):
    agent = _random_dqn(grid)
    r = agent.route(traffic, 0, 60, max_steps=300)
    edges = r["edges"]
    assert edges and int(grid.eu[edges[0]]) == 0
    for a, b in zip(edges[:-1], edges[1:]):
        assert grid.ev[a] == grid.eu[b]
    # masked values are -inf, never chosen
    from reinforcement_learning.environment.route_env import RouteEnv, encode_actions

    env = RouteEnv(grid, traffic)
    env.reset(0, 60)
    f, m, _ = encode_actions(env)
    q = agent.q_values(f, m)
    assert np.isneginf(q[~m]).all() and np.isfinite(q[m]).all()


def test_numpy_network_save_load(tmp_path, grid):
    agent = _random_dqn(grid, 3)
    p = tmp_path / "dqn.npz"
    agent.np_net.save(p, {"n_features": N_FEATURES, "config": {}, "graph_edges": grid.n_edges})
    net, meta = NumpyQNetwork.load(p)
    x = np.random.default_rng(1).normal(size=(8, N_FEATURES)).astype(np.float32)
    assert np.allclose(net(x), agent.np_net(x))


# ------------------------------------------------------------ PyTorch DQN
def test_dqn_torch_training_and_numpy_parity(grid, tmp_path):
    torch = pytest.importorskip("torch")
    from reinforcement_learning.dqn.agent import DQNConfig

    cfg = DQNConfig(episodes=120, batch_size=32, buffer_size=5000, warmup=200, target_update=100,
                    max_trip_km=3.0, max_steps=150, seed=3)
    agent = DQNAgent(grid, cfg)
    hist = agent.train(TrafficModel(grid, seed=3), checkpoint_path=tmp_path / "dqn")
    assert len(hist) == 120
    assert any(h["loss"] is not None for h in hist)
    assert (tmp_path / "dqn.pt").exists() and (tmp_path / "dqn.npz").exists()
    a_np = DQNAgent.load(grid, tmp_path / "dqn")
    a_t = DQNAgent.load(grid, tmp_path / "dqn", use_torch=True)
    x = np.random.default_rng(0).normal(size=(8, N_FEATURES)).astype(np.float32)
    m = np.ones(8, bool)
    assert np.allclose(a_np.q_values(x, m), a_t.q_values(x, m), atol=1e-4)
    r = a_np.route(TrafficModel(grid, seed=4), 0, 30, max_steps=400)
    assert r["edges"] and int(grid.eu[r["edges"][0]]) == 0
    del torch


def test_training_script_small_reproducible_run(tmp_path):
    out = tmp_path / "q.npz"
    prog = tmp_path / "progress.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "train_q_learning.py"), "--synthetic", "--episodes", "40",
           "--preferences", "fastest", "--output", str(out), "--progress-file", str(prog), "--seed", "11"]
    subprocess.run(cmd, check=True, capture_output=True, timeout=300)
    p = json.loads(prog.read_text())
    assert p["state"] == "completed" and p["current_episode"] == p["total_episodes"] == 120
    assert out.exists() and (tmp_path / "q_learning_history.json").exists()


def test_real_graph_routes_reach_destination(real_graph):
    """All algorithms on the real Hyderabad graph: Hitech City -> Charminar."""
    tr = TrafficModel(real_graph, "peak", seed=1)
    s, _ = real_graph.nearest_node(17.4504, 78.3810)
    t, _ = real_graph.nearest_node(17.3616, 78.4747)
    straight_km = real_graph.distance_m(s, t) / 1000
    d = dijkstra_route(real_graph, tr, s, t, "shortest")
    assert_valid_path(real_graph, d["edges"], s, t)
    m = evaluate_path(real_graph, tr, d["edges"], s, t, tr.clock_min)
    # shortest road route must be longer than the straight line but not absurdly so
    assert straight_km < m["distance_km"] < 2.5 * straight_km
    f = dijkstra_route(real_graph, tr, s, t, "fastest")  # may use the Outer Ring Road
    assert_valid_path(real_graph, f["edges"], s, t)
