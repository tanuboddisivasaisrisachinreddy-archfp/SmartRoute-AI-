"""Traffic model dynamics and the RL environment (incl. invalid actions)."""
import numpy as np
import pytest

from reinforcement_learning.environment.route_env import (MAX_ACTIONS, N_FEATURES, PREFERENCES, RouteEnv,
                                                          encode_actions)
from reinforcement_learning.environment.traffic import TrafficModel, time_of_day_profile


def test_scenarios_order_congestion(grid):
    means = [TrafficModel(grid, s, clock_min=18 * 60 + 30, seed=1).congestion_array().mean()
             for s in ("low", "moderate", "heavy", "peak")]
    assert means == sorted(means)


def test_congestion_bounds_and_travel_time(traffic):
    c = traffic.congestion_array()
    assert c.min() >= 0 and c.max() <= 0.95
    tt = traffic.travel_time_array()
    assert (tt >= traffic.graph.free_time_s - 1e-9).all()  # traffic never speeds you up
    e = 3
    assert traffic.edge_congestion(e) == pytest.approx(c[e], abs=1e-9)


def test_time_of_day_peaks():
    assert time_of_day_profile(18 * 60 + 45) > time_of_day_profile(3 * 60)
    assert time_of_day_profile(10 * 60) > time_of_day_profile(14 * 60 + 30) * 0.9


def test_incident_increases_and_recovers(grid):
    t = TrafficModel(grid, "low", seed=3)
    e = 10
    before = t.edge_congestion(e)
    inc = t.add_incident(float(grid.edge_mid_lat[e]), float(grid.edge_mid_lon[e]), radius_m=300, severity=0.8,
                         duration_min=20, recovery_min=10)
    assert t.edge_congestion(e) > before + 0.3
    assert inc.to_dict(t.clock_min)["status"] == "active"
    t.advance(25)  # in recovery
    mid = t.edge_congestion(e)
    t.advance(15)  # recovered and expired
    assert not t.incidents
    assert t.edge_congestion(e) < mid


def test_clear_incident_and_reset(traffic, grid):
    inc = traffic.add_incident(float(grid.lat[5]), float(grid.lon[5]))
    assert traffic.clear_incident(inc.id) == 1
    traffic.set_scenario("heavy")
    traffic.reset("low", 8 * 60)
    assert traffic.scenario == "low" and traffic.clock_min == 8 * 60 and not traffic.incidents


def test_surge_decays(traffic):
    lv = traffic.level()
    traffic.trigger_surge(0.4, 20)
    assert traffic.level() > lv + 0.3
    traffic.advance(60)
    assert traffic.summary()["surge"] < 0.1


def test_simulated_label(traffic):
    s = traffic.summary()
    assert s["mode"] == "simulation"
    assert s["label"] == "Simulated Traffic – Not Live Traffic"


def test_env_invalid_action_penalised_and_state_unchanged(grid, traffic):
    env = RouteEnv(grid, traffic)
    env.reset(0, grid.n_nodes - 1)
    invalid = int(np.setdiff1d(np.arange(grid.n_edges), grid.out_edges(0))[0])
    _, r, done, info = env.step(invalid)
    assert info["invalid"] and r == -PREFERENCES["fastest"].invalid_penalty
    assert env.node == 0 and not done


def test_env_valid_actions_exclude_uturn(grid, traffic):
    env = RouteEnv(grid, traffic)
    env.reset(13, grid.n_nodes - 1)
    e = int(env.valid_actions()[0])
    env.step(e)
    back = [a for a in env.valid_actions() if grid.ev[a] == 13]
    assert back == []  # immediate U-turn removed (node has other exits)


def test_env_reaches_goal_with_reward(grid, traffic):
    from backend.app.routing.dijkstra import dijkstra_route

    env = RouteEnv(grid, traffic)
    r = dijkstra_route(grid, traffic, 0, 50)
    env.reset(0, 50)
    for e in r["edges"]:
        _, rew, done, info = env.step(e)
    assert done and info["reached"]
    assert env.t > env.t0 and rew > 0


def test_feature_encoding_masks_padding(grid, traffic):
    env = RouteEnv(grid, traffic)
    env.reset(0, 100)
    f, m, ids = encode_actions(env)
    assert f.shape == (MAX_ACTIONS, N_FEATURES)
    k = int(m.sum())
    assert k == len(env.valid_actions())
    assert (ids[:k] >= 0).all() and (ids[k:] == -1).all()
    assert np.isfinite(f).all()
