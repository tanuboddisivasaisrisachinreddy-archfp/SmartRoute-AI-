"""Reinforcement-learning environment over the Hyderabad road graph.

MDP definition (see docs/rl_design.md for the full write-up)
------------------------------------------------------------
State  s_t  = (current node u, destination node g, straight-line distance
               d(u, g), network traffic level bucket b(t), local congestion,
               elapsed travel time, visited set)
Action a_t  = one outgoing road segment (u -> v) of the current intersection.
              Invalid actions (edges not leaving u) are never offered; the
              immediate U-turn back to the previous node is removed unless it
              is the only exit (dead end).
Reward r_t  = - w_time * tt_e(t) [min]
              - w_dist * len_e   [km]
              - w_cong * c_e(t) * tt_e(t) [congested minutes]
              + w_prog * (gamma * Phi(v) - Phi(u))      (potential-based shaping,
                                                        Phi(n) = -d(n, g) [km])
              + R_goal                       if v == g
              - P_revisit                    if v was already visited
              - P_invalid                    if the action is invalid (state unchanged)
              - P_timeout / P_dead_end       on truncation
Transition: the agent moves to v and the clock advances by tt_e(t), so the
            traffic it observes at the next intersection is time-dependent.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .config import ROAD_CLASSES
from .graph import RoadGraph
from .traffic import TrafficModel


@dataclass
class RewardConfig:
    w_time: float = 1.0
    w_dist: float = 0.25
    w_cong: float = 0.5
    w_prog: float = 2.0
    goal_reward: float = 20.0
    revisit_penalty: float = 3.0
    invalid_penalty: float = 5.0
    timeout_penalty: float = 15.0
    dead_end_penalty: float = 15.0
    gamma: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


PREFERENCES = {
    # fastest: minimise travel time and congested minutes
    "fastest": RewardConfig(w_time=1.0, w_dist=0.1, w_cong=0.6),
    # shortest: minimise kilometres
    "shortest": RewardConfig(w_time=0.15, w_dist=1.2, w_cong=0.1),
    # balanced: blend of both
    "balanced": RewardConfig(w_time=0.6, w_dist=0.6, w_cong=0.35),
}


def edge_cost(cfg: RewardConfig, tt_s: float, length_m: float, cong: float) -> float:
    """Non-shaped per-edge cost used by Dijkstra and for route scoring."""
    tt_min = tt_s / 60.0
    return cfg.w_time * tt_min + cfg.w_dist * length_m / 1000.0 + cfg.w_cong * cong * tt_min


class InvalidAction(Exception):
    pass


class RouteEnv:
    def __init__(self, graph: RoadGraph, traffic: TrafficModel, reward: RewardConfig | None = None,
                 max_steps: int = 600):
        self.g = graph
        self.traffic = traffic
        self.cfg = reward or PREFERENCES["fastest"]
        self.max_steps = max_steps
        self.source = self.goal = self.node = -1

    # ------------------------------------------------------------------ API
    def reset(self, source: int, goal: int, t0: float | None = None) -> dict:
        self.source, self.goal, self.node = int(source), int(goal), int(source)
        self.prev = -1
        self.t = self.traffic.clock_min if t0 is None else float(t0)
        self.t0 = self.t
        self.steps = 0
        self.visited = {self.node}
        self.path_nodes = [self.node]
        self.path_edges: list[int] = []
        self.total_reward = 0.0
        self.dist_to_goal = self.g.distances_to(self.goal) / 1000.0  # km, per node
        self.done = self.node == self.goal
        return self.state()

    def valid_actions(self, node: int | None = None, prev: int | None = None) -> np.ndarray:
        node = self.node if node is None else node
        prev = self.prev if prev is None else prev
        out = self.g.out_edges(node)
        if prev >= 0 and len(out) > 1:
            keep = self.g.ev[out] != prev
            if keep.any():
                out = out[keep]
        return out

    def potential(self, node: int) -> float:
        return -float(self.dist_to_goal[node])

    def state(self) -> dict:
        return {
            "node": self.node,
            "goal": self.goal,
            "distance_to_goal_km": float(self.dist_to_goal[self.node]),
            "traffic_level": self.traffic.level(self.t),
            "traffic_bucket": self.traffic.level_bucket(self.t),
            "elapsed_min": (self.t - self.t0),
            "steps": self.steps,
        }

    def step(self, edge_id: int) -> tuple[dict, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode finished; call reset()")
        edge_id = int(edge_id)
        out = self.g.out_edges(self.node)
        if edge_id not in out:
            # invalid action: penalise, keep state, count the step
            self.steps += 1
            r = -self.cfg.invalid_penalty
            self.total_reward += r
            trunc = self.steps >= self.max_steps
            self.done = trunc
            return self.state(), r, self.done, {"invalid": True, "truncated": trunc}
        v = int(self.g.ev[edge_id])
        cong = self.traffic.edge_congestion(edge_id, self.t)
        tt = self.traffic.edge_time_s(edge_id, self.t)
        r = -edge_cost(self.cfg, tt, self.g.length_m[edge_id], cong)
        r += self.cfg.w_prog * (self.cfg.gamma * self.potential(v) - self.potential(self.node))
        info = {"invalid": False, "edge": edge_id, "travel_time_s": tt, "congestion": cong,
                "revisit": v in self.visited, "reached": False, "truncated": False, "dead_end": False}
        if v in self.visited:
            r -= self.cfg.revisit_penalty
        self.prev, self.node = self.node, v
        self.t += tt / 60.0
        self.steps += 1
        self.visited.add(v)
        self.path_nodes.append(v)
        self.path_edges.append(edge_id)
        if v == self.goal:
            r += self.cfg.goal_reward
            self.done = True
            info["reached"] = True
        elif len(self.g.out_edges(v)) == 0:
            r -= self.cfg.dead_end_penalty
            self.done = True
            info["dead_end"] = True
        elif self.steps >= self.max_steps:
            r -= self.cfg.timeout_penalty
            self.done = True
            info["truncated"] = True
        self.total_reward += r
        return self.state(), r, self.done, info


# ---------------------------------------------------------------------------
# Feature encoding for the DQN (action-in architecture)
# ---------------------------------------------------------------------------
MAX_ACTIONS = 8
STATE_FEATURES = [
    "dist_to_goal_km/20", "traffic_level", "local_congestion", "elapsed_min/120", "steps/max_steps",
    "pref_time", "pref_dist", "pref_cong",
]
ACTION_FEATURES = (
    ["length_km", "travel_time_min/5", "congestion", "free_speed/80", "progress_ratio",
     "next_dist_to_goal/20", "delta_dist_km", "next_visited", "is_goal", "next_out_degree/4",
     "is_dead_end", "lanes/4"]
    + [f"class_{c}" for c in ROAD_CLASSES]
)
N_FEATURES = len(STATE_FEATURES) + len(ACTION_FEATURES)


def encode_actions(env: RouteEnv, actions: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (features[MAX_ACTIONS, N_FEATURES], mask[MAX_ACTIONS], action_ids[MAX_ACTIONS]).

    Each row describes one candidate outgoing edge together with the shared
    state features, so the network scores Q(s, a) per row.  Padding rows are
    masked out (variable action space)."""
    g = env.g
    if actions is None:
        actions = env.valid_actions()
    actions = actions[:MAX_ACTIONS]
    k = len(actions)
    feats = np.zeros((MAX_ACTIONS, N_FEATURES), dtype=np.float32)
    mask = np.zeros(MAX_ACTIONS, dtype=bool)
    ids = np.full(MAX_ACTIONS, -1, dtype=np.int64)
    if k == 0:
        return feats, mask, ids
    t = env.t
    tr = env.traffic
    d_here = env.dist_to_goal[env.node]
    cong = np.array([tr.edge_congestion(int(e), t) for e in actions])
    speed = np.maximum(4.0, g.speed_kph[actions] * (1 - 0.85 * cong))
    length_km = g.length_m[actions] / 1000.0
    tt_min = length_km / speed * 60.0
    nxt = g.ev[actions]
    d_next = env.dist_to_goal[nxt]
    delta = d_here - d_next
    progress = delta / np.maximum(length_km, 1e-3)
    cfg = env.cfg
    norm = max(cfg.w_time + cfg.w_dist + cfg.w_cong, 1e-6)
    state = np.array([
        d_here / 20.0, tr.level(t), float(cong.mean()), (t - env.t0) / 120.0, env.steps / env.max_steps,
        cfg.w_time / norm, cfg.w_dist / norm, cfg.w_cong / norm,
    ], dtype=np.float32)
    feats[:k, : len(STATE_FEATURES)] = state
    col = len(STATE_FEATURES)
    outdeg = g.out_degree[nxt]
    visited = np.array([n in env.visited for n in nxt], dtype=np.float32)
    act = np.stack([
        length_km, tt_min / 5.0, cong, g.speed_kph[actions] / 80.0, np.clip(progress, -1.5, 1.5),
        d_next / 20.0, np.clip(delta, -3, 3), visited, (nxt == env.goal).astype(np.float32),
        np.minimum(outdeg, 8) / 4.0, (outdeg == 0).astype(np.float32), g.lanes[actions] / 4.0,
    ], axis=1)
    feats[:k, col : col + act.shape[1]] = act
    onehot = np.zeros((k, len(ROAD_CLASSES)), dtype=np.float32)
    onehot[np.arange(k), g.road_class[actions].astype(int)] = 1.0
    feats[:k, col + act.shape[1]:] = onehot
    mask[:k] = True
    ids[:k] = actions
    return feats, mask, ids
