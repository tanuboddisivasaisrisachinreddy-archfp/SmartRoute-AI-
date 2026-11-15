"""Goal-conditioned tabular Q-learning on the Hyderabad road graph.

Q-table layout
--------------
Every action is a directed road segment e = (u -> v) and the primary state
component is the current intersection u, so a (state, action) pair is
identified by the edge id.  One table is kept per (preference, destination):

    Q(b, e) = Qbase[e] + Qres[b, e]        b = traffic-level bucket 0..3

``Qbase`` is shared by all traffic levels, ``Qres`` holds the
traffic-level-specific correction.  This is tabular Q-learning with two
binary "tilings" (Sutton & Barto, ch. 9): every update moves both entries, so
experience from one traffic level also informs the others, while the residual
still lets the policy differ between low and peak traffic.  With a single
4-way table the arterial graph of Hyderabad (tens of thousands of entries per
destination) would get only a handful of updates per entry.

Learning rule (one-step Q-learning, epsilon-greedy behaviour):

    delta = r + gamma * max_{e' in A(v)} Q(b', e') * (1 - terminal) - Q(b, e)
    Qbase[e]   += 0.5 * alpha_n * delta
    Qres[b, e] += 0.5 * alpha_n * delta,     alpha_n = max(alpha_min, alpha / sqrt(1 + n/5))

Initialisation: Qbase starts from a potential-consistent estimate of the
return (see ``initial_q``) instead of zeros, so unexplored road segments are
neither wildly optimistic nor pessimistic.

After each episode the transitions are replayed in reverse order (backward
replay) with the same update rule, which propagates the arrival reward along
100-step road paths much faster.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..environment.graph import RoadGraph
from ..environment.route_env import PREFERENCES, RouteEnv
from ..environment.traffic import TrafficModel

N_BUCKETS = 4


@dataclass
class QLearningConfig:
    episodes_per_goal: int = 3000
    alpha: float = 0.3
    alpha_min: float = 0.05
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.998  # multiplicative per episode
    max_steps: int = 500
    seed: int = 42
    start_radius_km: float = 18.0  # sample episode starts within this radius of the goal
    backward_replay: bool = True
    replay_passes: int = 2
    # initial value estimate (deliberately pessimistic: heavy congestion, 1.6x detour),
    # so the greedy policy prefers segments whose value was confirmed by experience
    detour_factor: float = 1.6
    init_congestion: float = 0.6
    bucket_init_congestion: tuple = (0.35, 0.55, 0.75, 0.9)

    def to_dict(self) -> dict:
        return asdict(self)


class QTable:
    def __init__(self, n_edges: int):
        self.q = np.zeros(n_edges, dtype=np.float32)               # Qbase[e]
        self.res = np.zeros((N_BUCKETS, n_edges), dtype=np.float32)  # Qres[b, e]
        self.n = np.zeros((N_BUCKETS, n_edges), dtype=np.uint32)     # visit counts per bucket
        self.episodes = 0
        self.successes = 0

    def values(self, b: int, edges: np.ndarray) -> np.ndarray:
        return self.q[edges] + self.res[b, edges]

    def visits(self, edges: np.ndarray) -> np.ndarray:
        return self.n[:, edges].sum(axis=0)


class QLearningAgent:
    name = "Q-Learning"

    def __init__(self, graph: RoadGraph, config: QLearningConfig | None = None):
        self.g = graph
        self.cfg = config or QLearningConfig()
        self.tables: dict[tuple[str, int], QTable] = {}
        self.meta: dict = {}

    # --------------------------------------------------------------- helpers
    def table(self, preference: str, goal: int, create: bool = True) -> QTable | None:
        key = (preference, int(goal))
        if key not in self.tables and create:
            tab = QTable(self.g.n_edges)
            base = self.initial_q(preference, int(goal), self.cfg.init_congestion)
            tab.q[:] = base
            # bucket-specific starting point: each traffic level gets its own pessimistic estimate
            for b, c in enumerate(self.cfg.bucket_init_congestion):
                tab.res[b] = self.initial_q(preference, int(goal), c) - base
            self.tables[key] = tab
        return self.tables.get(key)

    def initial_q(self, preference: str, goal: int, congestion: float | None = None) -> np.ndarray:
        """Potential-consistent initial estimate of the shaped return.

            Q0(u->v) = R_goal - c_km * (len_e + DETOUR * d(v, g)) + w_prog * d(u, g)

        i.e. the return if the rest of the trip were DETOUR x the straight-line
        distance at the median cost per km under heavy congestion.  Being
        pessimistic matters: rarely visited segments keep (close to) their
        initial value, and an optimistic value would lure the greedy policy
        into unexplored detours.
        """
        cfg = PREFERENCES[preference]
        g = self.g
        d = g.distances_to(goal) / 1000.0
        c0 = self.cfg.init_congestion if congestion is None else congestion
        speed = np.maximum(g.speed_kph * (1 - 0.85 * c0), 4.0)
        c_km = cfg.w_time * 60.0 / speed + cfg.w_dist + cfg.w_cong * c0 * 60.0 / speed
        c_bar = float(np.median(c_km))
        q0 = (cfg.goal_reward - c_bar * (g.length_m / 1000.0 + self.cfg.detour_factor * d[g.ev])
              + cfg.w_prog * d[g.eu])
        return q0.astype(np.float32)

    def has_table(self, preference: str, goal: int) -> bool:
        return (preference, int(goal)) in self.tables

    def _alpha(self, n: int) -> float:
        return max(self.cfg.alpha_min, self.cfg.alpha / float(np.sqrt(1.0 + n / 5.0)))

    def _fresh(self, env: RouteEnv, acts: np.ndarray) -> np.ndarray:
        if len(acts) > 1:  # avoid immediate loops when possible
            fresh = acts[[int(self.g.ev[a]) not in env.visited for a in acts]]
            if len(fresh):
                return fresh
        return acts

    def _select(self, env: RouteEnv, tab: QTable, eps: float, rng: np.random.Generator) -> int:
        acts = self._fresh(env, env.valid_actions())
        if rng.random() < eps:
            return int(acts[rng.integers(len(acts))])
        qv = tab.values(env.traffic.level_bucket(env.t), acts)
        best = np.flatnonzero(qv == qv.max())
        return int(acts[best[rng.integers(len(best))]])

    def _target(self, tab: QTable, v: int, b_next: int, r: float, terminal: bool, prev: int) -> float:
        if terminal:
            return r
        out = self.g.out_edges(v)
        if len(out) > 1:
            keep = self.g.ev[out] != prev
            if keep.any():
                out = out[keep]
        return r + self.cfg.gamma * float(tab.values(b_next, out).max()) if len(out) else r

    def _update(self, tab: QTable, b: int, e: int, target: float) -> None:
        a = 0.5 * self._alpha(int(tab.n[b, e]))
        delta = target - (tab.q[e] + tab.res[b, e])
        tab.q[e] += a * delta
        tab.res[b, e] += a * delta

    # -------------------------------------------------------------- training
    def run_episode(self, env: RouteEnv, tab: QTable, start: int, goal: int, eps: float,
                    rng: np.random.Generator) -> dict:
        env.reset(start, goal)
        trans = []
        while not env.done:
            b = env.traffic.level_bucket(env.t)
            u = env.node
            e = self._select(env, tab, eps, rng)
            _, r, done, info = env.step(e)
            v = env.node
            b2 = env.traffic.level_bucket(env.t)
            terminal = bool(info["reached"] or info["dead_end"])
            tab.n[b, e] += 1
            self._update(tab, b, e, self._target(tab, v, b2, r, terminal, u))
            trans.append((b, e, r, v, b2, terminal, u))
        for _ in range(self.cfg.replay_passes if self.cfg.backward_replay else 0):
            for b, e, r, v, b2, terminal, u in reversed(trans):
                self._update(tab, b, e, self._target(tab, v, b2, r, terminal, u))
        reached = env.node == goal
        tab.episodes += 1
        tab.successes += int(reached)
        return {"reward": float(env.total_reward), "steps": env.steps, "reached": bool(reached),
                "travel_time_min": float(env.t - env.t0)}

    def sample_start(self, goal: int, rng: np.random.Generator, radius_km: float,
                     focus: list[int] | None = None) -> int:
        if focus and rng.random() < 0.5:
            return int(focus[rng.integers(len(focus))])
        d = self.g.distances_to(goal) / 1000.0
        cand = np.flatnonzero((d <= radius_km) & (d > 0.3) & (self.g.out_degree > 0))
        if len(cand) == 0:
            cand = np.flatnonzero(self.g.out_degree > 0)
        return int(cand[rng.integers(len(cand))])

    def train_goal(self, traffic: TrafficModel, goal: int, preference: str = "fastest",
                   episodes: int | None = None, focus: list[int] | None = None,
                   time_budget_s: float | None = None, progress: Callable[[dict], None] | None = None,
                   rng: np.random.Generator | None = None) -> list[dict]:
        rng = rng or np.random.default_rng(self.cfg.seed + goal)
        episodes = episodes or self.cfg.episodes_per_goal
        env = RouteEnv(self.g, traffic, PREFERENCES[preference], max_steps=self.cfg.max_steps)
        tab = self.table(preference, goal)
        eps = self.cfg.epsilon_start
        history = []
        t_start = time.time()
        for ep in range(episodes):
            traffic.randomize(rng)
            start = self.sample_start(goal, rng, self.cfg.start_radius_km, focus)
            stats = self.run_episode(env, tab, start, goal, eps, rng)
            stats.update(episode=ep + 1, epsilon=eps, goal=goal, preference=preference)
            history.append(stats)
            if progress:
                progress(stats)
            eps = max(self.cfg.epsilon_end, eps * self.cfg.epsilon_decay)
            if time_budget_s and time.time() - t_start > time_budget_s:
                break
        return history

    # ------------------------------------------------------------- inference
    def route(self, traffic: TrafficModel, source: int, goal: int, preference: str = "fastest",
              t0: float | None = None, max_steps: int = 1500) -> dict:
        """Greedy policy rollout with loop avoidance.

        Steps taken at intersections where none of the candidate segments was
        ever updated during training are counted in ``fallback_steps`` (the
        initial value estimate decides there) and reported to the user."""
        tab = self.table(preference, goal, create=False)
        if tab is None:
            raise KeyError("no Q-table for this destination/preference")
        env = RouteEnv(self.g, traffic, PREFERENCES[preference], max_steps=max_steps)
        env.reset(source, goal, t0)
        fallback = 0
        while not env.done:
            acts = self._fresh(env, env.valid_actions())
            b = traffic.level_bucket(env.t)
            if not (tab.visits(acts) > 0).any():
                fallback += 1
            env.step(int(acts[int(np.argmax(tab.values(b, acts)))]))
        return {
            "edges": env.path_edges,
            "nodes": env.path_nodes,
            "reached": env.node == goal,
            "fallback_steps": fallback,
            "steps": env.steps,
            "table_episodes": tab.episodes,
            "table_success_rate": round(tab.successes / max(1, tab.episodes), 3),
            "table_coverage": round(float((tab.n.sum(axis=0) > 0).mean()), 3),
        }

    # ------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.tables)
        arrays = {}
        for i, k in enumerate(keys):
            t = self.tables[k]
            arrays[f"q_{i}"] = t.q
            arrays[f"r_{i}"] = t.res.astype(np.float16)
            arrays[f"n_{i}"] = np.minimum(t.n, 65535).astype(np.uint16)
        meta = {
            "algorithm": "tabular-q-learning (base + traffic-level residual)",
            "keys": [[p, int(gl)] for p, gl in keys],
            "episodes": [self.tables[k].episodes for k in keys],
            "successes": [self.tables[k].successes for k in keys],
            "config": self.cfg.to_dict(),
            "graph_edges": self.g.n_edges,
            "graph_signature": self.g.meta.get("signature"),
            **self.meta,
        }
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, graph: RoadGraph, path: Path) -> "QLearningAgent":
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        if meta["graph_edges"] != graph.n_edges:
            raise ValueError("Q-table was trained on a different road graph")
        cfg = dict(meta["config"])
        if "bucket_init_congestion" in cfg:
            cfg["bucket_init_congestion"] = tuple(cfg["bucket_init_congestion"])
        agent = cls(graph, QLearningConfig(**{k: v for k, v in cfg.items() if k in QLearningConfig.__dataclass_fields__}))
        for i, (p, gl) in enumerate(meta["keys"]):
            t = QTable(graph.n_edges)
            t.q = data[f"q_{i}"].astype(np.float32)
            t.res = data[f"r_{i}"].astype(np.float32)
            t.n = data[f"n_{i}"].astype(np.uint32)
            t.episodes = meta["episodes"][i]
            t.successes = meta["successes"][i]
            agent.tables[(p, int(gl))] = t
        agent.meta = {k: v for k, v in meta.items() if k not in ("keys", "episodes", "successes", "config")}
        return agent
