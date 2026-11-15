"""Deep Q-Network agent (PyTorch) for dynamic routing on the road graph.

Training ingredients
--------------------
* action-in Q-network (see network.py) with invalid-action masking
* experience replay buffer
* target network, hard-updated every ``target_update`` gradient steps
* Double-DQN Bellman target:
      a* = argmax_a' Q_online(s', a')   (masked)
      y  = r + gamma * (1 - terminal) * Q_target(s', a*)
* Huber (smooth L1) loss, Adam optimiser, gradient clipping
* epsilon-greedy exploration with exponential decay
* curriculum on trip length (short trips first)
* checkpointing of the best model by rolling success rate
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..environment.graph import RoadGraph
from ..environment.route_env import (ACTION_FEATURES, MAX_ACTIONS, N_FEATURES, PREFERENCES,
                                     STATE_FEATURES, RouteEnv, encode_actions)
from ..environment.traffic import TrafficModel
from .network import HIDDEN, NumpyQNetwork


@dataclass
class DQNConfig:
    episodes: int = 1500
    learning_rate: float = 5e-4
    gamma: float = 0.995
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.996
    batch_size: int = 128
    buffer_size: int = 40_000
    target_update: int = 500
    train_every: int = 2
    warmup: int = 1_000
    max_steps: int = 400
    min_trip_km: float = 1.5
    max_trip_km: float = 22.0
    curriculum_fraction: float = 0.4
    grad_clip: float = 10.0
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def _loop_free_mask(env: RouteEnv, mask: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Hide actions that re-enter an already visited node (unless nothing else)."""
    m = mask.copy()
    for k in np.flatnonzero(mask):
        if int(env.g.ev[ids[k]]) in env.visited:
            m[k] = False
    return m if m.any() else mask


class DQNAgent:
    name = "DQN"

    def __init__(self, graph: RoadGraph, config: DQNConfig | None = None):
        self.g = graph
        self.cfg = config or DQNConfig()
        self.np_net: NumpyQNetwork | None = None
        self.torch_net = None
        self.meta: dict = {}

    # ------------------------------------------------------------ inference
    def q_values(self, feats: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if self.torch_net is not None:
            import torch

            with torch.no_grad():
                q = self.torch_net(torch.from_numpy(feats)).squeeze(-1).numpy()
        else:
            q = self.np_net(feats)
        return np.where(mask, q, -np.inf)

    def route(self, traffic: TrafficModel, source: int, goal: int, preference: str = "fastest",
              t0: float | None = None, max_steps: int = 1500) -> dict:
        env = RouteEnv(self.g, traffic, PREFERENCES[preference], max_steps=max_steps)
        env.reset(source, goal, t0)
        q_trace = []
        while not env.done:
            feats, mask, ids = encode_actions(env)
            if not mask.any():
                break
            mask = _loop_free_mask(env, mask, ids)
            q = self.q_values(feats, mask)
            k = int(np.argmax(q))
            q_trace.append(float(q[k]))
            env.step(int(ids[k]))
        return {"edges": env.path_edges, "nodes": env.path_nodes, "reached": env.node == goal,
                "steps": env.steps, "mean_q": float(np.mean(q_trace)) if q_trace else 0.0}

    # -------------------------------------------------------------- training
    def sample_od(self, rng: np.random.Generator, max_km: float) -> tuple[int, int]:
        valid = np.flatnonzero(self.g.out_degree > 0)
        for _ in range(200):
            s = int(valid[rng.integers(len(valid))])
            d = self.g.distances_to(s) / 1000.0
            cand = np.flatnonzero((d >= self.cfg.min_trip_km) & (d <= max_km) & (self.g.in_degree > 0))
            if len(cand):
                return s, int(cand[rng.integers(len(cand))])
        raise RuntimeError("could not sample an origin/destination pair")

    def train(self, traffic: TrafficModel, progress: Callable[[dict], None] | None = None,
              checkpoint_path: Path | None = None, od_pairs: list[tuple[int, int]] | None = None,
              stop_flag: Callable[[], bool] | None = None) -> list[dict]:
        import torch
        import torch.nn.functional as F

        from .network import build_torch_qnet
        from .replay import ReplayBuffer

        cfg = self.cfg
        rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        online = build_torch_qnet(N_FEATURES)
        target = build_torch_qnet(N_FEATURES)
        target.load_state_dict(online.state_dict())
        opt = torch.optim.Adam(online.parameters(), lr=cfg.learning_rate)
        buf = ReplayBuffer(cfg.buffer_size, MAX_ACTIONS, N_FEATURES, seed=cfg.seed)
        env = RouteEnv(self.g, traffic, max_steps=cfg.max_steps)
        prefs = list(PREFERENCES)
        eps = cfg.epsilon_start
        steps_total = grad_steps = 0
        history: list[dict] = []
        best_rate = -1.0
        t_start = time.time()

        def learn() -> float:
            f, m, a, r, nf, nm, term = buf.sample(cfg.batch_size)
            f_t = torch.from_numpy(f)
            q = online(f_t).squeeze(-1)  # [B, K]
            q_sa = q.gather(1, torch.from_numpy(a).unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                nf_t = torch.from_numpy(nf)
                nm_t = torch.from_numpy(nm)
                q_next_online = online(nf_t).squeeze(-1).masked_fill(~nm_t, -1e9)
                a_star = q_next_online.argmax(1, keepdim=True)
                q_next_target = target(nf_t).squeeze(-1).gather(1, a_star).squeeze(1)
                no_next = (~nm_t.any(1)).float()
                done = torch.clamp(torch.from_numpy(term) + no_next, 0, 1)
                y = torch.from_numpy(r) + cfg.gamma * (1 - done) * q_next_target
            loss = F.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online.parameters(), cfg.grad_clip)
            opt.step()
            return float(loss.item())

        for ep in range(cfg.episodes):
            if stop_flag and stop_flag():
                break
            frac = min(1.0, ep / max(1, cfg.episodes * cfg.curriculum_fraction))
            max_km = cfg.min_trip_km + 2.5 + frac * (cfg.max_trip_km - cfg.min_trip_km - 2.5)
            traffic.randomize(rng)
            pref = prefs[int(rng.integers(len(prefs)))]
            env.cfg = PREFERENCES[pref]
            if od_pairs:
                s, gl = od_pairs[int(rng.integers(len(od_pairs)))]
            else:
                s, gl = self.sample_od(rng, max_km)
            env.reset(s, gl)
            feats, mask, ids = encode_actions(env)
            losses = []
            while not env.done and mask.any():
                m2 = _loop_free_mask(env, mask, ids)
                if rng.random() < eps:
                    k = int(rng.choice(np.flatnonzero(m2)))
                else:
                    with torch.no_grad():
                        q = online(torch.from_numpy(feats)).squeeze(-1).numpy()
                    k = int(np.argmax(np.where(m2, q, -np.inf)))
                _, r, done, info = env.step(int(ids[k]))
                if done:
                    nf = np.zeros_like(feats)
                    nm = np.zeros_like(mask)
                else:
                    nf, nm, nids = encode_actions(env)
                terminal = bool(info["reached"] or info["dead_end"])
                buf.push(feats, mask, k, r, nf, nm, terminal)
                steps_total += 1
                if len(buf) >= cfg.warmup and steps_total % cfg.train_every == 0:
                    losses.append(learn())
                    grad_steps += 1
                    if grad_steps % cfg.target_update == 0:
                        target.load_state_dict(online.state_dict())
                if not done:
                    feats, mask, ids = nf, nm, nids
            reached = env.node == gl
            stats = {
                "episode": ep + 1, "reward": round(env.total_reward, 3), "steps": env.steps,
                "reached": bool(reached), "epsilon": round(eps, 4),
                "loss": round(float(np.mean(losses)), 5) if losses else None,
                "trip_km": round(self.g.distance_m(s, gl) / 1000, 2), "preference": pref,
                "travel_time_min": round(env.t - env.t0, 2), "elapsed_s": round(time.time() - t_start, 1),
            }
            history.append(stats)
            window = history[-100:]
            rate = sum(h["reached"] for h in window) / len(window)
            stats["rolling_success"] = round(rate, 3)
            if progress:
                progress(stats)
            eps = max(cfg.epsilon_end, eps * cfg.epsilon_decay)
            if checkpoint_path and ep >= 100 and (ep + 1) % 50 == 0 and rate >= best_rate:
                best_rate = rate
                self._export(online, checkpoint_path, history, best_rate)
        self.torch_net = None
        self.np_net = NumpyQNetwork.from_torch(online)
        if checkpoint_path:
            final_rate = sum(h["reached"] for h in history[-100:]) / max(1, len(history[-100:]))
            if final_rate >= best_rate or not Path(checkpoint_path).exists():
                self._export(online, checkpoint_path, history, final_rate)
            else:
                self.load_into(checkpoint_path)
        return history

    def _export(self, online, path: Path, history: list[dict], rate: float) -> None:
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "algorithm": "double-dqn-action-in",
            "n_features": N_FEATURES, "max_actions": MAX_ACTIONS, "hidden": list(HIDDEN),
            "state_features": STATE_FEATURES, "action_features": ACTION_FEATURES,
            "config": self.cfg.to_dict(), "episodes_trained": len(history),
            "rolling_success": rate, "graph_edges": self.g.n_edges,
            "graph_signature": self.g.meta.get("signature"), "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        torch.save({"state_dict": online.state_dict(), "meta": meta}, path.with_suffix(".pt"))
        NumpyQNetwork.from_torch(online).save(path.with_suffix(".npz"), meta)
        self.meta = meta

    def load_into(self, path: Path) -> None:
        net, meta = NumpyQNetwork.load(Path(path).with_suffix(".npz"))
        self.np_net, self.meta = net, meta

    @classmethod
    def load(cls, graph: RoadGraph, path: Path, use_torch: bool = False) -> "DQNAgent":
        path = Path(path)
        net, meta = NumpyQNetwork.load(path.with_suffix(".npz"))
        if meta.get("n_features") != N_FEATURES:
            raise ValueError("DQN checkpoint feature layout does not match this code version")
        if meta.get("graph_edges") not in (None, graph.n_edges):
            # the DQN generalises across graphs, but warn through meta
            meta["graph_mismatch"] = True
        agent = cls(graph, DQNConfig(**meta["config"]))
        agent.np_net, agent.meta = net, meta
        if use_torch:
            import torch

            from .network import build_torch_qnet

            ckpt = torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=False)
            agent.torch_net = build_torch_qnet(N_FEATURES)
            agent.torch_net.load_state_dict(ckpt["state_dict"])
            agent.torch_net.eval()
        return agent

    @staticmethod
    def describe() -> dict:
        return {"state_features": STATE_FEATURES, "action_features": ACTION_FEATURES,
                "hidden": list(HIDDEN), "max_actions": MAX_ACTIONS, "n_features": N_FEATURES}


def history_to_json(history: list[dict], path: Path, meta: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"meta": meta, "history": history}, fh)
