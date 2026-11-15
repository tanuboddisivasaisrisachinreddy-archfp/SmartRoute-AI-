#!/usr/bin/env python3
"""Train goal-conditioned tabular Q-learning on the Hyderabad road graph.

    python scripts/train_q_learning.py --episodes 3000
    python scripts/train_q_learning.py --episodes 300 --preferences fastest --goals hitech-city,charminar

One Q-table is learned per (travel preference, destination landmark).  Routes
to other destinations are trained on demand by the API with a small, bounded
episode budget.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import _common  # noqa: F401
import numpy as np

from reinforcement_learning.environment.config import CHECKPOINT_DIR
from reinforcement_learning.environment.traffic import TrafficModel
from reinforcement_learning.q_learning.agent import QLearningAgent, QLearningConfig
from reinforcement_learning.training.loaders import landmark_goals, load_graph
from reinforcement_learning.training.progress import ProgressWriter

log = logging.getLogger("train_q")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=3000, help="episodes per (preference, destination)")
    p.add_argument("--alpha", "--learning-rate", dest="alpha", type=float, default=0.3)
    p.add_argument("--gamma", "--discount", dest="gamma", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.998)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preferences", default="fastest,balanced,shortest")
    p.add_argument("--goals", default="all", help="comma-separated landmark ids or 'all'")
    p.add_argument("--output", default=str(CHECKPOINT_DIR / "q_learning.npz"))
    p.add_argument("--progress-file", default=None)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test grid (tests only)")
    a = p.parse_args()

    np.random.seed(a.seed)
    graph = load_graph(a.synthetic)
    goals = landmark_goals(graph, a.synthetic)
    if a.goals != "all":
        wanted = set(a.goals.split(","))
        goals = [g for g in goals if g[0] in wanted]
    prefs = [x for x in a.preferences.split(",") if x]
    cfg = QLearningConfig(episodes_per_goal=a.episodes, alpha=a.alpha, gamma=a.gamma, epsilon_start=a.epsilon,
                          epsilon_end=a.epsilon_end, epsilon_decay=a.epsilon_decay, max_steps=a.max_steps,
                          seed=a.seed)
    agent = QLearningAgent(graph, cfg)
    out = Path(a.output)
    if out.exists():
        try:
            prev = QLearningAgent.load(graph, out)
            agent.tables.update({k: v for k, v in prev.tables.items()
                                 if k[0] not in prefs or k[1] not in {g for _, g in goals}})
        except Exception as exc:  # incompatible file -> start fresh
            log.warning("Ignoring existing checkpoint: %s", exc)
    total = a.episodes * len(goals) * len(prefs)
    progress = ProgressWriter(a.progress_file, "q_learning", total, {**cfg.to_dict(), "preferences": prefs,
                                                                     "goals": [g for g, _ in goals]})
    traffic = TrafficModel(graph, "moderate", seed=a.seed)
    history, summary = [], []
    t0 = time.time()
    for pref in prefs:
        for gid, node in goals:
            log.info("Training %s -> %s (node %d)", pref, gid, node)
            h = agent.train_goal(traffic, node, pref, episodes=a.episodes, progress=progress.update,
                                 rng=np.random.default_rng(a.seed + node))
            succ = float(np.mean([x["reached"] for x in h[-100:]]))
            summary.append({"preference": pref, "goal": gid, "node": node, "episodes": len(h),
                            "final_success_rate": round(succ, 3),
                            "final_avg_reward": round(float(np.mean([x["reward"] for x in h[-100:]])), 3)})
            log.info("  success(last100)=%.2f", succ)
            history.extend({**x, "goal_id": gid} for x in h)
    agent.meta = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "training_seconds": round(time.time() - t0, 1),
                  "summary": summary}
    agent.save(out)
    hist_path = out.with_name("q_learning_history.json")
    compact = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in x.items()} for x in history]
    hist_path.write_text(json.dumps({"meta": {**cfg.to_dict(), "summary": summary, "curve": progress.curve},
                                     "history": compact[-20000:]}))
    progress.write(state="completed", checkpoint=str(out), summary=summary)
    log.info("Saved %s and %s in %.1fs", out, hist_path, time.time() - t0)


if __name__ == "__main__":
    main()
