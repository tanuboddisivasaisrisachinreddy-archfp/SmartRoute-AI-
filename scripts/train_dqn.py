#!/usr/bin/env python3
"""Train the Double-DQN routing agent (PyTorch).

    python scripts/train_dqn.py --episodes 1500
    python scripts/train_dqn.py --episodes 200 --batch-size 64 --seed 7

Writes reinforcement_learning/checkpoints/dqn.pt (PyTorch state dict),
dqn.npz (exported weights for NumPy inference) and dqn_history.json.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import _common  # noqa: F401

from reinforcement_learning.dqn.agent import DQNAgent, DQNConfig, history_to_json
from reinforcement_learning.environment.config import CHECKPOINT_DIR
from reinforcement_learning.environment.traffic import TrafficModel
from reinforcement_learning.training.loaders import load_graph
from reinforcement_learning.training.progress import ProgressWriter

log = logging.getLogger("train_dqn")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=1500)
    p.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=5e-4)
    p.add_argument("--gamma", "--discount", dest="gamma", type=float, default=0.995)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--epsilon-end", type=float, default=0.05)
    p.add_argument("--epsilon-decay", type=float, default=0.996)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--buffer-size", type=int, default=40_000)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--max-trip-km", type=float, default=22.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output", default=str(CHECKPOINT_DIR / "dqn"))
    p.add_argument("--progress-file", default=None)
    p.add_argument("--synthetic", action="store_true", help="use the synthetic test grid (tests only)")
    a = p.parse_args()

    graph = load_graph(a.synthetic)
    cfg = DQNConfig(episodes=a.episodes, learning_rate=a.lr, gamma=a.gamma, epsilon_start=a.epsilon,
                    epsilon_end=a.epsilon_end, epsilon_decay=a.epsilon_decay, batch_size=a.batch_size,
                    buffer_size=a.buffer_size, target_update=a.target_update, max_steps=a.max_steps,
                    max_trip_km=a.max_trip_km if not a.synthetic else 5.0,
                    warmup=min(1000, a.batch_size * 4), seed=a.seed)
    agent = DQNAgent(graph, cfg)
    traffic = TrafficModel(graph, "moderate", seed=a.seed)
    progress = ProgressWriter(a.progress_file, "dqn", cfg.episodes, cfg.to_dict())
    t0 = time.time()
    out = Path(a.output)
    history = agent.train(traffic, progress=progress.update, checkpoint_path=out)
    history_to_json(history, out.with_name("dqn_history.json"),
                    {**cfg.to_dict(), "curve": progress.curve, "training_seconds": round(time.time() - t0, 1),
                     "final": agent.meta})
    progress.write(state="completed", checkpoint=str(out.with_suffix(".npz")))
    log.info("Done in %.1fs - rolling success %.2f", time.time() - t0, agent.meta.get("rolling_success", 0))


if __name__ == "__main__":
    main()
