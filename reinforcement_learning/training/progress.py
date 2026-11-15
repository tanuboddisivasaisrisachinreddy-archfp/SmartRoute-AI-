"""Training progress reporting shared by the CLI scripts and the API job runner.

The trainer writes a small JSON status file (atomic replace) that the FastAPI
process polls - training never runs inside a web request.
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path


class ProgressWriter:
    def __init__(self, path: str | Path | None, algorithm: str, total_episodes: int, config: dict,
                 every: int = 5):
        self.path = Path(path) if path else None
        self.algorithm = algorithm
        self.total = total_episodes
        self.config = config
        self.every = every
        self.rewards = deque(maxlen=100)
        self.success = deque(maxlen=100)
        self.losses = deque(maxlen=100)
        self.count = 0
        self.started = time.time()
        self.curve: list[dict] = []
        self.write(state="running")

    def update(self, stats: dict) -> None:
        self.count += 1
        self.rewards.append(stats["reward"])
        self.success.append(1.0 if stats["reached"] else 0.0)
        if stats.get("loss") is not None:
            self.losses.append(stats["loss"])
        self.last = stats
        if self.count % self.every == 0 or self.count == self.total:
            self.curve.append({"episode": self.count, "avg_reward": self.avg_reward, "completion_rate": self.rate,
                               "epsilon": stats.get("epsilon"), "loss": self.avg_loss,
                               "episode_reward": stats["reward"]})
            self.curve = self.curve[-400:]
            self.write(state="running")

    @property
    def avg_reward(self) -> float:
        return round(sum(self.rewards) / max(1, len(self.rewards)), 3)

    @property
    def rate(self) -> float:
        return round(sum(self.success) / max(1, len(self.success)), 3)

    @property
    def avg_loss(self):
        return round(sum(self.losses) / len(self.losses), 5) if self.losses else None

    def write(self, state: str, **extra) -> None:
        if not self.path:
            return
        last = getattr(self, "last", {})
        payload = {
            "algorithm": self.algorithm, "state": state, "current_episode": self.count,
            "total_episodes": self.total, "episode_reward": last.get("reward"), "average_reward": self.avg_reward,
            "epsilon": last.get("epsilon"), "loss": self.avg_loss, "completion_rate": self.rate,
            "elapsed_s": round(time.time() - self.started, 1), "config": self.config, "curve": self.curve,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), **extra,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, self.path)
