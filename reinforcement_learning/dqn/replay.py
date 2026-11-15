"""Fixed-size experience replay buffer (NumPy ring buffer)."""
from __future__ import annotations

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, max_actions: int, n_features: int, seed: int = 0):
        self.capacity = int(capacity)
        self.feats = np.zeros((capacity, max_actions, n_features), np.float32)
        self.mask = np.zeros((capacity, max_actions), bool)
        self.action = np.zeros(capacity, np.int64)
        self.reward = np.zeros(capacity, np.float32)
        self.next_feats = np.zeros((capacity, max_actions, n_features), np.float32)
        self.next_mask = np.zeros((capacity, max_actions), bool)
        self.terminal = np.zeros(capacity, np.float32)
        self.pos = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def push(self, feats, mask, action, reward, next_feats, next_mask, terminal) -> None:
        i = self.pos
        self.feats[i] = feats
        self.mask[i] = mask
        self.action[i] = action
        self.reward[i] = reward
        self.next_feats[i] = next_feats
        self.next_mask[i] = next_mask
        self.terminal[i] = float(terminal)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch: int):
        idx = self.rng.integers(0, self.size, size=batch)
        return (self.feats[idx], self.mask[idx], self.action[idx], self.reward[idx],
                self.next_feats[idx], self.next_mask[idx], self.terminal[idx])

    def __len__(self) -> int:
        return self.size
