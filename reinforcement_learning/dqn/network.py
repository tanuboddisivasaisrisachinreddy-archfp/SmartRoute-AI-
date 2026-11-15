"""Q-network definitions.

Architecture ("action-in" DQN, handles a variable action space):

    input  x[k] = [state features (8) | action features (18)]   for each
                  candidate outgoing edge k = 1..MAX_ACTIONS
    MLP    26 -> 128 -> ReLU -> 128 -> ReLU -> 64 -> ReLU -> 1
    output Q(s, a_k) for every row; padded rows are masked with -inf.

The same weights score every candidate, so intersections with 1..8 exits are
handled naturally and the network generalises to any origin/destination pair
in Hyderabad (no node ids are used as inputs).

The PyTorch module is used for training.  For inference the backend can use
an exact NumPy re-implementation of the forward pass (weights exported from
the PyTorch checkpoint) so the web service does not need to import torch,
which keeps memory inside Render's free-tier limit.  ``tests/test_dqn.py``
checks both give identical Q-values.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HIDDEN = (128, 128, 64)


def build_torch_qnet(n_features: int, hidden=HIDDEN):
    import torch.nn as nn

    layers = []
    prev = n_features
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class NumpyQNetwork:
    """Inference-only MLP with weights exported from the PyTorch model."""

    def __init__(self, weights: list[tuple[np.ndarray, np.ndarray]]):
        self.weights = weights

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = x.astype(np.float32)
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(self.weights):
            h = h @ w.T + b
            if i < last:
                np.maximum(h, 0, out=h)
        return h[..., 0]

    @classmethod
    def from_torch(cls, module) -> "NumpyQNetwork":
        import torch.nn as nn

        ws = []
        for layer in module:
            if isinstance(layer, nn.Linear):
                ws.append((layer.weight.detach().cpu().numpy().astype(np.float32),
                           layer.bias.detach().cpu().numpy().astype(np.float32)))
        return cls(ws)

    def save(self, path: Path, meta: dict) -> None:
        arrays = {}
        for i, (w, b) in enumerate(self.weights):
            arrays[f"w{i}"] = w
            arrays[f"b{i}"] = b
        np.savez_compressed(path, meta=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: Path) -> tuple["NumpyQNetwork", dict]:
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        n = len([k for k in data.files if k.startswith("w")])
        return cls([(data[f"w{i}"], data[f"b{i}"]) for i in range(n)]), meta
