"""Entropic Force Neural Network as a forecaster, ported from kausalflow/physics_models.

The original EFNN (physics_models/entropic_force_nn.py, PAPER.md) is a regression MLP with two thermodynamic pieces:
    activation  a(x) = T * tanh(x / T)              temperature-scaled, after Linear + BatchNorm
    loss        MSE - w * H(softmax(a_last / T))    rewards high entropy in the last hidden layer
The paper only tested it on static regression (y = x^2 and friends). Here it becomes a direct multi-horizon
forecaster: one global network per call, trained on every sliding window of the N histories, each window scaled
by its last value and the spread of its differences, mapping the last L points to the next h.

Three registered variants reproduce the paper's ablation (model comparison and entropy-weight study):
    efnn        entropic activation + entropy loss (T = 1, w = 0.1, the paper's defaults)
    efnn_noent  entropic activation, no entropy term (w = 0)
    efnn_mlp    ReLU and no entropy term: the plain MLP control with the same windows and training budget

Needs the `physics` extra (torch). Plain torch replaces PyTorch Lightning; the network and loss are unchanged.
"""

import numpy as np

from flab.registry import forecaster

EPS = 1e-10
MIN_TEMPERATURE = 0.01


def windows(Y: np.ndarray, L: int, h: int, max_samples: int, rng: np.random.Generator):
    """Every (input L, target h) window of every row, scaled; subsampled to at most max_samples."""
    N, n = Y.shape
    starts = np.arange(n - L - h + 1)
    rows, cols = np.repeat(np.arange(N), len(starts)), np.tile(starts, N)
    if len(rows) > max_samples:
        keep = rng.choice(len(rows), max_samples, replace=False)
        rows, cols = rows[keep], cols[keep]
    W = Y[rows[:, None], cols[:, None] + np.arange(L + h)]
    level, scale = window_scale(W[:, :L])
    return (W[:, :L] - level) / scale, (W[:, L:] - level) / scale


def window_scale(X: np.ndarray):
    """Level = last value, scale = mean absolute difference (the naive one-step error), floored."""
    level = X[:, -1:]
    scale = np.abs(np.diff(X, axis=1)).mean(axis=1, keepdims=True)
    floor = 1e-6 * np.maximum(np.abs(X).mean(axis=1, keepdims=True), 1.0)
    return level, np.maximum(scale, floor)


class EntropicForecaster:
    """Global EFNN over scaled windows. Built with no arguments by the registry; subclasses set the variant."""

    temperature = 1.0
    entropy_weight = 0.1
    entropic_activation = True
    hidden = (64, 32)
    max_lags = 64
    steps = 400
    batch = 256
    lr = 1e-3
    max_samples = 20_000
    seed = 0

    def __init__(self):
        import torch  # the `physics` extra; imported only when a benchmark asks for this model

        self.torch = torch

    def lags(self, n: int, h: int, period: int | None) -> int:
        L = max(2 * h, 2 * (period or 0), 8)
        return int(min(L, self.max_lags, n - h - 1))

    def net(self, L: int, h: int):
        torch, nn = self.torch, self.torch.nn
        T = max(self.temperature, MIN_TEMPERATURE)
        entropic = self.entropic_activation

        class EFNN(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
                self.norms = nn.ModuleList(nn.BatchNorm1d(b) for b in dims[1:])
                self.out = nn.Linear(dims[-1], h)

            def forward(self, x):
                for layer, bn in zip(self.layers, self.norms):
                    x = bn(layer(x))
                    x = T * torch.tanh(x / T) if entropic else torch.relu(x)
                return self.out(x), x

        return EFNN((L, *self.hidden))

    def loss(self, pred, target, last_hidden):
        F = self.torch.nn.functional
        base = F.mse_loss(pred, target)
        if self.entropy_weight == 0:
            return base
        T = max(self.temperature, MIN_TEMPERATURE)
        p = F.softmax(last_hidden / T, dim=-1)
        entropy = -(p * self.torch.log(p + EPS)).sum(dim=-1).mean()
        return base - self.entropy_weight * entropy

    def __call__(self, Y: np.ndarray, h: int, period: int | None = None) -> np.ndarray:
        torch = self.torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        N, n = Y.shape
        L = self.lags(n, h, period)
        X, T = windows(Y, L, h, self.max_samples, rng)
        X, T = torch.tensor(X, dtype=torch.float32), torch.tensor(T, dtype=torch.float32)

        model = self.net(L, h)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        model.train()
        for _ in range(self.steps):
            idx = torch.from_numpy(rng.integers(0, len(X), min(self.batch, len(X))))
            pred, last_hidden = model(X[idx])
            loss = self.loss(pred, T[idx], last_hidden)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        level, scale = window_scale(Y[:, -L:])
        with torch.no_grad():
            Z = torch.tensor((Y[:, -L:] - level) / scale, dtype=torch.float32)
            out = model(Z)[0].numpy().astype(float)
        return level + scale * out


@forecaster("efnn")
class EFNN(EntropicForecaster):
    pass


@forecaster("efnn_noent")
class EFNNNoEntropy(EntropicForecaster):
    entropy_weight = 0.0


@forecaster("efnn_mlp")
class EFNNMLP(EntropicForecaster):
    entropy_weight = 0.0
    entropic_activation = False
