"""Exact classic baselines. Every new forecaster should beat these before anything else.

naive and seasonal_naive also return quantiles from their own in-sample errors at each lag,
so the probabilistic scores have a reference too.
"""

import numpy as np

from horizoneval.prediction import Prediction, empirical_quantiles
from horizoneval.registry import forecaster


@forecaster("naive")
def naive(Y: np.ndarray, h: int, period: int | None = None) -> Prediction:
    point = np.repeat(Y[:, -1:], h, axis=1)
    return Prediction(point, empirical_quantiles(Y, point, np.arange(1, h + 1)))


@forecaster("seasonal_naive")
def seasonal_naive(Y: np.ndarray, h: int, period: int | None = None) -> Prediction:
    """The last full season repeated; falls back to naive without a period or with too little history."""
    m = period or 1
    if m <= 1 or Y.shape[1] < m:
        return naive(Y, h)
    reps = int(np.ceil(h / m))
    point = np.tile(Y[:, -m:], reps)[:, :h]
    lags = m * np.ceil(np.arange(1, h + 1) / m).astype(int)
    return Prediction(point, empirical_quantiles(Y, point, lags))


@forecaster("drift")
def drift(Y: np.ndarray, h: int, period: int | None = None) -> np.ndarray:
    slope = (Y[:, -1] - Y[:, 0]) / max(Y.shape[1] - 1, 1)
    return Y[:, -1:] + slope[:, None] * np.arange(1, h + 1)


@forecaster("mean")
def mean(Y: np.ndarray, h: int, period: int | None = None) -> np.ndarray:
    return np.repeat(Y.mean(axis=1, keepdims=True), h, axis=1)
