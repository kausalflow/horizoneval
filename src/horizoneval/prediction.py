"""What a forecaster returns: point forecasts, optionally with quantiles at the fixed levels QUANTILES.

A forecaster may return a plain (N, h) array (point only) or a Prediction. Probabilistic scores
(CRPS, interval coverage) are computed only for models that return quantiles.
"""

from dataclasses import dataclass

import numpy as np

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)


@dataclass
class Prediction:
    point: np.ndarray  # (N, h)
    quantiles: np.ndarray | None = None  # (N, h, len(levels)), non-decreasing along the last axis
    levels: tuple[float, ...] = QUANTILES  # quantile levels of the last axis, increasing

    def __post_init__(self):
        self.point = np.asarray(self.point, dtype=float)
        self.levels = tuple(float(q) for q in self.levels)
        if list(self.levels) != sorted(set(self.levels)):
            raise ValueError(f"quantile levels must be increasing and distinct, got {self.levels}")
        if self.quantiles is not None:
            q = np.sort(np.asarray(self.quantiles, dtype=float), axis=-1)  # repair crossings
            if q.shape != self.point.shape + (len(self.levels),):
                raise ValueError(f"quantiles have shape {q.shape}, expected {self.point.shape + (len(self.levels),)}")
            self.quantiles = q


def as_prediction(out) -> Prediction:
    return out if isinstance(out, Prediction) else Prediction(np.asarray(out, dtype=float))


def empirical_quantiles(Y: np.ndarray, point: np.ndarray, lags: np.ndarray) -> np.ndarray:
    """Quantiles from the method's own in-sample errors: step k of the forecast is y_{t+lag_k} - y_t shifted.

    For a forecaster that copies the value `lags[k]` steps back (naive: lag k; seasonal naive: m * ceil(k / m)),
    the in-sample errors at step k are Y[:, lag:] - Y[:, :-lag], and their quantiles shift the point forecast.
    """
    N, h = point.shape
    q = np.empty((N, h, len(QUANTILES)))
    for k, lag in enumerate(lags):
        if lag >= Y.shape[1]:
            q[:, k] = point[:, k : k + 1]  # no in-sample error at this lag: degenerate interval
            continue
        err = Y[:, lag:] - Y[:, :-lag]
        q[:, k] = point[:, k : k + 1] + np.quantile(err, QUANTILES, axis=1).T
    return q
