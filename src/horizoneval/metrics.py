"""Forecast scores. Inputs are (N, h) (quantiles (N, h, Q)); each per-series score returns (N,).

Scaled scores divide by the in-sample MAE of the one-step seasonal naive on the history (as in MASE),
so they average across series of different scales.
"""

import numpy as np

from horizoneval.prediction import QUANTILES, Prediction


def scale_of(hist, period=None):
    m = period if period and hist.shape[1] > period else 1
    return np.maximum(np.abs(hist[:, m:] - hist[:, :-m]).mean(axis=1), 1e-12)


def mae(F, T):
    return np.abs(F - T).mean(axis=1)


def mse(F, T):
    return ((F - T) ** 2).mean(axis=1)


def smape(F, T):
    denom = np.abs(F) + np.abs(T)
    return np.where(denom > 0, 2 * np.abs(F - T) / np.where(denom > 0, denom, 1), 0.0).mean(axis=1)


def pinball(Q, T):
    """Mean pinball loss over steps and quantile levels, (N,)."""
    q = np.asarray(QUANTILES)
    d = T[..., None] - Q
    return np.maximum(q * d, (q - 1) * d).mean(axis=(1, 2))


def crps(Q, T):
    """CRPS approximated on the quantile grid: twice the mean pinball loss."""
    return 2 * pinball(Q, T)


def interval(Q, lo: float, hi: float):
    return Q[..., QUANTILES.index(lo)], Q[..., QUANTILES.index(hi)]


def coverage(Q, T, lo=0.1, hi=0.9):
    L, U = interval(Q, lo, hi)
    return ((T >= L) & (T <= U)).mean(axis=1)


def series_scores(pred: Prediction, T, hist, period=None) -> dict[str, np.ndarray]:
    s = scale_of(hist, period)
    out = {"mae": mae(pred.point, T), "mse": mse(pred.point, T), "smape": smape(pred.point, T)}
    out["mase"] = out["mae"] / s
    if pred.quantiles is not None:
        out["crps_scaled"] = crps(pred.quantiles, T) / s
        out["coverage80"] = coverage(pred.quantiles, T, 0.1, 0.9)
        out["coverage90"] = coverage(pred.quantiles, T, 0.05, 0.95)
        L, U = interval(pred.quantiles, 0.1, 0.9)
        out["width80_scaled"] = (U - L).mean(axis=1) / s
    return out


def horizon_scores(pred: Prediction, T, hist, period=None) -> np.ndarray:
    """Scaled absolute error by forecast step, averaged over series: (h,)."""
    return (np.abs(pred.point - T) / scale_of(hist, period)[:, None]).mean(axis=0)
