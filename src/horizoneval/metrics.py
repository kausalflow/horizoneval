"""Forecast scores. Inputs are (N, h) (quantiles (N, h, Q)); each per-series score returns (N,).

Scaled scores divide by the in-sample MAE of the one-step seasonal naive on the history (as in MASE),
so they average across series of different scales. Quantile scores take the levels of the last axis
(default `QUANTILES`); interval scores are computed for the central intervals whose two levels are present.
"""

import numpy as np

from horizoneval.prediction import QUANTILES, Prediction

# central intervals reported when both of their quantiles are present: name -> (lower, upper)
INTERVALS = {"80": (0.1, 0.9), "90": (0.05, 0.95)}


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


def pinball(Q, T, levels=QUANTILES):
    """Mean pinball loss over steps and quantile levels, (N,)."""
    q = np.asarray(levels)
    d = T[..., None] - Q
    return np.maximum(q * d, (q - 1) * d).mean(axis=(1, 2))


def crps(Q, T, levels=QUANTILES):
    """CRPS approximated on the quantile grid: twice the mean pinball loss."""
    return 2 * pinball(Q, T, levels)


def _index(levels, q):
    """Position of level q in levels (to 1e-9), or None."""
    for i, lv in enumerate(levels):
        if abs(lv - q) < 1e-9:
            return i
    return None


def interval(Q, lo: float, hi: float, levels=QUANTILES):
    i, j = _index(levels, lo), _index(levels, hi)
    if i is None or j is None:
        raise KeyError(f"levels {tuple(levels)} do not contain both {lo} and {hi}")
    return Q[..., i], Q[..., j]


def coverage(Q, T, lo=0.1, hi=0.9, levels=QUANTILES):
    L, U = interval(Q, lo, hi, levels)
    return ((T >= L) & (T <= U)).mean(axis=1)


def scores(pred: Prediction, T, scale) -> dict[str, np.ndarray]:
    """Every per-series score of one forecast against the truth T, given each series' scale (N,)."""
    s = np.maximum(np.asarray(scale, dtype=float), 1e-12)
    out = {"mae": mae(pred.point, T), "mse": mse(pred.point, T), "smape": smape(pred.point, T)}
    out["mase"] = out["mae"] / s
    if pred.quantiles is not None:
        out["crps_scaled"] = crps(pred.quantiles, T, pred.levels) / s
        for name, (lo, hi) in INTERVALS.items():
            if _index(pred.levels, lo) is None or _index(pred.levels, hi) is None:
                continue
            out[f"coverage{name}"] = coverage(pred.quantiles, T, lo, hi, pred.levels)
        if "coverage80" in out:
            L, U = interval(pred.quantiles, *INTERVALS["80"], pred.levels)
            out["width80_scaled"] = (U - L).mean(axis=1) / s
    return out


def series_scores(pred: Prediction, T, hist, period=None) -> dict[str, np.ndarray]:
    return scores(pred, T, scale_of(hist, period))


def step_errors(pred: Prediction, T, scale) -> np.ndarray:
    """Scaled absolute error per series and forecast step: (N, h)."""
    return np.abs(pred.point - T) / np.maximum(np.asarray(scale, dtype=float), 1e-12)[:, None]


def horizon_scores(pred: Prediction, T, hist, period=None) -> np.ndarray:
    """Scaled absolute error by forecast step, averaged over series: (h,)."""
    return step_errors(pred, T, scale_of(hist, period)).mean(axis=0)


def cost(spec, pred: Prediction, T, scale) -> np.ndarray:
    """The user's own cost of a forecast, per series (N,), divided by the series' scale when there is one.

    spec  "linlin:a"   under-forecasting costs a per unit, over-forecasting 1 - a (stock-outs vs. overstock);
                       scored on the quantile at level a when the model gives it, else on the point forecast
          "pinball:q"  pinball loss of the quantile at level q (the point forecast when the model has no such level)
          a callable   f(truth (N, h), point (N, h)) -> (N,), any cost you can write down
    """
    if callable(spec):
        raw = np.asarray(spec(T, pred.point), dtype=float)
    else:
        kind, _, val = str(spec).partition(":")
        if kind not in ("linlin", "pinball") or not val:
            raise ValueError(f"unknown cost {spec!r}; use 'linlin:<a>', 'pinball:<q>' or a callable")
        a = float(val)
        if not 0 < a < 1:
            raise ValueError(f"the level in {spec!r} must be strictly between 0 and 1")
        i = _index(pred.levels, a) if pred.quantiles is not None else None
        F = pred.quantiles[..., i] if i is not None else pred.point
        d = T - F
        raw = np.maximum(a * d, (a - 1) * d).mean(axis=1)
    s = np.asarray(scale, dtype=float)
    return np.where(np.isfinite(s), raw / np.maximum(s, 1e-12), raw)
