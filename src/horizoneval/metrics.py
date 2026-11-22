"""Forecast scores. Inputs are (N, h) (quantiles (N, h, Q)); each per-series score returns (N,).

Scaled scores divide by the in-sample MAE of the one-step seasonal naive on the history (as in MASE),
so they average across series of different scales. Quantile scores take the levels of the last axis
(default `QUANTILES`); interval scores are computed for the central intervals whose two levels are present.
"""

import numpy as np

from horizoneval.prediction import QUANTILES, Prediction

# central intervals reported when both of their quantiles are present: name -> (lower, upper)
INTERVALS = {"80": (0.1, 0.9), "90": (0.05, 0.95)}


def _positive(scale):
    """Scales as floats, with NaN where a series' history never changed (scale 0): scaled scores are undefined there
    and left out of averages, as in GIFT-Eval, instead of exploding."""
    s = np.asarray(scale, dtype=float)
    return np.where(s > 0, s, np.nan)


def scale_of(hist, period=None):
    m = period if period and hist.shape[1] > period else 1
    return np.abs(hist[:, m:] - hist[:, :-m]).mean(axis=1)


def _mean(x, axis=1):
    """Mean over the observed steps (truth present): NaN where nothing was observed."""
    ok = np.isfinite(x)
    n = ok.sum(axis=axis)
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, np.where(ok, x, 0.0).sum(axis=axis) / np.maximum(n, 1), np.nan)


def observed(T):
    """Number of observed (non-missing) target steps per series, (N,)."""
    return np.isfinite(T).sum(axis=1)


def mae(F, T):
    return _mean(np.abs(F - T))


def mse(F, T):
    return _mean((F - T) ** 2)


SMAPE_ZERO = ("zero", "skip")


def smape(F, T, zero="zero"):
    """2 |F - T| / (|F| + |T|), between 0 and 2. Where truth and forecast are both 0 the ratio is 0 / 0:
    zero="zero" counts the step as a perfect forecast (M4, utilsforecast, the default); zero="skip" leaves it out
    (gluonts, hence GIFT-Eval)."""
    if zero not in SMAPE_ZERO:
        raise ValueError(f"smape_zero must be one of {SMAPE_ZERO}, got {zero!r}")
    denom = np.abs(F) + np.abs(T)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, 2 * np.abs(F - T) / np.where(denom > 0, denom, 1), 0.0 if zero == "zero" else np.nan)
    return _mean(np.where(np.isfinite(T), r, np.nan))


def pinball(Q, T, levels=QUANTILES):
    """Mean pinball loss over steps and quantile levels, (N,)."""
    q = np.asarray(levels)
    d = T[..., None] - Q
    loss = np.maximum(q * d, (q - 1) * d).mean(axis=-1)  # NaN where the truth is missing
    return _mean(loss)


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
    return _mean(np.where(np.isfinite(T), ((T >= L) & (T <= U)).astype(float), np.nan))


def scores(pred: Prediction, T, scale, smape_zero="zero") -> dict[str, np.ndarray]:
    """Every per-series score of one forecast against the truth T, given each series' scale (N,).

    Missing truth (NaN) is left out: each score averages the observed steps, and `n` counts them."""
    s = _positive(scale)
    out = {"n": observed(T), "mae": mae(pred.point, T), "mse": mse(pred.point, T),
           "smape": smape(pred.point, T, smape_zero)}
    if smape_zero == "skip":  # steps that entered the sMAPE, so that averages weight each such step once
        out["n_smape"] = (np.isfinite(T) & ((np.abs(pred.point) + np.abs(T)) > 0)).sum(axis=1)
    out["mase"] = out["mae"] / s
    if pred.quantiles is not None:
        out["crps_scaled"] = crps(pred.quantiles, T, pred.levels) / s
        for name, (lo, hi) in INTERVALS.items():
            if _index(pred.levels, lo) is None or _index(pred.levels, hi) is None:
                continue
            out[f"coverage{name}"] = coverage(pred.quantiles, T, lo, hi, pred.levels)
        if "coverage80" in out:
            L, U = interval(pred.quantiles, *INTERVALS["80"], pred.levels)
            out["width80_scaled"] = _mean(np.where(np.isfinite(T), U - L, np.nan)) / s
    return out


def series_scores(pred: Prediction, T, hist, period=None) -> dict[str, np.ndarray]:
    return scores(pred, T, scale_of(hist, period))


def step_errors(pred: Prediction, T, scale) -> np.ndarray:
    """Scaled absolute error per series and forecast step: (N, h)."""
    return np.abs(pred.point - T) / _positive(scale)[:, None]


def horizon_scores(pred: Prediction, T, hist, period=None) -> np.ndarray:
    """Scaled absolute error by forecast step, averaged over series: (h,)."""
    return np.nanmean(step_errors(pred, T, scale_of(hist, period)), axis=0)


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
        raw = _mean(np.maximum(a * d, (a - 1) * d))
    s = np.asarray(scale, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.isnan(s), raw, np.where(s > 0, raw / s, np.nan))
