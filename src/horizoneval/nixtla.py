"""horizoneval scores as utilsforecast losses, so they run inside Nixtla's own `evaluate`:

    from utilsforecast.evaluation import evaluate
    from horizoneval import nixtla
    evaluate(cv_df, metrics=[nixtla.mase(seasonality=24), nixtla.smape(), nixtla.crps_scaled(24)],
             train_df=train_df)

Each factory returns a callable with utilsforecast's loss signature (df, models, id_col, target_col, cutoff_col
[, train_df, time_col]) that returns one row per (cutoff, unique_id) and one column per model, computed by the same
code as `horizoneval.evaluate`. Quantile scores read `<model>-q-<p>`, `-lo-/-hi-` and `-median` columns, so
TimeCopilot frames work too. Needs only pandas; utilsforecast itself is not imported.
"""

import numpy as np
import pandas as pd

from horizoneval.frame import quantile_columns, scales
from horizoneval.metrics import scores
from horizoneval.prediction import Prediction

SCALED = {"mase", "crps_scaled", "width80_scaled"}


def _std(df, id_col, target_col, cutoff_col, time_col):
    cols = {id_col: "unique_id", target_col: "y", time_col: "ds"}
    if cutoff_col in df.columns:
        cols[cutoff_col] = "cutoff"
    d = df.rename(columns=cols)
    if "cutoff" not in d:  # a single forecast without cutoffs: one window per series
        d = d.assign(cutoff=d.groupby("unique_id")["ds"].transform("min") - _one(d["ds"]))
    return d


def _one(ds: pd.Series):
    return pd.Timedelta(0) if pd.api.types.is_datetime64_any_dtype(ds) else 0


def _score(name, df, models, seasonality, train_df, id_col, target_col, cutoff_col, time_col):
    d = _std(df, id_col, target_col, cutoff_col, time_col).sort_values(["unique_id", "cutoff", "ds"], kind="stable")
    keys = d.groupby(["unique_id", "cutoff"], sort=False).head(1)[["unique_id", "cutoff"]]
    G = len(keys)
    h = len(d) // G
    T = d["y"].to_numpy(float).reshape(G, h)
    if name in SCALED:
        tr = _std(train_df, id_col, target_col, "__none__", time_col)
        scale = scales(tr, keys["unique_id"].to_numpy(), keys["cutoff"].to_numpy(), np.full(G, seasonality))
    else:
        scale = np.ones(G)
    out = keys.reset_index(drop=True).copy()
    for m in models:
        point = d[m].to_numpy(float).reshape(G, h) if m in d else None
        qc = quantile_columns(d, m)
        Q = np.stack([d[c].to_numpy(float).reshape(G, h) for c in qc.values()], -1) if qc else None
        if point is None:  # quantile-only model: the median is the point
            point = Q[..., list(qc).index(0.5)] if Q is not None and 0.5 in qc else np.full((G, h), np.nan)
        s = scores(Prediction(point, Q, tuple(qc) if qc else ()), T, scale) if Q is not None else \
            scores(Prediction(point), T, scale)
        out[m] = s.get(name, np.full(G, np.nan))
    out = out.rename(columns={"unique_id": id_col, "cutoff": cutoff_col})
    if cutoff_col not in df.columns:
        out = out.drop(columns=cutoff_col)
    return out


def _factory(name, needs_train):
    def make(seasonality: int = 1):
        if needs_train:
            def loss(df, models, train_df, id_col="unique_id", target_col="y", cutoff_col="cutoff", time_col="ds"):
                return _score(name, df, models, seasonality, train_df, id_col, target_col, cutoff_col, time_col)
        else:
            def loss(df, models, id_col="unique_id", target_col="y", cutoff_col="cutoff", time_col="ds"):
                return _score(name, df, models, seasonality, None, id_col, target_col, cutoff_col, time_col)
        loss.__name__ = name
        loss.__qualname__ = name
        return loss

    make.__name__ = name
    make.__doc__ = f"horizoneval `{name}` as a utilsforecast loss" + (" (needs train_df)" if needs_train else "")
    return make


mase = _factory("mase", True)
crps_scaled = _factory("crps_scaled", True)
width80_scaled = _factory("width80_scaled", True)
mae = _factory("mae", False)
mse = _factory("mse", False)
smape = _factory("smape", False)
coverage80 = _factory("coverage80", False)
coverage90 = _factory("coverage90", False)
