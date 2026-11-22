"""The forecast frame: every forecast, from any tool, as one long table in Nixtla's `cross_validation` layout.

Columns
    unique_id  series id
    ds         target timestamp (datetimes, or integers for series without a calendar)
    cutoff     last timestamp the forecaster saw; one cutoff is one rolling window
    y          truth
    <model>    point forecast, one column per model
    <model>-q-<p>             quantile at level p, in percent (`tirex-q-10`, TimeCopilot) or as a fraction (`m-q-0.1`)
    <model>-lo-<l>, -hi-<l>   central interval at level l percent (statsforecast); read as quantiles (1 - l/100)/2
                              and 1 - (1 - l/100)/2
    <model>-median            read as quantile 0.5

The history ("train") frame is the same long format (unique_id, ds, y); scaled scores use, for each window, only the
rows with ds <= cutoff, as utilsforecast does. Series attributes (dataset, period) live in a small `series` table keyed
by unique_id, so the forecast frame stays readable by Nixtla tools.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from horizoneval.prediction import Prediction

KEYS = ["unique_id", "ds", "cutoff", "y"]
_SUFFIX = re.compile(r"^(?P<model>.+)-(?:(?P<kind>q|lo|hi)-(?P<val>\d+(?:\.\d+)?)|(?P<median>median))$")


class FrameError(ValueError):
    """A forecast frame that cannot be scored as given; the message says what to fix."""


def _parse(col: str):
    """(model, quantile level) for a probabilistic column, or None for a point column."""
    m = _SUFFIX.match(col)
    if not m:
        return None
    if m["median"]:
        return m["model"], 0.5
    v = float(m["val"])
    if m["kind"] == "q":
        return m["model"], v if "." in m["val"] and v < 1 else v / 100
    alpha = (1 - v / 100) / 2
    return m["model"], round(alpha if m["kind"] == "lo" else 1 - alpha, 10)


def models_of(df: pd.DataFrame) -> list[str]:
    """Point-forecast columns, in frame order. Probabilistic columns attach to the model they name."""
    return [c for c in df.columns if c not in KEYS and _parse(c) is None]


def quantile_columns(df: pd.DataFrame, model: str) -> dict[float, str]:
    """{level: column} of every quantile, interval or median column of `model`, sorted by level."""
    out = {}
    for c in df.columns:
        p = _parse(c)
        if p and p[0] == model:
            if p[1] in out and out[p[1]] != c:
                raise FrameError(f"columns {out[p[1]]!r} and {c!r} both give quantile {p[1]} of {model!r}")
            out[p[1]] = c
    return dict(sorted(out.items()))


def validate(df: pd.DataFrame, models: list[str] | None = None, group: pd.Series | None = None) -> list[str]:
    """Check that the frame can be scored and return its models. Raises FrameError with the reason.

    Every window must have the same horizon, within each group when `group` maps unique_id to a group (a dataset):
    M4's yearly and hourly series, for instance, are forecast 6 and 48 steps ahead."""
    missing = [c for c in KEYS if c not in df.columns]
    if missing:
        raise FrameError(f"the forecast frame needs columns {KEYS}; missing {missing}")
    models = models or models_of(df)
    if not models:
        raise FrameError("the forecast frame has no model column")
    for m in models:
        if m not in df.columns:
            raise FrameError(f"model {m!r} is not a column of the forecast frame")
    if df[["unique_id", "ds", "cutoff"]].isna().any().any():
        raise FrameError("unique_id, ds and cutoff must not be missing (a missing y is allowed: that step is not scored)")
    if not (df["ds"] > df["cutoff"]).all():
        bad = df.loc[~(df["ds"] > df["cutoff"]), ["unique_id", "ds", "cutoff"]].head(3)
        raise FrameError("leakage: every target must come after its cutoff (ds > cutoff); first offending rows:\n"
                         f"{bad.to_string(index=False)}")
    if df.duplicated(["unique_id", "cutoff", "ds"]).any():
        raise FrameError("each (unique_id, cutoff, ds) must appear once")
    sizes = df.groupby(["unique_id", "cutoff"]).size().rename("h").reset_index()
    sizes["group"] = sizes["unique_id"].map(group) if group is not None else 0
    per = sizes.groupby("group")["h"].nunique()
    if (per != 1).any():
        found = sorted(sizes.loc[sizes["group"].isin(per[per != 1].index), "h"].unique())
        raise FrameError(f"every (unique_id, cutoff) window must have the same horizon"
                         f"{' within a dataset' if group is not None else ''}; found {found}")
    return models


@dataclass
class Windows:
    """A frame as arrays: one row per (unique_id, cutoff) window, sorted by unique_id then cutoff."""

    unique_id: np.ndarray  # (G,)
    cutoff: np.ndarray  # (G,)
    truth: np.ndarray  # (G, h)
    df: pd.DataFrame  # the sorted frame the arrays were taken from

    @property
    def h(self) -> int:
        return self.truth.shape[1]

    def prediction(self, model: str) -> Prediction:
        G, h = self.truth.shape
        point = self.df[model].to_numpy(dtype=float).reshape(G, h)
        qcols = quantile_columns(self.df, model)
        if not qcols:
            return Prediction(point)
        Q = np.stack([self.df[c].to_numpy(dtype=float).reshape(G, h) for c in qcols.values()], axis=-1)
        return Prediction(point, Q, tuple(qcols))


def windows(df: pd.DataFrame) -> Windows:
    d = df.sort_values(["unique_id", "cutoff", "ds"], kind="stable").reset_index(drop=True)
    first = d.groupby(["unique_id", "cutoff"], sort=False).head(1)
    G = len(first)
    h = len(d) // G
    return Windows(first["unique_id"].to_numpy(), first["cutoff"].to_numpy(), d["y"].to_numpy(float).reshape(G, h), d)


def scales(train: pd.DataFrame, unique_id: np.ndarray, cutoff: np.ndarray, period: np.ndarray) -> np.ndarray:
    """In-sample MAE of the seasonal naive (lag = period, or 1) on each series' history up to each cutoff: (G,).

    Vectorised per series with cumulative sums, so it costs one pass over the history whatever the number of windows.
    Pairs with a missing value are left out (as gluonts does); windows without any complete pair get NaN.
    """
    out = np.full(len(unique_id), np.nan)
    t = train.sort_values(["unique_id", "ds"], kind="stable")
    groups = {k: g for k, g in t.groupby("unique_id", sort=False)}
    order = pd.Series(np.arange(len(unique_id))).groupby(unique_id)
    for uid, idx in order.groups.items():
        g = groups.get(uid)
        if g is None:
            continue
        y, ds = g["y"].to_numpy(float), g["ds"].to_numpy()
        for lag in np.unique(period[idx]):
            sel = np.asarray(idx)[period[idx] == lag]
            m = int(lag) if lag and lag > 1 and len(y) > lag else 1
            d = np.abs(y[m:] - y[:-m])
            ok = np.isfinite(d)
            cs = np.concatenate([[0.0], np.cumsum(np.where(ok, d, 0.0))])
            cn = np.concatenate([[0], np.cumsum(ok)])
            n_hist = np.searchsorted(ds, cutoff[sel], side="right")  # rows with ds <= cutoff
            k = np.clip(n_hist - m, 0, len(d))
            with np.errstate(invalid="ignore", divide="ignore"):
                out[sel] = np.where(cn[k] > 0, cs[k] / np.maximum(cn[k], 1), np.nan)
    return out


def from_arrays(model: str, Y: np.ndarray, cuts: list[int], preds: list[Prediction], ids: np.ndarray) -> pd.DataFrame:
    """Frame rows for one model on one (N, n) dataset with integer time: window k forecasts Y[:, c_k : c_k + h]."""
    parts = []
    for c, pred in zip(cuts, preds):
        N, h = pred.point.shape
        part = {"unique_id": np.repeat(ids, h), "ds": np.tile(np.arange(c, c + h), N),
                "cutoff": np.full(N * h, c - 1), "y": Y[:, c : c + h].ravel(), model: pred.point.ravel()}
        if pred.quantiles is not None:
            for j, q in enumerate(pred.levels):
                part[f"{model}-q-{_fmt(q)}"] = pred.quantiles[..., j].ravel()
        parts.append(pd.DataFrame(part))
    return pd.concat(parts, ignore_index=True)


def history(Y: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    N, n = Y.shape
    return pd.DataFrame({"unique_id": np.repeat(ids, n), "ds": np.tile(np.arange(n), N), "y": Y.ravel()})


def read(path) -> pd.DataFrame:
    """A frame from parquet or CSV; text timestamps in ds / cutoff become datetimes."""
    path = str(path)
    df = pd.read_parquet(path) if path.endswith((".parquet", ".pq")) else pd.read_csv(path)
    for c in ["ds", "cutoff"]:
        if c in df and df[c].dtype == object:
            df[c] = pd.to_datetime(df[c])
    if "unique_id" in df:
        df["unique_id"] = df["unique_id"].astype(str)
    return df


def _fmt(q: float) -> str:
    """Quantile level in percent as TimeCopilot writes it: 0.1 -> '10', 0.025 -> '2.5'."""
    p = round(q * 100, 6)
    return str(int(p)) if p == int(p) else f"{p:g}"


def to_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Rename symmetric quantile pairs to statsforecast's -lo-/-hi- columns (and q 0.5 to -median), which is what
    utilsforecast's `evaluate` reads; quantiles without a partner are dropped from the copy."""
    rename, drop = {}, []
    for m in models_of(df):
        qc = quantile_columns(df, m)
        for q, col in qc.items():
            if abs(q - 0.5) < 1e-9:
                rename[col] = f"{m}-median"
                continue
            partner = next((c for p, c in qc.items() if abs(p - (1 - q)) < 1e-9), None)
            if partner is None:
                drop.append(col)
                continue
            level = _fmt(abs(1 - 2 * q))
            rename[col] = f"{m}-{'lo' if q < 0.5 else 'hi'}-{level}"
    return df.drop(columns=drop).rename(columns=rename)
