"""Score any forecast frame: the single scoring path behind `horizoneval run` and `horizoneval score`.

    ev = evaluate(cv_df, train=train_df, period=24)        # a Nixtla / TimeCopilot cross_validation frame
    ev.leaderboard; ev.summary; ev.tests; ev.write("results/my-run")

Forecasts from horizoneval forecasters, TimeCopilot, statsforecast, neuralforecast or mlforecast all arrive here in
the same layout (see horizoneval.frame), so a number never depends on where the forecast was made.
"""

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from horizoneval.frame import FrameError, scales, validate, windows
from horizoneval.metrics import scores, step_errors
from horizoneval.stats import comparisons, friedman_nemenyi

SUMMARY_METRICS = ["mase", "smape", "crps_scaled", "coverage80", "coverage90", "width80_scaled"]


@dataclass
class Evaluation:
    scores: pd.DataFrame  # one row per model, dataset, window (cutoff) and series
    by_horizon: pd.DataFrame  # scaled error by forecast step
    summary: pd.DataFrame
    leaderboard: pd.DataFrame
    tests: pd.DataFrame
    ranks: dict
    config: dict = field(default_factory=dict)
    forecasts: pd.DataFrame | None = None
    train: pd.DataFrame | None = None
    series: pd.DataFrame | None = None

    def write(self, out, report=True) -> Path:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        self.scores.to_csv(out / "scores.csv", index=False)
        self.by_horizon.to_csv(out / "by_horizon.csv", index=False)
        self.summary.to_csv(out / "summary.csv", index=False)
        self.leaderboard.to_csv(out / "leaderboard.csv", index=False)
        self.tests.to_csv(out / "tests.csv", index=False)
        (out / "ranks.json").write_text(json.dumps(self.ranks, indent=2) + "\n")
        if self.forecasts is not None:
            self.forecasts.to_parquet(out / "forecasts.parquet", index=False)
        if self.train is not None:
            self.train.to_parquet(out / "train.parquet", index=False)
        if self.series is not None:
            self.series.to_csv(out / "series.csv", index=False)
        (out / "config.json").write_text(json.dumps(self.config, indent=2, default=str) + "\n")
        if report:
            from horizoneval.report import write_report

            write_report(out)
        return out


def fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:16]


def _series_table(df, series, period, dataset) -> pd.DataFrame:
    ids = pd.Index(df["unique_id"].unique(), name="unique_id")
    s = pd.DataFrame(index=ids)
    if series is not None:
        extra = series.drop_duplicates("unique_id").set_index("unique_id")
        missing = ids.difference(extra.index)
        if len(missing):
            raise FrameError(f"the series table has no row for {len(missing)} series, e.g. {list(missing[:3])}")
        s = s.join(extra[[c for c in ["dataset", "period"] if c in extra]])
    if "dataset" not in s:
        s["dataset"] = dataset
    if "period" not in s:
        s["period"] = period
    s["period"] = pd.to_numeric(s["period"], errors="coerce").fillna(1).astype(int)
    return s.reset_index()


def summarize(scores: pd.DataFrame, reference: str | None = "naive", rank_by: str = "mase") -> pd.DataFrame:
    metrics = [m for m in SUMMARY_METRICS if m in scores]
    summary = scores.groupby(["model", "dataset"])[metrics].mean()
    if reference and reference in scores["model"].unique():
        ref = scores[scores["model"] == reference].set_index(["dataset", "window", "series"])["mse"]
        s = scores.join(ref.rename("ref_mse"), on=["dataset", "window", "series"])
        s["skill"] = np.log(np.maximum(s["ref_mse"], 1e-300) / np.maximum(s["mse"], 1e-300))
        summary["skill"] = s.groupby(["model", "dataset"])["skill"].mean()
    summary = summary.reset_index()
    summary["rank"] = summary.groupby("dataset")[rank_by].rank()
    return summary


def leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    cols = {c: f"mean_{c}" for c in ["rank", "mase", "skill", "crps_scaled", "coverage80"] if c in summary}
    lb = summary.groupby("model")[list(cols)].mean().rename(columns=cols)
    lb["datasets"] = summary.groupby("model")["dataset"].nunique()
    return lb.sort_values("mean_rank").reset_index()


def evaluate(forecasts: pd.DataFrame, train: pd.DataFrame | None = None, *, models: list[str] | None = None,
             series: pd.DataFrame | None = None, period: int | None = None, dataset: str = "data",
             reference: str | None = "naive", config: dict | None = None) -> Evaluation:
    """Score every model column of a forecast frame.

    train     history in long format (unique_id, ds, y); needed for the scaled scores (MASE, scaled CRPS), which are
              NaN without it. Only rows with ds <= cutoff are used for each window.
    series    optional table (unique_id, dataset, period) when the frame mixes datasets or periods
    period    seasonal period for every series when `series` gives none (1 or None: non-seasonal)
    reference model the skill column and the paired tests are measured against, if present in the frame
    A model whose forecasts are missing or non-finite anywhere is dropped and listed under config["failures"].
    """
    models = validate(forecasts, models)
    if train is not None:
        lacking = set(forecasts["unique_id"].unique()) - set(train["unique_id"].unique())
        if lacking:
            raise FrameError(f"the train frame has no history for {len(lacking)} series, e.g. {sorted(lacking)[:3]}")
    meta = _series_table(forecasts, series, period, dataset)
    failures = {}
    rows, hrows = [], []
    for ds_name, ids in meta.groupby("dataset")["unique_id"]:
        part = forecasts[forecasts["unique_id"].isin(set(ids))]
        w = windows(part)
        per = meta.set_index("unique_id").loc[w.unique_id, "period"].to_numpy()
        scale = scales(train, w.unique_id, w.cutoff, per) if train is not None else np.full(len(w.unique_id), np.nan)
        window = pd.Series(w.cutoff).rank(method="dense").astype(int).to_numpy() - 1
        for m in models:
            try:
                pred = w.prediction(m)
            except ValueError as e:
                failures[m] = f"{type(e).__name__}: {e}"
                continue
            if not np.isfinite(pred.point).all() or (pred.quantiles is not None and not np.isfinite(pred.quantiles).all()):
                failures[m] = f"non-finite forecasts on {ds_name}"
                continue
            s = scores(pred, w.truth, scale)
            rows.append(pd.DataFrame({"model": m, "dataset": ds_name, "window": window, "cutoff": w.cutoff,
                                      "series": w.unique_id, **s}))
            err = step_errors(pred, w.truth, scale)
            hrows.append(pd.DataFrame({"model": m, "dataset": ds_name, "step": np.arange(1, w.h + 1),
                                       "mase": np.nanmean(err, axis=0) if np.isfinite(err).any() else np.nan}))
    if not rows:
        raise FrameError(f"no model could be scored: {failures}")
    sc = pd.concat(rows, ignore_index=True)
    sc = sc[~sc["model"].isin(failures)]
    by_h = pd.concat(hrows, ignore_index=True)
    by_h = by_h[~by_h["model"].isin(failures)].reset_index(drop=True)
    rank_by = "mase" if sc["mase"].notna().any() else "smape"  # MASE needs the history; sMAPE does not
    summary = summarize(sc, reference, rank_by)
    tests = comparisons(sc, reference, metrics=(rank_by, "crps_scaled"))
    if tests.empty:
        tests = pd.DataFrame(columns=["metric", "model", "dataset", "mean", "ci_low", "ci_high"])
    cfg = {"models": [m for m in models if m not in failures], "reference": reference, "failures": failures,
           "h": int(forecasts.groupby(["unique_id", "cutoff"]).size().iloc[0]),
           "windows": int(forecasts["cutoff"].nunique()), "rank_by": rank_by,
           "datasets": {d: {"series": int(len(g)), "period": int(g["period"].iloc[0])}
                        for d, g in meta.groupby("dataset")},
           "forecasts_sha256_16": fingerprint(forecasts),
           "train_sha256_16": fingerprint(train) if train is not None else None,
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version()}
    cfg.update(config or {})
    if failures:
        cfg["failures"] = {**cfg.get("failures", {}), **failures}
    return Evaluation(sc.reset_index(drop=True), by_h, summary, leaderboard(summary), tests,
                      friedman_nemenyi(summary, metric=rank_by), cfg, forecasts, train, meta)
