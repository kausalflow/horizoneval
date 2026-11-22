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
from horizoneval import choose
from horizoneval.metrics import cost as cost_of, scores, step_errors
from horizoneval.checks import preflight
from horizoneval.settings import settings as settings_of
from horizoneval.stats import comparisons, friedman_nemenyi

SUMMARY_METRICS = ["mase", "smape", "crps_scaled", "coverage80", "coverage90", "width80_scaled", "cost"]


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
    decision: dict | None = None  # which model to choose and how sure that is (horizoneval.choose)
    checks: list | None = None  # pre-flight findings on the raw forecasts (horizoneval.checks)
    ladder: dict | None = None  # where each model's skill comes from (horizoneval.ladder)

    def write(self, out, report=True, frames=True, scores=True) -> Path:
        """Write the run folder. frames=False skips forecasts.parquet, train.parquet and series.csv, scores=False skips the
        per-window scores.csv: for very large runs whose summaries are what gets committed."""
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        if scores:
            self.scores.to_csv(out / "scores.csv", index=False)
        self.by_horizon.to_csv(out / "by_horizon.csv", index=False)
        self.summary.to_csv(out / "summary.csv", index=False)
        self.leaderboard.to_csv(out / "leaderboard.csv", index=False)
        self.tests.to_csv(out / "tests.csv", index=False)
        (out / "ranks.json").write_text(json.dumps(self.ranks, indent=2) + "\n")
        if frames and self.forecasts is not None:
            self.forecasts.to_parquet(out / "forecasts.parquet", index=False)
        if frames and self.train is not None:
            self.train.to_parquet(out / "train.parquet", index=False)
        if frames and self.series is not None:
            self.series.to_csv(out / "series.csv", index=False)
        (out / "config.json").write_text(json.dumps(self.config, indent=2, default=str) + "\n")
        from horizoneval.result import dumps, to_result

        (out / "result.json").write_text(dumps(to_result(self)) + "\n")
        if self.checks is not None:
            (out / "checks.json").write_text(json.dumps(self.checks, indent=2) + "\n")
        if self.decision is not None:
            self.decision["confidence"].to_csv(out / "confidence.csv", index=False)
            (out / "decision.json").write_text(json.dumps(choose.to_json(self.decision), indent=2) + "\n")
        if self.ladder is not None:
            from horizoneval.ladder import to_json as ladder_json
            from horizoneval.result import clean

            self.ladder["rungs"].to_csv(out / "ladder_rungs.csv", index=False)
            self.ladder["models"].to_csv(out / "ladder_models.csv", index=False)
            (out / "ladder.json").write_text(json.dumps(clean(ladder_json(self.ladder)), indent=2) + "\n")
        if report:
            from horizoneval.report import write_report

            write_report(out)
        return out


def fingerprint(df: pd.DataFrame) -> str:
    """sha256 (16 hex) of a frame's content, whatever the order of its rows and columns."""
    d = df[sorted(df.columns)]
    d = d.sort_values(list(d.columns), kind="stable", ignore_index=True)
    return hashlib.sha256(pd.util.hash_pandas_object(d, index=False).to_numpy().tobytes()).hexdigest()[:16]


def protocol(forecasts: pd.DataFrame, meta: pd.DataFrame, metric: str) -> str:
    """Fingerprint of how the forecasts were evaluated, not of the numbers: series, cutoffs, horizon, periods,
    datasets and the deciding metric. Two runs with the same protocol fingerprint are comparable."""
    win = forecasts.groupby(["unique_id", "cutoff"]).size().rename("h").reset_index().astype(str)
    spec = json.dumps({"metric": metric})
    return hashlib.sha256((fingerprint(win) + fingerprint(meta.astype(str)) + spec).encode()).hexdigest()[:16]


def _series_table(df, series, period, dataset) -> pd.DataFrame:
    ids = pd.Index(df["unique_id"].unique(), name="unique_id")
    s = pd.DataFrame(index=ids)
    if series is not None:
        extra = series.drop_duplicates("unique_id").set_index("unique_id")
        missing = ids.difference(extra.index)
        if len(missing):
            raise FrameError(f"the series table has no row for {len(missing)} series, e.g. {list(missing[:3])}")
        s = s.join(extra)
    if "dataset" not in s:
        s["dataset"] = dataset
    if "period" not in s:
        s["period"] = period
    s["period"] = pd.to_numeric(s["period"], errors="coerce").fillna(1).astype(int)
    return s.reset_index()


def _horizon(forecasts, meta):
    """The horizon, or {dataset: horizon} when datasets differ."""
    h = forecasts.groupby(["unique_id", "cutoff"]).size().groupby(level=0).first()
    per = h.groupby(meta.set_index("unique_id")["dataset"].reindex(h.index)).first().astype(int)
    return int(per.iloc[0]) if per.nunique() == 1 else per.to_dict()


def _weighted_means(scores: pd.DataFrame, keys: list[str], metrics: list[str]) -> pd.DataFrame:
    """Means of per-window scores weighted by the steps each one averaged (n, or n_<metric> when a metric skips some
    steps); plain means when nothing is missing."""
    out = {}
    for m in metrics:
        col = f"n_{m}" if f"n_{m}" in scores else "n"
        w = scores[col] if col in scores else pd.Series(1.0, index=scores.index)
        ok = scores[m].notna() & (w > 0)
        num = (scores[m].where(ok, 0.0) * w.where(ok, 0)).groupby([scores[k] for k in keys]).sum()
        den = w.where(ok, 0).groupby([scores[k] for k in keys]).sum()
        out[m] = num / den.replace(0, np.nan)
    return pd.DataFrame(out)


def summarize(scores: pd.DataFrame, reference: str | None = "naive", rank_by: str = "mase") -> pd.DataFrame:
    metrics = [m for m in SUMMARY_METRICS if m in scores]
    summary = _weighted_means(scores, ["model", "dataset"], metrics)
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
             reference: str | None = "naive", config: dict | None = None, cost=None, segment: str | None = None,
             decide: bool = True, n_boot: int = 1000, seed: int = 0, smape_zero: str = "zero",
             ladder: bool = False) -> Evaluation:
    """Score every model column of a forecast frame.

    cost      your own cost of a forecast ('linlin:0.9', 'pinball:0.95' or f(truth, point) -> per-series cost); it
              becomes the `cost` score and decides the choice. See horizoneval.metrics.cost.
    segment   column of `series` whose groups may deserve different models (default: the datasets)
    smape_zero  sMAPE where truth and forecast are both 0: "zero" counts a perfect step (M4, utilsforecast), "skip"
              leaves it out (gluonts, GIFT-Eval)
    Missing truth (y NaN) is allowed: those steps are not scored, and means over windows weight each window by its
    number of observed steps (column n), so every observed point counts once.
    decide    add the decision (probability of best, winner's curse, segments; horizoneval.choose)
    A model whose forecasts are missing or non-finite anywhere is dropped and listed under config["failures"].
    ladder    split each model's skill over repeating the last value into what a simple rule gets, what a
              linear model fitted on the history adds, and what is left beyond linear, with the decision rule of
              the naive-forecaster paper (horizoneval.ladder; needs train)

    Args:
        forecasts: the forecast frame (unique_id, ds, cutoff, y, one column per model; see horizoneval.frame).
        train: history in long format (unique_id, ds, y); needed for the scaled scores (MASE, scaled CRPS), which
            are NaN without it. Only rows with ds <= cutoff are used for each window.
        models: model columns to score (default: every point-forecast column).
        series: optional table (unique_id, dataset, period) when the frame mixes datasets or periods.
        period: seasonal period for every series when `series` gives none (1 or None: non-seasonal).
        dataset: dataset name for every series when `series` gives none.
        reference: model the skill column and the paired tests are measured against, if present in the frame.
        config: extra settings recorded in config.json.

    Returns:
        An Evaluation with scores, by_horizon, summary, leaderboard, tests and ranks; `.write(out)` saves them.
    """
    meta = _series_table(forecasts, series, period, dataset)
    models = validate(forecasts, models, meta.set_index("unique_id")["dataset"])
    checks = preflight(forecasts, train, models, meta.set_index("unique_id")["dataset"])
    if train is not None:
        lacking = set(forecasts["unique_id"].unique()) - set(train["unique_id"].unique())
        if lacking:
            raise FrameError(f"the train frame has no history for {len(lacking)} series, e.g. {sorted(lacking)[:3]}")
    failures = {}
    rows, hrows = [], []
    for ds_name, ids in meta.groupby("dataset")["unique_id"]:
        part = forecasts[forecasts["unique_id"].isin(set(ids))]
        w = windows(part)
        per = meta.set_index("unique_id").loc[w.unique_id, "period"].to_numpy()
        scale = scales(train, w.unique_id, w.cutoff, per) if train is not None else np.full(len(w.unique_id), np.nan)
        # k-th window of each series (series may have different cutoffs, as in M4 where histories differ in length)
        window = pd.Series(w.cutoff).groupby(w.unique_id).rank(method="dense").astype(int).to_numpy() - 1
        for m in models:
            try:
                pred = w.prediction(m)
            except ValueError as e:
                failures[m] = f"{type(e).__name__}: {e}"
                continue
            if not np.isfinite(pred.point).all() or (pred.quantiles is not None and not np.isfinite(pred.quantiles).all()):
                failures[m] = f"non-finite forecasts on {ds_name}"
                continue
            s = scores(pred, w.truth, scale, smape_zero)
            if cost is not None:
                s["cost"] = cost_of(cost, pred, w.truth, scale)
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
    metric = "cost" if cost is not None else rank_by
    if segment is not None and segment not in meta:
        raise FrameError(f"segment {segment!r} is not a column of the series table")
    summary = summarize(sc, reference, rank_by)
    tests = comparisons(sc, reference, metrics=(rank_by, "crps_scaled"))
    if tests.empty:
        tests = pd.DataFrame(columns=["metric", "model", "dataset", "mean", "ci_low", "ci_high"])
    cfg = {"models": [m for m in models if m not in failures], "reference": reference, "failures": failures,
           "h": _horizon(forecasts, meta),
           "windows": int(forecasts["cutoff"].nunique()), "rank_by": rank_by,
           "datasets": {d: {"series": int(len(g)), "period": int(g["period"].iloc[0])}
                        for d, g in meta.groupby("dataset")},
           "forecasts_sha256_16": fingerprint(forecasts),
           "train_sha256_16": fingerprint(train) if train is not None else None,
           "protocol_sha256_16": protocol(forecasts, meta, metric), "decide_by": metric, "smape_zero": smape_zero,
           "missing_targets": int(forecasts["y"].isna().sum()),
           "cost": cost if isinstance(cost, str) or cost is None else getattr(cost, "__name__", "custom"),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "python": platform.python_version()}
    cfg["settings"] = settings_of(forecasts, train, meta, [m for m in models if m not in failures], reference,
                                  metric, cost, smape_zero)
    cfg["settings_sha256_16"] = hashlib.sha256(json.dumps(cfg["settings"], sort_keys=True, default=str)
                                               .encode()).hexdigest()[:16]
    cfg.update(config or {})
    if failures:
        cfg["failures"] = {**cfg.get("failures", {}), **failures}
    sc = sc.reset_index(drop=True)
    decision = None
    if decide:
        seg = meta.set_index("unique_id")[segment] if segment is not None else None
        decision = choose.decide(sc, metric, seg, n_boot=n_boot, seed=seed)
    lad = None
    if ladder:
        from horizoneval.ladder import ladder as ladder_of

        lad = ladder_of(forecasts, train, meta, [m for m in models if m not in failures])
    return Evaluation(sc, by_h, summary, leaderboard(summary), tests,
                      friedman_nemenyi(summary, metric=rank_by), cfg, forecasts, train, meta, decision, checks, lad)
