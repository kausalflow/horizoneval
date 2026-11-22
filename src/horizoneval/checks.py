"""Pre-flight checks on raw forecasts, before anything is scored.

Most wrong forecast comparisons are wrong before the first score: a forecast that copied the truth, forecasts in
thousands against truth in units, quantiles that cross, a history that disagrees with the truth, two model columns
that are the same model. `preflight` looks for each and says what it found, in plain language:

    severity  "error": the scores of that model would mislead; "warning": look before trusting them; "info": context

`horizoneval.evaluate` runs it on every frame (Evaluation.checks, checks.json, the report), and
`horizoneval check forecasts.parquet --train train.parquet` runs it alone. It never changes the numbers.
"""

import numpy as np
import pandas as pd

from horizoneval.frame import FrameError, KEYS, models_of, quantile_columns, validate


def _row(check, severity, message, model=None, share=None):
    return {"check": check, "severity": severity, "model": model, "share": share, "message": message}


def preflight(forecasts: pd.DataFrame, train: pd.DataFrame | None = None, models: list[str] | None = None,
              group: pd.Series | None = None) -> list[dict]:
    out = []
    try:
        models = validate(forecasts, models, group)
    except FrameError as e:
        return [_row("structure", "error", str(e))]
    d = forecasts
    y = d["y"].to_numpy(float)
    obs = np.isfinite(y)
    if not obs.all():
        out.append(_row("missing_truth", "info", f"{(~obs).mean():.1%} of target steps have no truth; they are not "
                        "scored", share=float((~obs).mean())))
    flat_truth = np.nanstd(y) == 0
    ay = np.nanmedian(np.abs(y[obs])) if obs.any() else np.nan
    cols = {}
    for m in models:
        f = d[m].to_numpy(float)
        cols[m] = f
        bad = ~np.isfinite(f)
        if bad.any():
            n_series = d.loc[bad, "unique_id"].nunique()
            out.append(_row("missing_forecasts", "error", f"{m} has no forecast for {bad.mean():.1%} of steps "
                            f"({n_series} series); it cannot be scored as given", m, float(bad.mean())))
            continue
        same = np.isclose(f[obs], y[obs], rtol=0, atol=1e-12 * max(ay, 1))
        if not flat_truth and same.mean() >= 0.5:
            out.append(_row("copies_truth", "error", f"{m} equals the truth on {same.mean():.0%} of steps: it has "
                            "almost certainly seen the values it forecasts (look-ahead)", m, float(same.mean())))
        af = np.median(np.abs(f))
        if ay > 0 and af > 0 and not 0.1 <= af / ay <= 10:
            out.append(_row("units", "warning", f"{m}'s forecasts are typically {af / ay:.3g} times the size of the "
                            "truth: different units or a transform (log, thousands) not undone?", m, float(af / ay)))
        if np.ptp(f) == 0 and not flat_truth:
            out.append(_row("constant", "warning", f"{m} forecasts the same value everywhere", m))
        q = quantile_columns(d, m)
        if len(q) >= 2:
            Q = d[list(q.values())].to_numpy(float)
            crossed = (np.diff(Q, axis=1) < 0).any(axis=1)
            if crossed.any():
                out.append(_row("quantile_crossing", "warning", f"{m}'s quantiles cross on {crossed.mean():.1%} of "
                                "steps; they are sorted before scoring", m, float(crossed.mean())))
            lo, hi = Q.min(axis=1), Q.max(axis=1)
            outside = (f < lo) | (f > hi)
            if outside.mean() > 0.2:
                out.append(_row("point_outside_quantiles", "warning", f"{m}'s point forecast lies outside its own "
                                f"quantile range on {outside.mean():.0%} of steps: are the columns from the same "
                                "model?", m, float(outside.mean())))
    names = list(cols)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if np.array_equal(cols[a], cols[b]):
                out.append(_row("duplicate_model", "warning", f"{a} and {b} are identical forecasts", f"{a}, {b}"))
    if train is not None:
        j = d[["unique_id", "ds", "y"]].merge(train[["unique_id", "ds", "y"]], on=["unique_id", "ds"],
                                              suffixes=("", "_train"))
        if len(j):
            diff = ~np.isclose(j["y"], j["y_train"], equal_nan=True, rtol=1e-9)
            if diff.any():
                out.append(_row("truth_disagrees_with_history", "error", f"for {diff.mean():.1%} of the steps that "
                                "also appear in the history, y differs from the history's value: the two tables are "
                                "not the same data", share=float(diff.mean())))
        late = train.merge(d.groupby("unique_id")["cutoff"].min().rename("first_cutoff").reset_index(), on="unique_id")
        if (late["ds"] > late["first_cutoff"]).any():
            out.append(_row("history_beyond_cutoff", "info", "the history runs past the first cutoff; only rows up to "
                            "each window's cutoff are used for scaling, so nothing leaks"))
    win = d.groupby(["unique_id", "cutoff"]).size().reset_index()
    n_series, n_windows = d["unique_id"].nunique(), win.groupby("unique_id").size().max()
    if n_series < 10:
        out.append(_row("few_series", "info", f"only {n_series} series: differences between models will rarely be "
                        "significant"))
    if pd.api.types.is_datetime64_any_dtype(d["ds"]):
        steps = d.sort_values(["unique_id", "cutoff", "ds"]).groupby(["unique_id", "cutoff"])["ds"].diff().dropna()
        if steps.nunique() > 1 and steps.nunique() / max(len(steps), 1) > 0 and steps.mode().size:
            irregular = (steps != steps.mode().iloc[0]).mean()
            if irregular > 0.01:
                out.append(_row("irregular_steps", "warning", f"{irregular:.1%} of forecast steps are not one regular "
                                "time step apart: gaps or a mixed frequency", share=float(irregular)))
    return out


def summary(checks: list[dict]) -> str:
    if not checks:
        return "All pre-flight checks passed."
    n = {s: sum(c["severity"] == s for c in checks) for s in ["error", "warning", "info"]}
    return f"{n['error']} errors, {n['warning']} warnings, {n['info']} notes."
