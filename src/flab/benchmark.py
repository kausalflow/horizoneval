"""Rolling-origin benchmark: every model on every dataset, scored per series and window, summarized and tested.

Protocol: each series has n = context + h + (windows - 1) * step points. Window k forecasts h steps from cutoff
c_k = n - h - (windows - 1 - k) * step and sees only Y[:, :c_k]. The datasets are drawn once from the seed and
fingerprinted, so a run is reproducible from its config.json.
skill = mean log(MSE_ref / MSE_model) in nats over series and windows: > 0 beats the reference.
"""

import hashlib
import json
import platform
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from flab.data import expand, load_dataset
from flab.metrics import horizon_scores, series_scores
from flab.prediction import as_prediction
from flab.registry import load
from flab.stats import comparisons, friedman_nemenyi


def cutoffs(n: int, h: int, windows: int, step: int) -> list[int]:
    cs = [n - h - (windows - 1 - k) * step for k in range(windows)]
    if cs[0] < 2:
        raise ValueError("not enough history for the requested windows")
    return cs


def fingerprint(Y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(Y, dtype=np.float64).tobytes()).hexdigest()[:16]


def evaluate(model, name: str, ds, h: int, windows: int, step: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, hrows = [], []
    for k, c in enumerate(cutoffs(ds.Y.shape[1], h, windows, step)):
        hist, truth = ds.Y[:, :c], ds.Y[:, c : c + h]
        pred = as_prediction(model(hist, h, ds.period))
        if pred.point.shape != truth.shape:
            raise ValueError(f"{name} returned shape {pred.point.shape}, expected {truth.shape}")
        if not np.isfinite(pred.point).all():
            raise ValueError(f"{name} returned non-finite forecasts on {ds.name}")
        s = series_scores(pred, truth, hist, ds.period)
        rows.append(pd.DataFrame({"model": name, "dataset": ds.name, "window": k, "series": np.arange(len(truth)), **s}))
        hrows.append(pd.DataFrame({"model": name, "dataset": ds.name, "window": k, "step": np.arange(1, h + 1),
                                   "mase": horizon_scores(pred, truth, hist, ds.period)}))
    return pd.concat(rows, ignore_index=True), pd.concat(hrows, ignore_index=True)


def summarize(scores: pd.DataFrame, reference: str = "naive") -> pd.DataFrame:
    metrics = [m for m in ["mase", "smape", "crps_scaled", "coverage80", "coverage90", "width80_scaled"] if m in scores]
    summary = scores.groupby(["model", "dataset"])[metrics].mean()
    if reference in scores["model"].unique():
        ref = scores[scores["model"] == reference].set_index(["dataset", "window", "series"])["mse"]
        s = scores.join(ref.rename("ref_mse"), on=["dataset", "window", "series"])
        s["skill"] = np.log(np.maximum(s["ref_mse"], 1e-300) / np.maximum(s["mse"], 1e-300))
        summary["skill"] = s.groupby(["model", "dataset"])["skill"].mean()
    summary = summary.reset_index()
    summary["rank"] = summary.groupby("dataset")["mase"].rank()
    return summary


def leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    cols = {c: f"mean_{c}" for c in ["rank", "mase", "skill", "crps_scaled", "coverage80"] if c in summary}
    lb = summary.groupby("model")[list(cols)].mean().rename(columns=cols)
    lb["datasets"] = summary.groupby("model")["dataset"].nunique()
    return lb.sort_values("mean_rank").reset_index()


def run(models, datasets, h=12, windows=5, step=None, context=256, n_series=30, seed=0, out=None,
        reference="naive", report=True):
    step = step or h
    n = context + h + (windows - 1) * step
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(([reference] if reference else []) + list(models)))
    loaded = [load_dataset(d, n_series, n, rng) for d in expand(datasets)]
    all_scores, all_h, timings, failures = [], [], [], {}
    for name in names:
        t0 = time.time()
        try:
            model = load(name)
            parts = [evaluate(model, name, ds, h, windows, step) for ds in loaded]
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"  FAIL  {name}: {failures[name]}")
            traceback.print_exc(limit=2)
            continue
        all_scores += [p[0] for p in parts]
        all_h += [p[1] for p in parts]
        timings.append({"model": name, "seconds": round(time.time() - t0, 2)})
        print(f"  done  {name} ({timings[-1]['seconds']}s)")
    if not all_scores:
        raise RuntimeError("every model failed")
    scores = pd.concat(all_scores, ignore_index=True)
    by_h = pd.concat(all_h, ignore_index=True).groupby(["model", "dataset", "step"], as_index=False)["mase"].mean()
    summary = summarize(scores, reference)
    lb = leaderboard(summary).merge(pd.DataFrame(timings), on="model", how="left")
    tests = comparisons(scores, reference)
    ranks = friedman_nemenyi(summary)
    config = dict(models=names, datasets={d.name: {"period": d.period, "shape": list(d.Y.shape),
                                                   "sha256_16": fingerprint(d.Y)} for d in loaded},
                  h=h, windows=windows, step=step, context=context, n_series=n_series, seed=seed,
                  reference=reference, failures=failures, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  python=platform.python_version())
    if out:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        scores.to_csv(out / "scores.csv", index=False)
        by_h.to_csv(out / "by_horizon.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        lb.to_csv(out / "leaderboard.csv", index=False)
        tests.to_csv(out / "tests.csv", index=False)
        (out / "ranks.json").write_text(json.dumps(ranks, indent=2) + "\n")
        (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        if report:
            from flab.report import write_report

            write_report(out)
    return lb, summary, scores
