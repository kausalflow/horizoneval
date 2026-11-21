"""Rolling-origin benchmark: every model on every dataset, scored per series and window, summarized and tested.

Protocol: each series has n = context + h + (windows - 1) * step points. Window k forecasts h steps from cutoff
c_k = n - h - (windows - 1 - k) * step and sees only Y[:, :c_k]. The datasets are drawn once from the seed and
fingerprinted, so a run is reproducible from its config.json.
skill = mean log(MSE_ref / MSE_model) in nats over series and windows: > 0 beats the reference.
Forecasts are collected into one forecast frame (Nixtla cross_validation layout, horizoneval.frame) and scored by
horizoneval.evaluate, the same path that scores forecasts made by any other tool.
"""

import hashlib
import platform
import time
import traceback

import numpy as np
import pandas as pd

from horizoneval.data import expand, load_dataset
from horizoneval.evaluate import evaluate as evaluate_frame
from horizoneval.evaluate import leaderboard, summarize  # noqa: F401  (kept importable from here)
from horizoneval.frame import from_arrays, history
from horizoneval.prediction import as_prediction
from horizoneval.registry import load


def cutoffs(n: int, h: int, windows: int, step: int) -> list[int]:
    cs = [n - h - (windows - 1 - k) * step for k in range(windows)]
    if cs[0] < 2:
        raise ValueError("not enough history for the requested windows")
    return cs


def fingerprint(Y: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(Y, dtype=np.float64).tobytes()).hexdigest()[:16]


def series_ids(ds) -> np.ndarray:
    return np.array([f"{ds.name}#{i:04d}" for i in range(ds.Y.shape[0])])


def forecast(model, name: str, ds, h: int, windows: int, step: int) -> pd.DataFrame:
    """Forecast frame rows of one model on one dataset; the model sees only Y[:, :c] for each cutoff c."""
    cuts, preds = cutoffs(ds.Y.shape[1], h, windows, step), []
    for c in cuts:
        pred = as_prediction(model(ds.Y[:, :c], h, ds.period))
        if pred.point.shape != (ds.Y.shape[0], h):
            raise ValueError(f"{name} returned shape {pred.point.shape}, expected {(ds.Y.shape[0], h)}")
        if not np.isfinite(pred.point).all():
            raise ValueError(f"{name} returned non-finite forecasts on {ds.name}")
        preds.append(pred)
    return from_arrays(name, ds.Y, cuts, preds, series_ids(ds))


def run(models, datasets, h=12, windows=5, step=None, context=256, n_series=30, seed=0, out=None,
        reference="naive", report=True):
    step = step or h
    n = context + h + (windows - 1) * step
    rng = np.random.default_rng(seed)
    names = list(dict.fromkeys(([reference] if reference else []) + list(models)))
    loaded = [load_dataset(d, n_series, n, rng) for d in expand(datasets)]
    frames, timings, failures = {}, [], {}
    for name in names:
        t0 = time.time()
        try:
            model = load(name)
            frames[name] = pd.concat([forecast(model, name, ds, h, windows, step) for ds in loaded], ignore_index=True)
        except Exception as e:  # noqa: BLE001
            failures[name] = f"{type(e).__name__}: {e}"
            print(f"  FAIL  {name}: {failures[name]}")
            traceback.print_exc(limit=2)
            continue
        timings.append({"model": name, "seconds": round(time.time() - t0, 2)})
        print(f"  done  {name} ({timings[-1]['seconds']}s)")
    if not frames:
        raise RuntimeError("every model failed")
    keys = ["unique_id", "ds", "cutoff", "y"]
    fc = None
    for f in frames.values():
        fc = f if fc is None else fc.merge(f, on=keys, how="outer")
    train = pd.concat([history(ds.Y, series_ids(ds)) for ds in loaded], ignore_index=True)
    series = pd.concat([pd.DataFrame({"unique_id": series_ids(ds), "dataset": ds.name, "period": ds.period or 1})
                        for ds in loaded], ignore_index=True)
    config = dict(models=names, datasets={d.name: {"period": d.period, "shape": list(d.Y.shape),
                                                   "sha256_16": fingerprint(d.Y)} for d in loaded},
                  h=h, windows=windows, step=step, context=context, n_series=n_series, seed=seed,
                  reference=reference, failures=failures, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                  python=platform.python_version())
    ev = evaluate_frame(fc, train, series=series, reference=reference, config=config)
    ev.leaderboard = ev.leaderboard.merge(pd.DataFrame(timings), on="model", how="left")
    if out:
        ev.write(out, report=report)
    return ev.leaderboard, ev.summary, ev.scores
