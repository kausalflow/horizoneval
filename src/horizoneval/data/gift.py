"""GIFT-Eval loader (install with the `gift` extra), from extrapolation-as-forecasts/experiments/gift_data.py.

Download one configuration per --include (this hf CLI drops all but the last value of a repeated pattern list):
    uv run hf download Salesforce/GiftEval --repo-type=dataset --local-dir data/gift-eval --include "electricity/H/*"
Each folder is a `datasets` save with columns item_id, start, freq, target.
"""

import re
from pathlib import Path

import numpy as np


def period_of(freq: str) -> int | None:
    """The calendar period in samples: points per day for sub-daily data, 7 for daily, 12 for monthly, 4 for quarterly."""
    m = re.match(r"(\d*)\s*([A-Za-z]+)", freq.strip())  # 'W-SUN', 'A-DEC': the anchor after the dash is ignored
    k, unit = int(m.group(1) or 1), m.group(2).upper()
    if unit in ("T", "MIN"):
        return 1440 // k if 1440 % k == 0 else None
    if unit == "H":
        return 24 // k if 24 % k == 0 else None
    return {"D": 7, "M": 12, "MS": 12, "ME": 12, "Q": 4, "QS": 4, "QE": 4}.get(unit)


def load_gift(name: str, n_series: int, n_total: int, rng, root="data/gift-eval"):
    """(Y, period): the last n_total points of n_series series of dataset `name` ('electricity/H', 'm4_hourly')."""
    from datasets import load_from_disk

    path = Path(root) / name
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: download it as shown in horizoneval/data/gift.py")
    ds = load_from_disk(str(path))
    freq = ds[0]["freq"]
    series = []
    for t in ds["target"]:
        arr = np.asarray(t, dtype=float)
        for v in arr if arr.ndim == 2 else [arr]:  # multivariate targets become several series
            if len(v) >= n_total and np.isfinite(v[-n_total:]).all() and v[-n_total:].std() > 0:
                series.append(v[-n_total:])
    if not series:
        raise ValueError(f"{name}: no series of length >= {n_total} without gaps")
    idx = rng.choice(len(series), size=min(n_series, len(series)), replace=False)
    return np.stack([series[i] for i in sorted(idx)]), period_of(freq)
