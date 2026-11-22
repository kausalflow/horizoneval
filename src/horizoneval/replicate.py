"""Reproduce published benchmark results with horizoneval's own scoring, as evidence that its numbers are right.

    uv run --extra replicate horizoneval replicate m4          # downloads M4 into data/m4 (about 300 MB)
    uv run --extra gift horizoneval replicate gift-eval        # needs the GIFT_CONFIGS below in data/gift-eval

M4: every series of every frequency is scored through `horizoneval.evaluate`, for (a) the forecasts that M4
participants submitted, read from their own files, and (b) horizoneval's baselines run on the same histories. The
official sMAPE and MASE per frequency and in total are then compared with ours in replication.csv.
Conventions: M4's sMAPE is 200 |e| / (|y| + |f|), i.e. 100 x horizoneval's smape; MASE uses the in-sample seasonal
naive at the period in m4.FREQS; totals are means over all 100,000 series.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from horizoneval.data import m4
from horizoneval.evaluate import evaluate
from horizoneval.prediction import as_prediction
from horizoneval.registry import load


def _run_forecaster(name, train, h, period):
    """Point forecasts (N, h) of a registered forecaster on ragged histories, one call per history length."""
    f = load(name)
    out = np.empty((len(train), h))
    lengths = np.array([len(t) for t in train])
    for n in np.unique(lengths):
        idx = np.flatnonzero(lengths == n)
        out[idx] = as_prediction(f(np.stack([train[i] for i in idx]), h, period)).point
    return out


def m4_frames(freqs=tuple(m4.FREQS), models=("naive", "seasonal_naive"), submissions=tuple(m4.SUBMISSIONS)):
    """(forecasts, train, series) frames for the M4 frequencies, integer time per series."""
    fcs, trains, series = [], [], []
    subs = {s: m4.submission(s) for s in submissions}
    for freq in freqs:
        h, period = m4.FREQS[freq]
        ids, train, test = m4.load(freq)
        n = np.array([len(t) for t in train])
        cols = {"unique_id": np.repeat(ids, h), "ds": (n[:, None] + np.arange(h)).ravel(),
                "cutoff": np.repeat(n - 1, h), "y": test.ravel()}
        for s, df in subs.items():
            F = df.reindex(ids)[[f"F{k}" for k in range(1, h + 1)]].to_numpy(float)
            cols[m4.SUBMISSIONS[s]] = F.ravel()
        for name in models:
            cols[name] = _run_forecaster(name, train, h, period).ravel()
        fcs.append(pd.DataFrame(cols))
        trains.append(pd.DataFrame({"unique_id": np.repeat(ids, n), "ds": np.concatenate([np.arange(k) for k in n]),
                                    "y": np.concatenate(train)}))
        series.append(pd.DataFrame({"unique_id": ids, "dataset": freq, "period": period}))
    return pd.concat(fcs, ignore_index=True), pd.concat(trains, ignore_index=True), pd.concat(series, ignore_index=True)


def compare_m4(scores: pd.DataFrame) -> pd.DataFrame:
    """Ours vs. published sMAPE and MASE per method and frequency (and in total), with OWA against Naive2."""
    pub = m4.published().set_index("method")
    names = {v: k for k, v in m4.SUBMISSIONS.items()} | {"naive": "Naive", "seasonal_naive": "sNaive"}
    per = scores.groupby(["model", "dataset"])[["smape", "mase"]].mean()
    tot = scores.groupby("model")[["smape", "mase"]].mean().assign(dataset="Total").set_index("dataset", append=True)
    ours = pd.concat([per, tot]).reset_index()
    ours["smape"] *= 100
    rows = []
    for _, r in ours.iterrows():
        method = names.get(r["model"])
        if method not in pub.index:
            continue
        for metric in ["smape", "mase"]:
            p = float(pub.loc[method, f"{metric}_{r['dataset']}"])
            digits = 5 if r["dataset"] == "Total" and metric == "smape" else 6 if r["dataset"] == "Total" else 3
            rows.append({"method": method, "model": r["model"], "frequency": r["dataset"], "metric": metric,
                         "ours": r[metric], "published": p, "diff": r[metric] - p,
                         "published_digits": digits, "match": abs(r[metric] - p) <= 0.5 * 10 ** -digits + 1e-9})
    return pd.DataFrame(rows)


def m4_run(out="results/replication-m4", n_boot=200):
    fc, train, series = m4_frames()
    ev = evaluate(fc, train, series=series, reference="m4_Naive2", n_boot=n_boot,
                  config={"replication": "M4 point forecasts, all 100,000 series", "source_url": m4.BASE})
    out = Path(out)
    ev.write(out, frames=False, scores=False)
    cmp = compare_m4(ev.scores)
    cmp.to_csv(out / "replication.csv", index=False)
    (out / "replication.json").write_text(json.dumps(
        {"matched": int(cmp["match"].sum()), "compared": len(cmp),
         "max_abs_diff": {m: float(g["diff"].abs().max()) for m, g in cmp.groupby("metric")}}, indent=2) + "\n")
    return ev, cmp


# ---- GIFT-Eval ------------------------------------------------------------------------------------------------------
# The published numbers are the leaderboard's all_results.csv of Seasonal_Naive and Naive
# (huggingface.co/spaces/Salesforce/GIFT-Eval, results/<model>/all_results.csv). GIFT-Eval fills missing history
# for the forecaster (gluonts LastValueImputation) and leaves missing targets out of every metric; it scores sMAPE with
# gluonts' convention (0/0 steps left out), so the replication runs with smape_zero="skip".
GIFT_CONFIGS = ["m4_yearly", "m4_quarterly", "m4_monthly", "m4_weekly", "m4_daily", "m4_hourly",
                "ett1/H", "ett2/H", "ett1/D", "ett2/D", "ett1/W", "ett2/W", "solar/H", "solar/D",
                "us_births/D", "saugeenday/D", "hospital", "covid_deaths", "LOOP_SEATTLE/H", "LOOP_SEATTLE/D",
                "M_DENSE/H", "M_DENSE/D", "SZ_TAXI/H", "jena_weather/H", "jena_weather/D",
                # with missing values
                "electricity/H", "electricity/D", "electricity/W", "kdd_cup_2018_with_missing/H",
                "kdd_cup_2018_with_missing/D", "car_parts_with_missing", "restaurant", "hierarchical_sales/D",
                "bitbrains_rnd/H"]
# configurations GIFT-Eval also scores at the medium and long terms (its notebooks' med_long_datasets)
GIFT_MED_LONG = {"electricity/15T", "electricity/H", "solar/10T", "solar/H", "kdd_cup_2018_with_missing/H",
                 "LOOP_SEATTLE/5T", "LOOP_SEATTLE/H", "SZ_TAXI/15T", "M_DENSE/H", "ett1/15T", "ett1/H", "ett2/15T",
                 "ett2/H", "jena_weather/10T", "jena_weather/H", "bitbrains_fast_storage/5T", "bitbrains_rnd/5T",
                 "bizitobs_application", "bizitobs_service", "bizitobs_l2c/5T", "bizitobs_l2c/H"}
GIFT_NAMES = {"saugeenday": "saugeen", "temperature_rain_with_missing": "temperature_rain",
              "kdd_cup_2018_with_missing": "kdd_cup_2018", "car_parts_with_missing": "car_parts"}
GIFT_METRICS = {"mase": "eval_metrics/MASE[0.5]", "smape": "eval_metrics/sMAPE[0.5]", "mae": "eval_metrics/MAE[0.5]",
                "mse": "eval_metrics/MSE[0.5]"}


def last_value_imputation(y: np.ndarray) -> np.ndarray:
    """gluonts LastValueImputation: carry the last observed value forward; leading gaps take the first observed."""
    y = np.asarray(y, dtype=float)
    miss = np.isnan(y)
    if not miss.any():
        return y
    if miss.all():
        return np.zeros_like(y)
    idx = np.maximum.accumulate(np.where(~miss, np.arange(len(y)), 0))
    out = y[idx]
    lead = np.isnan(out)
    out[lead] = y[np.flatnonzero(~miss)[0]]
    return out


def gift_key(name, term, freq):
    """The dataset key of GIFT-Eval's results table, e.g. 'm4_yearly/A/short', 'ett1/H/medium', 'hospital/M/short'."""
    base = name.split("/")[0]
    f = freq.split("-")[0]
    return f"{GIFT_NAMES.get(base, base).lower()}/{f}/{term}"


def gift_frames(configs=GIFT_CONFIGS, models=("naive", "seasonal_naive"), root="data/gift-eval"):
    """(forecasts, train, series) for every term of each configuration, following GIFT-Eval's test protocol."""
    from horizoneval.data import gift

    fcs, trains, series = [], [], []
    for name in configs:
        ids, ser, freq = gift.test_series(name, root)
        lengths = np.array([len(s) for s in ser])
        period = gift.gluonts_seasonality(freq)
        terms = list(gift.TERMS) if name in GIFT_MED_LONG else ["short"]
        for term in terms:
            h, W = gift.protocol(name, term, freq, lengths.min())
            key = gift_key(name, term, freq)
            uids = np.char.add(f"{key}#", ids)
            rows = []
            for k in range(W):
                start = lengths - h * (W - k)  # first target index of window k, per series
                hist = [last_value_imputation(s[:c]) for s, c in zip(ser, start)]
                cols = {"unique_id": np.repeat(uids, h), "ds": (start[:, None] + np.arange(h)).ravel(),
                        "cutoff": np.repeat(start - 1, h),
                        "y": np.concatenate([s[c:c + h] for s, c in zip(ser, start)])}
                for m in models:
                    cols[m] = _run_forecaster(m, hist, h, period).ravel()
                rows.append(pd.DataFrame(cols))
            fcs += rows
            trains.append(pd.DataFrame({"unique_id": np.repeat(uids, lengths - h),
                                        "ds": np.concatenate([np.arange(n - h) for n in lengths]),
                                        "y": np.concatenate([s[:-h] for s in ser])}))
            series.append(pd.DataFrame({"unique_id": uids, "dataset": key, "period": period}))
    return pd.concat(fcs, ignore_index=True), pd.concat(trains, ignore_index=True), pd.concat(series, ignore_index=True)


def compare_gift(scores: pd.DataFrame) -> pd.DataFrame:
    from horizoneval.data import gift  # noqa: F401  (documents where the protocol lives)

    here = Path(__file__).parent / "data"
    pub = {"seasonal_naive": pd.read_csv(here / "gift_published_seasonal_naive.csv").set_index("dataset"),
           "naive": pd.read_csv(here / "gift_published_naive.csv").set_index("dataset")}
    from horizoneval.evaluate import _weighted_means

    ours = _weighted_means(scores, ["model", "dataset"], list(GIFT_METRICS)).reset_index()
    rows = []
    for _, r in ours.iterrows():
        p = pub.get(r["model"])
        if p is None or r["dataset"] not in p.index:
            continue
        for metric, col in GIFT_METRICS.items():
            theirs = float(p.loc[r["dataset"], col])
            rel = abs(r[metric] - theirs) / max(abs(theirs), 1e-12)
            rows.append({"model": r["model"], "dataset": r["dataset"], "metric": metric, "ours": r[metric],
                         "published": theirs, "rel_diff": rel, "match": rel < 1e-6})
    return pd.DataFrame(rows)


def gift_run(out="results/replication-gift-eval", root="data/gift-eval", n_boot=200):
    fc, train, series = gift_frames(root=root)
    ev = evaluate(fc, train, series=series, reference="naive", n_boot=n_boot, smape_zero="skip",
                  config={"replication": "GIFT-Eval seasonal naive and naive",
                          "source_url": "huggingface.co/datasets/Salesforce/GiftEval"})
    out = Path(out)
    ev.write(out, frames=False, scores=False)
    cmp = compare_gift(ev.scores)
    cmp.to_csv(out / "replication.csv", index=False)
    (out / "replication.json").write_text(json.dumps(
        {"matched": int(cmp["match"].sum()), "compared": len(cmp),
         "max_rel_diff": {m: float(g["rel_diff"].max()) for m, g in cmp.groupby("metric")}}, indent=2) + "\n")
    return ev, cmp
