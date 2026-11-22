"""Every setting that can change a score, fingerprinted, and a diff that says why two runs disagree.

Two evaluations of "the same" forecasts often give different numbers because something underneath changed: the
series, the cutoffs, the horizon, the seasonal period behind MASE, a convention such as how sMAPE treats 0/0, the
truth itself, or one model's forecasts. `settings` records all of them in config.json; `diff` compares two runs and
names, in plain language, each setting that differs and the scores it moved.

    horizoneval diff results/run-a results/run-b
"""

import hashlib
import json
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

from horizoneval.frame import KEYS, quantile_columns

CONVENTIONS = {
    "mase_scale": "in-sample MAE of the seasonal naive on the history up to each cutoff; pairs with a missing value "
                  "left out; a flat history gives no scaled score",
    "crps": "twice the mean pinball loss over the model's quantile levels",
    "missing_truth": "steps with missing truth are not scored; means weight each observed step once",
    "quantile_crossing": "quantiles are sorted before scoring",
}

# what each setting means, for the diff
EXPLAIN = {
    "smape_zero": "sMAPE counts steps where truth and forecast are both 0 differently",
    "period": "the seasonal period behind MASE and scaled CRPS differs",
    "h": "the forecast horizon differs",
    "windows": "the number of rolling windows differs",
    "cutoffs": "the cutoffs (the dates forecasts were made from) differ",
    "series": "the set of series differs",
    "truth": "the truth (y) differs: the data changed",
    "history": "the history used to scale MASE differs",
    "levels": "the quantile levels of a model differ, which changes CRPS and coverage",
    "forecasts": "the forecasts of a model differ",
    "reference": "the reference model differs, which changes skill and the paired tests",
    "decide_by": "the choice is made by a different metric",
    "cost": "the user's cost differs",
    "conventions": "a scoring convention changed",
    "versions": "software versions differ",
}


def _sha(obj) -> str:
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        b = pd.util.hash_pandas_object(obj, index=False).to_numpy().tobytes()
    else:
        b = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(b).hexdigest()[:16]


def _version(pkg):
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def settings(forecasts: pd.DataFrame, train: pd.DataFrame | None, meta: pd.DataFrame, models: list[str],
             reference=None, decide_by=None, cost=None, smape_zero="zero") -> dict:
    d = forecasts.sort_values(["unique_id", "cutoff", "ds"], kind="stable", ignore_index=True)
    ds_of = meta.set_index("unique_id")["dataset"]
    hist = {}
    if train is not None:
        t = train[["unique_id", "ds", "y"]]
        t = t.assign(_d=t["unique_id"].map(ds_of)).sort_values(["unique_id", "ds"], kind="stable")
        hist = {k: _sha(g[["unique_id", "ds", "y"]]) for k, g in t.groupby("_d", sort=False)}
    datasets = {}
    for name, g in d.groupby(d["unique_id"].map(ds_of), sort=True):
        win = g.groupby(["unique_id", "cutoff"]).size()
        datasets[str(name)] = {
            "series": int(g["unique_id"].nunique()), "series_sha": _sha(pd.Series(sorted(g["unique_id"].unique()))),
            "h": int(win.iloc[0]), "windows": int(win.groupby(level=0).size().max()),
            "cutoffs": _sha(win.reset_index()[["unique_id", "cutoff"]]),
            "period": int(meta.loc[meta["dataset"] == name, "period"].iloc[0]),
            "truth": _sha(g["y"].astype(float)),
            "history": hist.get(name),
        }
    per_model = {}
    for m in models:
        q = quantile_columns(d, m)
        per_model[m] = {"levels": [round(x, 6) for x in q], "forecasts": _sha(d[[m, *q.values()]].astype(float))}
    return {"datasets": datasets, "models": per_model, "reference": reference, "decide_by": decide_by,
            "cost": cost if cost is None or isinstance(cost, str) else getattr(cost, "__name__", "custom"),
            "smape_zero": smape_zero, "conventions": CONVENTIONS,
            "versions": {p: _version(p) for p in ["horizoneval", "numpy", "pandas", "scipy"]}}


def _load(run) -> tuple[dict, pd.DataFrame | None]:
    if isinstance(run, dict):
        return run, None
    if hasattr(run, "config"):
        return run.config, run.summary
    p = Path(run)
    cfg = json.loads((p / "config.json").read_text())
    summ = pd.read_csv(p / "summary.csv") if (p / "summary.csv").exists() else None
    return cfg, summ


def diff(a, b, metrics=("mase", "smape", "crps_scaled")) -> dict:
    """Why runs a and b (folders, Evaluations or configs) disagree: each differing setting, in plain language, and the
    scores that moved for (model, dataset) pairs both runs share."""
    (ca, sa), (cb, sb) = _load(a), _load(b)
    A, B = ca.get("settings"), cb.get("settings")
    if A is None or B is None:
        which = " and ".join(n for n, x in [("the first run", A), ("the second run", B)] if x is None)
        return {"same_settings": None, "causes": [], "moved": [], "text": [
            f"Cannot compare settings: {which} recorded none (made before horizoneval recorded them). Re-score it "
            "from its forecasts.parquet with `horizoneval score` to compare."]}
    causes = []

    def add(key, where, va, vb):
        causes.append({"setting": key, "where": where, "a": va, "b": vb, "why": EXPLAIN.get(key, key)})

    for key in ["smape_zero", "reference", "decide_by", "cost", "conventions"]:
        if A.get(key) != B.get(key):
            add(key, "run", A.get(key), B.get(key))
    da, db = A.get("datasets", {}), B.get("datasets", {})
    for name in sorted(set(da) | set(db)):
        if name not in da or name not in db:
            add("series", name, name in da, name in db)
            continue
        for key in ["series_sha", "h", "windows", "cutoffs", "period", "truth", "history"]:
            if da[name].get(key) != db[name].get(key):
                k = "series" if key == "series_sha" else key
                add(k, name, da[name].get(key) if key != "series_sha" else da[name]["series"],
                    db[name].get(key) if key != "series_sha" else db[name]["series"])
    ma, mb = A.get("models", {}), B.get("models", {})
    for m in sorted(set(ma) & set(mb)):
        for key in ["levels", "forecasts"]:
            if ma[m].get(key) != mb[m].get(key):
                add(key, m, ma[m].get(key), mb[m].get(key))
    if A.get("versions") != B.get("versions"):
        add("versions", "run", A.get("versions"), B.get("versions"))
    moved = []
    if sa is not None and sb is not None:
        j = sa.merge(sb, on=["model", "dataset"], suffixes=("_a", "_b"))
        for m in metrics:
            if f"{m}_a" not in j:
                continue
            d = j[["model", "dataset", f"{m}_a", f"{m}_b"]].dropna()
            d = d[~np.isclose(d[f"{m}_a"], d[f"{m}_b"], rtol=1e-9, atol=1e-12)]
            for _, r in d.iterrows():
                moved.append({"metric": m, "model": r["model"], "dataset": r["dataset"], "a": float(r[f"{m}_a"]),
                              "b": float(r[f"{m}_b"])})
    same = not causes
    text = ["The two runs were evaluated the same way." if same else
            "The runs differ in " + "; ".join(sorted({c["why"] for c in causes})) + "."]
    if moved and same:
        text.append("Scores moved although every recorded setting is the same: report this as a bug.")
    return {"same_settings": same, "causes": causes, "moved": moved, "text": text}
