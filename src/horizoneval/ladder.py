"""Where a model's skill comes from: the skill ladder of the naive-forecaster paper.

A model's gain over repeating the last value is split into rungs that horizoneval builds itself from the history:

    total = log MSE(naive) / MSE(model)                         skill over the identity, in nats
          = F  log MSE(naive)   / MSE(classic)                  free: the best simple rule (drift, seasonal naive, ...)
          + L  log MSE(classic) / MSE(linear)                   linear: a least-squares AR we fit on the history
          + R  log MSE(linear)  / MSE(model)                    beyond linear: what the model adds on top

(Ma, "Naive forecasters: an operator theory of what a forecast trusts", Result 4(b); code and checks in
emptymalei/extrapolation-as-forecasts, paper_naive/). A gain of x nats is a squared error lower by 1 - exp(-x).

R counts as evidence that the model does something a linear forecaster cannot only if, in every horizon group, the
linear reference is saturated (doubling the memory of the dense AR filters moves their validation skill by less than
0.1 nats), R exceeds twice its standard error over series, and R exceeds the offset a fitted linear reference leaves
even against the exact optimum (0.01 / 0.03 / 0.05 nats at steps 1 / 2-4 / 5+, measured on the paper's calibration
case and checked in tests/test_ladder.py). R <= 0 is conclusive the other way: the model does no better than a
linear one. The rule can prove "no more than linear" and can bound, but not certify, "more than linear".

At one step ahead the paper gives model-free values to compare with, read from the history alone: F at h = 1 is
log Var(diff y) / Var(u) for the invariant u of the chosen classic, and L at h = 1 is the price of naivety, the log
spectral flatness log A/G of u, estimated by a long AR (biased up by its in-sample optimism, about p/(n-p) nats) and
by a Welch periodogram (biased down by its resolution). They are values to compare with, not ceilings.

The rungs are forecast columns (ladder_naive, ladder_classic, ladder_linear) added to the forecast frame and scored
by `horizoneval.evaluate` like any model, so there is no second scoring path.

Protocol, per dataset (all of it before each series' first cutoff, so nothing is fitted on the scored windows):
    context    n = the last min(512, available) points before each cutoff
    classic    per series and step, the simplest of naive, drift, last_change, mean, seasonal_naive, seasonal_drift
               within one standard error of the best (one-SE rule) on up to 24 windows inside the history
    linear     per step, the candidate with the smallest validation MSE pooled over series, on the last 8 of those
               windows, among the chosen classic and AR filters on the differences (dense lags 4..96), on the
               seasonal differences, and on dilated seasonal lag sets
Missing history values are filled with the last observed value.
"""

import numpy as np
import pandas as pd

from horizoneval.frame import FrameError, KEYS, windows

CLASSICS = ["naive", "drift", "last_change", "mean", "seasonal_naive", "seasonal_drift"]
RUNGS = ["ladder_naive", "ladder_classic", "ladder_linear"]
OFFSET = {"1": 0.01, "2-4": 0.03, "5+": 0.05}  # nats; null expectation of R for an exactly optimal model
SATURATION = 0.1  # nats
CONTEXT = 512
N_SELECT, N_VALIDATE = 24, 24


def groups(h: int) -> dict[str, np.ndarray]:
    """Horizon groups (0-based step indices) the decision rule is read on: 1, 2-4, 5+ (as far as h reaches)."""
    out = {"1": np.array([0])}
    if h >= 2:
        out["2-4"] = np.arange(1, min(h, 4))
    if h >= 5:
        out["5+"] = np.arange(4, h)
    return out


def nats_to_percent(x: float) -> float:
    """A gain of x nats as the percentage by which squared error falls (negative: it rises)."""
    return 100 * (1 - np.exp(-x))


# --------------------------------------------------------------------------------------------------------------------
# The rungs as forecasters: (N, n) histories -> (N, h) forecasts
# --------------------------------------------------------------------------------------------------------------------


def classics(Y: np.ndarray, h: int, period: int) -> dict[str, np.ndarray]:
    """Point forecasts (N, h) of the grade-0 classics (no fitted constant beyond a mean or a slope)."""
    N, n = Y.shape
    k = np.arange(1, h + 1)
    last = Y[:, -1:]
    out = {"naive": np.repeat(last, h, axis=1),
           "drift": last + ((Y[:, -1] - Y[:, 0]) / (n - 1))[:, None] * k,
           "last_change": last + (Y[:, -1] - Y[:, -2])[:, None] * k,
           "mean": np.repeat(Y.mean(axis=1, keepdims=True), h, axis=1)}
    m = period
    if m > 1 and n >= 2 * m:
        idx = n - m + (k - 1) % m
        out["seasonal_naive"] = Y[:, idx]
        c = (Y[:, m:] - Y[:, :-m]).mean(axis=1)
        out["seasonal_drift"] = Y[:, idx] + c[:, None] * ((k - 1) // m + 1)
    return out


def ar_forecast(Y: np.ndarray, h: int, m: int, lags: np.ndarray, ridge: float = 1e-6, chunk: int = 128) -> np.ndarray:
    """Least-squares AR with intercept on u = y_t - y_{t-m} at the given lags, forecast by recursion and rebuilt as
    y_{n+k} = y_{n+k-m} + u_{n+k}; fitted per series, vectorised over series. (N, n) -> (N, h)."""
    out = np.empty((Y.shape[0], h))
    L = int(lags.max())
    for a in range(0, Y.shape[0], chunk):
        Yc = Y[a:a + chunk]
        U = Yc[:, m:] - Yc[:, :-m]
        T = U.shape[1] - L
        X = np.stack([np.ones((len(Yc), T))] + [U[:, L - lag:U.shape[1] - lag] for lag in lags], axis=2)
        A = np.einsum("mti,mtj->mij", X, X) + ridge * T * np.eye(X.shape[2])
        coef = np.linalg.solve(A, np.einsum("mti,mt->mi", X, U[:, L:])[..., None])[..., 0]
        u, y = U[:, -L:], Yc[:, -m:]
        for _ in range(h):
            x = np.concatenate([np.ones((len(Yc), 1)), u[:, -lags]], axis=1)
            un = (coef * x).sum(axis=1, keepdims=True)
            u = np.concatenate([u[:, 1:], un], axis=1)
            y = np.concatenate([y, y[:, -m:-m + 1] if m > 1 else y[:, -1:]], axis=1)
            y[:, -1] += un[:, 0]
        out[a:a + chunk] = y[:, -h:]
    return out


def lag_sets(period: int, n: int) -> dict[str, tuple[int, np.ndarray]]:
    """Candidate linear references (m, lags) that fit a context of n points."""
    sets = {f"ar{p}_diff": (1, np.arange(1, p + 1)) for p in (4, 8, 16, 32, 64, 96)}
    if period > 1:
        m = period
        sets.update({f"ar{p}_sdiff{m}": (m, np.arange(1, p + 1)) for p in (4, 8, 16)})
        around = lambda c, w=2: np.arange(max(1, c - w), c + w + 1)  # noqa: E731
        dil = np.unique(np.concatenate([np.arange(1, 9), around(m), around(2 * m)]))
        sets[f"dilated_{m}"] = (1, dil)
        sets[f"dilated_{m}_week"] = (1, np.unique(np.concatenate([dil, around(7 * m)])))
    return {k: (m, lags) for k, (m, lags) in sets.items() if lags.max() + m + 50 <= n}


def one_se(err: np.ndarray) -> np.ndarray:
    """(C, N, W, h) squared errors of candidates listed simplest first -> (N, h) index of the simplest candidate
    within one standard error (over windows) of the best."""
    mean, se = err.mean(axis=2), err.std(axis=2) / np.sqrt(err.shape[2])
    best = np.argmin(mean, axis=0)
    thresh = np.take_along_axis(mean, best[None], 0)[0] + np.take_along_axis(se, best[None], 0)[0]
    return np.argmax(mean <= thresh[None], axis=0)


def _pick(fc: dict[str, np.ndarray], names: list[str], choice: np.ndarray) -> np.ndarray:
    """(N, h) forecasts assembled from per-series, per-step choices among names; fc values are (N, h)."""
    stacked = np.stack([fc[c] for c in names])
    return np.take_along_axis(stacked, choice[None], 0)[0]


# --------------------------------------------------------------------------------------------------------------------
# Model-free values at h = 1
# --------------------------------------------------------------------------------------------------------------------


def invariant(Y: np.ndarray, classic: str, period: int) -> np.ndarray:
    """The series whose one-step error the classic makes: S(B) y for an anchored classic, centred for a projected
    one ((1-B)^2 y for last_change, centred y for mean)."""
    m = {"mean": 0, "seasonal_naive": period, "seasonal_drift": period}.get(classic, 1)
    u = Y[:, m:] - Y[:, :-m] if m else Y.copy()
    if classic == "last_change":
        u = np.diff(u, axis=1)
    if classic in ("drift", "mean", "seasonal_drift"):
        u = u - u.mean(axis=1, keepdims=True)
    return u


def log_flatness_ar(u: np.ndarray, p: int = 64, ridge: float = 1e-8) -> tuple[np.ndarray, float]:
    """log Var(u) / in-sample one-step MSE of a least-squares AR(p): an estimate of log A/G, and its expected
    in-sample optimism p / (n - p) nats."""
    u = u - u.mean(axis=1, keepdims=True)
    n = u.shape[1]
    p = max(1, min(p, n // 4))
    X = np.stack([u[:, p - lag:n - lag] for lag in range(1, p + 1)], axis=2)
    A = np.einsum("mti,mtj->mij", X, X) + ridge * (n - p) * np.eye(p)
    coef = np.linalg.solve(A, np.einsum("mti,mt->mi", X, u[:, p:])[..., None])[..., 0]
    mse = ((u[:, p:] - np.einsum("mti,mi->mt", X, coef)) ** 2).mean(axis=1)
    return np.log((u ** 2).mean(axis=1) / np.maximum(mse, 1e-300)), p / (n - p)


def log_flatness_welch(u: np.ndarray, seg: int = 128) -> np.ndarray:
    """log A/G of the spectrum of u from a Welch average of K periodograms, with the bias of the log of an average
    of K exponentials removed. NaN when fewer than two segments fit."""
    from scipy.special import digamma

    u = u - u.mean(axis=1, keepdims=True)
    K = u.shape[1] // seg
    if K < 2:
        return np.full(u.shape[0], np.nan)
    s = u[:, :K * seg].reshape(len(u), K, seg)
    spec = (np.abs(np.fft.rfft(s, axis=2)) ** 2 / seg).mean(axis=1)[:, 1:]
    spec = np.maximum(spec, 1e-300)
    return np.log(spec.mean(axis=1)) - np.log(spec).mean(axis=1) - (np.log(K) - digamma(K))


# --------------------------------------------------------------------------------------------------------------------
# Building the rungs on a forecast frame
# --------------------------------------------------------------------------------------------------------------------


def _series_arrays(train: pd.DataFrame) -> dict:
    t = train.sort_values(["unique_id", "ds"], kind="stable")
    out = {}
    for uid, g in t.groupby("unique_id", sort=False):
        y = g["y"].astype(float).ffill().bfill().to_numpy()
        out[uid] = (y, g["ds"].to_numpy())
    return out


def _cut(arrays, uids, ends, n, h=0):
    """(len, n) histories ending before position `ends` and, with h, the (len, h) truth that follows."""
    H = np.stack([arrays[u][0][e - n:e] for u, e in zip(uids, ends)])
    T = np.stack([arrays[u][0][e:e + h] for u, e in zip(uids, ends)]) if h else None
    return H, T


def rungs(forecasts: pd.DataFrame, train: pd.DataFrame, meta: pd.DataFrame, context: int = CONTEXT):
    """The forecast frame with the three rung columns added, and a per-dataset record of what was chosen.

    Returns (frame, info): info[dataset] holds the protocol, the classic chosen per step, the linear reference per
    step, the saturation of the dense references per horizon group and the model-free values at h = 1, or
    {"skipped": reason} when the history is too short to build the rungs.
    """
    clash = [c for c in RUNGS if c in forecasts.columns]
    if clash:
        raise FrameError(f"the forecast frame already has columns named {clash}")
    arrays = _series_arrays(train)
    period_of = meta.set_index("unique_id")["period"]
    parts, info = [], {}
    for ds_name, ids in meta.groupby("dataset")["unique_id"]:
        part = forecasts[forecasts["unique_id"].isin(set(ids))]
        w = windows(part)
        h, period = w.h, int(period_of.loc[w.unique_id[0]])
        ends = np.array([np.searchsorted(arrays[u][1], c, side="right") for u, c in zip(w.unique_id, w.cutoff)])
        first = pd.Series(ends).groupby(w.unique_id).min()
        n_min = max(64, 2 * period + 16)
        S = int(min(N_SELECT, max(0, (int(first.min()) - n_min) // h)))
        n = int(min(context, int(first.min()) - S * h)) if S else 0
        if S < 4 or n < n_min or ends.min() < n:
            info[ds_name] = {"skipped": f"too little history before the first cutoff for the ladder (needs at least "
                                        f"{n_min} points plus 4 windows of {h}; the shortest series has "
                                        f"{int(first.min())})"}
            continue
        uids = first.index.to_numpy()
        N = len(uids)
        # selection windows inside the history: S windows of h steps ending right before each series' first cutoff
        sel_ends = [first.to_numpy() - (S - j) * h for j in range(S)]
        sel = [_cut(arrays, uids, e, n, h) for e in sel_ends]
        cl_sel = [classics(H, h, period) for H, _ in sel]
        names = [c for c in CLASSICS if c in cl_sel[0]]
        err = np.stack([np.stack([(f[c] - T) ** 2 for f, (_, T) in zip(cl_sel, sel)], axis=1) for c in names])
        choice = one_se(err)  # (N, h)
        # linear references, chosen per step on the last validation windows, pooled over series
        V = min(N_VALIDATE, S)
        cands = lag_sets(period, n)
        val_mse, val_naive = {}, None
        for (H, T), f in zip(sel[-V:], cl_sel[-V:]):
            fc = {"classic": _pick(f, names, choice)}
            fc.update({c: ar_forecast(H, h, m, lags) for c, (m, lags) in cands.items()})
            for c, v in fc.items():
                val_mse[c] = val_mse.get(c, 0) + ((v - T) ** 2).mean(axis=0) / V
            val_naive = (0 if val_naive is None else val_naive) + ((f["naive"] - T) ** 2).mean(axis=0) / V
        cnames = list(val_mse)
        best = np.argmin(np.stack([val_mse[c] for c in cnames]), axis=0)  # (h,)
        skill = {c: np.log(np.maximum(val_naive, 1e-300) / np.maximum(v, 1e-300)) for c, v in val_mse.items()}
        dense = sorted(int(c[2:-5]) for c in cands if c.endswith("_diff"))
        saturated = {}
        for g, steps in groups(h).items():
            if len(dense) >= 2:
                hi = dense[-1]
                lo = [p for p in dense if p <= hi // 2]
                gain = float(np.mean(skill[f"ar{hi}_diff"][steps] - np.max([skill[f"ar{p}_diff"][steps] for p in lo],
                                                                           axis=0)))
                saturated[g] = {"saturated": gain < SATURATION, "memory": [max(lo), hi], "gain": gain}
            else:
                saturated[g] = {"saturated": False, "memory": dense, "gain": None}
        # the rungs on the scored windows
        pos = {u: i for i, u in enumerate(uids)}
        rows = np.array([pos[u] for u in w.unique_id])
        Ht, _ = _cut(arrays, w.unique_id, ends, n)
        cl = classics(Ht, h, period)
        classic_fc = _pick(cl, names, choice[rows])
        lin = {"classic": classic_fc}
        for c in set(cnames[b] for b in best) - {"classic"}:
            lin[c] = ar_forecast(Ht, h, *cands[c])
        linear_fc = np.stack([lin[cnames[best[k]]][:, k] for k in range(h)], axis=1)
        d = w.df[KEYS].copy()
        d["ladder_naive"], d["ladder_classic"], d["ladder_linear"] = (cl["naive"].ravel(), classic_fc.ravel(),
                                                                      linear_fc.ravel())
        parts.append(d)
        # model-free values at h = 1 on the context before each series' first cutoff
        H0, _ = _cut(arrays, uids, first.to_numpy(), n)
        F1, L1, W1 = np.full(N, np.nan), np.full(N, np.nan), np.full(N, np.nan)
        optimism = np.nan
        d1 = np.diff(H0, axis=1)
        for c in names:
            idx = np.flatnonzero(choice[:, 0] == names.index(c))
            if not len(idx):
                continue
            u = invariant(H0[idx], c, period)
            with np.errstate(divide="ignore", invalid="ignore"):
                F1[idx] = np.log((d1[idx] ** 2).mean(axis=1) / (u ** 2).mean(axis=1))
            L1[idx], optimism = log_flatness_ar(u)
            W1[idx] = log_flatness_welch(u)
        share = [{nm: int((choice[:, k] == j).sum()) for j, nm in enumerate(names) if (choice[:, k] == j).any()}
                 for k in range(h)]
        info[ds_name] = {
            "context": n, "selection_windows": S, "validation_windows": V, "period": period, "series": N,
            "classic_share": share, "linear_reference": [cnames[b] for b in best],
            "linear_candidates": sorted(cnames), "saturation": saturated,
            "available_h1": {"F": _finite_mean(F1), "L_ar": _finite_mean(L1), "L_ar_optimism": optimism,
                             "L_welch": _finite_mean(W1)}}
    if not parts:
        return None, info
    lad = pd.concat(parts, ignore_index=True)
    out = forecasts.merge(lad.drop(columns=["y"]), on=["unique_id", "ds", "cutoff"], how="inner")
    return out, info


def _finite_mean(x):
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


# --------------------------------------------------------------------------------------------------------------------
# Scoring the rungs and reading the decision rule
# --------------------------------------------------------------------------------------------------------------------


def _per_series_mse(frame, meta, models, steps):
    """(model, dataset, series) -> MSE over the windows at the given steps, through horizoneval.evaluate."""
    from horizoneval.evaluate import evaluate

    f = frame.sort_values(["unique_id", "cutoff", "ds"], kind="stable")
    step = f.groupby(["unique_id", "cutoff"]).cumcount().to_numpy()
    sub = f[np.isin(step, steps)]
    ev = evaluate(sub, None, models=models, series=meta, reference=None, decide=False)
    s = ev.scores.assign(se=ev.scores["mse"] * ev.scores["n"])
    g = s.groupby(["model", "dataset", "series"])
    return (g["se"].sum() / g["n"].sum().replace(0, np.nan)).rename("mse")


def _mean_se(x: pd.Series):
    x = x[np.isfinite(x)]
    return float(x.mean()), (float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan)


def ladder(forecasts: pd.DataFrame, train: pd.DataFrame, meta: pd.DataFrame, models: list[str],
           context: int = CONTEXT) -> dict:
    """The skill ladder of every model: rungs (F, L per dataset and horizon group), models (total skill and what is
    left beyond linear, with the verdict of the decision rule), the model-free values at h = 1, and plain sentences."""
    if train is None:
        raise FrameError("the ladder needs the history (train): its rungs are fitted on it")
    frame, info = rungs(forecasts, train, meta, context)
    rung_rows, model_rows = [], []
    if frame is not None:
        h_of = {d: len(i["linear_reference"]) for d, i in info.items() if "skipped" not in i}
        hmax = max(h_of.values())
        allg = {**groups(hmax), "all": np.arange(hmax)}
        for g, steps in allg.items():
            mse = _per_series_mse(frame, meta, RUNGS + models, steps).unstack("model")
            for ds_name, t in mse.groupby(level="dataset"):
                if ds_name not in h_of or (g != "all" and g not in groups(h_of[ds_name])):
                    continue
                lg = np.log(t.clip(lower=1e-300))
                F, Fse = _mean_se(lg["ladder_naive"] - lg["ladder_classic"])
                L, Lse = _mean_se(lg["ladder_classic"] - lg["ladder_linear"])
                sat = info[ds_name]["saturation"].get(g, {}).get("saturated") if g != "all" else None
                FL, FLse = _mean_se(lg["ladder_naive"] - lg["ladder_linear"])
                rung_rows.append({"dataset": ds_name, "group": g, "F": F, "F_se": Fse, "L": L, "L_se": Lse,
                                  "F_plus_L": FL, "F_plus_L_se": FLse, "saturated": sat})
                for m in models:
                    tot, tse = _mean_se(lg["ladder_naive"] - lg[m])
                    R, Rse = _mean_se(lg["ladder_linear"] - lg[m])
                    off = OFFSET.get(g)
                    model_rows.append({"dataset": ds_name, "model": m, "group": g, "total": tot, "total_se": tse,
                                       "beyond_linear": R, "beyond_linear_se": Rse, "offset": off,
                                       "verdict": None if g == "all" else _verdict(R, Rse, off, sat)})
    rung_df, model_df = pd.DataFrame(rung_rows), pd.DataFrame(model_rows)
    for df, cols in ((rung_df, ["F", "L", "F_plus_L"]), (model_df, ["total", "beyond_linear"])):
        for c in cols:
            if c in df:
                df[f"{c}_pct"] = nats_to_percent(df[c].to_numpy(float))
    credited = {}
    for (ds_name, m), t in (model_df.groupby(["dataset", "model"]) if len(model_df) else []):
        v = t.loc[t["group"] != "all", "verdict"]
        credited.setdefault(ds_name, {})[m] = ("beyond linear" if (v == "beyond linear").all() else
                                               "no better than linear" if (v == "no better than linear").all() else
                                               "not shown")
    out = {"rungs": rung_df, "models": model_df, "credited": credited, "datasets": info}
    out["flags"] = flags(out)
    out["text"] = narrate(out)
    return out


NEAR_PERFECT = float(np.log(100))  # nats: squared error 100 times smaller than repeating the last value
CONSISTENT_F = 0.5  # nats: measured F at step 1 against its value read from the history (paper: within 0.4)
FEW_SERIES = 10


def flags(lad: dict) -> list[dict]:
    """Deterministic sanity checks on a ladder, each with the numbers it rests on: what an explanation must mention.

    severity "warning": read the numbers with this caveat; "info": an independent check that agrees (evidence the
    numbers can be trusted); "error": the ladder could not be built."""
    out = []
    r, m = lad["rungs"], lad["models"]
    for ds_name, i in lad["datasets"].items():
        def add(flag, severity, message, **values):
            out.append({"dataset": ds_name, "flag": flag, "severity": severity, "message": message, "values": values})

        if "skipped" in i:
            add("skipped", "error", i["skipped"])
            continue
        rd = r[r["dataset"] == ds_name].set_index("group")
        if i["series"] < FEW_SERIES:
            add("few_series", "warning", f"only {i['series']} series: standard errors over series are rough and the "
                "verdicts can flip with a few more series", series=i["series"])
        top = rd.drop(index="all", errors="ignore")["F_plus_L"]
        if len(top) and top.max() > NEAR_PERFECT:
            g = top.idxmax()
            add("near_perfect", "warning",
                f"at steps {g} the linear model's squared error is {np.exp(top.max()):.0f} times smaller than repeating "
                "the last value: the series is nearly deterministic (smooth or periodic). There the attainable linear "
                "gain has no finite bound (it is set by the noise floor and by how much memory the model gets), so "
                "read these rungs as 'almost perfectly forecastable' rather than as a precise amount, and compare "
                "models by ratios, not by percentages, which are all close to 100%.",
                group=g, F_plus_L=float(top.max()), ratio=float(np.exp(top.max())))
        unsat = [g for g, s in i["saturation"].items() if not s["saturated"]]
        if unsat:
            add("not_saturated", "warning",
                f"the linear reference is not saturated at steps {', '.join(unsat)} (more memory still helped it by "
                f"{max(i['saturation'][g]['gain'] for g in unsat):.2f} nats on the history): L there is a lower bound "
                "of what a linear model can do, and no model can be credited with skill beyond linear on this dataset",
                groups=unsat, gain={g: i["saturation"][g]["gain"] for g in unsat})
        for c, what in (("F", "the simple rule"), ("L", "the linear model")):
            for g, row in rd.drop(index="all", errors="ignore").iterrows():
                se = row[f"{c}_se"]
                if np.isfinite(se) and row[c] < -2 * se and row[c] < -0.05:
                    add(f"negative_{c}", "warning",
                        f"{what} chosen on the history did worse on the scored windows at steps {g} ({row[c]:.3f} "
                        "nats): the scored period behaves differently from the history it was chosen on",
                        group=g, value=float(row[c]), se=float(se))
        av = i["available_h1"]
        if "1" in rd.index and av.get("F") is not None:
            F1, diff = float(rd.loc["1", "F"]), float(rd.loc["1", "F"] - av["F"])
            ok = abs(diff) <= CONSISTENT_F
            add("check_F_h1", "info" if ok else "warning",
                f"independent check: F at step 1 measured on the scored windows is {F1:.3f} nats, and read from the "
                f"history alone without forecasting it is {av['F']:.3f} nats; " + (
                    "they agree" if ok else f"they differ by {abs(diff):.2f} nats, so the history and the scored "
                    "period differ, or the series are too few or too short for a stable estimate"),
                measured=F1, from_history=float(av["F"]), difference=diff)
        if "1" in rd.index and None not in (av.get("F"), av.get("L_ar"), av.get("L_welch")):
            # F + L at step 1 is the stable quantity (the split depends on the simple rule chosen), compared with the
            # history's F plus the price of naivety, whose two estimates are biased in opposite directions
            FL = float(rd.loc["1", "F_plus_L"])
            se = float(rd.loc["1", "F_plus_L_se"])
            lo, hi = sorted([float(av["F"] + av["L_welch"]), float(av["F"] + av["L_ar"])])
            slack = 2 * (se if np.isfinite(se) else 0) + 0.05
            inside = lo - slack <= FL <= hi + slack
            near = len(top) and top.max() > NEAR_PERFECT
            add("check_FL_h1", "info" if inside else "warning",
                f"independent check: F + L at step 1 measured on the scored windows is {FL:.3f} nats; read from the "
                f"history's spectrum without forecasting it (the price of naivety) it is between {lo:.3f} and "
                f"{hi:.3f} nats; " + ("the measured value is consistent with it" if inside else
                                      "the measured value is outside that range" + (
                                          ", as expected for a nearly deterministic series, where the bound does "
                                          "not hold" if near else ", so read the step-1 rungs with care")),
                measured=FL, se=se, low=lo, high=hi)
        md = m[(m["dataset"] == ds_name) & (m["group"] != "all")]
        for mod, t in md.groupby("model"):
            if (t["verdict"] == "beyond linear").any() and not (t["verdict"] == "beyond linear").all():
                groups_ = list(t.loc[t["verdict"] == "beyond linear", "group"])
                add("partial_beyond_linear", "warning",
                    f"{mod} passes the beyond-linear test only at steps {', '.join(groups_)}; the paper's rule counts "
                    "it only when it holds in every horizon group, because a run reads many cells and one passing "
                    "alone can be chance", model=mod, groups=groups_)
    return out


def _verdict(R, Rse, offset, saturated):
    if not np.isfinite(R):
        return None
    if R <= 0:
        return "no better than linear"
    if not saturated:
        return "reference not saturated"
    if np.isfinite(Rse) and R > 2 * Rse and R > offset:
        return "beyond linear"
    return "not shown"


def _pct(x):
    p = nats_to_percent(x)
    if p >= 0:
        return f"{p:.1f}% lower" if p > 99 else f"{p:.0f}% lower"
    return f"{-p:.0f}% higher"


def narrate(lad: dict) -> list[str]:
    """One plain sentence per dataset and model, for readers who are not forecasters."""
    text = []
    for ds_name, i in lad["datasets"].items():
        if "skipped" in i:
            text.append(f"{ds_name}: no skill ladder: {i['skipped']}.")
            continue
        r = lad["rungs"]
        r = r[(r["dataset"] == ds_name) & (r["group"] == "all")].iloc[0]
        share = {}
        for s in i["classic_share"]:
            for k, v in s.items():
                share[k] = share.get(k, 0) + v
        top = max(share, key=share.get).replace("_", " ")
        base = (f"{ds_name}: a simple rule (mostly {top}) has {_pct(r['F'])} squared error than repeating the last "
                f"value, and a linear model fitted on the history has {_pct(r['F_plus_L'])}.")
        text.append(base)
        m = lad["models"]
        for _, row in m[(m["dataset"] == ds_name) & (m["group"] == "all")].iterrows():
            v = lad["credited"][ds_name][row["model"]]
            s = f"{row['model']} has {_pct(row['total'])} squared error than repeating the last value; "
            if v == "beyond linear":
                s += ("it goes beyond the linear model in every horizon group by more than fitting noise explains, "
                      "which is evidence of skill a linear model cannot reach.")
            elif v == "no better than linear":
                s += "it does no better than the linear model, so a linear model gets the same or more."
            else:
                s += (f"against the linear model it has {_pct(row['beyond_linear'])} squared error, which is not "
                      "enough in every horizon group to rule out fitting noise: no skill beyond linear is shown.")
            text.append(s)
    return text


def to_json(lad: dict) -> dict:
    return {"rungs": lad["rungs"], "models": lad["models"], "credited": lad["credited"],
            "datasets": lad["datasets"], "flags": lad.get("flags", []), "text": lad["text"]}


# --------------------------------------------------------------------------------------------------------------------
# Calibration: the one case with a known optimum (the paper's AR differences), and a case with real beyond-linear skill
# --------------------------------------------------------------------------------------------------------------------

PAPER = {"L_h1_population": float(-np.log(1 - 0.5 ** 2)), "L_h1_paper": 0.309, "L_h1_paper_se": 0.078,
         "offset_paper": {"1": 0.008, "2-4": 0.027, "5+": 0.046}, "L_ar_paper": 0.419, "L_welch_paper": 0.256}


def _case(Y, models, h, W, name, period=1):
    from horizoneval.evaluate import _series_table
    from horizoneval.frame import from_arrays, history
    from horizoneval.prediction import Prediction

    n = Y.shape[1]
    cuts = [n - (W - j) * h for j in range(W)]
    ids = np.array([f"{name}#{i:03d}" for i in range(len(Y))])
    fc = None
    for m, fn in models.items():
        part = from_arrays(m, Y, cuts, [Prediction(fn(Y[:, :c], h)) for c in cuts], ids)
        fc = part if fc is None else fc.merge(part, on=KEYS)
    return fc, history(Y, ids), _series_table(fc, None, period, name)


def ar_differences(N, n, rng, phi=0.5):
    """Series whose differences are AR(1) with coefficient phi: the paper's calibration case."""
    e = rng.normal(size=(N, n))
    u = np.zeros_like(e)
    for t in range(1, n):
        u[:, t] = phi * u[:, t - 1] + e[:, t]
    return np.cumsum(u, axis=1)


def exact_ar(H, h, phi=0.5):
    """The exact forecaster of ar_differences: u_{n+k} = phi^k u_n, cumulated."""
    return H[:, -1:] + np.cumsum(phi ** np.arange(1, h + 1))[None] * (H[:, -1] - H[:, -2])[:, None]


def calibrate(seeds: int = 12, N: int = 30, h: int = 16, W: int = 8) -> tuple[pd.DataFrame, dict]:
    """Run the ladder on the paper's calibration case over several seeds: the exact optimum scored as a model. Its
    residual beyond the fitted linear reference is the offset (the null expectation of R), L at h = 1 should be the
    population value -log(1 - phi^2), and the exact model must never be credited with skill beyond linear."""
    rows = []
    n = CONTEXT + N_SELECT * h + W * h
    for seed in range(seeds):
        Y = ar_differences(N, n, np.random.default_rng(seed))
        fc, tr, meta = _case(Y, {"exact": exact_ar}, h, W, "ar_differences")
        lad = ladder(fc, tr, meta, ["exact"])
        m, r = lad["models"].set_index("group"), lad["rungs"].set_index("group")
        av = lad["datasets"]["ar_differences"]["available_h1"]
        row = {"seed": seed, "credited": lad["credited"]["ar_differences"]["exact"],
               "L_ar_h1": av["L_ar"], "L_welch_h1": av["L_welch"]}
        for g in groups(h):
            row.update({f"F_{g}": r.loc[g, "F"], f"L_{g}": r.loc[g, "L"], f"offset_{g}": m.loc[g, "beyond_linear"],
                        f"saturated_{g}": r.loc[g, "saturated"]})
        rows.append(row)
    d = pd.DataFrame(rows)
    num = d.select_dtypes("number").drop(columns="seed")
    summary = {"seeds": seeds, "series": N, "h": h, "windows": W, "context": CONTEXT,
               "mean": num.mean().to_dict(), "se": num.sem().to_dict(),
               "credited_beyond_linear": int((d["credited"] == "beyond linear").sum()),
               "offset_used": OFFSET, "paper": PAPER}
    return d, summary
