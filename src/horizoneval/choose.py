"""Help people choose a forecast: which model to use on their data, how sure that choice is, and what it costs.

Everything here reads the per-window scores that `horizoneval.evaluate` produced; nothing is scored twice.

- confidence      probability that each model is the best one, and the expected loss of picking it, from a
                  bootstrap over series (the windows of one series stay together)
- winners_curse   choose the way a backtest chooses, but score the choice on data it was not chosen on: later
                  windows (several windows), or the other half of the series (one window). The gap between the
                  backtest's best score and this honest score is the optimism of picking the winner.
- segments        does choosing a model per segment (dataset, region, product line) beat one model for everything,
                  on data the choices were not made on? Recommended only when the gain is significant.
- disagreement    when the user's own cost picks a different model than MASE or CRPS would, and what that is worth
- decide          all of the above per dataset, plus a plain-language verdict

The statistics are checked by simulation in `horizoneval.verify` and tests/test_choose.py.
"""

import numpy as np
import pandas as pd
from scipy import stats

from horizoneval.stats import verify_claim

ALL = "(all)"  # pooled group when the scores cover several datasets


def _pct(x: float) -> str:
    return "under 0.1%" if abs(x) < 0.0005 else f"{100 * x:.1f}%"


def loss_tensor(scores: pd.DataFrame, metric: str):
    """(windows, series, models) array of `metric`, with the window, series and model labels."""
    s = scores.dropna(subset=[metric])
    keys = list(zip(s["dataset"], s["series"]))
    s = s.assign(_unit=pd.MultiIndex.from_tuples(keys).to_flat_index() if keys else [])
    cube = s.pivot_table(index=["window", "_unit"], columns="model", values=metric, aggfunc="mean", dropna=False)
    windows = np.sort(s["window"].unique())
    units = pd.Index(s["_unit"].unique())
    models = list(cube.columns)
    cube = cube.reindex(pd.MultiIndex.from_product([windows, units], names=["window", "_unit"]))
    L = cube.to_numpy(float).reshape(len(windows), len(units), len(models))
    return L, windows, units, models


def _bootstrap_means(X: np.ndarray, n_boot: int, rng) -> np.ndarray:
    """(n_boot, M) means of X (S, M) over series resampled with replacement, in memory-bounded chunks."""
    S = len(X)
    out, chunk = [], max(1, 2_000_000 // max(S, 1))
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        counts = rng.multinomial(S, np.full(S, 1 / S), size=b)
        out.append(counts @ X / S)
    return np.concatenate(out)


def confidence(L: np.ndarray, models: list[str], n_boot=1000, seed=0) -> pd.DataFrame:
    """Per model: mean loss, probability of being best, expected loss of picking it (relative to the best).

    Series are the resampling unit; a series' windows are averaged first so that they stay together.
    """
    X = np.nanmean(L, axis=0)  # (S, M)
    X = X[np.isfinite(X).all(axis=1)]
    rng = np.random.default_rng(seed)
    B = _bootstrap_means(X, n_boot, rng)
    best = B.min(axis=1, keepdims=True)
    p_best = np.bincount(B.argmin(axis=1), minlength=len(models)) / n_boot
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(best > 0, B / best - 1, 0.0)
    out = pd.DataFrame({"model": models, "mean": X.mean(axis=0), "p_best": p_best,
                        "expected_loss": (B - best).mean(axis=0), "expected_loss_rel": rel.mean(axis=0),
                        "series": len(X)})
    return out.sort_values("mean").reset_index(drop=True)


def _folds(S: int, seed: int):
    """Two halves of the series, each used once to choose and once to score."""
    perm = np.random.default_rng(seed).permutation(S)
    a = np.zeros(S, bool)
    a[perm[: S // 2]] = True
    return [(a, ~a), (~a, a)]


def honest_losses(L: np.ndarray, groups: np.ndarray | None = None, seed=0):
    """Loss of the model a backtest would choose, scored on data it was not chosen on: (S,) per series.

    Several windows: before each window w >= 1, choose on windows < w; score on window w. One window: choose on one
    half of the series, score on the other, both ways. With `groups`, the choice is made separately per group (from
    that group's series only; groups without data fall back to the overall choice).
    Returns (losses per series, the choice made each time, the mode used).
    """
    W, S, M = L.shape
    g = np.zeros(S, int) if groups is None else np.asarray(groups)
    labels = np.unique(g)

    def pick(block, members):  # block (w, S, M) of selection data
        overall = np.nanargmin(np.nanmean(block, axis=(0, 1)))
        out = np.full(S, overall)
        for lab in labels:
            sel = members & (g == lab)
            if sel.any() and np.isfinite(block[:, sel]).any():
                out[g == lab] = np.nanargmin(np.nanmean(block[:, sel], axis=(0, 1)))
        return out

    choices = []
    if W >= 2:
        e = np.full((W - 1, S), np.nan)
        for w in range(1, W):
            c = pick(L[:w], np.ones(S, bool))
            e[w - 1] = L[w, np.arange(S), c]
            choices.append(c)
        with np.errstate(all="ignore"):
            return np.nanmean(e, axis=0), choices, "later windows"
    e = np.full(S, np.nan)
    for sel, ev in _folds(S, seed):
        c = pick(L, sel)
        idx = np.flatnonzero(ev)
        with np.errstate(all="ignore"):
            e[idx] = np.nanmean(L[:, idx, c[idx]], axis=0)
        choices.append(c)
    return e, choices, "other half of the series"


def winners_curse(L: np.ndarray, models: list[str], seed=0) -> dict:
    """How much the backtest flatters its winner: its best score vs. the honest score of choosing that way."""
    W = L.shape[0]
    honest, choices, mode = honest_losses(L, seed=seed)
    ev = L[1:] if W >= 2 else L  # the data the honest score was measured on
    with np.errstate(all="ignore"):
        per_model = np.nanmean(ev, axis=(0, 1))
    final = int(np.nanargmin(np.nanmean(L, axis=(0, 1))))
    backtest = float(np.nanmin(per_model))
    h = float(np.nanmean(honest))
    stable = float(np.mean([np.all(c == final) for c in choices]))
    return {"winner": models[final], "backtest_score": backtest, "honest_score": h,
            "optimism_rel": h / backtest - 1 if backtest > 0 else np.nan, "same_choice_share": stable, "mode": mode,
            "rounds": len(choices)}


def segments(L: np.ndarray, models: list[str], labels: np.ndarray, alpha=0.05, seed=0) -> dict:
    """Per-segment winners, and whether choosing per segment beats one overall choice on unseen data."""
    labels = np.asarray(labels)
    one, _, mode = honest_losses(L, None, seed)
    per, choices, _ = honest_losses(L, labels, seed)
    ok = np.isfinite(one) & np.isfinite(per)
    d = per[ok] - one[ok]
    if len(d) >= 3 and d.std(ddof=1) > 0:
        t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
        p = float(2 * stats.t.sf(abs(t), len(d) - 1))
    else:
        p = 1.0 if len(d) and np.allclose(d, 0) else np.nan
    gain = 1 - per[ok].mean() / one[ok].mean() if one[ok].mean() > 0 else np.nan
    winners = {}
    for lab in np.unique(labels):
        sel = labels == lab
        with np.errstate(all="ignore"):
            winners[str(lab)] = models[int(np.nanargmin(np.nanmean(L[:, sel], axis=(0, 1))))]
    return {"winners": winners, "one_model_score": float(one[ok].mean()), "per_segment_score": float(per[ok].mean()),
            "gain_rel": float(gain), "p_value": p, "recommend": bool(len(set(winners.values())) > 1 and gain > 0
                                                                     and p < alpha), "mode": mode}


def disagreement(scores: pd.DataFrame, metric: str, others=("mase", "crps_scaled")) -> list[dict]:
    """Where `metric` (usually the user's cost) and the standard scores pick different models."""
    rows = []
    for ds, g in _groups(scores):
        means = g.groupby("model")[[c for c in [metric, *others] if c in g]].mean()
        if metric not in means or means[metric].isna().all():
            continue
        mine = means[metric].idxmin()
        for o in others:
            if o == metric or o not in means or means[o].isna().all():
                continue
            theirs = means[o].dropna().idxmin()
            if theirs != mine and np.isfinite(means.loc[theirs, metric]):
                rows.append({"dataset": ds, "metric": metric, "pick": mine, "standard": o, "standard_pick": theirs,
                             "extra_cost_rel": float(means.loc[theirs, metric] / means.loc[mine, metric] - 1)})
    return rows


def _groups(scores):
    yield from scores.groupby("dataset")
    if scores["dataset"].nunique() > 1:
        yield ALL, scores


def decide(scores: pd.DataFrame, metric: str, segment_of: pd.Series | None = None, n_boot=1000, seed=0,
           alpha=0.05) -> dict:
    """The decision report: per dataset (and pooled), confidence, winner's curse, and a verdict; plus segment and
    cost checks. `segment_of` maps series (unique_id) to a segment label; by default the datasets are the segments."""
    conf, curse, verdicts, claims = [], {}, {}, {}
    for ds, g in _groups(scores):
        L, _, units, models = loss_tensor(g, metric)
        if not len(units) or len(models) < 1:
            continue
        c = confidence(L, models, n_boot, seed).assign(dataset=ds)
        conf.append(c)
        curse[ds] = wc = winners_curse(L, models, seed) if len(models) > 1 else None
        claims[ds] = cl = claim(L, models, c, alpha, seed) if len(models) > 1 else None
        verdicts[ds] = verdict(c, wc, metric, cl)
    out = {"metric": metric, "confidence": pd.concat(conf, ignore_index=True) if conf else pd.DataFrame(),
           "winners_curse": curse, "claims": claims, "verdict": verdicts, "segments": None,
           "disagreement": disagreement(scores, metric) if metric not in ("mase", "crps_scaled") else []}
    L, _, units, models = loss_tensor(scores, metric)
    if segment_of is not None:
        labels = np.array([segment_of.get(u[1], "?") for u in units])
    elif scores["dataset"].nunique() > 1:
        labels = np.array([u[0] for u in units])
    else:
        labels = None
    if labels is not None and len(np.unique(labels)) > 1 and len(models) > 1:
        out["segments"] = segments(L, models, labels, alpha, seed)
    out["text"] = narrate(out)
    return out


def claim(L: np.ndarray, models: list[str], conf: pd.DataFrame, alpha=0.05, seed=0) -> dict:
    """Is the best model really better than the runner-up? The paired test, checked by simulating equally good
    forecasters on this data (horizoneval.stats.verify_claim)."""
    top, second = conf["model"].iloc[0], conf["model"].iloc[1]
    D = L[:, :, models.index(top)] - L[:, :, models.index(second)]
    keep = np.isfinite(D).any(axis=0)
    if np.allclose(D[:, keep][np.isfinite(D[:, keep])], 0):
        return {"better": top, "than": second, "identical": True, "verified": False}
    return {"better": top, "than": second, **verify_claim(D[:, keep], alpha=alpha, seed=seed)}


def verdict(conf: pd.DataFrame, curse: dict | None, metric: str, cl: dict | None = None) -> str:
    top = conf.sort_values(["p_best", "mean"], ascending=[False, True]).iloc[0]
    if len(conf) == 1:
        return f"Only {top['model']} was scored."
    s = f"Choose {top['model']}. By {metric} it is the best model with probability {top['p_best']:.2f}"
    s += (", in every resample of your series." if top["p_best"] >= 0.995 else
          f"; counting the cases where another model is better, picking it costs on average "
          f"{_pct(top['expected_loss_rel'])} more than picking the best would.")
    close = conf[(conf["model"] != top["model"]) & (conf["p_best"] >= 0.2)]
    if len(close):
        s += f" {', '.join(close['model'])} could also be best: the data cannot separate them yet."
    if cl:
        if cl.get("identical"):
            s += f" {cl['better']} and {cl['than']} give identical forecasts here."
        elif cl["verified"]:
            s += (f" {cl['better']} beats {cl['than']} (p = {cl['p_robust']:.3g}, confirmed by simulating equally "
                  f"good forecasters on your data).")
        elif cl["p_t"] < 0.05:
            s += (f" The usual test says {cl['better']} beats {cl['than']} (p = {cl['p_t']:.3g}), but on your data that "
                  f"test raises false alarms in {100 * cl['false_alarm_rate']:.0f}% of simulations of equally good "
                  f"forecasters; the difference is not confirmed (p = {cl['p_robust']:.2g} once that is accounted "
                  f"for).")
        else:
            s += f" {cl['better']} and {cl['than']} are not significantly different (p = {cl['p_t']:.2g})."
    if curse and np.isfinite(curse.get("optimism_rel", np.nan)):
        o = curse["optimism_rel"]
        if o > 0.005:
            s += (f" Backtests flatter the winner: choosing this way scores {curse['honest_score']:.3g} on "
                  f"{curse['mode']}, not {curse['backtest_score']:.3g} ({_pct(o)} worse).")
        else:
            s += f" Choosing this way holds up on {curse['mode']} (within {_pct(max(o, 0))})."
    return s


def narrate(d: dict) -> list[str]:
    lines = [f"{ds}: {v}" for ds, v in d["verdict"].items()]
    seg = d.get("segments")
    if seg:
        w = ", ".join(f"{k}: {v}" for k, v in seg["winners"].items())
        if seg["recommend"]:
            lines.append(f"Choose per segment ({w}): {100 * seg['gain_rel']:.1f}% better than one model on "
                         f"{seg['mode']} (p = {seg['p_value']:.3f}).")
        else:
            lines.append(f"One model for everything is enough: choosing per segment ({w}) changes the score by "
                         f"{100 * seg['gain_rel']:.1f}% on {seg['mode']}, which is not a reliable gain "
                         f"(p = {seg['p_value']:.2f}).")
    for r in d.get("disagreement", []):
        lines.append(f"{r['dataset']}: by your cost ({r['metric']}) pick {r['pick']}; {r['standard']} would pick "
                     f"{r['standard_pick']}, which costs you {100 * r['extra_cost_rel']:.1f}% more.")
    return lines


def to_json(d: dict) -> dict:
    """The decision as plain JSON."""
    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, list):
            return [clean(v) for v in x]
        if isinstance(x, (np.floating, float)):
            return None if not np.isfinite(x) else float(x)
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    out = {k: v for k, v in d.items() if k != "confidence"}
    out["confidence"] = d["confidence"].to_dict(orient="records")
    return clean(out)
