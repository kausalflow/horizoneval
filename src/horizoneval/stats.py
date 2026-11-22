"""Formal comparisons.

- bootstrap_ci: percentile bootstrap of a mean score over series (windows of one series stay together).
- paired_test: Diebold–Mariano-style test of equal accuracy against a reference. The loss differential is
  averaged over the windows of each series, so dependence between windows stays inside a series; series are
  treated as independent and compared with a t test on the per-series differentials.
- friedman_nemenyi: Friedman test over datasets and the Nemenyi critical difference of mean ranks (Demšar 2006).
- verify_claim: checks a claimed difference on the user's own data by simulating equally good forecasters, with a
  p value that holds when series or windows are not independent (horizoneval.verify checks the check).
"""

import numpy as np
import pandas as pd
from scipy import stats


def per_series(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Mean of `metric` over windows: one row per (model, dataset, series)."""
    return scores.groupby(["model", "dataset", "series"], as_index=False)[metric].mean()


def bootstrap_ci(values: np.ndarray, n_boot=2000, alpha=0.05, seed=0) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return tuple(np.quantile(means, [alpha / 2, 1 - alpha / 2]))


def paired_test(model_vals: np.ndarray, ref_vals: np.ndarray) -> tuple[float, float]:
    """(t statistic, two-sided p) for mean(model - ref) = 0. Negative t: the model has lower loss."""
    d = model_vals - ref_vals
    d = d[np.isfinite(d)]
    if len(d) < 3 or d.std(ddof=1) == 0:
        return (np.nan, np.nan)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return float(t), float(2 * stats.t.sf(abs(t), df=len(d) - 1))


def comparisons(scores: pd.DataFrame, reference: str, metrics=("mase", "crps_scaled")) -> pd.DataFrame:
    """Per model and dataset: mean, bootstrap CI, and the paired test against the reference, for each metric."""
    rows = []
    for metric in [m for m in metrics if m in scores]:
        ps = per_series(scores.dropna(subset=[metric]), metric)
        ref = ps[ps["model"] == reference].set_index(["dataset", "series"])[metric]
        for (model, dataset), g in ps.groupby(["model", "dataset"]):
            v = g[metric].to_numpy()
            lo, hi = bootstrap_ci(v)
            row = {"metric": metric, "model": model, "dataset": dataset, "mean": v.mean(), "ci_low": lo, "ci_high": hi}
            if model != reference and (dataset, g["series"].iloc[0]) in ref.index:
                r = ref.loc[dataset].reindex(g["series"]).to_numpy()
                row["t_vs_ref"], row["p_vs_ref"] = paired_test(v, r)
            rows.append(row)
    return pd.DataFrame(rows)


def friedman_nemenyi(summary: pd.DataFrame, metric="mase", alpha=0.05) -> dict:
    """Friedman over datasets (blocks) and models (treatments), and the Nemenyi critical difference."""
    table = summary.pivot(index="dataset", columns="model", values=metric).dropna()
    n, k = table.shape
    out = {"metric": metric, "datasets": n, "models": k, "mean_ranks": table.rank(axis=1).mean().sort_values().to_dict()}
    if n >= 2 and k >= 3:
        out["friedman_stat"], out["friedman_p"] = map(float, stats.friedmanchisquare(*table.to_numpy().T))
    if n >= 2 and k >= 2:
        q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
        out["critical_difference"] = float(q * np.sqrt(k * (k + 1) / (6 * n)))
    return out


def _t(per_series: np.ndarray) -> np.ndarray:
    """t statistics of the mean of per-series differences, along the last axis (NaN-aware)."""
    n = np.isfinite(per_series).sum(axis=-1)
    m = np.nanmean(per_series, axis=-1)
    sd = np.nanstd(per_series, axis=-1, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return m / (sd / np.sqrt(n))


def two_way_p(D: np.ndarray) -> tuple[float, float, int]:
    """(mean, p, df) for mean(D) = 0 with standard errors robust to correlation within series and within windows
    (two-way clustering, Cameron, Gelbach & Miller 2011), with small-sample corrections and min(S, W) - 1 degrees of
    freedom. With one window it is the paired t test on per-series means."""
    ok = np.isfinite(D)
    N = ok.sum()
    W, S = D.shape
    mu = D[ok].mean()
    e = np.where(ok, D - mu, 0.0)
    vs = (e.sum(axis=0) ** 2).sum() / N**2 * S / max(S - 1, 1)
    if W < 2:
        df = S - 1
        v = vs
    else:
        vw = (e.sum(axis=1) ** 2).sum() / N**2 * W / max(W - 1, 1)
        vc = (e**2).sum() / N**2
        v = vs + vw - vc
        v = v if v > 0 else max(vs, vw)
        df = min(S, W) - 1
    if v <= 0 or df < 1:
        return float(mu), float("nan"), int(df)
    return float(mu), float(2 * stats.t.sf(abs(mu) / np.sqrt(v), df)), int(df)


def verify_claim(D: np.ndarray, alpha=0.05, n_sims=500, seed=0) -> dict:
    """Check a claim "A and B differ" on the user's own data by simulating equally good forecasters.

    D (windows, series): loss of A minus loss of B. The paired t test on per-series means assumes the series are
    independent. When they are not (a shock that hits every series at the same cutoff), it claims winners too often.
    The observed differences are centred, which makes A and B equally good while keeping the dependence of the data,
    and resampled by series and by window at once (pigeonhole bootstrap); the t test is run on each simulated null.

    p_t               the paired t test on per-series means
    false_alarm_rate  share of the simulated nulls in which that t test claims a difference at `alpha`
    p_robust          p value robust to correlation within series and within windows (two_way_p)
    verified          both below alpha. When the false-alarm rate is well above alpha, trust p_robust, not p_t.
    """
    D = np.asarray(D, dtype=float)
    W, S = D.shape
    per = np.nanmean(D, axis=0)
    t_obs = float(_t(per))
    p_t = float(2 * stats.t.sf(abs(t_obs), df=max(np.isfinite(per).sum() - 1, 1)))
    mu, p_rob, df = two_way_p(D)
    Dc = D - np.nanmean(D)
    rng = np.random.default_rng(seed)
    crit = stats.t.ppf(1 - alpha / 2, df=max(S - 1, 1))
    alarms = []
    for _ in range(n_sims):
        si = rng.integers(0, S, S)
        wi = rng.integers(0, W, W) if W > 1 else np.zeros(1, int)
        with np.errstate(all="ignore"):
            t = _t(np.nanmean(Dc[np.ix_(wi, si)], axis=0))
        if np.isfinite(t):
            alarms.append(abs(t) > crit)
    far = float(np.mean(alarms)) if alarms else float("nan")
    return {"mean_diff": mu, "t": t_obs, "p_t": p_t, "false_alarm_rate": far, "p_robust": p_rob, "df_robust": df,
            "verified": bool(p_t < alpha and p_rob < alpha), "windows": W, "series": S}
