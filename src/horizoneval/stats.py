"""Formal comparisons.

- bootstrap_ci: percentile bootstrap of a mean score over series (windows of one series stay together).
- paired_test: Diebold–Mariano-style test of equal accuracy against a reference. The loss differential is
  averaged over the windows of each series, so dependence between windows stays inside a series; series are
  treated as independent and compared with a t test on the per-series differentials.
- friedman_nemenyi: Friedman test over datasets and the Nemenyi critical difference of mean ranks (Demšar 2006).
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
