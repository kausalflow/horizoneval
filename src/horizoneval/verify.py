"""Simulations that check the decision statistics in horizoneval.choose, where the right answer is known.

Each check draws many synthetic score tables with a known truth and reports how the statistic behaved:

- p_best_null         identical models: each should be best in about 1/K of resamples on average
- p_best_calibration  of the choices made with probability of best near p, about a share p should be right
- winners_curse       identical models: the backtest's best score is optimistic; the honest score is not
- segments            the per-segment choice is recommended rarely when it does not help, and often when it does
- claims              equally good forecasters whose differences share a shock per window: the plain t test claims
                      winners far too often, the robust p value of stats.verify_claim does not

`horizoneval verify` prints them; tests/test_choose.py holds them to tolerances.
"""

import numpy as np
import pandas as pd

from horizoneval import choose


def table(L: np.ndarray, models=None, datasets=None) -> pd.DataFrame:
    """A scores table from a loss array (windows, series, models), as horizoneval.evaluate writes it."""
    W, S, M = L.shape
    models = models or [f"m{j}" for j in range(M)]
    w, s, m = np.meshgrid(np.arange(W), np.arange(S), np.arange(M), indexing="ij")
    ds = np.asarray(datasets)[s.ravel()] if datasets is not None else "data"
    return pd.DataFrame({"model": np.asarray(models)[m.ravel()], "dataset": ds, "window": w.ravel(),
                         "cutoff": w.ravel(), "series": [f"s{i:04d}" for i in s.ravel()], "mase": L.ravel()})


def draw(rng, W, S, means, noise=0.3, series_sd=0.5):
    """Positive losses: a series effect shared by all models, a model mean, and independent noise."""
    base = 1 + series_sd * rng.random((1, S, 1))
    return base * np.exp(np.asarray(means)[None, None, :] + noise * rng.standard_normal((W, S, len(means))))


def p_best_null(n_sims=200, K=3, W=3, S=40, seed=0) -> dict:
    rng = np.random.default_rng(seed)
    p = np.array([choose.confidence(draw(rng, W, S, np.zeros(K)), list(range(K)), n_boot=300, seed=i)
                  .sort_values("model")["p_best"].to_numpy() for i in range(n_sims)])
    return {"expected": 1 / K, "mean_p_best": p.mean(axis=0).tolist(), "max_abs_error": float(np.abs(p.mean(0) - 1 / K).max())}


def p_best_calibration(n_sims=400, W=2, S=30, spread=0.15, seed=1) -> pd.DataFrame:
    """Two models whose true difference is drawn around zero. Bins the winner's p_best and the share of right picks."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_sims):
        delta = rng.uniform(-spread, spread)
        c = choose.confidence(draw(rng, W, S, [0.0, delta]), [0, 1], n_boot=300, seed=i)
        top = c.sort_values("p_best", ascending=False).iloc[0]
        rows.append({"p_best": top["p_best"], "right": int(top["model"]) == (0 if delta > 0 else 1)})
    df = pd.DataFrame(rows)
    df["bin"] = pd.cut(df["p_best"], [0.5, 0.7, 0.9, 0.99, 1.0], include_lowest=True)
    return df.groupby("bin", observed=True).agg(n=("right", "size"), p_best=("p_best", "mean"), right=("right", "mean")).reset_index()


def winners_curse(n_sims=100, K=6, W=4, S=30, seed=2) -> dict:
    """Identical models with true mean loss mu: the honest score should average mu, the backtest's best below it."""
    rng = np.random.default_rng(seed)
    noise = 0.3
    mu = np.exp(noise ** 2 / 2) * 1.25  # E[base] * E[lognormal]
    bt, ho = [], []
    for i in range(n_sims):
        r = choose.winners_curse(draw(rng, W, S, np.zeros(K), noise), list(range(K)), seed=i)
        bt.append(r["backtest_score"]); ho.append(r["honest_score"])
    bt, ho = np.array(bt), np.array(ho)
    return {"true_mean": mu, "backtest_mean": float(bt.mean()), "honest_mean": float(ho.mean()),
            "honest_se": float(ho.std(ddof=1) / np.sqrt(n_sims))}


def segments(n_sims=100, W=3, S=60, effect=0.0, seed=3) -> float:
    """Share of simulations in which choosing per segment is recommended. Two segments; with effect > 0 each
    segment has a different best model, with effect 0 both models are equal everywhere."""
    rng = np.random.default_rng(seed)
    labels = np.repeat([0, 1], S // 2)
    hits = 0
    for i in range(n_sims):
        L = draw(rng, W, S, [0.0, 0.0])
        L[:, labels == 0, 0] *= np.exp(-effect)
        L[:, labels == 1, 1] *= np.exp(-effect)
        hits += choose.segments(L, ["a", "b"], labels, seed=i)["recommend"]
    return hits / n_sims


def claims(n_sims=200, W=5, S=60, shock=1.0, noise=1.0, seed=4) -> dict:
    """False-alarm rates at 5% for equally good forecasters, with a common shock per window (sd `shock`)."""
    from horizoneval.stats import verify_claim

    rng = np.random.default_rng(seed)
    naive, calibrated = [], []
    for i in range(n_sims):
        D = shock * rng.standard_normal((W, 1)) + noise * rng.standard_normal((W, S))
        r = verify_claim(D, n_sims=200, seed=i)
        naive.append(r["p_t"] < 0.05)
        calibrated.append(r["p_robust"] < 0.05)
    return {"t_test_false_alarms": float(np.mean(naive)), "robust_false_alarms": float(np.mean(calibrated))}


def run() -> dict:
    return {"p_best_null": p_best_null(), "p_best_calibration": p_best_calibration().to_dict(orient="records"),
            "winners_curse": winners_curse(), "segments_false_alarm_rate": segments(effect=0.0),
            "segments_power": segments(effect=0.3), "claims_with_shared_shocks": claims(),
            "claims_independent": claims(shock=0.0)}
