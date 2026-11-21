import json

import numpy as np
import pandas as pd
import pytest

from horizoneval import FrameError, evaluate, nixtla
from horizoneval.benchmark import run
from horizoneval.cli import main
from horizoneval.frame import models_of, quantile_columns, read, to_intervals, validate

H, CUTS, N, PERIOD = 6, 3, 4, 7


def nixtla_frames(seed=0, quantiles="q"):
    """A daily cross_validation frame in Nixtla's layout: two models, three cutoffs, four series."""
    rng = np.random.default_rng(seed)
    n = 60
    ds = pd.date_range("2024-01-01", periods=n, freq="D")
    train, rows = [], []
    for i in range(N):
        y = 10 + np.sin(2 * np.pi * np.arange(n) / PERIOD) * (i + 1) + rng.normal(0, 0.3, n).cumsum()
        train.append(pd.DataFrame({"unique_id": f"s{i}", "ds": ds, "y": y}))
        for k in range(CUTS):
            c = n - H - (CUTS - 1 - k) * H  # first target index
            r = pd.DataFrame({"unique_id": f"s{i}", "ds": ds[c : c + H], "cutoff": ds[c - 1], "y": y[c : c + H]})
            r["Naive"] = y[c - 1]
            r["SeasonalNaive"] = y[c - PERIOD : c][np.arange(H) % PERIOD]
            spread = np.abs(np.diff(y[:c])).mean()
            for q, z in [(0.1, -1.2816), (0.5, 0.0), (0.9, 1.2816)]:
                col = {"q": f"SeasonalNaive-q-{int(q * 100)}", "frac": f"SeasonalNaive-q-{q}"}[quantiles]
                r[col] = r["SeasonalNaive"] + z * spread
            rows.append(r)
    return pd.concat(rows, ignore_index=True), pd.concat(train, ignore_index=True)


def test_quantile_columns_in_every_convention():
    cols = ["unique_id", "ds", "cutoff", "y", "A", "A-q-10", "A-q-0.9", "A-lo-80", "A-median", "B", "B-hi-95"]
    df = pd.DataFrame(columns=cols)
    assert models_of(df) == ["A", "B"]
    with pytest.raises(FrameError):  # q-10 and lo-80 both claim the 0.1 quantile
        quantile_columns(df, "A")
    assert quantile_columns(df.drop(columns="A-lo-80"), "A") == {0.1: "A-q-10", 0.5: "A-median", 0.9: "A-q-0.9"}
    assert quantile_columns(df, "B") == {0.975: "B-hi-95"}


def test_validate_refuses_leakage_duplicates_and_ragged_windows():
    fc, _ = nixtla_frames()
    validate(fc)
    leak = fc.copy()
    leak.loc[0, "cutoff"] = leak.loc[0, "ds"]
    with pytest.raises(FrameError, match="leakage"):
        validate(leak)
    with pytest.raises(FrameError, match="once"):
        validate(pd.concat([fc, fc.head(1)]))
    with pytest.raises(FrameError, match="same horizon"):
        validate(fc.drop(index=0))
    with pytest.raises(FrameError, match="missing"):
        validate(fc.drop(columns="cutoff"))


def test_mase_matches_utilsforecast():
    from utilsforecast import losses

    fc, train = nixtla_frames()
    ev = evaluate(fc, train, period=PERIOD, reference="Naive")
    theirs = losses.mase(fc, ["Naive", "SeasonalNaive"], seasonality=PERIOD, train_df=train)
    ours = ev.scores.pivot_table(index=["series", "cutoff"], columns="model", values="mase")
    theirs = theirs.set_index(["unique_id", "cutoff"]).sort_index()
    for m in ["Naive", "SeasonalNaive"]:
        assert np.allclose(ours[m].sort_index().to_numpy(), theirs[m].to_numpy())


def test_our_losses_run_inside_utilsforecast_evaluate():
    from utilsforecast.evaluation import evaluate as uf_evaluate

    fc, train = nixtla_frames()
    point = fc[["unique_id", "ds", "cutoff", "y", "Naive", "SeasonalNaive"]]
    res = uf_evaluate(point, metrics=[nixtla.mase(PERIOD), nixtla.smape()], train_df=train)
    ev = evaluate(fc, train, period=PERIOD)
    for metric in ["mase", "smape"]:
        r = res[res["metric"] == metric].set_index(["unique_id", "cutoff"]).sort_index()
        ours = ev.scores.pivot_table(index=["series", "cutoff"], columns="model", values=metric).sort_index()
        assert np.allclose(r["SeasonalNaive"].to_numpy(), ours["SeasonalNaive"].to_numpy())


def test_quantile_conventions_give_identical_scores():
    pct, train = nixtla_frames(quantiles="q")
    frac, _ = nixtla_frames(quantiles="frac")
    intervals = to_intervals(pct)
    assert {"SeasonalNaive-lo-80", "SeasonalNaive-hi-80", "SeasonalNaive-median"} <= set(intervals.columns)
    runs = [evaluate(f, train, period=PERIOD).scores for f in (pct, frac, intervals)]
    for s in runs[1:]:
        for c in ["crps_scaled", "coverage80", "width80_scaled"]:
            assert np.allclose(runs[0][c].dropna(), s[c].dropna())
    assert runs[0]["coverage80"].notna().any() and "coverage90" not in runs[0]


def test_a_broken_model_is_dropped_and_recorded():
    fc, train = nixtla_frames()
    fc["Broken"] = fc["Naive"]
    fc.loc[3, "Broken"] = np.nan
    ev = evaluate(fc, train, period=PERIOD)
    assert "Broken" in ev.config["failures"] and "Broken" not in set(ev.scores["model"])


def test_scaled_scores_need_history_but_the_rest_do_not():
    fc, _ = nixtla_frames()
    ev = evaluate(fc)
    assert ev.scores["mase"].isna().all() and ev.scores["smape"].notna().all()


def test_a_run_rescored_from_its_saved_frame_gives_the_same_numbers(tmp_path):
    run(["seasonal_naive", "drift"], ["synthetic:random_walk", "synthetic:seasonal_trend"],
        h=6, windows=3, context=60, n_series=8, out=tmp_path)
    for f in ["forecasts.parquet", "train.parquet", "series.csv"]:
        assert (tmp_path / f).exists(), f
    again = evaluate(pd.read_parquet(tmp_path / "forecasts.parquet"), pd.read_parquet(tmp_path / "train.parquet"),
                     series=pd.read_csv(tmp_path / "series.csv"), reference="naive")
    before = pd.read_csv(tmp_path / "summary.csv").sort_values(["model", "dataset"]).reset_index(drop=True)
    after = again.summary.sort_values(["model", "dataset"]).reset_index(drop=True)
    num = before.select_dtypes("number").columns
    assert np.allclose(before[num].to_numpy(), after[num].to_numpy(), equal_nan=True)


def test_cli_scores_an_external_csv(tmp_path):
    fc, train = nixtla_frames()
    fc.to_csv(tmp_path / "cv.csv", index=False)
    train.to_parquet(tmp_path / "train.parquet", index=False)
    assert read(tmp_path / "cv.csv")["ds"].dtype.kind == "M"
    main(["score", str(tmp_path / "cv.csv"), "--train", str(tmp_path / "train.parquet"), "--period", str(PERIOD),
          "--reference", "Naive", "--out", str(tmp_path / "run")])
    cfg = json.loads((tmp_path / "run" / "config.json").read_text())
    assert cfg["reference"] == "Naive" and (tmp_path / "run" / "report.html").exists()
    lb = pd.read_csv(tmp_path / "run" / "leaderboard.csv")
    assert lb.iloc[0]["model"] == "SeasonalNaive"


def test_a_frame_without_reference_or_history_still_gets_a_report(tmp_path):
    fc, _ = nixtla_frames()
    ev = evaluate(fc, reference=None)
    assert ev.config["rank_by"] == "smape"
    ev.write(tmp_path)
    assert (tmp_path / "report.html").exists()
