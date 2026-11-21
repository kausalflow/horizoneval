import json

import numpy as np

from horizoneval import available, forecaster
from horizoneval.benchmark import cutoffs, run


def test_cutoffs_are_rolling_and_end_at_the_horizon():
    assert cutoffs(100, 10, 3, 10) == [70, 80, 90]


def test_registry_discovers_baselines_and_models():
    names = available()
    for n in ["naive", "seasonal_naive", "drift", "damped_drift", "chronos_bolt_small"]:
        assert n in names


def test_run_writes_scores_tests_and_report(tmp_path):
    lb, summary, scores = run(["seasonal_naive", "drift", "damped_drift"],
                              ["synthetic:random_walk", "synthetic:seasonal_trend", "synthetic:ar_differences"],
                              h=6, windows=3, context=60, n_series=8, out=tmp_path)
    assert set(lb["model"]) == {"naive", "seasonal_naive", "drift", "damped_drift"}
    for f in ["scores.csv", "summary.csv", "by_horizon.csv", "tests.csv", "ranks.json", "config.json", "report.html"]:
        assert (tmp_path / f).exists(), f
    # the seasonal series are where seasonal naive must beat naive
    s = summary.set_index(["model", "dataset"])
    assert s.loc[("seasonal_naive", "synthetic:seasonal_trend"), "skill"] > 0
    assert "critical_difference" in json.loads((tmp_path / "ranks.json").read_text())


def test_run_is_reproducible(tmp_path):
    a = run(["drift"], ["synthetic:random_walk"], h=4, windows=2, context=30, n_series=5, out=tmp_path / "a")[2]
    b = run(["drift"], ["synthetic:random_walk"], h=4, windows=2, context=30, n_series=5, out=tmp_path / "b")[2]
    assert np.allclose(a["mase"], b["mase"])
    ca, cb = (json.loads((tmp_path / x / "config.json").read_text())["datasets"] for x in "ab")
    assert ca == cb


def test_a_failing_model_is_recorded_not_fatal(tmp_path):
    @forecaster("broken_for_test")
    def broken(Y, h, period=None):
        return np.zeros((Y.shape[0], h + 1))

    run(["broken_for_test"], ["synthetic:random_walk"], h=4, windows=2, context=30, n_series=4, out=tmp_path)
    assert "broken_for_test" in json.loads((tmp_path / "config.json").read_text())["failures"]
