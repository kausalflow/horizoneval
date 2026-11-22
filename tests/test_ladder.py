"""The skill ladder (horizoneval.ladder) on cases with known answers, from the naive-forecaster paper."""

import numpy as np
import pytest

from horizoneval import evaluate, tools
from horizoneval.ladder import _case, ar_differences, exact_ar, groups, log_flatness_ar, log_flatness_welch, nats_to_percent
from horizoneval.result import dumps, to_result, validate

H, W = 16, 8
N_TOTAL = 512 + 24 * H + W * H


def naive(H_, h):
    return np.repeat(H_[:, -1:], h, axis=1)


def test_groups_and_percent():
    assert list(groups(1)) == ["1"] and list(groups(4)) == ["1", "2-4"] and list(groups(16)) == ["1", "2-4", "5+"]
    assert nats_to_percent(0) == 0 and abs(nats_to_percent(np.log(2)) - 50) < 1e-9


def test_random_walk_has_nothing_to_find():
    """Every rung is zero on a random walk, and the identity itself is 'no better than linear'."""
    Y = np.cumsum(np.random.default_rng(0).normal(size=(20, N_TOTAL)), axis=1)
    fc, tr, meta = _case(Y, {"Naive": naive}, H, W, "rw")
    lad = evaluate(fc, tr, series=meta, reference=None, decide=False, ladder=True).ladder
    r = lad["rungs"]
    assert (r["F"].abs() < 0.03).all() and (r["L"].abs() < 0.03).all()
    assert lad["credited"]["rw"]["Naive"] != "beyond linear"


def test_ar_differences_recover_the_population_value_and_the_optimum_is_not_credited():
    """L at h = 1 is -log(1 - phi^2) = 0.288; the exact forecaster sits within the offset of the fitted linear
    reference, so it is never credited with skill beyond linear; the two model-free estimates bracket 0.288."""
    Y = ar_differences(30, N_TOTAL, np.random.default_rng(1))
    fc, tr, meta = _case(Y, {"exact": exact_ar, "Naive": naive}, H, W, "ar")
    lad = evaluate(fc, tr, series=meta, reference=None, decide=False, ladder=True).ladder
    r = lad["rungs"].set_index("group")
    pop = -np.log(1 - 0.25)
    assert abs(r.loc["1", "L"] - pop) < 3 * r.loc["1", "L_se"]
    assert abs(r.loc["1", "F"]) < 0.1
    assert lad["credited"]["ar"]["exact"] != "beyond linear"
    m = lad["models"].set_index(["model", "group"])
    assert abs(m.loc[("exact", "1"), "beyond_linear"]) < 0.05
    # the identity scored as a model: its total is exactly 0 and what is left beyond linear is -(F + L)
    for g in ["1", "2-4", "5+", "all"]:
        assert m.loc[("Naive", g), "total"] == pytest.approx(0, abs=1e-12)
    assert lad["credited"]["ar"]["Naive"] != "beyond linear"
    av = lad["datasets"]["ar"]["available_h1"]
    assert av["L_welch"] < pop + 0.1 and av["L_ar"] > pop - 0.05


def test_price_of_naivety_estimates():
    rng = np.random.default_rng(2)
    white = rng.normal(size=(50, 1024))
    est, opt = log_flatness_ar(white)
    assert abs(est.mean() - opt) < 0.03  # white noise: only the in-sample optimism of the fit
    assert abs(log_flatness_welch(white).mean()) < 0.02
    phi, e = 0.5, rng.normal(size=(50, 4096))
    u = np.zeros_like(e)
    for t in range(1, e.shape[1]):
        u[:, t] = phi * u[:, t - 1] + e[:, t]
    assert abs(log_flatness_welch(u).mean() - (-np.log(1 - phi ** 2))) < 0.03


def test_a_nonlinear_map_is_credited_beyond_linear():
    """The logistic map has a nearly flat spectrum, so no linear model helps, while its own map forecasts it almost
    exactly a few steps ahead: the decision rule credits it in every horizon group."""
    rng, h, N = np.random.default_rng(3), 4, 30
    n = 512 + 24 * h + W * h
    f = lambda x: 4 * x * (1 - x)  # noqa: E731
    X = np.empty((N, n))
    X[:, 0] = rng.uniform(0.1, 0.9, N)
    for t in range(1, n):
        X[:, t] = np.clip(f(X[:, t - 1]) + 1e-4 * rng.normal(size=N), 1e-6, 1 - 1e-6)

    def mapf(H_, h):
        x, out = H_[:, -1], []
        for _ in range(h):
            x = f(x)
            out.append(x)
        return np.stack(out, axis=1)

    fc, tr, meta = _case(X, {"map": mapf}, h, W, "logistic")
    ev = evaluate(fc, tr, series=meta, reference=None, decide=False, ladder=True)
    lad = ev.ladder
    assert lad["credited"]["logistic"]["map"] == "beyond linear"
    assert (lad["rungs"]["L"].abs() < 0.05).all()
    assert "beyond the linear model" in " ".join(lad["text"])
    doc = to_result(ev)
    validate(doc)
    assert doc["ladder"]["credited"]["logistic"]["map"] == "beyond linear"


def test_too_little_history_is_skipped_with_a_reason():
    Y = np.cumsum(np.random.default_rng(4).normal(size=(10, 120)), axis=1)
    fc, tr, meta = _case(Y, {"Naive": naive}, 12, 2, "short")
    lad = evaluate(fc, tr, series=meta, reference=None, decide=False, ladder=True).ladder
    assert "skipped" in lad["datasets"]["short"] and lad["rungs"].empty
    assert "no skill ladder" in lad["text"][0]


def test_ladder_through_every_front_door(tmp_path):
    from horizoneval.cli import main
    import contextlib
    import io

    Y = ar_differences(10, 512 + 24 * 4 + 4 * 4, np.random.default_rng(5))
    fc, tr, meta = _case(Y, {"exact": exact_ar, "Naive": naive}, 4, 4, "data")
    fc.to_csv(tmp_path / "cv.csv", index=False)
    tr.to_parquet(tmp_path / "train.parquet", index=False)
    from horizoneval.frame import read

    python = dumps(to_result(evaluate(read(tmp_path / "cv.csv"), read(tmp_path / "train.parquet"), reference=None,
                                      ladder=True)))
    registry = tools.call_json("score", {"forecasts": str(tmp_path / "cv.csv"), "train": str(tmp_path / "train.parquet"),
                                         "ladder": True, "out": str(tmp_path / "run")})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["score", str(tmp_path / "cv.csv"), "--train", str(tmp_path / "train.parquet"), "--ladder",
              "--out", str(tmp_path / "cli"), "--json"])
    assert python == registry == buf.getvalue().strip() == (tmp_path / "run" / "result.json").read_text().strip()
    assert (tmp_path / "run" / "ladder.json").exists() and "Where the skill comes from" in (
        tmp_path / "run" / "report.html").read_text()
