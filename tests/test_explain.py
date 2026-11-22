"""Explanations are built from the result alone, and invented numbers are caught."""

import contextlib
import io
import json

import numpy as np
import pytest

from horizoneval import evaluate, tools
from horizoneval.cli import main
from horizoneval.explain import check_numbers, explain
from horizoneval.ladder import _case, ar_differences, exact_ar
from horizoneval.result import to_result, validate


def naive(H, h):
    return np.repeat(H[:, -1:], h, axis=1)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    d = tmp_path_factory.mktemp("run")
    rng = np.random.default_rng(0)
    n = 512 + 24 * 8 + 4 * 8
    t = np.arange(n)
    periodic = 10 * np.sin(2 * np.pi * t / 12)[None] + 0.01 * rng.normal(size=(12, n))  # nearly deterministic
    Y = np.vstack([ar_differences(12, n, rng), periodic])
    fc, tr, meta = _case(Y, {"exact": exact_ar, "Naive": naive}, 8, 4, "x", period=12)
    meta["dataset"] = ["ar"] * 12 + ["periodic"] * 12
    ev = evaluate(fc, tr, series=meta, reference="Naive", ladder=True)
    ev.write(d)
    return d, to_result(ev)


def test_explanation_carries_answer_caveats_trust_and_facts(run):
    d, doc = run
    validate(doc)
    ex = explain(doc)
    assert ex["answer"][: len(doc["verdict"])] == doc["verdict"]
    flags = {(c["flag"]) for c in ex["caveats"] if "flag" in c}
    assert "near_perfect" in flags  # the periodic series
    assert any("independent check" in t for t in ex["trust"])
    ids = {f["id"] for f in ex["facts"]}
    assert "ladder.periodic.1.F" in ids and "choice.ar.p_best" in ids
    f = next(f for f in ex["facts"] if f["id"] == "ladder.periodic.1.F_plus_L")
    assert f["percent_lower_squared_error"] > 99 and f["ci95"][0] < f["value"] < f["ci95"][1]
    assert json.loads((d / "explanation.json").read_text())["facts"] == json.loads(json.dumps(ex["facts"]))


def test_invented_numbers_are_caught(run):
    _, doc = run
    f = next(f for f in explain(doc)["facts"] if f["id"] == "ladder.periodic.1.F_plus_L")
    v, pct = f["value"], f["percent_lower_squared_error"]
    good = (f"On the periodic series the linear model gains {v:.2f} nats ({v:.1f} rounded), a squared error "
            f"{pct:.1f}% lower, over steps 2-4 and 5+ with 12 series.")
    assert check_numbers(good, doc) == {"ok": True, "unsupported": [], "checked": 3}
    bad = check_numbers(good + " It is 37.4% better than ETS and 3.217 nats above TimeGPT.", doc)
    assert not bad["ok"] and bad["unsupported"] == ["37.4 %", "3.217"]


def test_every_front_door_explains_the_same(run):
    d, doc = run
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["explain", str(d)])
    assert buf.getvalue().strip() == tools.call_json("explain", {"run": str(d)})
    assert json.loads(buf.getvalue()) == json.loads(json.dumps(explain(doc)))
    (d / "draft.txt").write_text("It is 37.4% better.")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(["explain", str(d), "--check", str(d / "draft.txt")])
    assert json.loads(buf.getvalue())["unsupported"] == ["37.4 %"]
    names = {t["name"] for t in tools.definitions()}
    assert {"explain", "check_explanation"} <= names
