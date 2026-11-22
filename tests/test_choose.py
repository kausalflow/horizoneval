"""The decision layer and the safety net under it: simulations with a known answer, differential tests against
utilsforecast, metamorphic tests (changes that must not, or must predictably, change the numbers) and fingerprints."""

import numpy as np
import pandas as pd
import pytest

from horizoneval import choose, evaluate, verify
from horizoneval.evaluate import fingerprint

from test_frame import PERIOD, nixtla_frames


# ---- simulations with a known answer -------------------------------------------------------------------------------

def test_identical_models_are_each_best_about_one_in_k():
    r = verify.p_best_null(n_sims=150)
    assert r["max_abs_error"] < 0.03, r


def test_probability_of_best_is_calibrated():
    cal = verify.p_best_calibration(n_sims=400)
    assert (cal["n"] >= 30).all()
    assert np.allclose(cal["right"], cal["p_best"], atol=0.1), cal


def test_backtests_flatter_the_winner_and_the_honest_score_does_not():
    r = verify.winners_curse(n_sims=100)
    assert r["backtest_mean"] < r["true_mean"] - 0.02  # picking the best of six equal models looks better than it is
    assert abs(r["honest_mean"] - r["true_mean"]) < 4 * r["honest_se"] + 0.005


def test_per_segment_choice_is_recommended_only_when_it_helps():
    assert verify.segments(n_sims=100, effect=0.0) <= 0.1
    assert verify.segments(n_sims=50, effect=0.3) >= 0.9


def test_winners_curse_with_one_window_uses_the_other_half_of_the_series():
    rng = np.random.default_rng(0)
    r = choose.winners_curse(verify.draw(rng, 1, 40, np.zeros(4)), list("abcd"))
    assert r["mode"] == "other half of the series" and r["rounds"] == 2


# ---- end to end: forecasts whose quality we set ---------------------------------------------------------------------

def oracle_frames(noise=(0.1, 1.0, 1.0), n_series=30, seed=0):
    """Forecasts = truth + noise of known size: the first model is best by construction."""
    rng = np.random.default_rng(seed + 1000)  # not the stream that drew the series
    fc, train = nixtla_frames(seed)
    fc = pd.concat([fc.assign(unique_id=fc["unique_id"] + f"-{k}") for k in range(n_series // 4)], ignore_index=True)
    train = pd.concat([train.assign(unique_id=train["unique_id"] + f"-{k}") for k in range(n_series // 4)],
                      ignore_index=True)
    fc = fc[["unique_id", "ds", "cutoff", "y"]].copy()
    for name, sd in zip(["good", "bad1", "bad2"], noise):
        fc[name] = fc["y"] + rng.normal(0, sd, len(fc))
    return fc, train


def test_the_oracle_model_is_chosen_with_confidence():
    fc, train = oracle_frames()
    ev = evaluate(fc, train, period=PERIOD, reference=None)
    conf = ev.decision["confidence"]
    assert conf.iloc[0]["model"] == "good" and conf.iloc[0]["p_best"] > 0.99
    assert ev.decision["text"][0].startswith("data: Choose good.")


def test_equal_models_are_reported_as_inseparable():
    fc, train = oracle_frames(noise=(1.0, 1.0, 1.0))
    ev = evaluate(fc, train, period=PERIOD, reference=None)
    assert ev.decision["confidence"]["p_best"].max() < 0.9
    assert "cannot separate" in ev.decision["text"][0]


def test_your_cost_can_pick_another_model_and_we_say_so():
    fc, train = oracle_frames(noise=(0.5, 0.5, 0.5))
    fc["biased_up"] = fc["good"] + 1.5  # worse by MASE, but never short: cheap when shortfalls cost 19 times more
    ev = evaluate(fc, train, period=PERIOD, reference=None, cost="linlin:0.95")
    flips = ev.decision["disagreement"]
    assert flips and flips[0]["pick"] == "biased_up" and flips[0]["standard"] == "mase"
    assert ev.config["decide_by"] == "cost" and "cost" in ev.summary


def test_segments_from_the_series_table(tmp_path):
    fc, train = oracle_frames(noise=(0.5, 0.5, 0.5))
    ids = fc["unique_id"].unique()
    region = pd.Series(np.where(np.arange(len(ids)) % 2 == 0, "north", "south"), index=ids)
    north = fc["unique_id"].map(region) == "north"
    fc.loc[north, "bad1"] = fc.loc[north, "y"] + 0.05 * (fc.loc[north, "bad1"] - fc.loc[north, "y"])
    fc.loc[~north, "bad2"] = fc.loc[~north, "y"] + 0.05 * (fc.loc[~north, "bad2"] - fc.loc[~north, "y"])
    series = pd.DataFrame({"unique_id": ids, "region": region.to_numpy()})
    ev = evaluate(fc, train, series=series, period=PERIOD, reference=None, segment="region")
    seg = ev.decision["segments"]
    assert seg["winners"] == {"north": "bad1", "south": "bad2"} and seg["recommend"]
    ev.write(tmp_path)
    assert (tmp_path / "decision.json").exists() and "What to choose" in (tmp_path / "report.html").read_text()


# ---- differential tests against utilsforecast ----------------------------------------------------------------------

def test_point_losses_match_utilsforecast():
    from utilsforecast import losses

    fc, train = nixtla_frames()
    ev = evaluate(fc, train, period=PERIOD, decide=False)
    ours = ev.scores.set_index(["series", "cutoff", "model"]).sort_index()
    for ours_col, fn, factor in [("mae", losses.mae, 1), ("mse", losses.mse, 1), ("smape", losses.smape, 2)]:
        theirs = fn(fc, ["Naive", "SeasonalNaive"]).melt(["unique_id", "cutoff"], var_name="model")
        theirs = theirs.set_index(["unique_id", "cutoff", "model"]).sort_index()["value"]
        # utilsforecast's sMAPE is |e| / (|f| + |y|) in [0, 1]; ours is the M4 convention 2|e| / (|f| + |y|) in [0, 2]
        assert np.allclose(ours[ours_col].to_numpy(), factor * theirs.to_numpy()), ours_col


def test_quantile_scores_match_utilsforecast():
    from utilsforecast import losses

    fc, train = nixtla_frames()
    ev = evaluate(fc, train, period=PERIOD, decide=False)
    s = ev.scores[ev.scores["model"] == "SeasonalNaive"].set_index(["series", "cutoff"]).sort_index()
    cols = ["SeasonalNaive-q-10", "SeasonalNaive-q-50", "SeasonalNaive-q-90"]
    mq = losses.mqloss(fc, {"SeasonalNaive": cols}, np.array([0.1, 0.5, 0.9]))
    mq = mq.set_index(["unique_id", "cutoff"]).sort_index()["SeasonalNaive"]
    crps_raw = s["crps_scaled"] * s["mae"] / s["mase"]
    assert np.allclose(crps_raw.to_numpy(), 2 * mq.to_numpy())
    iv = fc.rename(columns={"SeasonalNaive-q-10": "SeasonalNaive-lo-80", "SeasonalNaive-q-90": "SeasonalNaive-hi-80"})
    cov = losses.coverage(iv, ["SeasonalNaive"], level=80).set_index(["unique_id", "cutoff"]).sort_index()
    assert np.allclose(s["coverage80"].to_numpy(), cov["SeasonalNaive"].to_numpy())


# ---- metamorphic tests ---------------------------------------------------------------------------------------------

def _summary(ev):
    return ev.summary.sort_values(["model", "dataset"]).reset_index(drop=True)


def test_row_order_and_series_names_do_not_matter():
    fc, train = nixtla_frames()
    base = evaluate(fc, train, period=PERIOD, reference="Naive")
    shuffled = fc.sample(frac=1, random_state=1)
    renamed = {f"s{i}": f"z{9 - i}" for i in range(4)}
    again = evaluate(shuffled.assign(unique_id=shuffled["unique_id"].map(renamed)),
                     train.assign(unique_id=train["unique_id"].map(renamed)), period=PERIOD, reference="Naive")
    pd.testing.assert_frame_equal(_summary(base), _summary(again))
    assert fingerprint(fc) == fingerprint(shuffled) and base.config["protocol_sha256_16"] != ""


def test_units_and_level_do_not_change_scaled_scores():
    fc, train = nixtla_frames()
    base = evaluate(fc, train, period=PERIOD, reference="Naive")
    vals = [c for c in fc.columns if c not in ("unique_id", "ds", "cutoff")]
    for f in (lambda v: 1000 * v, lambda v: v + 50):  # change of units; change of level
        t = evaluate(fc.assign(**{c: f(fc[c]) for c in vals}), train.assign(y=f(train["y"])), period=PERIOD,
                     reference="Naive")
        for c in ["mase", "crps_scaled", "coverage80", "width80_scaled", "skill"]:
            assert np.allclose(_summary(base)[c], _summary(t)[c], equal_nan=True), c
    big = evaluate(fc.assign(**{c: 1000 * fc[c] for c in vals}), train.assign(y=1000 * train["y"]), period=PERIOD)
    assert np.allclose(1000 * evaluate(fc, train, period=PERIOD).scores["mae"], big.scores["mae"])


def test_adding_a_model_leaves_the_others_untouched():
    fc, train = nixtla_frames()
    base = evaluate(fc, train, period=PERIOD, reference="Naive").scores
    more = evaluate(fc.assign(Extra=fc["Naive"] + 1), train, period=PERIOD, reference="Naive").scores
    more = more[more["model"] != "Extra"].reset_index(drop=True)
    pd.testing.assert_frame_equal(base.sort_values(["model", "series", "cutoff"]).reset_index(drop=True),
                                  more.sort_values(["model", "series", "cutoff"]).reset_index(drop=True))


def test_a_perfect_forecast_scores_zero_and_wins():
    fc, train = nixtla_frames()
    ev = evaluate(fc.assign(Perfect=fc["y"]), train, period=PERIOD, reference="Naive")
    p = ev.scores[ev.scores["model"] == "Perfect"]
    assert (p[["mae", "mse", "smape", "mase"]] == 0).all().all()
    assert ev.decision["confidence"].iloc[0]["model"] == "Perfect"


def test_fingerprints_see_value_changes_and_protocol_sees_cutoff_changes():
    fc, train = nixtla_frames()
    changed = fc.copy()
    changed.loc[0, "Naive"] += 1e-9
    assert fingerprint(changed) != fingerprint(fc)
    a = evaluate(fc, train, period=PERIOD).config
    b = evaluate(changed, train, period=PERIOD).config
    assert a["protocol_sha256_16"] == b["protocol_sha256_16"] and a["forecasts_sha256_16"] != b["forecasts_sha256_16"]
    c = evaluate(fc[fc["cutoff"] != fc["cutoff"].min()], train, period=PERIOD).config
    assert c["protocol_sha256_16"] != a["protocol_sha256_16"]


def test_bad_costs_are_refused():
    fc, train = nixtla_frames()
    with pytest.raises(ValueError):
        evaluate(fc, train, period=PERIOD, cost="linlin:2")
    with pytest.raises(ValueError):
        evaluate(fc, train, period=PERIOD, cost="asymmetric")


def test_a_series_whose_history_never_changed_has_no_scaled_score():
    fc, train = nixtla_frames()
    train.loc[train["unique_id"] == "s0", "y"] = 5.0
    ev = evaluate(fc, train, period=PERIOD)
    s0 = ev.scores[ev.scores["series"] == "s0"]
    assert s0["mase"].isna().all() and s0["mae"].notna().all()
    assert np.isfinite(ev.summary["mase"]).all()  # averages leave it out instead of exploding


def test_claims_hold_up_when_a_shock_hits_every_series_at_once():
    shared = verify.claims(n_sims=150)
    assert shared["t_test_false_alarms"] > 0.5  # the usual test claims winners that do not exist
    assert shared["robust_false_alarms"] <= 0.1
    independent = verify.claims(n_sims=150, shock=0.0)
    assert independent["robust_false_alarms"] <= 0.1 and independent["t_test_false_alarms"] <= 0.1


def test_the_simulated_false_alarm_rate_flags_dependent_data():
    from horizoneval.stats import verify_claim

    rng = np.random.default_rng(0)
    shocked = verify_claim(rng.standard_normal((5, 1)) + rng.standard_normal((5, 60)))
    calm = verify_claim(rng.standard_normal((5, 60)))
    assert shocked["false_alarm_rate"] > 0.2 > calm["false_alarm_rate"]


def test_the_verdict_states_whether_the_winner_is_confirmed():
    fc, train = oracle_frames()
    ev = evaluate(fc, train, period=PERIOD, reference=None)
    cl = ev.decision["claims"]["data"]
    assert cl["better"] == "good" and cl["verified"] and "confirmed by simulating" in ev.decision["text"][0]


def test_diff_names_why_two_runs_disagree():
    from horizoneval.settings import diff

    fc, train = nixtla_frames()
    base = evaluate(fc, train, period=PERIOD, reference="Naive")
    assert diff(base, evaluate(fc.sample(frac=1, random_state=0), train, period=PERIOD, reference="Naive"))["same_settings"]
    other_period = diff(base, evaluate(fc, train, period=12, reference="Naive"))
    assert [c["setting"] for c in other_period["causes"]] == ["period"] and other_period["moved"]
    tweaked = fc.copy()
    tweaked["Naive"] += 0.1
    d = diff(base, evaluate(tweaked, train, period=PERIOD, reference="Naive"))
    assert [(c["setting"], c["where"]) for c in d["causes"]] == [("forecasts", "Naive")]
    d = diff(base, evaluate(fc, train, period=PERIOD, reference="Naive", smape_zero="skip"))
    assert d["causes"][0]["setting"] == "smape_zero"
    fewer = diff(base, evaluate(fc[fc["cutoff"] != fc["cutoff"].min()], train, period=PERIOD, reference="Naive"))
    assert {"windows", "cutoffs", "truth"} <= {c["setting"] for c in fewer["causes"]}


# ---- pre-flight checks ---------------------------------------------------------------------------------------------

def _checks(fc, train=None):
    from horizoneval.checks import preflight

    return {(c["check"], c["model"]) for c in preflight(fc, train)}


def test_preflight_passes_a_clean_frame_and_catches_the_classic_mistakes():
    fc, train = nixtla_frames()
    assert not {k for k in _checks(fc, train) if k[0] not in ("few_series", "history_beyond_cutoff")}
    bad = fc.assign(Cheat=fc["y"], InThousands=fc["Naive"] / 1000, Copy=fc["Naive"])
    bad.loc[::3, "SeasonalNaive-q-90"] = bad.loc[::3, "SeasonalNaive-q-10"] - 1
    found = _checks(bad, train)
    assert ("copies_truth", "Cheat") in found
    assert ("units", "InThousands") in found
    assert ("duplicate_model", "Naive, Copy") in found
    assert ("quantile_crossing", "SeasonalNaive") in found
    disagreeing = pd.concat([train, fc[["unique_id", "ds", "y"]].assign(y=fc["y"] + 1)])  # history with other values
    assert ("truth_disagrees_with_history", None) in _checks(fc, disagreeing)


def test_preflight_findings_reach_the_run_folder(tmp_path):
    fc, train = nixtla_frames()
    ev = evaluate(fc.assign(Cheat=fc["y"]), train, period=PERIOD, reference="Naive")
    assert any(c["check"] == "copies_truth" for c in ev.checks)
    ev.write(tmp_path)
    assert "Before you trust these numbers" in (tmp_path / "report.html").read_text()
