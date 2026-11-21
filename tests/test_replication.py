"""Published results reproduced by horizoneval's scoring. Needs the data on disk (see horizoneval.replicate);
skipped otherwise. The full runs are committed under results/replication-m4 and results/replication-gift-eval."""

from pathlib import Path

import pytest

from horizoneval import evaluate

M4 = Path("data/m4/Dataset/Train/Hourly-train.csv")
GIFT = Path("data/gift-eval/ett1/H")


@pytest.mark.skipif(not M4.exists(), reason="M4 data not downloaded")
def test_m4_hourly_and_weekly_match_the_official_table():
    pytest.importorskip("libarchive")
    from horizoneval import replicate

    fc, train, series = replicate.m4_frames(freqs=("Hourly", "Weekly"))
    ev = evaluate(fc, train, series=series, reference="m4_Naive2", decide=False)
    cmp = replicate.compare_m4(ev.scores)
    cmp = cmp[cmp["frequency"] != "Total"]
    assert len(cmp) == 28 and cmp["match"].all(), cmp[~cmp["match"]]


@pytest.mark.skipif(not GIFT.exists(), reason="GIFT-Eval data not downloaded")
def test_gift_eval_ett1_hourly_matches_the_leaderboard():
    pytest.importorskip("datasets")
    from horizoneval import replicate

    fc, train, series = replicate.gift_frames(configs=["ett1/H"])
    ev = evaluate(fc, train, series=series, reference="naive", decide=False)
    cmp = replicate.compare_gift(ev.scores)
    assert len(cmp) == 24 and cmp["match"].all(), cmp[~cmp["match"]]
