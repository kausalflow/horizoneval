import numpy as np

from horizoneval.baselines import drift, naive, seasonal_naive
from horizoneval.prediction import QUANTILES


def test_naive_repeats_last_value_with_ordered_quantiles():
    Y = np.cumsum(np.random.default_rng(0).normal(size=(3, 50)), axis=1)
    p = naive(Y, 5)
    assert np.allclose(p.point, Y[:, -1:])
    assert p.quantiles.shape == (3, 5, len(QUANTILES))
    assert (np.diff(p.quantiles, axis=-1) >= 0).all()


def test_seasonal_naive_repeats_last_season():
    Y = np.tile(np.arange(4.0), 10)[None]
    assert np.allclose(seasonal_naive(Y, 6, period=4).point, [[0, 1, 2, 3, 0, 1]])


def test_drift_extends_the_line():
    Y = np.arange(10.0)[None]
    assert np.allclose(drift(Y, 3), [[10, 11, 12]])
