import numpy as np

from flab.metrics import coverage, crps, series_scores
from flab.prediction import QUANTILES, Prediction


def test_perfect_forecast_scores_zero():
    T = np.ones((2, 4))
    q = np.repeat(T[..., None], len(QUANTILES), axis=-1)
    s = series_scores(Prediction(T, q), T, np.arange(20.0)[None].repeat(2, 0))
    assert np.allclose(s["mase"], 0) and np.allclose(s["crps_scaled"], 0) and np.allclose(s["coverage80"], 1)


def test_crps_of_point_mass_is_close_to_absolute_error():
    T = np.zeros((1, 1))
    q = np.full((1, 1, len(QUANTILES)), 2.0)
    assert np.isclose(crps(q, T)[0], 2 * np.mean([qq * 2 for qq in QUANTILES]))
    assert coverage(q, T)[0] == 0
