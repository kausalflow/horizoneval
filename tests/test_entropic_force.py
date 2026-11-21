import numpy as np
import pytest

pytest.importorskip("torch")

from horizoneval.models.entropic_force import EFNN, EFNNMLP, window_scale, windows  # noqa: E402
from horizoneval.registry import load  # noqa: E402


def test_windows_are_scaled_by_last_value_and_naive_error():
    Y = np.arange(40.0)[None] * 3 + 5  # constant step 3
    X, T = windows(Y, L=8, h=4, max_samples=1000, rng=np.random.default_rng(0))
    assert X.shape == (40 - 12 + 1, 8) and T.shape == (29, 4)
    assert np.allclose(X[:, -1], 0) and np.allclose(T, [1, 2, 3, 4])


def test_window_scale_floors_constant_series():
    _, s = window_scale(np.full((2, 10), 7.0))
    assert (s > 0).all()


@pytest.mark.parametrize("cls", [EFNN, EFNNMLP])
def test_efnn_forecasts_shape_and_learns_a_sine(cls):
    t = np.arange(200)
    rng = np.random.default_rng(1)
    Y = np.sin(2 * np.pi * (t[None] + rng.integers(0, 12, (8, 1))) / 12) * 5 + 10
    F = cls()(Y[:, :188], 12, period=12)
    assert F.shape == (8, 12) and np.isfinite(F).all()
    naive_err = np.abs(Y[:, 188:] - Y[:, 187:188]).mean()
    assert np.abs(F - Y[:, 188:]).mean() < 0.5 * naive_err


def test_variants_are_registered():
    for name in ["efnn", "efnn_noent", "efnn_mlp"]:
        assert callable(load(name))


def test_short_history_falls_back_to_naive():
    Y = np.arange(28.0).reshape(2, 14)
    for n in (14, 10, 3):
        F = EFNN()(Y[:, :n], 12)
        assert F.shape == (2, 12) and np.allclose(F, Y[:, n - 1 : n])


def test_window_subsampling_is_bounded():
    Y = np.random.default_rng(0).normal(size=(50, 500))
    X, T = windows(Y, L=8, h=4, max_samples=100, rng=np.random.default_rng(0))
    assert X.shape == (100, 8) and T.shape == (100, 4)
