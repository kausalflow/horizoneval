"""Template: drift with its slope shrunk by its own t-statistic.

Copy this file to start a new forecaster. The contract:
    f(Y, h, period) -> F,  Y: (N, n) history,  F: (N, h) forecasts,  period: samples per season or None.
Rows are independent series; vectorize over them. Fit whatever you need from Y inside the call.
"""

import numpy as np

from flab.registry import forecaster


@forecaster("damped_drift")
def damped_drift(Y: np.ndarray, h: int, period: int | None = None) -> np.ndarray:
    dY = np.diff(Y, axis=1)
    m = dY.shape[1]
    slope = dY.mean(axis=1)
    t2 = m * slope**2 / np.maximum(dY.var(axis=1, ddof=1), 1e-12)
    shrink = np.clip(1 - 1 / np.maximum(t2, 1e-12), 0, 1)  # James-Stein-like: keep the drift only when it is clear
    return Y[:, -1:] + (shrink * slope)[:, None] * np.arange(1, h + 1)
