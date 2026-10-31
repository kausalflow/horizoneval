"""Ground-truth synthetic series: sampled ODE systems, linear nulls, and seasonal series.

The ODE systems and linear nulls come from extrapolation-as-forecasts/experiments/systems.py.
"""

import numpy as np
from scipy.integrate import solve_ivp


def lorenz(t, s, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def rossler(t, s, a=0.2, b=0.2, c=5.7):
    x, y, z = s
    return [-y - z, x + a * y, b + z * (x - c)]


def van_der_pol(t, s, mu=2.0):
    x, v = s
    return [v, mu * (1 - x**2) * v - x]


# (vector field, typical initial state, approximate mean period)
ODES = {
    "lorenz": (lorenz, [1.0, 1.0, 1.0], 0.75),
    "rossler": (rossler, [1.0, 1.0, 0.0], 5.9),
    "van_der_pol": (van_der_pol, [2.0, 0.0], 7.6),
}


def sample_ode(name, n, N, rng, pts_per_period=12, noise_frac=0.01, burn_periods=20):
    """N trajectories of the first coordinate, sampled at dt = period / pts_per_period."""
    f, s0, period = ODES[name]
    dt = period / pts_per_period
    out = np.empty((N, n))
    for i in range(N):
        s_init = np.asarray(s0) + rng.normal(scale=0.5, size=len(s0))
        t_eval = burn_periods * period + dt * np.arange(n)
        sol = solve_ivp(f, (0, t_eval[-1]), s_init, t_eval=t_eval, method="DOP853", rtol=1e-9, atol=1e-9)
        out[i] = sol.y[0]
    scale = out.std(axis=1, keepdims=True)
    return out + noise_frac * scale * rng.normal(size=out.shape)


def random_walk(n, N, rng):
    return np.cumsum(rng.normal(size=(N, n)), axis=1)


def ar_differences(n, N, rng, phi=0.5):
    e = rng.normal(size=(N, n + 100))
    u = np.zeros_like(e)
    for t in range(1, e.shape[1]):
        u[:, t] = phi * u[:, t - 1] + e[:, t]
    return np.cumsum(u[:, 100:], axis=1)


def seasonal_trend(n, N, rng, period=12):
    """Level + trend + a fixed seasonal shape + AR(1) noise: the case seasonal baselines are built for."""
    t = np.arange(n)
    shape = rng.normal(size=(N, period))
    season = np.take_along_axis(shape, np.broadcast_to(t % period, (N, n)), axis=1)
    trend = rng.normal(scale=0.05, size=(N, 1)) * t
    noise = ar_differences(n, N, rng, phi=0.3) * 0.1
    return 10 + trend + 2 * season + noise


# name -> (generator(n, N, rng), period)
SYNTHETIC = {
    "random_walk": (random_walk, None),
    "ar_differences": (ar_differences, None),
    "seasonal_trend": (seasonal_trend, 12),
    "lorenz": (lambda n, N, rng: sample_ode("lorenz", n, N, rng), 12),
    "rossler": (lambda n, N, rng: sample_ode("rossler", n, N, rng), 12),
    "van_der_pol": (lambda n, N, rng: sample_ode("van_der_pol", n, N, rng), 12),
}
