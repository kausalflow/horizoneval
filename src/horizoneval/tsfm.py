"""Foundation models and statistical references through TimeCopilot (install with the `tsfm` extra).

Registered only by name: nothing is imported or downloaded until a benchmark loads one.
"""

import logging
import warnings

import numpy as np
import pandas as pd

from horizoneval.prediction import QUANTILES, Prediction
from horizoneval.registry import register_factory

FREQ = "D"  # the models see a plain daily index; the true period is passed to the seasonal statistical models

SPECS = {
    "chronos_bolt_small": ("foundation", "Chronos", {"repo_id": "amazon/chronos-bolt-small"}),
    "moirai2_small": ("foundation", "Moirai", {"repo_id": "Salesforce/moirai-2.0-R-small"}),
    "tirex": ("foundation", "TiRex", {"repo_id": "NX-AI/TiRex"}),
    "timesfm3": ("foundation", "TimesFM", {"repo_id": "google/timesfm-3.0-pytorch"}),
    "auto_arima": ("stats", "AutoARIMA", {}),
    "auto_ets": ("stats", "AutoETS", {}),
}


def to_long(Y: np.ndarray) -> pd.DataFrame:
    """(N, n) array -> TimeCopilot long format (unique_id, ds, y)."""
    N, n = Y.shape
    ids = np.repeat([f"s{i:05d}" for i in range(N)], n)
    ds = np.tile(pd.date_range("2000-01-01", periods=n, freq=FREQ), N)
    return pd.DataFrame({"unique_id": ids, "ds": ds, "y": Y.ravel()})


class TimeCopilotModel:
    def __init__(self, alias: str):
        warnings.filterwarnings("ignore")
        logging.disable(logging.INFO)
        import importlib

        from timecopilot import TimeCopilotForecaster

        self.alias = alias
        self.kind, cls_name, self.kwargs = SPECS[alias]
        self.cls = getattr(importlib.import_module(f"timecopilot.models.{self.kind}"), cls_name)
        self._tcf_cls = TimeCopilotForecaster
        self._cache: dict = {}

    def _tcf(self, period: int | None):
        # statistical models are seasonal when the period is short enough for statsforecast (<= 24)
        m = period if (self.kind == "stats" and period and period <= 24) else 1
        key = m if self.kind == "stats" else None
        if key not in self._cache:
            extra = {"season_length": m} if self.kind == "stats" else {}
            self._cache[key] = self._tcf_cls(models=[self.cls(alias=self.alias, **self.kwargs, **extra)])
        return self._cache[key]

    def __call__(self, Y: np.ndarray, h: int, period: int | None = None) -> Prediction:
        fc = self._tcf(period).forecast(df=to_long(Y), h=h, freq=FREQ, quantiles=list(QUANTILES))

        def wide(col):
            return fc.pivot(index="unique_id", columns="ds", values=col).sort_index().to_numpy()

        q_cols = [f"{self.alias}-q-{int(q * 100)}" for q in QUANTILES]
        quantiles = np.stack([wide(c) for c in q_cols], axis=-1) if all(c in fc for c in q_cols) else None
        return Prediction(wide(self.alias), quantiles)


for _alias in SPECS:
    register_factory(_alias, lambda a=_alias: TimeCopilotModel(a))
