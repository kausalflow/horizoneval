"""horizoneval: evaluate time series forecasts from any tool, and invent forecasters to evaluate.

Score a forecast frame (Nixtla / TimeCopilot cross_validation layout: unique_id, ds, cutoff, y, one column per model):

    from horizoneval import evaluate
    ev = evaluate(cv_df, train=train_df, period=24, reference="SeasonalNaive")

A forecaster is any callable  f(Y, h, period) -> F  with Y of shape (N, n) and F of shape (N, h).
Register it with @forecaster("name") in a module under horizoneval/models/ and it shows up in the benchmark.
"""

from horizoneval.evaluate import Evaluation, evaluate
from horizoneval.frame import FrameError
from horizoneval.registry import available, forecaster, load

__all__ = ["Evaluation", "FrameError", "available", "evaluate", "forecaster", "load"]
