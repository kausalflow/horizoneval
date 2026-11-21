# Python API

The public entry points are importable from the package root:

```python
from horizoneval import evaluate, Evaluation, FrameError, forecaster, available, load
```

## Scoring

::: horizoneval.evaluate.evaluate

::: horizoneval.evaluate.Evaluation

## Choosing a forecast

::: horizoneval.choose

::: horizoneval.verify

## Forecast frame

::: horizoneval.frame
    options:
      members: [FrameError, validate, models_of, quantile_columns, read, to_intervals, windows, Windows]

## Scores

::: horizoneval.metrics

## Formal comparisons

::: horizoneval.stats

## Nixtla losses

::: horizoneval.nixtla

## Forecasters

::: horizoneval.registry
    options:
      members: [forecaster, available, load, register_factory]

::: horizoneval.prediction
    options:
      members: [QUANTILES, Prediction, as_prediction]

::: horizoneval.baselines
