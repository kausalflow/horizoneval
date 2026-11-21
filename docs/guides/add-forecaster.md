# Add a forecaster

A forecaster is one function. Copy `src/horizoneval/models/damped_drift.py` to a new file in
`src/horizoneval/models/`; every module there is picked up automatically.

## The contract

```python
import numpy as np
from horizoneval.registry import forecaster


@forecaster("my_model")
def my_model(Y: np.ndarray, h: int, period: int | None = None) -> np.ndarray:
    # Y: (N, n) histories, one series per row
    # h: steps to forecast
    # period: samples per season, or None
    return F  # (N, h) point forecasts
```

- **Vectorize over series.** Rows are independent series; work on the whole array with numpy rather than looping.
- **Fit inside the call.** The function sees only the history before the cutoff, which is what keeps the evaluation
  honest.
- **Return ranges if you can.** Return a [`Prediction`](../reference/api.md#horizoneval.prediction.Prediction)
  with `point` (N, h) and `quantiles` (N, h, Q) at the levels `horizoneval.prediction.QUANTILES`
  (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95) and the model is also scored on CRPS and coverage.

## The template

```python
--8<-- "src/horizoneval/models/damped_drift.py"
```

## Models with state

For anything trained, cached or loaded from a checkpoint, decorate a class instead. It is built with no arguments,
and its instances are called like the function above:

```python
@forecaster("my_net")
class MyNet:
    def __init__(self):
        self.model = load_weights()   # runs only when a benchmark asks for "my_net"

    def __call__(self, Y, h, period=None):
        return self.model.predict(Y, h)
```

## Test it

```bash
uv run horizoneval run --models naive seasonal_naive my_model --data synthetic --out results/my-model-first-look
```

A model that crashes, returns the wrong shape or produces non-finite values is listed under `failures` and the run
continues, so a broken model never hides the others. Compare against the baselines first: a new forecaster should
beat `naive` and `seasonal_naive` before anything else. Never edit the baselines to make a model look better.
