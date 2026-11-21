# forecaster-lab

A place to invent forecasters and test them against baselines with formal, diverse and reproducible evaluations.
Every run writes plain CSVs and a self-contained static HTML report.

```bash
uv sync                       # core: numpy, pandas, scipy
uv run horizoneval list              # forecasters, datasets, suites
uv run horizoneval run --models seasonal_naive drift damped_drift --data synthetic --out results/my-run
open results/my-run/report.html
uv run horizoneval index             # results/index.html over all runs
```

Foundation models and statistical references (Chronos-Bolt, Moirai 2, TiRex, TimesFM 3, AutoARIMA, AutoETS) come
through [TimeCopilot](https://github.com/TimeCopilot/timecopilot), pinned to the same version as
`extrapolation-as-forecasts`:

```bash
uv sync --extra tsfm
uv run horizoneval run --models chronos_bolt_small tirex auto_ets damped_drift --data synthetic
```

## Add a forecaster

Copy `src/horizoneval/models/damped_drift.py` to a new file in `src/horizoneval/models/`. Every module there is picked up
automatically. The contract:

```python
@forecaster("my_model")
def my_model(Y, h, period=None):   # Y: (N, n) histories, one series per row; period: samples per season or None
    return F                       # (N, h) point forecasts, or Prediction(point, quantiles) for probabilistic ones
```

A class works too (`@forecaster("x")` on a class built with no arguments whose instances are callable), which is the
place for anything trained or cached. Quantiles are at the fixed levels `horizoneval.prediction.QUANTILES`.

## Physics-inspired forecasters

`src/horizoneval/models/entropic_force.py` ports the Entropic Force Neural Network from
[kausalflow/physics_models](https://github.com/kausalflow/physics_models) (temperature-scaled activation
T·tanh(x/T) and an entropy bonus on the last hidden layer) into a direct multi-horizon forecaster trained on scaled
sliding windows. Three variants reproduce the paper's ablation: `efnn`, `efnn_noent` (no entropy term) and `efnn_mlp`
(ReLU, no entropy term). They need torch:

```bash
uv sync --extra physics
uv run horizoneval run --models seasonal_naive efnn efnn_noent efnn_mlp --data synthetic
```

First results: `results/physics-efnn-synthetic/` (see its `NOTES.md`).

## Evaluation

| | |
|---|---|
| Protocol | Rolling origin: `--windows` cutoffs every `--step` points, each forecasting `--h` steps from `--context`+ points of history |
| Point scores | MASE, sMAPE, MAE, MSE per series and window; MASE by forecast step |
| Probabilistic | CRPS (quantile grid, scaled like MASE), 80% and 90% interval coverage, 80% interval width |
| Skill | mean log(MSE<sub>ref</sub> / MSE<sub>model</sub>) in nats against `--reference` (default `naive`) |
| Uncertainty | 95% bootstrap intervals over series |
| Tests | Paired test against the reference on per-series losses (Diebold–Mariano style), Friedman test over datasets, Nemenyi critical difference of mean ranks |
| Reproducibility | Datasets drawn once from `--seed`; `config.json` records every setting and a sha256 fingerprint of each dataset |

A model that crashes or returns the wrong shape or non-finite values is recorded in `config.json` under `failures`
and the run continues.

## Data

- `synthetic`: random walk, AR(1) differences, seasonal series with trend, and the Lorenz, Rössler and Van der Pol
  systems sampled at 12 points per period (`src/horizoneval/data/synthetic.py`).
- `gift-small`: five GIFT-Eval configurations. Install with `uv sync --extra gift` and download each into
  `data/gift-eval/` with one `--include` per configuration (see `src/horizoneval/data/gift.py`). Any other configuration
  works as `gift:<dataset>/<freq>`.

Add a suite in `src/horizoneval/data/__init__.py`.

## Outputs of a run

| File | Contents |
|---|---|
| `scores.csv` | every score per model, dataset, window and series |
| `by_horizon.csv` | scaled error by forecast step |
| `summary.csv` | means per model and dataset, with rank |
| `leaderboard.csv` | means over datasets, sorted by mean rank |
| `tests.csv` | bootstrap intervals and paired tests per model, dataset and metric |
| `ranks.json` | Friedman test and Nemenyi critical difference |
| `config.json` | settings, dataset fingerprints, failures |
| `report.html` | everything above as one static page, light and dark |

Commit a run folder to keep it; `results/index.html` links them all.
