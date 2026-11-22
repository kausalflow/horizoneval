# horizoneval

`horizoneval` evaluates time series forecasts from any tool (horizoneval's own forecasters, TimeCopilot, Nixtla's
statsforecast / neuralforecast / mlforecast) with formal, diverse and reproducible evaluations, and is the place to
invent forecasters and test them against baselines. Every run writes plain CSVs, the forecasts themselves and a
self-contained static HTML report.

**Docs:** https://kausalflow.com/horizoneval/ (quickstart, plain-language guide to every score, API reference).
Build them locally with `uv sync --group docs && uv run zensical serve`.

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

## Evaluate forecasts made anywhere

Every forecast becomes one long table in the layout of Nixtla's `cross_validation` output, the *forecast frame*:
`unique_id`, `ds`, `cutoff`, `y` and one column per model, with optional quantiles as `<model>-q-<p>` (TimeCopilot,
`p` in percent or as a fraction), `<model>-lo-<l>` / `<model>-hi-<l>` (statsforecast) or `<model>-median`. Scaled
scores (MASE, scaled CRPS) also need the history in long format (`unique_id`, `ds`, `y`); for each window only the
rows with `ds <= cutoff` are used.

```python
from horizoneval import evaluate

cv = sf.cross_validation(df=train_df, h=24, n_windows=5)          # or TimeCopilotForecaster(...).cross_validation(...)
ev = evaluate(cv, train=train_df, period=24, reference="SeasonalNaive")
ev.leaderboard; ev.tests; ev.write("results/my-run")              # same files and report as `horizoneval run`
```

```bash
uv run horizoneval score cv.parquet --train train.parquet --period 24 --reference SeasonalNaive --out results/my-run
```

A frame that mixes datasets or periods takes a `series` table (`unique_id`, `dataset`, `period`). The frame is
checked before scoring: a target at or before its cutoff (leakage), a duplicated row or windows of different lengths
stop the run with the reason; a model with missing or non-finite forecasts is dropped and listed under `failures`.
`horizoneval run` builds the same frame from its own forecasters and scores it through the same code, so a number
never depends on where the forecast was made.

The scores also run inside Nixtla's own evaluation, as utilsforecast losses:

```python
from utilsforecast.evaluation import evaluate as uf_evaluate
from horizoneval import nixtla

uf_evaluate(cv, metrics=[nixtla.mase(24), nixtla.smape(), nixtla.coverage80()], train_df=train_df)
```

utilsforecast's `evaluate` reads intervals only as `-lo-`/`-hi-` columns; `horizoneval.frame.to_intervals` renames
TimeCopilot's `-q-` columns to that form.

## Choose a forecast

Every evaluation ends with a decision, not only a ranking (`ev.decision`, `decision.json`, the top of the report):
the probability that each model is the best one for your series and the expected loss of picking it; the winner's
curse, i.e. how the chosen model scores on windows or series it was not chosen on; whether a model per segment beats one
model for everything on unseen data (`segment="region"`); and, with `cost="linlin:0.9"` or any cost you write, whether
your cost picks a different model than MASE or CRPS. These statistics are checked by simulation in
`horizoneval.verify`. See [Choosing a forecast](docs/explain/choosing.md).

## Our numbers match the published ones

`horizoneval replicate m4` scores all 100,000 M4 series, from the teams' own submitted forecasts and from our
baselines, and reproduces all 98 published sMAPE and MASE values of the official table. `horizoneval replicate
gift-eval` does the same for the GIFT-Eval leaderboard's seasonal naive and naive: all 400 numbers on 34 datasets, including
the ones with missing values, within one part in a million. Runs: `results/replication-m4`,
`results/replication-gift-eval`; details in [docs/explain/replication.md](docs/explain/replication.md).

## Use from AI assistants

One tool registry (`horizoneval.tools`) behind every front door: `horizoneval-mcp` (MCP server, `mcp` extra), the
Agent Skill in `skills/horizoneval/`, `--json` on the command line, and function-calling definitions
(`tools.definitions("anthropic" | "openai")`). All return the same versioned `result.json` document; a test checks
they are byte-identical. See [docs/guides/ai-assistants.md](docs/guides/ai-assistants.md).

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
| `forecasts.parquet` | the forecast frame: every forecast of every model, re-scorable with `horizoneval score` |
| `train.parquet` | the history the scaled scores used |
| `series.csv` | dataset and period of each series |
| `scores.csv` | every score per model, dataset, window (cutoff) and series |
| `by_horizon.csv` | scaled error by forecast step |
| `summary.csv` | means per model and dataset, with rank |
| `leaderboard.csv` | means over datasets, sorted by mean rank |
| `tests.csv` | bootstrap intervals and paired tests per model, dataset and metric |
| `ranks.json` | Friedman test and Nemenyi critical difference |
| `decision.json`, `confidence.csv` | which model to choose, probability of best, expected loss, winner's curse, segments |
| `config.json` | settings, dataset fingerprints, failures |
| `report.html` | everything above as one static page, light and dark |

Commit a run folder to keep it; `results/index.html` links them all.

## License

Apache License 2.0; see [LICENSE](LICENSE).
