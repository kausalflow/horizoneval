# Run a benchmark

`horizoneval run` draws datasets, runs the forecasters you name on rolling windows, and scores them through the same
code as `horizoneval score`.

```bash
uv run horizoneval run --models seasonal_naive drift damped_drift --data synthetic --out results/my-run
```

## Choosing forecasters

`uv run horizoneval list` prints every registered name. Built in:

| Name | What it does | Needs |
|---|---|---|
| `naive` | repeats the last value | |
| `seasonal_naive` | repeats the last full season | |
| `drift` | extends the line from the first to the last value | |
| `mean` | the average of the history | |
| `damped_drift` | drift, shrunk towards naive when the slope is not clearly there (the template for new models) | |
| `auto_arima`, `auto_ets` | classical statistical models through TimeCopilot | `tsfm` extra |
| `chronos_bolt_small`, `moirai2_small`, `tirex`, `timesfm3` | pretrained foundation models through TimeCopilot | `tsfm` extra |
| `efnn`, `efnn_noent`, `efnn_mlp` | Entropic Force neural network and its ablations | `physics` extra |

`naive` and `seasonal_naive` also give uncertainty bands (from their own past errors), so they are scored on CRPS
and coverage too.

## Choosing data

| `--data` value | Data |
|---|---|
| `synthetic` | all six synthetic datasets: random walk, AR(1) differences, seasonal series with trend, and the Lorenz, Rössler and Van der Pol systems |
| `synthetic:<name>` | one of them, for example `synthetic:lorenz` |
| `gift-small` | five GIFT-Eval configurations (electricity, solar, M4 hourly and daily, Jena weather); needs the `gift` extra and a download |
| `gift:<dataset>/<freq>` | any GIFT-Eval configuration, for example `gift:electricity/H` |

To download a GIFT-Eval configuration:

```bash
uv sync --extra gift
uv run hf download Salesforce/GiftEval --repo-type=dataset --local-dir data/gift-eval --include "electricity/H/*"
```

## Choosing the protocol

| Option | Default | Meaning |
|---|---|---|
| `--h` | 12 | steps forecast at each cutoff |
| `--windows` | 5 | cutoffs per series |
| `--step` | `h` | distance between cutoffs |
| `--context` | 256 | history length before the first cutoff |
| `--n-series` | 30 | series drawn per dataset |
| `--seed` | 0 | random seed for drawing data; same seed, same data |
| `--reference` | `naive` | model the skill column and paired tests compare with |

## Keeping results

A run writes a folder of CSVs, the forecasts and `report.html` ([what each file holds](../reference/outputs.md)).
Commit the folder under `results/` to keep it: a comparison claim in this project needs its run folder committed and
its reference stated. `uv run horizoneval index` builds `results/index.html` linking all runs.
