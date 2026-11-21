# Command line

All commands run through uv: `uv run horizoneval <command>`.

## `horizoneval list`

Prints the registered forecasters, the synthetic datasets and the dataset suites.

## `horizoneval run`

Benchmark forecasters on rolling windows. See [Run a benchmark](../guides/benchmark.md).

| Option | Default | Meaning |
|---|---|---|
| `--models NAME ...` | required | forecasters to run |
| `--data SPEC ...` | `synthetic` | suites, `synthetic:<name>` or `gift:<dataset>/<freq>` |
| `--h` | 12 | forecast horizon |
| `--windows` | 5 | rolling windows per series |
| `--step` | `h` | distance between cutoffs |
| `--context` | 256 | history length before the first cutoff |
| `--n-series` | 30 | series per dataset |
| `--seed` | 0 | seed for drawing data |
| `--reference` | `naive` | model the skill column is measured against |
| `--out` | `results/<timestamp>` | results folder |

## `horizoneval score FORECASTS`

Score a [forecast frame](frame.md) made by any tool. See
[Score forecasts from Nixtla or TimeCopilot](../guides/score-forecasts.md).

| Option | Default | Meaning |
|---|---|---|
| `FORECASTS` | required | parquet or CSV with `unique_id`, `ds`, `cutoff`, `y` and one column per model |
| `--train` | none | history (`unique_id`, `ds`, `y`), needed for MASE and scaled CRPS |
| `--series` | none | table (`unique_id`, `dataset`, `period`) |
| `--period` | 1 | seasonal period of every series |
| `--dataset` | `data` | dataset name when `--series` gives none |
| `--reference` | none | model column for skill and paired tests |
| `--out` | `results/<timestamp>` | results folder |
| `--cost` | none | your own cost (`linlin:0.9`, `pinball:0.95`); it decides the choice. See [Choosing a forecast](../explain/choosing.md) |
| `--segment` | none | column of `--series` whose groups may each deserve their own model |

## `horizoneval report RUN_DIR`

Rebuild `report.html` of a run folder from its CSVs.

## `horizoneval index [RESULTS_DIR]`

Build `index.html` linking every run in `RESULTS_DIR` (default `results`).
