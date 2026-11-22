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
| `--smape-zero` | `zero` | sMAPE where truth and forecast are both 0: `zero` counts a perfect step, `skip` leaves it out |
| `--json` | off | print the result document instead of a table |
| `--segment` | none | column of `--series` whose groups may each deserve their own model |
| `--ladder` | off | split each model's skill into a simple rule, a linear model fitted on the history and what is left beyond linear ([explained here](../explain/skill-ladder.md)); needs `--train` |

## `horizoneval explain RUN`

Prints what an AI assistant needs to explain a run folder to a non-forecaster (the `explain` tool): the answer, how
to read each table, caveats, why to trust it, and every quotable number. With `--check draft.txt`, prints the numbers
in a draft explanation that the result does not support (the `check_explanation` tool).

## `horizoneval replicate BENCHMARK`

Reproduce published results with horizoneval's scoring and write the comparison to `replication.csv`. See
[Do our numbers match the published ones?](../explain/replication.md).

| Option | Default | Meaning |
|---|---|---|
| `BENCHMARK` | required | `m4` (needs `--extra replicate`; downloads the data), `gift-eval` (needs `--extra gift` and the data in `data/gift-eval`) or `ladder` (the skill ladder's calibration on simulated series; no data needed) |
| `--out` | `results/replication-<benchmark>` | results folder |

## `horizoneval check FORECASTS [--train TRAIN]`

Pre-flight checks on a forecast frame without scoring it; exits with 1 when a check finds an error. See
[Can you trust a comparison?](../explain/trust.md).

## `horizoneval diff RUN_A RUN_B`

Every setting that differs between two run folders, in plain language, and the scores it moved.

## `horizoneval report RUN_DIR`

Rebuild `report.html` of a run folder from its CSVs.

## `horizoneval index [RESULTS_DIR]`

Build `index.html` linking every run in `RESULTS_DIR` (default `results`).
