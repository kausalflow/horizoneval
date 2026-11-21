# Score forecasts from Nixtla or TimeCopilot

You do not need horizoneval's forecasters to use its evaluation. If a tool can produce rolling-window forecasts,
horizoneval can score them. The output of Nixtla's and TimeCopilot's `cross_validation` is already in the right
layout.

## From statsforecast

```python
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, SeasonalNaive
from horizoneval import evaluate

train_df = pd.read_parquet("train.parquet")             # unique_id, ds, y
sf = StatsForecast(models=[AutoETS(season_length=24), SeasonalNaive(season_length=24)], freq="h")
cv = sf.cross_validation(df=train_df, h=24, n_windows=5, level=[80, 90])

ev = evaluate(cv, train=train_df, period=24, reference="SeasonalNaive")
print(ev.leaderboard)
ev.write("results/statsforecast-hourly")                # CSVs, forecasts.parquet, report.html
```

`level=[80, 90]` adds `AutoETS-lo-80`, `AutoETS-hi-80` and so on; horizoneval reads them as quantiles and scores
CRPS and coverage. Without `level`, only the point scores are computed.

## From TimeCopilot

```python
from timecopilot import TimeCopilotForecaster

tcf = TimeCopilotForecaster(models=[...])
cv = tcf.cross_validation(df=train_df, h=24, n_windows=5, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
ev = evaluate(cv, train=train_df, period=24, reference="SeasonalNaive")
```

TimeCopilot's `-q-<p>` quantile columns are read directly.

## From anything else

Write a table with the columns of the [forecast frame](../reference/frame.md) (`unique_id`, `ds`, `cutoff`, `y`,
one column per model) to parquet or CSV, and score it from the command line:

```bash
uv run horizoneval score cv.parquet --train train.parquet --period 24 --reference SeasonalNaive --out results/my-run
```

## What the arguments do

| Argument | Why it matters |
|---|---|
| `train` | The history, in the same long format (`unique_id`, `ds`, `y`). MASE and scaled CRPS divide by how much each series moves in its own history, so they need it; for each window only rows with `ds <= cutoff` are used. Without it those two scores are empty. |
| `period` | The seasonal period (24 for hourly data with a daily cycle, 7 for daily, 12 for monthly). It sets the scale of MASE. Leave it out for non-seasonal data. |
| `reference` | The model column every other model is compared with (skill and paired tests). Use the forecast your team relies on today, or a seasonal naive. |
| `series` | A table (`unique_id`, `dataset`, `period`) when one frame mixes datasets or seasonal periods. Ranks and the Friedman test are computed over datasets. |
| `dataset` | The name of the single dataset when `series` is not given. |

## Run horizoneval's scores inside Nixtla's evaluation

If your pipeline already uses `utilsforecast.evaluation.evaluate`, the same scores are available as losses:

```python
from utilsforecast.evaluation import evaluate as uf_evaluate
from horizoneval import nixtla

uf_evaluate(cv, metrics=[nixtla.mase(24), nixtla.smape(), nixtla.coverage80()], train_df=train_df)
```

They are computed by the same code as `horizoneval.evaluate`. utilsforecast's `evaluate` reads intervals only as
`-lo-`/`-hi-` columns; `horizoneval.frame.to_intervals` renames TimeCopilot's `-q-` columns to that form.

## Checks before scoring

The frame is validated first. A target at or before its cutoff (leakage), a duplicated row, or windows of different
lengths stop the run with a [`FrameError`](../reference/api.md#horizoneval.frame.FrameError) that says what to fix.
A model with missing or non-finite forecasts is dropped and listed under `failures` in `config.json`; the remaining
models are still scored.
