# Outputs of a run

`horizoneval run`, `horizoneval score` and `Evaluation.write` all write the same folder.

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
| `decision.json` | which model to choose: the verdict, probability of best, winner's curse, segments, cost disagreements |
| `confidence.csv` | probability of best and expected loss per model and dataset |
| `config.json` | settings, failures, and fingerprints of the forecasts, the history and the protocol |
| `report.html` | everything above as one static page, light and dark |

Commit a run folder to keep it; `uv run horizoneval index` builds `results/index.html` linking them all.
