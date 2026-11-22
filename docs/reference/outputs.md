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
| `ladder.json`, `ladder_rungs.csv`, `ladder_models.csv` | with `--ladder`: each model's skill split into a simple rule (F), a linear model (L) and beyond linear, with the verdict of the decision rule |
| `explanation.json` | the answer in plain sentences, how to read each table, caveats, why to trust it, and every quotable number with its percentage and 95% interval (`horizoneval explain`) |
| `decision.json` | which model to choose: the verdict, probability of best, winner's curse, segments, cost disagreements |
| `result.json` | the versioned result document every entry point returns (see [Use from an AI assistant](../guides/ai-assistants.md)) |
| `checks.json` | pre-flight findings on the raw forecasts |
| `confidence.csv` | probability of best and expected loss per model and dataset |
| `config.json` | settings, failures, fingerprints, and `settings`: everything that can change a score (compare runs with `horizoneval diff`) |
| `report.html` | everything above as one static page, light and dark |

Commit a run folder to keep it; `uv run horizoneval index` builds `results/index.html` linking them all.
