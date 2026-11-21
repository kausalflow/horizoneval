# Reading a report

Every run writes `report.html`: one file, no internet needed, light and dark mode. Open it in a browser or publish
the run folder on any static host. Here is what each section tells you, top to bottom. An
[example report](../examples/example-synthetic-baselines.html) is published with these docs.

## Header and failures

The run name, when it was made and its settings. If any model failed (crashed, returned the wrong shape, or produced
missing values), a red line names it and gives the reason. The other models are still scored.

## Leaderboard

One row per model, ordered by **mean rank** over datasets (1 is best). The columns:

| Column | Meaning |
|---|---|
| `mean_rank` | average place by MASE across datasets; lower is better |
| `mean_mase` | average scaled error; below 1 beats the in-sample naive benchmark |
| `mean_skill` | improvement over the reference in nats; 0 is equal, positive is better (green), negative worse (red) |
| `mean_crps_scaled` | quality of the forecast range, scaled like MASE; empty for models without ranges |
| `mean_coverage80` | how often the truth fell inside the 80% band; should be near 0.80 |
| `datasets` | how many datasets the model was scored on |
| `seconds` | time the model took |

Under the table: the Friedman p-value and the **critical difference**. If two models' mean ranks differ by less than
the critical difference, the data cannot tell them apart at the 5% level.

!!! tip "When rank and average disagree"
    Rank counts how often a model wins; the average counts by how much. A model that wins narrowly on most datasets
    and fails badly on one can have the best rank and a poor average. Look at both, then at the per-dataset tables.

## Per dataset

One table per score (MASE, and scaled CRPS when there are ranges). For each dataset and model:

- `mean`: the average score over series,
- `ci`: the 95% bootstrap interval of that average,
- `t_vs_ref`, `p_vs_ref`: the paired test against the reference. Negative t (green) means lower error than the
  reference; p below 0.05 means the difference is unlikely to be luck.

## Error by forecast step

One small chart per dataset: scaled error at each step ahead, one line per model. Lines that rise steeply belong to
models that are only good for the near future.

## All scores

The per-dataset averages of every score, with the rank of each model within the dataset. This is `summary.csv`.

## Reproduce

A fingerprint (sha256) of each dataset as it was drawn. If you rerun with the same settings and get the same
fingerprints, you scored the same data. All settings are in `config.json` next to the report.
