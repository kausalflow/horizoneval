# Do our numbers match the published ones?

A tool that helps you choose forecasts must compute scores exactly the way the field does. We check this by
reproducing published results with horizoneval's own scoring: the same data, the same forecasts, and the published
number at the end. Both runs are committed, with every comparison, under `results/replication-m4` and
`results/replication-gift-eval`, and their reports are published with these docs:
[M4](../examples/replication-m4.html), [GIFT-Eval](../examples/replication-gift-eval.html).

## M4: all 98 published numbers reproduced

The [M4 competition](https://github.com/Mcompetitions/M4-methods) forecast 100,000 series. Its organisers published
the forecasts every team submitted and the official sMAPE and MASE per frequency (yearly to hourly) and in total.

We scored, on all 100,000 series:

- the forecasts submitted by the three best teams (Smyl; Montero-Manso et al.; Pawlikowski et al.) and the
  organisers' Naive and Naive2 benchmarks, read from their own files;
- horizoneval's `naive` and `seasonal_naive` forecasters, run on the same histories.

Every one of the 98 published numbers is reproduced to the precision it was published with (three decimals per
frequency; five or six in total). The largest difference is 0.0005, which is rounding. Examples:

| Method | Published sMAPE | Ours | Published MASE | Ours |
|---|---|---|---|---|
| Smyl (winner) | 11.374 | 11.374 | 1.536 | 1.536 |
| Montero-Manso et al. | 11.720 | 11.720 | 1.551 | 1.551 |
| Naive2 | 13.564 | 13.564 | 1.912 | 1.912 |
| sNaive (our `seasonal_naive`) | 14.657 | 14.657 | 2.057 | 2.057 |

M4 writes sMAPE in percent, 200 |e| / (|y| + |f|): 100 times horizoneval's `smape`.

The same run also shows the decision layer at work: Smyl's method is the best choice overall with probability 1.00,
but choosing a different team's method per frequency is 2.4% better on series it was not chosen on (p < 0.001).

```bash
uv run --extra replicate horizoneval replicate m4      # downloads M4 into data/m4
```

## GIFT-Eval: seasonal naive and naive on the leaderboard

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) publishes per-configuration results for every model.
We follow its test protocol (prediction length per frequency and term, the number of rolling test windows, gluonts'
seasonal period for both the seasonal naive and MASE), run horizoneval's `seasonal_naive` and `naive`, and compare
MASE, sMAPE, MAE and MSE with the leaderboard on 18 datasets whose series have no missing values (24 dataset and
term combinations, 192 published numbers).

184 of the 192 are reproduced to within one part in a million (the data are stored as 32-bit floats). The other 8
are one documented convention: sMAPE on datasets with stretches of exact zeros (solar at night, days without COVID
deaths). When both the truth and the forecast are 0, gluonts leaves the point out of the average; horizoneval, M4
and utilsforecast count it as a perfect forecast (0). Leaving those points out reproduces the published values
exactly (for example 0.6200 for seasonal naive on solar/H/short).

Two things this replication taught us, both now part of horizoneval:

- **Series with different histories.** M4 and GIFT-Eval forecast each series from its own cutoff and with its own
  horizon per frequency. The forecast frame now accepts a different horizon per dataset and counts windows per
  series.
- **Series whose history never changed.** MASE divides by how much the history moves; for a flat history that is 0.
  horizoneval now leaves such series out of scaled averages, as GIFT-Eval does, instead of reporting an enormous
  number.

GIFT-Eval's copy of M4 yearly has 22,974 series rather than the 23,000 of the original, so its yearly numbers differ
slightly from the M4 table (MASE 3.966 against 3.974). We reproduce both, each on its own data.

```bash
uv run --extra gift horizoneval replicate gift-eval    # needs the configurations in data/gift-eval
```

Configurations with missing values (electricity, KDD Cup, and others) are not yet included: gluonts masks missing
targets, and horizoneval's frame does not accept them yet.
