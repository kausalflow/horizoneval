# Do our numbers match the published ones?

A tool that helps you choose forecasts must compute scores exactly the way the field does. We check this by
reproducing published results with horizoneval's own scoring: the same data, the same forecasts, and the published
number at the end. Both runs are committed, with every comparison, under `results/replication-m4` and
`results/replication-gift-eval`, and their reports are published with these docs:
the [M4 report](../examples/replication-m4.html), and GIFT-Eval's comparison table in
`results/replication-gift-eval/replication.csv`.

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

## GIFT-Eval: all 400 leaderboard numbers reproduced

[GIFT-Eval](https://huggingface.co/spaces/Salesforce/GIFT-Eval) publishes per-configuration results for every model.
We follow its test protocol (prediction length per frequency and term, the number of rolling test windows, gluonts'
seasonal period for both the seasonal naive and MASE, gluonts' filling of missing history before forecasting), run
horizoneval's `seasonal_naive` and `naive`, and compare MASE, sMAPE, MAE and MSE with the leaderboard on 34 datasets,
9 of them with missing values (electricity, KDD Cup 2018, car parts, restaurant, hierarchical sales, bitbrains): 50
dataset and term combinations, 400 published numbers.

All 400 are reproduced to within one part in a million (the data are stored as 32-bit floats). Two of GIFT-Eval's
conventions differ from M4's, and horizoneval offers both:

- **Missing truth.** Steps without a true value are not scored, and averages weight every observed step once.
- **sMAPE when truth and forecast are both 0** (solar at night, days without COVID deaths). gluonts leaves such steps
  out; M4 and utilsforecast count them as perfect. horizoneval's default is M4's; `smape_zero="skip"` gives
  GIFT-Eval's, which the replication uses. With the default, the 8 sMAPE values on solar and COVID deaths differ,
  as expected (for example 0.258 instead of 0.620 for seasonal naive on solar/H/short).

Two things this replication taught us, both now part of horizoneval:

- **Series with different histories.** M4 and GIFT-Eval forecast each series from its own cutoff and with its own
  horizon per frequency. The forecast frame now accepts a different horizon per dataset and counts windows per
  series.
- **Missing values.** Frames with gaps in the truth or the history are scored, with the conventions above.
- **Series whose history never changed.** MASE divides by how much the history moves; for a flat history that is 0.
  horizoneval now leaves such series out of scaled averages, as GIFT-Eval does, instead of reporting an enormous
  number.

GIFT-Eval's copy of M4 yearly has 22,974 series rather than the 23,000 of the original, so its yearly numbers differ
slightly from the M4 table (MASE 3.966 against 3.974). We reproduce both, each on its own data.

```bash
uv run --extra gift horizoneval replicate gift-eval    # needs the configurations in data/gift-eval
```
