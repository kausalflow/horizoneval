# Glossary

**Cutoff.** The last point in time a forecaster may see. Everything after it is hidden until scoring.

**Coverage.** The share of true values that fell inside a forecast's band. An 80% band should catch about 80%.

**CRPS.** Continuous ranked probability score: one number for how good a whole forecast distribution (its ranges,
not only its middle) is. Lower is better. horizoneval reports it scaled like MASE.

**Critical difference.** The smallest gap in mean rank at which two models can be told apart at the 5% level
(Nemenyi test).

**Dataset.** A group of series scored together, for example "hourly electricity". Ranks and the Friedman test are
computed over datasets.

**Forecast frame.** The one table horizoneval scores: `unique_id`, `ds`, `cutoff`, `y` and one column per model. The
same layout as Nixtla's `cross_validation` output. See [Forecast frame](../reference/frame.md).

**Forecaster.** Anything that turns a history into a forecast. In horizoneval, a function `f(Y, h, period)`.

**Horizon (h).** How many steps ahead are forecast.

**Leakage.** Using information from after the cutoff. A frame with a target at or before its cutoff is refused.

**MASE.** Mean absolute scaled error: the average miss divided by the typical seasonal (or one-step) change in the
series' own history. Below 1 beats that simple benchmark.

**Naive forecast.** "The future equals the last value." The default reference.

**Nats.** The unit of a natural logarithm. A skill of 1 nat means the squared error is e ≈ 2.7 times smaller.

**p-value.** How surprising the observed difference would be if the two models were truly equally good. Below 0.05 is
the usual line for "not luck".

**Period (seasonality).** Number of points in one season: 24 for hourly data with a daily cycle, 7 for daily data
with a weekly cycle, 12 for monthly data with a yearly cycle. `1` or none means no season.

**Quantile.** A forecast of the value below which the truth should fall a given share of the time. The 0.9 quantile
should be exceeded 10% of the time.

**Reference.** The forecast every other model is compared with. `naive` by default; choose your own.

**Seasonal naive.** "The future equals the same time one season ago."

**Skill.** log(reference error / model error), in nats. Positive: better than the reference.

**sMAPE.** Symmetric mean absolute percentage error: the miss as a share of the values, between 0 and 2.

**Window.** One (series, cutoff) pair: one forecast of `h` steps from one point in time.
