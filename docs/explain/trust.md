# Can you trust a comparison?

Most wrong forecast comparisons go wrong before or around the scores, not in them. horizoneval checks three things
that other tools leave to you.

## Before scoring: pre-flight checks

Every evaluation first looks at the raw forecasts. The findings appear at the top of the report under **Before you
trust these numbers**, in `checks.json`, and in `ev.checks`. Run them alone with:

```bash
horizoneval check cv.parquet --train train.parquet     # exits with 1 if there is an error
```

| Check | What it catches |
|---|---|
| `copies_truth` (error) | a model equals the truth on most steps: it has seen the values it forecasts |
| `truth_disagrees_with_history` (error) | the forecast table and the history give different values for the same date |
| `missing_forecasts` (error) | a model has gaps, so it cannot be compared on the same steps |
| `units` (warning) | forecasts typically 10 times larger or smaller than the truth: thousands, logs, a transform not undone |
| `quantile_crossing` (warning) | a lower quantile above a higher one |
| `point_outside_quantiles` (warning) | the point forecast outside the model's own range: columns from different models? |
| `duplicate_model` (warning) | two columns with identical forecasts |
| `constant` (warning) | the same value everywhere |
| `irregular_steps` (warning) | forecast steps not one regular time step apart |
| `missing_truth`, `few_series`, `history_beyond_cutoff` (notes) | context that changes how to read the results |

The checks never change a number. Leakage (a target on or before its cutoff) is refused outright.

## When claiming a winner: verified significance

"A is significantly better than B" usually rests on a paired test that treats every series as independent. Series
rarely are: a holiday, a price change or a heat wave hits all of them at the same cutoff. Then the usual test claims
winners that do not exist.

For the best model and the runner-up, horizoneval makes the two equally good on your own data (by centring their
differences, which keeps every dependence in the data), simulates many such cases, and counts how often the usual
test would still claim a winner. It also computes a p value that holds when series or windows move together
(two-way clustered by series and by window). The verdict says which:

> m4_118_Smyl beats m4_245_MonteroManso (p = 0.0001, confirmed by simulating equally good forecasters on your data).

> The usual test says A beats B (p = 0.01), but on your data that test raises false alarms in 64% of simulations of
> equally good forecasters; the difference is not confirmed (p = 0.31 once that is accounted for).

`horizoneval.verify.claims` checks the check: with a shock shared by all series in each window, the usual test
claims a winner in about 77% of simulations of equally good forecasters; horizoneval's p value does so in about 5%.

## When two runs disagree: the settings fingerprint and diff

Every run records, in `config.json` under `settings`, everything that can change a score: per dataset the series,
cutoffs, horizon, number of windows, seasonal period, a fingerprint of the truth and of the history; per model its
quantile levels and a fingerprint of its forecasts; the reference, the deciding metric, the cost, the sMAPE
convention, the scoring conventions and the software versions. `settings_sha256_16` fingerprints all of it.

```bash
horizoneval diff results/run-a results/run-b
```

names each setting that differs in plain language ("the seasonal period behind MASE differs", "the forecasts of
model X differ") and lists the scores that moved. When scores moved but every setting is the same, it says to report
a bug.

## Conventions you can choose

| Setting | Options | Default |
|---|---|---|
| `smape_zero` | `"zero"`: a step where truth and forecast are both 0 is a perfect forecast (M4, utilsforecast); `"skip"`: leave it out (gluonts, GIFT-Eval) | `"zero"` |

Missing truth is never scored, and averages weight every observed step once, as GIFT-Eval does.
