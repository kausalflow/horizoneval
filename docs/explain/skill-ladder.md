# Where does a model's skill come from?

A forecast that beats "repeat the last value" is not yet a reason to use an expensive model. Much of the gain is
often available from a simple rule (follow the trend, repeat last week), and much of the rest from a plain linear
model fitted on the history. The skill ladder tells you how much of a model's gain is which, and whether anything
is left that a linear model cannot reach.

It comes from L. Ma, *Naive forecasters: an operator theory of what a forecast trusts* (code and checks in
[extrapolation-as-forecasts](https://github.com/emptymalei/extrapolation-as-forecasts), `paper_naive/`).

## Ask for it

```bash
horizoneval score cv.parquet --train train.parquet --period 24 --ladder --out results/my-run
```

or `evaluate(forecasts, train, period=24, ladder=True)` in Python, or `"ladder": true` in the `score` tool. It needs
the history: horizoneval builds the rungs itself from it.

## The three rungs

Skill is measured in **nats** over repeating the last value: `log(MSE of repeating / MSE of the model)`. A gain of
`x` nats is a squared error lower by `1 − e^(−x)`: 0.1 nats is about 10% lower, 0.7 nats about 50%, 2.3 nats 90%.
The gains add up along the ladder:

| Rung | What it is | Built by horizoneval as |
|---|---|---|
| **F**, free | the best simple rule | per series, the simplest of naive, drift, last change, mean, seasonal naive and seasonal drift within one standard error of the best, on up to 24 windows inside the history |
| **L**, linear | what a linear model adds on top | per forecast step, the best of the chosen rule and least-squares autoregressions on the differences (4 to 96 lags), on the seasonal differences and on seasonal lag sets, chosen on the same windows |
| **beyond linear** | what your model adds on top of L | `log(MSE of the linear model / MSE of your model)` |

Everything is chosen and fitted on history before each series' first cutoff, so nothing is tuned on the windows
being scored. The rungs are added to the forecast table as the columns `ladder_naive`, `ladder_classic` and
`ladder_linear` and scored by the same `evaluate` as every other model.

## How to read the table

Each row is one dataset and one **group** of forecast steps: `1` is one step ahead, `2-4` and `5+` further ahead.
The row `all` pools squared errors over every step, so it is dominated by the far steps, where errors are largest;
it is not the average of the other rows.

- `F`, `L`, `F_plus_L` describe your **data**: how much a simple rule, and then a linear model, beat repeating the
  last value. `*_pct` gives the same number as a percentage of squared error removed.
- `total` and `beyond_linear` (second table) belong to each **model**.
- `*_se` is the standard error over series. A value within about two standard errors of 0 is not distinguishable
  from 0; the 95% interval is the value ± 1.96 standard errors.
- `saturated` says whether more memory had stopped helping the linear model on the history. If not, L is a lower
  bound and no model can be credited with skill beyond linear on that dataset.

Big numbers are not errors. 7.47 nats at steps 2–4 on van der Pol means the simple rule (seasonal naive, since the
oscillation repeats) has a squared error e^7.47 ≈ 1,750 times smaller than repeating the last value: the series is
a nearly noise-free cycle. The run flags such datasets as `near_perfect`: there the gain a linear model can reach
has no finite bound (it is set by the noise floor), so read it as "almost perfectly forecastable", not as a precise
amount, and compare models by ratios, since all percentages are close to 100%.

## Can you trust it?

Four things back the numbers, and the run reports each of them in `ladder.flags` and in the report:

1. **One scoring path.** The rungs are forecast columns scored by `horizoneval.evaluate`, the function that
   reproduces all 98 M4 and all 400 GIFT-Eval published numbers.
2. **A known answer.** On the paper's calibration case the linear rung recovers the true value and the exact optimal
   model is never credited (below).
3. **An independent check on every run.** At one step ahead, F and F + L can be read from the history alone,
   without forecasting the scored windows (variance ratios and the spectrum, the paper's Result 4). `check_F_h1`
   and `check_FL_h1` compare them with the measured values. On the synthetic example all six datasets agree on F,
   van der Pol included (5.819 measured against 5.773 from the history).
4. **Sanity flags** that say when to read with care: `near_perfect`, `not_saturated`, `few_series`,
   `negative_F` / `negative_L` (a rule chosen on the history did worse on the scored period, so the two differ),
   and `partial_beyond_linear`.

## When does "beyond linear" count?

A fitted linear model is a little worse than the best possible linear model, so even a model that is exactly
optimal, and linear, shows a small positive gain over it. The paper's decision rule therefore credits a model with
skill beyond linear only if, **in every horizon group** (step 1, steps 2–4, steps 5 and later):

1. the linear model is **saturated**: giving the autoregressions twice the memory improves them by less than 0.1
   nats, so the reference is not simply too short;
2. the gain beyond linear exceeds **twice its standard error** over series;
3. it exceeds the **offset** a fitted linear model leaves against the exact optimum: 0.01, 0.03 and 0.05 nats for
   the three groups.

A gain at or below zero is conclusive the other way: the model does **no better than linear**. Anything in between
is reported as **not shown**. The rule can prove that a model achieves no more than a linear one; it can bound, but
not certify, that it achieves more.

## How big could the linear gain be? The price of naivety

At one step ahead, the paper gives values read from the history alone, without fitting any forecaster
(`ladder.datasets[...].available_h1`):

- `F`: `log Var(Δy) / Var(u)`, where `u` is what the chosen simple rule fails to predict;
- `L_ar` and `L_welch`: the **price of naivety**, `log A/G`, the distance of the spectrum of `u` from flat. A simple
  rule is optimal only when `u` is white noise; the more structure `u` has, the more a linear model can gain. The
  long-autoregression estimate is biased up by its in-sample optimism (`L_ar_optimism`, about `p/(n−p)` nats), the
  periodogram estimate biased down, so they usually bracket the truth.

These are values to compare with, not ceilings: on nearly deterministic series (smooth physical systems) the linear
gain has no finite bound.

## Checked against the paper

`horizoneval replicate ladder` runs the paper's calibration case, series whose differences are AR(1) with
coefficient 0.5, with the exact optimal forecaster scored as a model, over 12 seeds of 30 series
(`results/replication-ladder`):

| | horizoneval | paper | truth |
|---|---|---|---|
| L at step 1 | 0.298 ± 0.012 | 0.309 ± 0.078 | 0.288 |
| price of naivety, long AR | 0.417 | 0.419 | biased up |
| price of naivety, periodogram | 0.255 | 0.256 | biased down |
| offset at steps 1 / 2–4 / 5+ | 0.001 / 0.014 / 0.033 | 0.008 / 0.027 / 0.046 | |
| exact model credited beyond linear | 0 of 12 seeds | | never |

Our offsets are below the paper's, so the paper's thresholds are conservative here. The tests
(`tests/test_ladder.py`) add a random walk, where every rung is zero, and the logistic map, whose spectrum is nearly
flat so no linear model helps, while its own map forecast is credited with skill beyond linear in every group.

An example on the synthetic suite: [results/ladder-synthetic](../examples/ladder-synthetic.html).
