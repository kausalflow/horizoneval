# Choosing a forecast

A leaderboard tells you which model scored best on the past. What you need to know is which model to use from now
on, and how sure you can be. Every evaluation in horizoneval ends with a short answer to that, at the top of the
report under **What to choose**, in `decision.json`, and in `ev.decision` in Python:

> Choose B. By MASE it is the best model with probability 0.82; counting the cases where another model is better,
> picking it costs on average 0.4% more than picking the best would. Backtests flatter the winner: choosing this way
> scores 1.27 on later windows, not 1.25 (1.6% worse).

Each part of that answer comes from one check.

## How sure is the choice?

Your series are a sample. With a different sample, would the same model win? horizoneval redraws your series many
times (a bootstrap; all windows of a series stay together) and counts how often each model comes out best.

- **Probability of best** (`p_best`): the share of redraws in which the model is best. 0.5 between two models means
  the data cannot tell them apart.
- **Expected loss** (`expected_loss_rel`): how much worse than the best a model is on average over the redraws. A
  model can be best only half the time and still be a safe choice, if it is never much worse when it loses.

When another model is best in at least a fifth of the redraws, the answer says so: you may need more series or
windows before the choice is clear.

## Is the winner really better?

The answer also says whether the best model beats the runner-up, with a test checked on your own data by
simulating equally good forecasters. See [Can you trust a comparison?](trust.md).

## Will it hold up?

Picking the model with the best backtest score and then quoting that score is optimistic: among several models,
the one that got lucky tends to win. This is the winner's curse. horizoneval measures it by choosing the way a
backtest chooses, but scoring the choice on data it was not chosen on:

- with several windows, before each window it chooses using only the earlier windows, then scores that choice on
  the window;
- with one window, it chooses on half of your series and scores on the other half, both ways round.

The **honest score** is what you should expect from the model you pick. When it is clearly worse than the
backtest's best score, the answer tells you how much.

## One model, or a model per group?

If your series fall into groups (regions, product lines, the datasets of a run), a different model may win in each.
Choosing per group is only worth it if it helps on data the choices were not made on, so horizoneval scores both
strategies the honest way above and tests the difference per series. It recommends choosing per group only when the
gain is positive and significant at 5%.

```python
ev = evaluate(cv, train=train_df, series=series_df, segment="region")
```

`series_df` has one row per `unique_id` and a `region` column. Without `segment`, the datasets of a run are the groups.

## Your cost, not ours

MASE and CRPS treat over- and under-forecasting the same. If running out of stock costs you nine times more than
holding too much, give horizoneval that cost and it decides by it:

```python
ev = evaluate(cv, train=train_df, cost="linlin:0.9")     # shortfalls cost 0.9 per unit, overshoots 0.1
ev = evaluate(cv, train=train_df, cost="pinball:0.95")   # scores the 95% quantile if the model has one
ev = evaluate(cv, train=train_df, cost=lambda y, f: ...) # any per-series cost of truth and point forecast
```

```bash
horizoneval score cv.parquet --train train.parquet --cost linlin:0.9
```

When your cost picks a different model than MASE or CRPS would, the answer says which, and how much more the
standard pick would cost you.

## How we know these numbers are right

The checks above are themselves checked by simulation, where the right answer is known
(`horizoneval.verify`, run in the test suite):

- with identical models, each is best in about one redraw in K;
- of the choices made with probability of best near 0.8, about 80% are right;
- with identical models, the backtest's best score is optimistic and the honest score is not;
- choosing per group is recommended in at most about 5% of simulations where it does not help, and almost always
  when it does.

The scores themselves are tested against utilsforecast, and against changes that must not move them (row order,
series names, units, adding another model). Each run records fingerprints of its forecasts, its history and its
protocol (series, cutoffs, horizon, periods, deciding metric) in `config.json`, so two runs can be compared only when
they were evaluated the same way.
