# See it work

This page follows one real run from start to finish: five simple forecasters on six synthetic datasets, scored the way
you would score your own. Every chart and sentence below is read from the committed run folder
[`results/ladder-synthetic`](https://github.com/kausalflow/horizoneval/tree/main/results/ladder-synthetic) when these
docs are built, and the [full report](examples/ladder-synthetic.html) is published too.

To make the same run yourself (about a minute on a laptop):

```bash
uv run horizoneval run --models naive seasonal_naive drift damped_drift mean \
    --data synthetic --h 16 --windows 8 --context 896 --ladder --out results/ladder-synthetic
```

<div class="hz-stage"><b>1</b> Bring forecasts</div>

## The question

We have six kinds of series: a random walk, a noisy trend, a seasonal series with trend, and three chaotic systems
(Lorenz, Rössler, Van der Pol). Thirty series of each. Five candidate forecasters, from "repeat the last value"
(`naive`) to "repeat last season" (`seasonal_naive`) and "extend the trend" (`drift`). **Which one should we use, and
how sure can we be?**

Here horizoneval makes the forecasts itself. With your own forecasts you skip this step and hand over the table
from statsforecast, TimeCopilot or anything else ([how](guides/score-forecasts.md)).

<div class="hz-stage"><b>2</b> Go back in time</div>

## Backtest: forecast the past, then look

Each series is cut at eight points in time. At every cut the forecasters see only the history and predict the next 16
points; then the hidden part is revealed and the miss is measured. Here are the last three windows of one Van der
Pol series, which repeats a cycle every 12 points:

<div class="hz-figure-card">
--8<-- "docs/assets/figures/hero-backtest.svg"
</div>

`seasonal_naive` (indigo) repeats the last cycle and hugs the truth; `naive` (dashed orange) draws a flat line. The shaded band is
seasonal naive's 80% range: the truth should fall inside it about 80% of the time, and horizoneval checks that it
does ([coverage](explain/evaluation.md#step-2-measure-each-miss)).

Before scoring, the table is checked: a forecast for a time at or before its own cutoff (a peek at the answer),
duplicated rows or windows of different lengths stop the run with the reason
([pre-flight checks](explain/trust.md#before-scoring-pre-flight-checks)).

<div class="hz-stage"><b>3</b> Score every miss</div>

## Skill over the reference

Every model is compared with a reference forecast, here `naive`. Skill is measured in nats: 0 is the same as the
reference, +0.69 is half the squared error, +2.3 a tenth.

<div class="hz-figure-card">
--8<-- "docs/assets/figures/skill.svg"
</div>

`seasonal_naive` makes roughly seven times less squared error than `naive` on average; `mean` is worse than doing
nothing clever. Averages hide where the error comes from, so horizoneval also shows the error at each step ahead:

<div class="hz-figure-card">
--8<-- "docs/assets/figures/horizon.svg"
<p class="hz-caption">Scaled error (MASE per step) on the seasonal-with-trend dataset, averaged over series and windows.</p>
</div>

The further ahead, the bigger every miss, but `seasonal_naive` stays far below the rest at every step: it is a good
one-step model *and* a good 16-step model here.

<div class="hz-stage"><b>4</b> Reach a verdict</div>

## How sure is the choice?

On the easy datasets the answer is clear. On a noisy one it is not, and horizoneval says so instead of crowning a
winner. Resampling the 30 AR-differences series:

<div class="hz-figure-card">
--8<-- "docs/assets/figures/pbest.svg"
</div>

`naive` is the best in about six resamples out of ten, `damped_drift` in three: the data cannot separate them yet,
and picking `naive` costs about 0.2% more error on average than always picking the true best would. That is the
honest answer, and it is the one the report gives ([choosing a forecast](explain/choosing.md)).

## Where does the skill come from?

`--ladder` builds two extra references from each series' own history: the best simple rule (*free*) and a plain
linear model on top (*linear*). The share of error they already remove tells you how much room is left for anything
more sophisticated:

<div class="hz-figure-card">
--8<-- "docs/assets/figures/ladder.svg"
</div>

On Van der Pol and the seasonal series, simple rules plus a linear model already remove almost all of the error. A
neural network would have to beat *that*, not `naive`, to justify itself
([the skill ladder](explain/skill-ladder.md)).

<div class="hz-stage"><b>5</b> Say it in plain words</div>

## The explanation, and its check

`horizoneval explain results/ladder-synthetic` returns the answer as sentences, the caveats that must be said, and
every number a write-up may quote. `--check` reads a draft and flags any number the result does not support. Below
are the run's own answer and a draft with one invented number:

--8<-- "docs/assets/figures/explain.html"

An AI assistant connected through the [MCP server or the Agent Skill](guides/ai-assistants.md) is told to explain
only from `explain` and to run the check before it replies.

## What's in the run folder

| File | What it answers |
|---|---|
| `report.html` | the full report as one static page ([open it](examples/ladder-synthetic.html)) |
| `leaderboard.csv`, `summary.csv` | which model is best overall and per dataset |
| `confidence.csv`, `decision.json` | how sure the choice is, and whether it holds up on later windows |
| `tests.csv`, `ranks.json` | paired tests, bootstrap intervals, Friedman and Nemenyi |
| `ladder_rungs.csv`, `ladder_models.csv` | where the skill comes from |
| `explanation.json`, `result.json` | the plain-language answer and the full machine-readable result |
| `forecasts.parquet`, `train.parquet`, `config.json` | everything needed to re-score and check the run |

All files are described in [Outputs of a run](reference/outputs.md).
