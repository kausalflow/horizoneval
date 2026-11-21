# Quickstart

## Install

horizoneval uses [uv](https://docs.astral.sh/uv/) and Python 3.11 or 3.12.

```bash
git clone https://github.com/kausalflow/horizoneval
cd horizoneval
uv sync                 # core only: numpy, pandas, scipy, pyarrow
```

Heavier features are extras, installed only when you need them:

| Extra | Adds | Install |
|---|---|---|
| `tsfm` | foundation models and statistical references through TimeCopilot (Chronos-Bolt, Moirai 2, TiRex, TimesFM 3, AutoARIMA, AutoETS) | `uv sync --extra tsfm` |
| `gift` | the GIFT-Eval benchmark datasets | `uv sync --extra gift` |
| `physics` | the physics-inspired Entropic Force neural forecasters (torch) | `uv sync --extra physics` |

## 1. See what is available

```bash
uv run horizoneval list
```

This prints every registered forecaster, the synthetic datasets and the dataset suites.

## 2. Run a first benchmark

```bash
uv run horizoneval run --models naive seasonal_naive drift damped_drift --data synthetic --out results/first-run
```

Four simple forecasters are tested on six synthetic datasets (random walk, seasonal series with trend, chaotic
systems and more). Each series is cut at five points in time; at each cut, every forecaster sees only the past and
predicts the next 12 values. The terminal ends with the leaderboard:

```text
         model  mean_rank  mean_mase  mean_skill  mean_crps_scaled  mean_coverage80  datasets  seconds
  damped_drift      2.333     10.369       0.009               NaN              NaN         6    0.010
seasonal_naive      2.333      1.654       2.153             0.943            0.807         6    0.050
         drift      2.500     10.439      -0.002               NaN              NaN         6    0.010
         naive      2.833     10.376       0.000             4.958            0.840         6    0.060
```

How to read it, briefly (the [full explanation](explain/evaluation.md) covers every column):

- **mean_rank** is the model's average place over the six datasets (1 is best). Lower is better.
- **mean_mase** is the error relative to a simple in-sample benchmark; below 1 is better than that benchmark.
- **mean_skill** is how much better than the reference (`naive`) the model is, in "nats": 0 is the same, positive is
  better. `seasonal_naive` at 2.15 has roughly e^2.15 ≈ 8.6 times less squared error than `naive` on average.
- **NaN** in the probabilistic columns means that forecaster gives no uncertainty bands, so there is nothing to score.

!!! note "Rank and average can disagree"
    `damped_drift` ties with `seasonal_naive` on mean rank while its mean MASE is six times larger. Rank counts how
    often a model wins, the mean counts by how much. One dataset with huge errors moves the mean but not the rank.
    The report shows both so you can see which story holds.

## 3. Open the report

```bash
open results/first-run/report.html        # macOS; use xdg-open on Linux or double-click the file
```

The report is one self-contained HTML file with no external assets. [Reading a report](explain/report.md) walks
through it section by section.

## 4. Score forecasts you already have

If you made forecasts with statsforecast, TimeCopilot or anything else, save them in the
[forecast frame](reference/frame.md) layout and score them:

```python
from horizoneval import evaluate

ev = evaluate(cv, train=train_df, period=24, reference="SeasonalNaive")
ev.leaderboard
ev.write("results/my-run")   # same files and report as `horizoneval run`
```

or from the command line:

```bash
uv run horizoneval score cv.parquet --train train.parquet --period 24 --reference SeasonalNaive --out results/my-run
```

See [Score forecasts from Nixtla or TimeCopilot](guides/score-forecasts.md) for the details.
