---
name: horizoneval
description: Evaluate time series forecasts and help choose one. Use when someone has forecasts (from Nixtla, TimeCopilot, statsforecast, their own code or a spreadsheet) and asks which is best, whether a model really beats another, or why two evaluations disagree.
---

# Choosing a forecast with horizoneval

horizoneval scores forecasts made by any tool and answers: which model should I use on my data, and how sure is
that? Every number comes from its deterministic core; your job is to run it, read its result and explain it in plain
language. Never compute or estimate scores yourself.

## Run it

Install once: `pip install horizoneval` (or `uv add horizoneval`). Then, in order:

1. **Check the forecasts before scoring them.**
   `horizoneval check forecasts.csv --train history.csv --json`
   Exit code 1 means an error was found. Report every `error` and `warning` to the user before anything else:
   a forecast that copied the truth, wrong units, or history that disagrees with the truth makes the scores
   meaningless.
2. **Score them.**
   `horizoneval score forecasts.csv --train history.csv --period 7 --out results/run --json`
   - `--period`: the seasonal period in steps (7 for daily data with a weekly pattern, 24 for hourly, 12 for
     monthly). Ask the user if unsure; it changes MASE.
   - `--cost linlin:0.9` when shortfalls cost the user more than overshoots (here 9 times); `pinball:0.95` for a
     95% service level. The cost then decides the choice.
   - `--series series.csv --segment region` to test whether a model per group beats one model.
   - `--reference <model>` for skill scores and paired tests against a baseline.
3. **Compare two runs** that disagree: `horizoneval diff results/a results/b --json`.

The forecast file is one row per series, target time and forecast origin: columns `unique_id`, `ds`, `cutoff`, `y`
and one column per model (Nixtla's `cross_validation` output as is). Quantiles: `<model>-q-<p>` or
`<model>-lo-<l>` / `<model>-hi-<l>`. The history file has `unique_id`, `ds`, `y`.

An MCP server with the same tools is available: `horizoneval-mcp` (install `horizoneval[mcp]`).

## Read the result

The JSON document (`result.json`, contract version in `horizoneval_result`) has:

- `verdict`: the answer in plain language, one line per dataset. Lead with it, close to verbatim.
- `choice[dataset]`: `model`, `p_best` (probability it is the best model for these series), `expected_loss_rel`
  (how much worse than the best it is on average), `claim` (winner vs runner-up: `verified` is true only when the
  difference survives a test checked on this data), `winners_curse` (the honest score of choosing this way).
- `segments`: whether to use a different model per group (`recommend`).
- `cost_disagreement`: when the user's cost picks a different model than MASE or CRPS.
- `checks`: pre-flight findings. `failures`: models that could not be scored, and why.
- `summary`, `leaderboard`: every score. `run`: settings and fingerprints.

## Explain it

- Say which model to choose, how sure that is, and what it costs to be wrong, before any metric name.
- Translate: MASE below 1 beats the naive forecast "repeat the last value"; CRPS scores the whole forecast range;
  coverage80 should be near 0.80.
- If `claim.verified` is false, do not say one model is better: say the data cannot separate them yet.
- If `p_best` is below about 0.7, say the choice is uncertain and that more series or windows would help.
- Mention a winner's curse larger than a few percent: the backtest flatters the chosen model.
- Never round away a check marked `error`.
