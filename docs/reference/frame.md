# Forecast frame

Every forecast, from any tool, is scored as one long table in the layout of Nixtla's `cross_validation` output.
One row is one forecast target: one series, one cutoff, one future time.

| Column | Meaning |
|---|---|
| `unique_id` | series id |
| `ds` | the time being forecast (datetimes, or integers for series without a calendar) |
| `cutoff` | the last time the forecaster saw; one (series, cutoff) pair is one window |
| `y` | the truth |
| `<model>` | point forecast, one column per model |
| `<model>-q-<p>` | quantile at level `p`, in percent (`tirex-q-10`, as TimeCopilot writes it) or as a fraction (`m-q-0.1`) |
| `<model>-lo-<l>`, `<model>-hi-<l>` | central interval at level `l` percent (statsforecast); read as the quantiles (1 − l/100)/2 and 1 − (1 − l/100)/2 |
| `<model>-median` | read as quantile 0.5 |

Example, two models, one series, one window of three steps:

| unique_id | ds | cutoff | y | naive | AutoETS | AutoETS-lo-80 | AutoETS-hi-80 |
|---|---|---|---|---|---|---|---|
| shop_1 | 2026-01-08 | 2026-01-07 | 102 | 98 | 101.2 | 95.0 | 107.5 |
| shop_1 | 2026-01-09 | 2026-01-07 | 97 | 98 | 100.8 | 93.9 | 108.1 |
| shop_1 | 2026-01-10 | 2026-01-07 | 105 | 98 | 101.5 | 93.6 | 109.4 |

## History (`train`)

The same long format, `unique_id`, `ds`, `y`. For each window only the rows with `ds <= cutoff` are used, as in
utilsforecast.

## Series table (optional)

`unique_id`, `dataset`, `period`: which dataset each series belongs to and its seasonal period, for frames that mix
them. Keeping these out of the forecast frame keeps it readable by Nixtla tools.

## Checks

- every `ds` must be after its `cutoff` (no leakage),
- no duplicated (`unique_id`, `cutoff`, `ds`) rows,
- every window has the same number of steps.

A frame that fails a check raises [`FrameError`](api.md#horizoneval.frame.FrameError) with the reason.
