# horizoneval

**horizoneval tells you how good a forecast is, in words a non-forecaster can follow, and backs every sentence with a
formal test.**

You bring forecasts, made with any tool (horizoneval's own forecasters, [TimeCopilot](https://github.com/TimeCopilot/timecopilot),
Nixtla's statsforecast, neuralforecast or mlforecast, or a spreadsheet). horizoneval puts them in one table, scores
them all with the same code, compares them against a simple reference forecast, and writes a static report you can
open in any browser or publish as a web page.

<div class="grid cards" markdown>

- **New here?** Start with the [Quickstart](quickstart.md): install, run a first benchmark and open its report in
  five minutes.
- **Not a forecaster?** Read [How the evaluation works](explain/evaluation.md). It explains every number in the
  report without formulas first, then with them.
- **Already have forecasts?** [Score forecasts from Nixtla or TimeCopilot](guides/score-forecasts.md) takes the
  output of `cross_validation` as it is.
- **Inventing a model?** [Add a forecaster](guides/add-forecaster.md) is one function and one decorator.

</div>

## What you get

| Question you might ask | Where horizoneval answers it |
|---|---|
| Which model is best overall? | the **leaderboard**, ordered by mean rank over datasets |
| Is it *really* better, or just lucky? | **paired tests** against the reference and **bootstrap intervals** |
| Better than what? | every score is relative to a **reference** forecast, `naive` by default |
| How far ahead can I trust it? | **error by forecast step**, from one step ahead to the full horizon |
| Are its uncertainty bands honest? | **coverage**: how often the truth landed inside the 80% and 90% bands |
| Can someone else check this? | every run writes plain CSVs, the forecasts themselves and a `config.json` with data fingerprints |

## The three ideas behind it

1. **One table for every forecast.** Every forecast, from any tool, becomes a long table in the layout of Nixtla's
   `cross_validation` output (the [forecast frame](reference/frame.md)). A number never depends on where the forecast
   was made.
2. **One scoring path.** `horizoneval run` (benchmark our forecasters) and `horizoneval score` (score yours) both go
   through [`horizoneval.evaluate`](reference/api.md#horizoneval.evaluate.evaluate). There is no second way to
   compute a score.
3. **Always against a reference.** "MASE 0.8" means little on its own. "20% less error than repeating the last
   value, and the difference holds up in a paired test" is something you can act on.

## Example reports

These are real reports from runs committed under `results/` in the repository:

- [Classic baselines on synthetic data](examples/example-synthetic-baselines.html)
- [Physics-inspired neural forecasters on synthetic data](examples/physics-efnn-synthetic.html)

## License

horizoneval is open source under the [Apache License 2.0](https://github.com/kausalflow/horizoneval/blob/main/LICENSE).
