---
hide:
  - navigation
  - toc
---

<div class="hz-hero" markdown>

<span class="hz-eyebrow">Open-source forecast evaluation</span>

# Know which forecast to trust, <em>and why</em>.

<p class="hz-lede">horizoneval scores forecasts from any tool with one set of rules, tells you in plain words which one
to use and how sure that choice is, and backs every sentence with a formal test. No forecasting degree needed to read
the answer.</p>

<div class="hz-cta" markdown>
[See it work](tour.md){ .md-button .md-button--primary }
[Quickstart](quickstart.md){ .md-button }
<span class="hz-install">uv add git+https://github.com/kausalflow/horizoneval</span>
</div>

<div class="hz-figure-card">
--8<-- "docs/assets/figures/hero-backtest.svg"
<p class="hz-caption">A real backtest from <code>results/ladder-synthetic</code>: at each cutoff the forecasters see only
the past, predict the next 16 points, and are scored against what actually happened.</p>
</div>

</div>

<h2 class="hz-section-title">How it works</h2>
<p class="hz-section-sub">Five steps from a pile of forecasts to an answer you can repeat to your manager, and check.</p>

<div class="hz-pipeline" markdown>

<div class="hz-step" markdown>
### Bring forecasts
Made anywhere: statsforecast, TimeCopilot, neuralforecast, mlforecast, your own model, or a spreadsheet.
[Score yours →](guides/score-forecasts.md){ .hz-more }
</div>

<div class="hz-step" markdown>
### One table
Every forecast becomes one long table in Nixtla's `cross_validation` layout. Leakage, duplicates and gaps are caught
before anything is scored.
[The frame →](reference/frame.md){ .hz-more }
</div>

<div class="hz-step" markdown>
### Backtest scores
Each forecast is compared with what really happened, window by window: size of the miss, honesty of the bands,
skill over a simple reference.
[The scores →](explain/evaluation.md){ .hz-more }
</div>

<div class="hz-step" markdown>
### A verdict
Which model to choose, the probability it really is the best, whether the win survives on later data, and where
its skill comes from.
[Choosing →](explain/choosing.md){ .hz-more }
</div>

<div class="hz-step" markdown>
### Plain words
The result explains itself in sentences, lists the caveats, and a checker flags any number a write-up made up.
[With AI →](guides/ai-assistants.md){ .hz-more }
</div>

</div>

<h2 class="hz-section-title">Answers, not just numbers</h2>
<p class="hz-section-sub">Every chart below is drawn from the same committed run when these docs are built.</p>

<div class="hz-split" markdown>
<div markdown>
<span class="hz-q">Is it better than doing nothing clever?</span>

### Skill over a simple reference

"MASE 0.8" means little on its own. horizoneval measures every model against a reference, here *repeat the last
value*, and says how many times less squared error it makes. A model that cannot beat the reference is not worth
its cost yet.

[How skill is measured →](explain/evaluation.md#step-3-compare-with-a-reference)
</div>
<div class="hz-figure-card">
--8<-- "docs/assets/figures/skill.svg"
<p class="hz-caption">Mean skill over six synthetic datasets, in nats over <code>naive</code>.</p>
</div>
</div>

<div class="hz-split hz-flip" markdown>
<div class="hz-figure-card">
--8<-- "docs/assets/figures/pbest.svg"
<p class="hz-caption">Probability of being the best model on the AR-differences dataset, from resampling its 30 series.</p>
</div>
<div markdown>
<span class="hz-q">How sure is the choice?</span>

### Probability of being the best

A leaderboard always has a winner, even when the data cannot tell the top models apart. horizoneval resamples your
series and reports how often each model comes out on top, and how much picking it costs when it doesn't.

[Choosing a forecast →](explain/choosing.md)
</div>
</div>

<div class="hz-split" markdown>
<div markdown>
<span class="hz-q">Where does the skill come from?</span>

### The skill ladder

Much of a fancy model's gain is often available from a simple rule or a plain linear model. The ladder splits the
gain into *free*, *linear* and *beyond linear*, so you know whether the expensive part is earning its keep.

[The skill ladder →](explain/skill-ladder.md)
</div>
<div class="hz-figure-card">
--8<-- "docs/assets/figures/ladder.svg"
<p class="hz-caption">Share of squared error (over repeating the last value) removed by the best simple rule and by a
linear model fitted on the history.</p>
</div>
</div>

<div class="hz-split hz-flip" markdown>
<div>
--8<-- "docs/assets/figures/explain.html"
</div>
<div markdown>
<span class="hz-q">Can I explain it to someone else?</span>

### It explains itself, and checks the explanation

`horizoneval explain` turns a result into sentences a non-forecaster can follow, with every caveat that must be
said. `--check` reads a draft, written by you or an AI assistant, and flags every number the result does not
support. The flagged number here was made up on purpose.

[Use from an AI assistant →](guides/ai-assistants.md)
</div>
</div>

<h2 class="hz-section-title">Built to be trusted</h2>
<p class="hz-section-sub">The rules that make a horizoneval number mean the same thing everywhere.</p>

<div class="hz-grid" markdown>
<div class="hz-tile" markdown>
### One scoring path
`horizoneval run`, `horizoneval score`, the Python API, the MCP server and the Agent Skill all go through the same
`evaluate`. A number never depends on where the forecast was made.
</div>
<div class="hz-tile" markdown>
### Formal tests, plainly said
Paired tests against the reference, bootstrap intervals, Friedman and Nemenyi ranks, and a check that the winner
holds up on later windows.
</div>
<div class="hz-tile" markdown>
### Matches the published numbers
All 98 published M4 numbers and all 400 GIFT-Eval leaderboard numbers are reproduced by horizoneval's own scoring.
[Replication →](explain/replication.md)
</div>
<div class="hz-tile" markdown>
### Reproducible to the byte
Every run writes plain CSVs, the forecasts themselves and fingerprints of data and settings. `horizoneval diff`
says why two runs disagree.
[Trust →](explain/trust.md)
</div>
</div>

<h2 class="hz-section-title">Works with your stack</h2>
<div class="hz-logos">
<span>Nixtla statsforecast</span><span>neuralforecast</span><span>mlforecast</span><span>utilsforecast</span>
<span>TimeCopilot</span><span>GIFT-Eval</span><span>M4</span><span>MCP server</span><span>Agent Skill</span>
<span>CLI with JSON</span><span>Python</span>
</div>

<div class="hz-final" markdown>
## Try it on your own forecasts

```bash
uv add git+https://github.com/kausalflow/horizoneval
uv run horizoneval score cv.parquet --train train.parquet --period 24 --reference SeasonalNaive --out results/my-run
```

[Take the tour](tour.md){ .md-button .md-button--primary }
[Example report](examples/ladder-synthetic.html){ .md-button }

<p class="hz-caption">horizoneval is open source under the
<a href="https://github.com/kausalflow/horizoneval/blob/main/LICENSE">Apache License 2.0</a>.</p>
</div>
