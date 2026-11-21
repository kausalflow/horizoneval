# How the evaluation works

This page explains what horizoneval does to a forecast, first in plain words, then precisely. You do not need to
know any forecasting to follow the plain-words parts.

## The idea in one paragraph

A forecast is only worth something if it would have worked *before* you knew the answer. So horizoneval goes back
in time: it hides the end of each series, asks every forecaster to predict the hidden part from the past alone, then
uncovers it and measures the miss. It does this several times per series, at different points in time, and on many
series. Then it asks two questions: *how big are the misses compared with a very simple forecast?* and *is the
difference large and consistent enough that it is not luck?*

## Step 1: pretend it is the past (rolling origin)

Take a series of daily sales. Pick a date, the **cutoff**. Everything up to the cutoff is the **history**; the next
`h` days (the **horizon**) are hidden. Every forecaster sees the history and predicts those `h` days.

Then the cutoff moves forward by `step` points and it happens again. Each (series, cutoff) pair is a **window**.

```text
history ───────────────────────┤ cutoff 1 ├─ h hidden ─┤
history ─────────────────────────────┤ cutoff 2 ├─ h hidden ─┤
history ───────────────────────────────────┤ cutoff 3 ├─ h hidden ─┤
```

Why several cutoffs? One cutoff can be lucky or unlucky (a holiday, a shock). Several cutoffs give a fairer picture.
A forecast for a time at or before its own cutoff would be cheating (**leakage**), and horizoneval refuses to score
a table that contains one.

In `horizoneval run` these are the `--windows`, `--step`, `--h` and `--context` options. If you bring your own
forecasts, the `cutoff` column of your table says where each window starts.

## Step 2: measure each miss

For each window horizoneval compares the forecast with what actually happened and computes several scores. Each one
answers a slightly different question.

| Score | Plain question | Good value |
|---|---|---|
| **MASE** | How big is the miss, compared with the typical change from one season (or one step) to the next in this series' own past? | below 1 beats that simple benchmark; lower is better |
| **sMAPE** | How big is the miss as a percentage of the values? | lower is better, between 0 and 2 |
| **MAE**, **MSE** | How big is the miss in the series' own units (MSE squares it, so big misses count more)? | lower is better; not comparable across series |
| **CRPS (scaled)** | For forecasts that give a range, how good is the whole range, not just the middle? | lower is better; scaled like MASE |
| **coverage80 / coverage90** | How often did the truth land inside the forecast's 80% (90%) band? | close to 0.80 (0.90). Much lower: the bands are overconfident. Much higher: they are too wide to be useful |
| **width80 (scaled)** | How wide is the 80% band? | narrower is better, *if* coverage is right |

**Why scale?** A miss of 10 units is huge for a series that moves by 1 a day and tiny for one that moves by 1,000.
MASE and scaled CRPS divide by how much the series typically changes in its own history before the cutoff, so a
series of sales in millions and a series of temperatures can be averaged together.

??? info "Precise definitions"
    For a window with truth $y_1,\dots,y_h$, point forecast $\hat y_1,\dots,\hat y_h$ and history
    $x_1,\dots,x_n$ (everything up to the cutoff), with seasonal period $m$ (or $m=1$):

    $$\text{scale} = \frac{1}{n-m}\sum_{t=m+1}^{n} |x_t - x_{t-m}|$$

    $$\text{MASE} = \frac{1}{h}\sum_{k=1}^{h} \frac{|y_k - \hat y_k|}{\text{scale}}, \qquad
    \text{sMAPE} = \frac{1}{h}\sum_{k=1}^{h} \frac{2|y_k-\hat y_k|}{|y_k|+|\hat y_k|}$$

    CRPS is approximated on the quantile grid $\tau \in \{0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95\}$ (or the levels
    in your frame) as twice the mean pinball loss
    $\rho_\tau(u) = \max(\tau u, (\tau-1)u)$, $u = y_k - q_{\tau,k}$, then divided by the scale.
    Coverage80 uses the 0.1 and 0.9 quantiles, coverage90 the 0.05 and 0.95 quantiles.
    Code: [`horizoneval.metrics`](../reference/api.md#horizoneval.metrics).

## Step 3: compare with a reference

A score on its own is hard to judge. Is MASE 0.9 good? It depends on how hard the series are. So horizoneval always
compares with a **reference** forecast, by default `naive` ("tomorrow will be like today"). If a clever model cannot
beat that, it is not useful yet.

The **skill** column says by how much a model beats the reference:

- **0**: same error as the reference.
- **positive**: less error. A skill of 0.69 means about half the squared error (e^0.69 ≈ 2), 2.3 about a tenth.
- **negative**: worse than the reference.

??? info "Precise definition"
    For each window, $\text{skill} = \log(\text{MSE}_\text{ref} / \text{MSE}_\text{model})$, averaged over windows
    and series. It is measured in nats: the log of the ratio of squared errors. Log ratios add up and are symmetric
    (twice as good is +0.69, twice as bad is −0.69), which plain ratios are not.

Pick a reference that matters to you with `--reference` (benchmarks) or `reference=` (your own forecasts), for
example the forecast your team uses today.

## Step 4: is the difference real?

A model can win on average by luck: a few series where it happened to do well. horizoneval runs three checks.

**Bootstrap interval.** "If we had drawn a different set of series like these, where would the average score land?"
horizoneval resamples the series 2,000 times and reports the range holding 95% of the averages. Two models whose
intervals overlap a lot are hard to tell apart.

**Paired test against the reference.** "On the *same* series, does this model consistently miss by less than the
reference?" Comparing on the same series removes the noise of some series being harder than others. A small
**p** (below 0.05 is the usual line) says the difference is unlikely to be luck. A negative **t** means the model
has the lower error (shown in green in the report), positive means higher (red).

**Friedman test and critical difference.** "Across datasets, do the models' ranks differ at all, and which ones can
we tell apart?" Models whose mean ranks differ by less than the **critical difference** are not distinguishable at
the 5% level.

??? info "Precise definitions"
    - Bootstrap: percentile bootstrap of the mean over series, 2,000 resamples. All windows of a series stay together,
      because windows of the same series are not independent.
    - Paired test: Diebold–Mariano style. The loss differential (model minus reference) is averaged over the windows
      of each series; the per-series differentials are compared with zero by a t test. Series are treated as
      independent.
    - Friedman test over datasets (blocks) and models (treatments) on mean MASE; Nemenyi critical difference
      $q_\alpha \sqrt{k(k+1)/(6n)}$ for $k$ models and $n$ datasets (Demšar, 2006).
    - Code: [`horizoneval.stats`](../reference/api.md#horizoneval.stats).

## Step 5: how far ahead can you trust it?

Forecasts usually get worse the further ahead they look. The **error by forecast step** chart shows the scaled
error at step 1, 2, ..., h. A model that is excellent at step 1 and worse than naive at step 12 is a good
next-day model and a poor two-week model.

## What can go wrong, and what horizoneval does

| Problem | What happens |
|---|---|
| A forecast for a time at or before its cutoff (leakage) | the run stops and says which rows |
| Duplicated rows, or windows of different lengths | the run stops and says why |
| A model with missing or infinite forecasts anywhere | the model is dropped and listed under `failures` in `config.json`; the others are still scored |
| A model that crashes during `horizoneval run` | same: recorded under `failures`, the run continues |
| No history given | MASE and scaled CRPS are empty (their scale needs the history); the other scores still work |
