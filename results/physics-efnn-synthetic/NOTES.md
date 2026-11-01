# Entropic Force NN on the synthetic suite

Source: kausalflow/physics_models (EFNN, PAPER.md), ported to `src/flab/models/entropic_force.py`.

```bash
uv sync --extra physics
uv run flab run --models naive seasonal_naive drift damped_drift efnn efnn_noent efnn_mlp --data synthetic --out results/physics-efnn-synthetic
```

h = 12, 5 windows, context 256, 30 series per dataset, seed 0, reference `naive`. CPU torch 2.15.

MASE by dataset:

| model | ar_diff | lorenz | random_walk | rossler | seasonal_trend | van_der_pol |
|---|---|---|---|---|---|---|
| efnn | 3.506 | **0.452** | 2.354 | 0.478 | **0.793** | 3.907 |
| efnn_noent | 3.507 | 0.452 | 2.353 | 0.476 | 0.798 | 3.908 |
| efnn_mlp | 3.568 | 0.535 | **2.351** | **0.290** | 0.813 | 2.894 |
| seasonal_naive | 3.601 | 1.094 | 2.359 | 1.033 | 1.079 | **0.995** |
| naive | 3.601 | 1.072 | 2.359 | 2.812 | 4.436 | 46.313 |

Paired Wilcoxon over the 30 series (per-series mean MASE):

- efnn vs efnn_noent (the entropy term): differences of at most 0.005 MASE; only Rössler is significant (p = 3e-4) and
  there the entropy term is slightly worse. With T = 1 the hidden units are T·tanh(x/T) ∈ [-1, 1], so the softmax
  over 32 units is close to uniform whatever the weights do, its entropy stays near log 32, and the term barely moves
  the gradient. The thermodynamic loss is effectively inactive at the paper's defaults.
- efnn vs efnn_mlp (tanh-temperature activation vs ReLU): better on Lorenz (-0.08, p = 1e-4), worse on Rössler
  (+0.19, p = 2e-9) and Van der Pol (+1.01, p = 2e-8), no significant difference elsewhere.
- All three neural variants beat naive by a wide margin on the dynamical systems and on seasonal_trend, but
  seasonal_naive still wins Van der Pol, and nothing beats the naive on the random walk (as it should be).
