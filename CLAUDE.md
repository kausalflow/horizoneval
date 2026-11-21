# forecaster-lab

Workspace for inventing forecasters and benchmarking them. See README.md for the contract and the evaluation.

- Python via uv (`uv sync`, `uv run pytest -q`). Heavy deps are extras: `tsfm` (timecopilot, pinned), `gift` (datasets).
- New forecasters go in `src/horizoneval/models/<name>.py` with `@forecaster("<name>")`; never edit the baselines to make a model look better.
- Keep code simple and vectorized over series (numpy, (N, n) -> (N, h)); outputs are CSV/JSON plus the static report.
- A comparison claim needs the run folder committed under `results/` and the reference stated.
- Related research code: emptymalei/extrapolation-as-forecasts (naive-forecaster theory, TSFM probes).
