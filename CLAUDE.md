# horizoneval

Evaluate time series forecasts from any tool, and invent forecasters to evaluate (repository: kausalflow/horizoneval). See README.md for the forecast frame, the contract and the evaluation.

- Python via uv (`uv sync`, `uv run pytest -q`). Heavy deps are extras: `tsfm` (timecopilot, pinned), `gift` (datasets).
- New forecasters go in `src/horizoneval/models/<name>.py` with `@forecaster("<name>")`; never edit the baselines to make a model look better.
- All scoring goes through `horizoneval.evaluate` on the forecast frame (`horizoneval.frame`, Nixtla `cross_validation` layout); never add a second scoring path.
- Keep code simple and vectorized over series (numpy, (N, n) -> (N, h)); outputs are CSV/JSON plus the static report.
- A comparison claim needs the run folder committed under `results/` and the reference stated.
- Docs live in `docs/` (MkDocs Material + mkdocstrings, `uv sync --group docs`, `uv run mkdocs build --strict`); written for non-forecasters first. The `docs` workflow deploys main to GitHub Pages.
- Related research code: emptymalei/extrapolation-as-forecasts (naive-forecaster theory, TSFM probes).
