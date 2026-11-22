"""The result contract: one versioned JSON document per evaluation, the same from every entry point.

The Python API (`to_result(ev)`), the run folder (`result.json`), the command line (`horizoneval score --json`), the
tool registry (`horizoneval.tools`) and the MCP server (`horizoneval-mcp`) all return this document, serialised by
`dumps`, so an AI assistant, a script and a person reading the report see the same numbers. It is validated against
`schema/result.schema.json`; RESULT_VERSION changes when a field changes meaning or disappears.

It holds only what depends on the inputs (no timestamps, file paths or run times), so the same forecasts give a
byte-identical document wherever they are scored.
"""

import json
import math
from importlib import resources

import numpy as np
import pandas as pd

RESULT_VERSION = "1.0"
CHECKS_VERSION = "1.0"
_RUN_KEYS = ["decide_by", "rank_by", "reference", "smape_zero", "cost", "h", "windows", "datasets", "models",
             "missing_targets", "settings_sha256_16", "protocol_sha256_16", "forecasts_sha256_16", "train_sha256_16"]
_DROP_COLUMNS = ("seconds",)  # run times differ between machines and runs


def clean(x):
    """Plain JSON values: numpy scalars to Python, NaN and infinities to null, tables to lists of records."""
    if isinstance(x, pd.DataFrame):
        return clean(x.drop(columns=[c for c in x.columns if c in _DROP_COLUMNS]).to_dict(orient="records"))
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x))
    return x


def dumps(doc: dict) -> str:
    """The canonical serialisation: sorted keys, no NaN, compact. Every entry point writes exactly this."""
    return json.dumps(clean(doc), sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)


def to_result(ev) -> dict:
    """The result document of an Evaluation."""
    d = ev.decision or {}
    conf = d.get("confidence", pd.DataFrame())
    choice = {}
    for ds, v in (d.get("verdict") or {}).items():
        c = conf[conf["dataset"] == ds].sort_values(["p_best", "mean"], ascending=[False, True])
        if len(c):
            top = c.iloc[0]
            choice[ds] = {"model": top["model"], "p_best": top["p_best"], "expected_loss_rel": top["expected_loss_rel"],
                          "claim": (d.get("claims") or {}).get(ds), "winners_curse": (d.get("winners_curse") or {}).get(ds),
                          "verdict": v}
    run = {k: ev.config.get(k) for k in _RUN_KEYS}
    doc = {"horizoneval_result": RESULT_VERSION, "verdict": d.get("text", []), "choice": choice,
           "confidence": conf, "segments": d.get("segments"), "cost_disagreement": d.get("disagreement", []),
           "leaderboard": ev.leaderboard, "summary": ev.summary, "checks": ev.checks or [],
           "failures": ev.config.get("failures", {}), "run": run}
    return json.loads(dumps(doc))


def schema() -> dict:
    return json.loads(resources.files("horizoneval").joinpath("schema/result.schema.json").read_text())


def validate(doc: dict) -> None:
    """Raise jsonschema.ValidationError when doc does not follow the result contract (needs jsonschema)."""
    import jsonschema

    jsonschema.validate(doc, schema())
