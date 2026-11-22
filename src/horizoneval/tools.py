"""The tools horizoneval offers to people, scripts and AI assistants: one registry, every front door built from it.

Each tool is a function of JSON arguments returning a JSON document, with a JSON Schema for its input. The command
line (`--json`), the MCP server (`horizoneval-mcp`), the Agent Skill and function-calling definitions
(`definitions()`) are thin adapters over `call`, so they cannot drift apart: tests/test_tools.py checks that every
path returns byte-identical results.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from horizoneval.result import CHECKS_VERSION, dumps, to_result


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[..., dict]


def _score(forecasts, train=None, series=None, period=None, dataset="data", reference=None, cost=None, segment=None,
           smape_zero="zero", ladder=False, out=None):
    from horizoneval.evaluate import evaluate
    from horizoneval.frame import read

    ev = evaluate(read(forecasts), read(train) if train else None, series=read(series) if series else None,
                  period=period, dataset=dataset, reference=reference, cost=cost, segment=segment,
                  smape_zero=smape_zero, ladder=ladder, config={"source": {"forecasts": forecasts, "train": train}})
    if out:
        ev.write(out)
    return to_result(ev)


def _check(forecasts, train=None):
    from horizoneval.checks import preflight, summary
    from horizoneval.frame import read

    found = preflight(read(forecasts), read(train) if train else None)
    return {"horizoneval_checks": CHECKS_VERSION, "summary": summary(found), "checks": found,
            "ok": not any(c["severity"] == "error" for c in found)}


def _diff(run_a, run_b):
    from horizoneval.settings import diff

    return diff(run_a, run_b)


def _read_result(run):
    return json.loads((Path(run) / "result.json").read_text())


def _explain(run):
    from horizoneval.explain import explain

    return explain(_read_result(run))


def _check_explanation(run, text):
    from horizoneval.explain import check_numbers

    return check_numbers(text, _read_result(run))


def _forecasters():
    from horizoneval.registry import available

    return {"forecasters": available()}


_PATH = {"type": "string", "description": "path to a parquet or CSV file"}
TOOLS = {t.name: t for t in [
    Tool("score",
         "Score forecasts made by any tool and say which model to choose. Input: a forecast table in Nixtla's "
         "cross_validation layout (unique_id, ds, cutoff, y, one column per model; quantiles as <model>-q-<p> or "
         "<model>-lo-<l>/-hi-<l>). Returns the decision (verdict, probability each model is best, verified claim, "
         "winner's curse), every score, pre-flight checks, and the settings fingerprint.",
         {"type": "object", "required": ["forecasts"], "additionalProperties": False, "properties": {
             "forecasts": _PATH,
             "train": {**_PATH, "description": "history (unique_id, ds, y); needed for MASE and scaled CRPS"},
             "series": {**_PATH, "description": "optional table (unique_id, dataset, period, segment columns)"},
             "period": {"type": "integer", "minimum": 1, "description": "seasonal period of every series"},
             "dataset": {"type": "string", "description": "dataset name when series gives none"},
             "reference": {"type": "string", "description": "model column for skill and paired tests"},
             "cost": {"type": "string", "description": "the user's cost, e.g. 'linlin:0.9' (shortfalls cost 9 times "
                                                       "overshoots) or 'pinball:0.95'; it decides the choice"},
             "segment": {"type": "string", "description": "column of series whose groups may each deserve a model"},
             "smape_zero": {"enum": ["zero", "skip"], "description": "sMAPE where truth and forecast are both 0"},
             "ladder": {"type": "boolean", "description": "also say where each model's skill comes from: how much a "
                                                       "simple rule already gets, how much a linear model fitted on "
                                                       "the history adds, and whether anything beyond linear is "
                                                       "shown (needs train)"},
             "out": {"type": "string", "description": "folder to write the run (report.html, CSVs, result.json)"}}},
         _score),
    Tool("check",
         "Pre-flight checks on a forecast table without scoring it: look-ahead (forecasts that copied the truth), "
         "history that disagrees with the truth, missing forecasts, units, crossing quantiles, duplicate models, "
         "irregular time steps.",
         {"type": "object", "required": ["forecasts"], "additionalProperties": False,
          "properties": {"forecasts": _PATH, "train": _PATH}},
         _check),
    Tool("diff",
         "Explain why two evaluation runs disagree: every setting that differs (series, cutoffs, horizon, seasonal "
         "period, truth, a model's forecasts, conventions) in plain language, and the scores it moved.",
         {"type": "object", "required": ["run_a", "run_b"], "additionalProperties": False,
          "properties": {"run_a": {"type": "string", "description": "run folder"},
                         "run_b": {"type": "string", "description": "run folder"}}},
         _diff),
    Tool("read_result",
         "Read the result of an earlier run from its folder (result.json).",
         {"type": "object", "required": ["run"], "additionalProperties": False,
          "properties": {"run": {"type": "string", "description": "run folder"}}},
         _read_result),
    Tool("explain",
         "Everything needed to explain a run's result to someone who is not a forecaster: the answer in plain "
         "sentences, how to read each table (what nats, F, L, beyond_linear, p_best mean), the caveats that must be "
         "mentioned, why the numbers can be trusted (independent checks that agreed, replications), and a flat list "
         "of facts: every number an explanation may quote, with its percentage and 95% interval. Explain only from "
         "this; then run check_explanation on the draft.",
         {"type": "object", "required": ["run"], "additionalProperties": False,
          "properties": {"run": {"type": "string", "description": "run folder (holds result.json)"}}},
         _explain),
    Tool("check_explanation",
         "Check a draft explanation of a run against its result: lists every number in the text that the result does "
         "not support, at the precision it is written with. Fix or remove each one before replying.",
         {"type": "object", "required": ["run", "text"], "additionalProperties": False,
          "properties": {"run": {"type": "string", "description": "run folder (holds result.json)"},
                         "text": {"type": "string", "description": "the draft explanation"}}},
         _check_explanation),
    Tool("list_forecasters",
         "List the forecasters horizoneval can run itself (baselines, invented models, foundation-model wrappers).",
         {"type": "object", "additionalProperties": False, "properties": {}},
         _forecasters),
]}


def call(name: str, arguments: dict | None = None) -> dict:
    """Run a tool and return its JSON document (canonical: what every adapter returns)."""
    if name not in TOOLS:
        raise KeyError(f"unknown tool {name!r}; tools: {', '.join(TOOLS)}")
    return json.loads(dumps(TOOLS[name].fn(**(arguments or {}))))


def call_json(name: str, arguments: dict | None = None) -> str:
    return dumps(call(name, arguments))


def definitions(style: str = "mcp") -> list[dict]:
    """Tool definitions for function calling: 'mcp' / 'anthropic' (name, description, input_schema) or 'openai'."""
    if style in ("mcp", "anthropic"):
        return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in TOOLS.values()]
    if style == "openai":
        return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                                  "parameters": t.input_schema}} for t in TOOLS.values()]
    raise ValueError("style must be 'mcp', 'anthropic' or 'openai'")
