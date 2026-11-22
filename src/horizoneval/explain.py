"""Explain a result to someone who is not a forecaster, and check that an explanation invents no numbers.

An AI assistant explaining horizoneval's output gets two tools (horizoneval.tools: `explain`, `check_explanation`):

    explain(result)              the answer in plain sentences, how to read every table, the caveats that must be
                                 mentioned, why the numbers can be trusted, and a flat list of facts (every number an
                                 explanation may quote, with its unit, its percentage and its 95% interval)
    check_numbers(text, result)  every number in a draft explanation that the result does not support; an assistant
                                 runs it before replying and fixes what it lists

Everything here is computed from the result document (result.json) alone, deterministically, so two assistants
explaining the same result start from the same facts, and a sentence can always be traced to a field.
"""

import math
import re

import numpy as np

EXPLANATION_VERSION = "1.0"

HOW_TO_READ = {
    "choice": [
        "The verdict names the model to use on these series and how sure that is. p_best is the share of resamples "
        "of the series in which the model is the best one: 0.9 means it wins 9 times out of 10 when the set of "
        "series is redrawn. expected_loss_rel is how much worse than the best it is on average, counting the cases "
        "where another model wins: 0.01 means 1% worse.",
        "claim.verified is true only when the winner's lead over the runner-up survives a test whose false-alarm "
        "rate was checked on this very data. winners_curse is the score of the chosen model on windows or series it "
        "was not chosen on: the honest number to expect next time.",
    ],
    "ladder": [
        "Every number is skill in nats over repeating the last value: log(MSE of repeating the last value / MSE). "
        "Higher is better, 0 means no better than repeating the last value, negative means worse. x nats means a "
        "squared error lower by 1 - e^-x (column *_pct) or e^x times smaller: 0.1 nats is about 10% lower, 0.7 "
        "about 50% lower (2 times smaller), 2.3 about 90% lower (10 times smaller), 4.6 about 99% lower (100 times "
        "smaller).",
        "The gains add up: total = F + L + beyond_linear. F is what the best simple rule gets (drift, seasonal naive, "
        "the mean, ... chosen per series on the history). L is what a linear model fitted on the history adds on top "
        "of that rule. beyond_linear is what the model adds on top of the linear model. F and L describe the data, "
        "not any model; beyond_linear and total belong to the model.",
        "group is the forecast step: 1 is one step ahead, 2-4 and 5+ further ahead. The row 'all' pools squared "
        "errors over every step, so it is dominated by the far steps, where errors are largest; it is not the "
        "average of the other rows.",
        "*_se is the standard error over series: a value within about two standard errors of 0 is not distinguishable "
        "from 0. saturated says whether giving the linear model more memory had stopped helping it on the history; "
        "if not, L is a lower bound.",
        "verdict, per group: 'beyond linear' (the model beats the linear model by more than twice the standard error "
        "and more than the offset a fitted linear model leaves, with a saturated reference), 'no better than linear' "
        "(it does not beat it at all), 'not shown' (ahead, but not by enough), 'reference not saturated'. credited "
        "is the overall answer: 'beyond linear' only if every group says so.",
    ],
}

TRUST = [
    "Every score, including the ladder's rungs, goes through one function (horizoneval.evaluate); there is no "
    "second way to compute a number.",
    "That scoring reproduces published results: all 98 numbers of the official M4 table and all 400 GIFT-Eval "
    "leaderboard numbers of seasonal naive and naive (results/replication-m4, results/replication-gift-eval).",
    "The ladder reproduces its paper's calibration: on series with a known optimum the linear rung recovers the true "
    "value (0.298 against 0.288 nats) and the exact optimal model is never credited with skill beyond linear "
    "(0 of 12 seeds; results/replication-ladder, horizoneval replicate ladder).",
    "The decision statistics (probability of best, winner's curse, verified claims) are checked by simulation on "
    "cases with known answers (horizoneval.verify, tests/).",
]

RULES = [
    "Lead with the answer (the 'answer' sentences), in plain language, close to verbatim.",
    "Quote only numbers that appear in 'facts', 'answer', 'caveats' or 'trust'; round them, never compute new ones "
    "except converting nats to percent with the *_pct fields already given.",
    "Mention every caveat with severity 'warning' or 'error' that concerns what the user asked about.",
    "Say how sure the result is (p_best, standard errors, intervals); do not present a difference within two "
    "standard errors as real.",
    "Before replying, run check_explanation on the draft and remove or fix every number it lists.",
]


def _pct(x):
    return 100 * (1 - math.exp(-x)) if x is not None and math.isfinite(x) else None


def _times(x):
    return math.exp(x) if x is not None and math.isfinite(x) and abs(x) < 700 else None


def _fact(id_, label, value, unit, se=None):
    f = {"id": id_, "label": label, "value": value, "unit": unit}
    if value is None:
        return f
    if unit == "nats":
        f["percent_lower_squared_error"] = _pct(value)
        f["times_smaller_squared_error"] = _times(value)
    if se is not None and math.isfinite(se):
        f["se"] = se
        f["ci95"] = [value - 1.96 * se, value + 1.96 * se]
        if unit == "nats":
            f["ci95_percent"] = [_pct(value - 1.96 * se), _pct(value + 1.96 * se)]
    return f


def facts(doc: dict) -> list[dict]:
    """Every number an explanation may quote, labelled in plain words."""
    out = []
    for ds, c in (doc.get("choice") or {}).items():
        out.append(_fact(f"choice.{ds}.p_best", f"{ds}: probability that {c['model']} is the best model",
                         c.get("p_best"), "probability"))
        out.append(_fact(f"choice.{ds}.expected_loss_rel", f"{ds}: how much worse than the best {c['model']} is "
                         "on average", c.get("expected_loss_rel"), "fraction"))
    for row in doc.get("summary") or []:
        for k in ("mase", "smape", "crps_scaled", "coverage80", "skill", "cost"):
            if row.get(k) is not None:
                out.append(_fact(f"summary.{row['dataset']}.{row['model']}.{k}", f"{row['dataset']}: {k} of "
                                 f"{row['model']}", row[k], "nats" if k == "skill" else k))
    lad = doc.get("ladder") or {}
    names = {"F": "the best simple rule (F)", "L": "what a linear model adds on top (L)",
             "F_plus_L": "the linear model in all (F + L)"}
    for r in lad.get("rungs") or []:
        for k, name in names.items():
            out.append(_fact(f"ladder.{r['dataset']}.{r['group']}.{k}", f"{r['dataset']}, steps {r['group']}: "
                             f"{name}", r.get(k), "nats", r.get(f"{k}_se")))
    for r in lad.get("models") or []:
        for k, name in (("total", "total skill"), ("beyond_linear", "skill beyond the linear model")):
            f = _fact(f"ladder.{r['dataset']}.{r['group']}.{r['model']}.{k}", f"{r['dataset']}, steps {r['group']}: "
                      f"{name} of {r['model']}", r.get(k), "nats", r.get(f"{k}_se"))
            if k == "beyond_linear":
                f["verdict"], f["offset"] = r.get("verdict"), r.get("offset")
            out.append(f)
    for fl in lad.get("flags") or []:
        for k, v in (fl.get("values") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                out.append(_fact(f"flag.{fl['dataset']}.{fl['flag']}.{k}", f"{fl['dataset']}: {k} ({fl['flag']})",
                                 v, "nats" if k in ("measured", "from_history", "low", "high", "F_plus_L", "value",
                                                    "difference", "se") else k))
    return out


def explain(doc: dict) -> dict:
    """What an assistant needs to explain a result: answer, how to read it, caveats, trust, facts and rules."""
    lad = doc.get("ladder") or {}
    flags = lad.get("flags") or []
    caveats = [{"severity": c["severity"], "message": c["message"], "check": c["check"]}
               for c in doc.get("checks") or [] if c.get("severity") in ("warning", "error")]
    caveats += [{"severity": f["severity"], "message": f"{f['dataset']}: {f['message']}", "flag": f["flag"]}
                for f in flags if f["severity"] in ("warning", "error")]
    trust = TRUST + [f"{f['dataset']}: {f['message']}" for f in flags if f["severity"] == "info"]
    read = {"choice": HOW_TO_READ["choice"]}
    if lad:
        read["ladder"] = HOW_TO_READ["ladder"]
    return {"horizoneval_explanation": EXPLANATION_VERSION, "answer": list(doc.get("verdict") or []) + list(
        lad.get("text") or []), "how_to_read": read, "caveats": caveats, "trust": trust, "facts": facts(doc),
        "rules": RULES}


# --------------------------------------------------------------------------------------------------------------------
# Grounding: no number in an explanation that the result does not support
# --------------------------------------------------------------------------------------------------------------------

_NUMBER = re.compile(r"(?<![\w.])([-−]?\d{1,3}(?:,\d{3})+|[-−]?\d+(?:\.\d+)?)(\s*(?:%|percent|times|x\b|×))?")
_IN_TEXT = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _numbers(x, out):
    if isinstance(x, dict):
        for v in x.values():
            _numbers(v, out)
    elif isinstance(x, (list, tuple)):
        for v in x:
            _numbers(v, out)
    elif isinstance(x, bool) or x is None:
        return out
    elif isinstance(x, (int, float)):
        if math.isfinite(x):
            out.add(float(x))
    elif isinstance(x, str):
        out.update(float(m) for m in _IN_TEXT.findall(x))
    return out


def supported_numbers(doc: dict) -> np.ndarray:
    """Every number the result supports: its own values (and those inside its sentences), the facts' percentages,
    ratios and intervals, and fractions read as percent."""
    vals = _numbers(doc, set())
    _numbers(explain(doc), vals)
    vals |= {v * 100 for v in list(vals) if abs(v) <= 1}
    vals |= {abs(v) for v in list(vals)}
    return np.array(sorted(vals))


def check_numbers(text: str, doc: dict) -> dict:
    """Numbers in `text` that `doc` does not support, at the precision they are written with.

    A number is supported when a value of the result rounds to it (0.82 or 0.8 for 0.822; 56% for 0.5603 read as a
    fraction or for a *_pct field), or, written with at most two significant digits ('about 12,000 times'), lies
    within 5% of one. Small whole numbers that are not percentages or ratios (counts, steps, '2-4') are not checked."""
    allowed = supported_numbers(doc)
    bad, checked = [], 0
    for m in _NUMBER.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").strip()
        t = float(raw.replace(",", "").replace("−", "-"))
        if "." not in raw and abs(t) <= 24 and not unit:
            continue
        checked += 1
        dec = len(raw.split(".")[1]) if "." in raw else 0
        half = 0.5 * 10 ** -dec + 1e-9
        ok = bool(np.any(np.abs(allowed - abs(t)) <= half))
        digits = raw.replace(",", "").replace("-", "").replace("−", "").replace(".", "").lstrip("0").rstrip("0")
        if not ok and len(digits) <= 2 and abs(t) >= 10:
            ok = bool(np.any(np.abs(allowed - abs(t)) <= 0.05 * abs(t)))
        if not ok:
            bad.append(raw + (f" {unit}" if unit else ""))
    return {"ok": not bad, "unsupported": bad, "checked": checked}
