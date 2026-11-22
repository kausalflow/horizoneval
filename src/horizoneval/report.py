"""Static HTML reports: one self-contained report.html per run, and an index over all runs.

No server and no external assets: open the files directly, or publish the folder (GitHub Pages, any static host).
"""

import html
import json
from pathlib import Path

import pandas as pd

CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e2e1dc;--card:#fff;--good:#1a7f4b;--bad:#b3261e;
--s1:#3366cc;--s2:#dc3912;--s3:#109618;--s4:#ff9900;--s5:#990099;--s6:#0099c6;--s7:#dd4477;--s8:#66aa00}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--muted:#a09f99;--line:#33322f;--card:#1e1e1c;
--good:#5cc98f;--bad:#f2867e;--s1:#7aa2f7;--s2:#f7768e;--s3:#9ece6a;--s4:#e0af68;--s5:#bb9af7;--s6:#7dcfff;--s7:#ff9e64;--s8:#c3e88d}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,-apple-system,sans-serif;margin:0 auto;max-width:1100px;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 8px}h3{font-size:14px;margin:16px 0 4px}
.muted{color:var(--muted)}.wrap{overflow-x:auto}table{border-collapse:collapse;font-variant-numeric:tabular-nums;width:100%}
th,td{border-bottom:1px solid var(--line);padding:4px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}
th{color:var(--muted);font-weight:600}.good{color:var(--good)}.bad{color:var(--bad)}
.grid{display:grid;gap:16px;grid-template-columns:repeat(auto-fill,minmax(320px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px}
svg text{fill:var(--muted);font-size:10px}.legend span{display:inline-block;margin-right:12px}
td.txt,th.txt{text-align:left}code{font-size:12px}a{color:var(--s1)}
"""

COLORS = [f"var(--s{i})" for i in range(1, 9)]


def _fmt(v, digits=3):
    if isinstance(v, float):
        return "" if pd.isna(v) else f"{v:.{digits}f}"
    return html.escape(str(v))


def _table(df: pd.DataFrame, classes: dict | None = None) -> str:
    text_cols = {c for c in df.columns if df[c].dtype == object}
    head = "".join(f'<th class="{"txt" if c in text_cols else ""}">{html.escape(str(c))}</th>' for c in df.columns)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            cls = (classes or {}).get(c, lambda v: "")(r[c]) or ("txt" if c in text_cols else "")
            cells.append(f'<td class="{cls}">{_fmt(r[c])}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="wrap"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def _line_chart(df: pd.DataFrame, models: list[str], w=320, h=180, pad=28) -> str:
    """Scaled absolute error by forecast step, one line per model."""
    xs, ys = df["step"], df["mase"]
    x0, x1, y0, y1 = xs.min(), max(xs.max(), xs.min() + 1), 0.0, ys.max() * 1.05 or 1
    sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - pad - 8)  # noqa: E731
    sy = lambda y: h - pad + 8 - (y - y0) / (y1 - y0) * (h - pad)  # noqa: E731
    parts = [f'<line x1="{pad}" y1="{sy(0)}" x2="{w - 8}" y2="{sy(0)}" stroke="var(--line)"/>']
    for t in [0.0, y1 / 2, y1 / 1.05]:
        parts.append(f'<text x="2" y="{sy(t) + 3}">{t:.2f}</text>')
    parts.append(f'<text x="{pad}" y="{h - 2}">step {x0}</text><text x="{w - 40}" y="{h - 2}">{x1}</text>')
    for i, m in enumerate(models):
        g = df[df["model"] == m].sort_values("step")
        if g.empty:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(g["step"], g["mase"]))
        parts.append(f'<polyline fill="none" stroke="{COLORS[i % 8]}" stroke-width="1.6" points="{pts}"/>')
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="scaled error by step">{"".join(parts)}</svg>'


def write_report(run_dir) -> Path:
    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "config.json").read_text())
    lb = pd.read_csv(run_dir / "leaderboard.csv")
    summary = pd.read_csv(run_dir / "summary.csv")
    tests = pd.read_csv(run_dir / "tests.csv")
    by_h = pd.read_csv(run_dir / "by_horizon.csv")
    ranks = json.loads((run_dir / "ranks.json").read_text())
    models = list(lb["model"])
    ref = cfg["reference"]

    skill_cls = {"mean_skill": lambda v: "good" if v > 0 else "bad" if v < 0 else ""}
    out = [f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Forecast benchmark</title><style>{CSS}</style></head><body>",
           f"<h1>Forecast benchmark</h1><p class=muted>{html.escape(run_dir.name)} · {cfg.get('created', '')} · "
           + (f"h={cfg['h']}, {cfg['windows']} windows every {cfg['step']}, context {cfg['context']}, "
              f"{cfg['n_series']} series per dataset, seed {cfg['seed']}" if "context" in cfg else
              f"scored from {html.escape(str(cfg.get('source', {}).get('forecasts', 'a forecast frame')))}: "
              f"h={cfg.get('h')}, {cfg.get('windows')} windows")
           + f" · reference <code>{html.escape(str(ref))}</code></p>"]
    if cfg.get("failures"):
        out.append("<p class=bad>Failed: " + ", ".join(f"<code>{html.escape(k)}</code> ({html.escape(v)})"
                                                     for k, v in cfg["failures"].items()) + "</p>")

    if (run_dir / "checks.json").exists():
        chk = [c for c in json.loads((run_dir / "checks.json").read_text()) if c["severity"] != "info"]
        if chk:
            out.append("<h2>Before you trust these numbers</h2>" + "".join(
                f'<p class="{"bad" if c["severity"] == "error" else ""}">{html.escape(c["severity"].upper())}: '
                f'{html.escape(c["message"])}</p>' for c in chk))
    if (run_dir / "decision.json").exists():
        dec = json.loads((run_dir / "decision.json").read_text())
        out.append("<h2>What to choose</h2>" + "".join(f"<p>{html.escape(t)}</p>" for t in dec.get("text", [])))
        conf = pd.read_csv(run_dir / "confidence.csv")
        out.append(f"<p class=muted>By {html.escape(dec['metric'])}. p_best: share of bootstrap resamples of the series "
                   "in which the model is best. expected_loss_rel: how much worse than the best it is on average, "
                   "counting the resamples where it is not best.</p>")
        out.append(_table(conf[["dataset", "model", "mean", "p_best", "expected_loss_rel", "series"]]))
    if (run_dir / "ladder.json").exists():
        lad = json.loads((run_dir / "ladder.json").read_text())
        out.append("<h2>Where the skill comes from</h2>" + "".join(f"<p>{html.escape(t)}</p>" for t in lad["text"]))
        out.append("<p class=muted>Skill in nats over repeating the last value, log(MSE<sub>naive</sub>/MSE); x nats "
                   "is a squared error lower by 1 &minus; e<sup>&minus;x</sup>. F: a simple rule chosen per series "
                   "(drift, seasonal naive, ...). L: a least-squares linear model fitted on the history. "
                   "beyond_linear: what the model adds on top of L; it counts only if the linear model is saturated, "
                   "it exceeds twice its standard error and the offset a fitted linear model leaves "
                   "(Ma, <i>Naive forecasters</i>, Result 4 and the decision rule).</p>")
        if lad["rungs"]:
            out.append(_table(pd.DataFrame(lad["rungs"])))
            m = pd.DataFrame(lad["models"])
            out.append(_table(m[m["group"] != "all"], {"beyond_linear": lambda v: "good" if v > 0 else "bad"}))
        from horizoneval.explain import HOW_TO_READ

        out.append("<h3>How to read these tables</h3>" + "".join(f"<p class=muted>{html.escape(t)}</p>"
                                                                  for t in HOW_TO_READ["ladder"]))
        warn = [f for f in lad.get("flags", []) if f["severity"] != "info"]
        good = [f for f in lad.get("flags", []) if f["severity"] == "info"]
        if warn:
            out.append("<h3>Read with care</h3>" + "".join(
                f"<p>{html.escape(f['dataset'])}: {html.escape(f['message'])}</p>" for f in warn))
        if good:
            out.append("<h3>Independent checks that agree</h3>" + "".join(
                f"<p class=good>{html.escape(f['dataset'])}: {html.escape(f['message'])}</p>" for f in good))
    out.append("<h2>Leaderboard</h2><p class=muted>Mean rank by " + cfg.get("rank_by", "mase").upper() + " over datasets (lower is better); skill is "
               f"mean log(MSE<sub>{ref}</sub>/MSE<sub>model</sub>) in nats (higher is better); CRPS is scaled like MASE; "
               "coverage80 should be near 0.80.</p>")
    out.append(_table(lb, skill_cls))
    if "critical_difference" in ranks:
        fr = f", Friedman p = {ranks['friedman_p']:.3g}" if "friedman_p" in ranks else ""
        out.append(f"<p class=muted>{ranks['datasets']} datasets, {ranks['models']} models{fr}. Nemenyi critical "
                   f"difference of mean ranks at 5%: {ranks['critical_difference']:.2f}. Models whose mean ranks "
                   "differ by less are not distinguishable at this level.</p>")

    out.append(f"<h2>Per dataset</h2><p class=muted>Mean with 95% bootstrap interval over series; p is the paired "
               f"test against <code>{ref}</code> on per-series losses (Diebold–Mariano style).</p>")
    for metric in tests["metric"].unique():
        t = tests[tests["metric"] == metric].copy()
        t["ci"] = t.apply(lambda r: f"[{r.ci_low:.3f}, {r.ci_high:.3f}]", axis=1)
        cols = ["dataset", "model", "mean", "ci"] + [c for c in ["t_vs_ref", "p_vs_ref"] if c in t]
        t = t[cols].sort_values(["dataset", "mean"])
        cls = {"t_vs_ref": lambda v: "good" if v < 0 else "bad" if v > 0 else ""}
        out.append(f"<h3>{html.escape(metric)}</h3>" + _table(t, cls))

    out.append("<h2>Error by forecast step</h2><p class=muted>Scaled absolute error (MASE per step), averaged over "
               "series and windows.</p><p class=legend>" + "".join(
                   f'<span style="color:{COLORS[i % 8]}">■ {html.escape(m)}</span>' for i, m in enumerate(models)) + "</p>")
    out.append('<div class="grid">' + "".join(
        f'<div class="card"><h3>{html.escape(d)}</h3>{_line_chart(g, models)}</div>'
        for d, g in by_h.groupby("dataset")) + "</div>")

    out.append("<h2>All scores</h2>" + _table(summary.sort_values(["dataset", "rank"])))
    out.append("<h2>Reproduce</h2><p class=muted>Dataset fingerprints (sha256 of the drawn arrays):</p>" + _table(
        pd.DataFrame([{"dataset": k, **v} for k, v in cfg["datasets"].items()])))
    out.append("</body></html>")
    path = run_dir / "report.html"
    path.write_text("\n".join(out))
    return path


def build_index(results_dir="results") -> Path:
    """results/index.html: every run with its config and winner, newest first."""
    results_dir = Path(results_dir)
    rows = []
    for cfg_path in sorted(results_dir.glob("*/config.json"), reverse=True):
        run = cfg_path.parent
        cfg = json.loads(cfg_path.read_text())
        lb = pd.read_csv(run / "leaderboard.csv")
        if not (run / "report.html").exists():
            write_report(run)
        rows.append(f"<tr><td><a href='{html.escape(run.name)}/report.html'>{html.escape(run.name)}</a></td>"
                    f"<td>{len(cfg['datasets'])}</td><td>{len(lb)}</td><td>{cfg['h']}</td>"
                    f"<td>{html.escape(lb['model'].iloc[0])}</td></tr>")
    page = (f"<!doctype html><html lang=en><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
            f"initial-scale=1'><title>Forecast runs</title><style>{CSS}</style></head><body><h1>Forecast runs</h1>"
            "<div class=wrap><table><thead><tr><th>run</th><th>datasets</th><th>models</th><th>h</th><th>best mean rank</th>"
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div></body></html>")
    path = results_dir / "index.html"
    path.write_text(page)
    return path
