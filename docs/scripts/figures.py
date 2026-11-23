"""Build the docs figures from committed run folders: plain SVG and HTML snippets, no plotting library.

    uv run python docs/scripts/figures.py            # writes docs/assets/figures/*

Every number drawn comes from a run folder under results/ (default: results/ladder-synthetic), so the pictures in the
docs are real outputs and change when the run is re-committed. Colours are CSS classes (hz-*), styled in
docs/stylesheets/extra.css for light and dark mode, so the snippets are inlined into pages rather than linked as images.
"""

import html
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results" / "ladder-synthetic"
OUT = ROOT / "docs" / "assets" / "figures"

# one colour slot per model, the same on every figure
SLOT = {"seasonal_naive": 1, "naive": 2, "drift": 3, "damped_drift": 4, "mean": 5}


def esc(s) -> str:
    return html.escape(str(s))


def short(dataset: str) -> str:
    return dataset.replace("synthetic:", "").replace("_", " ")


def svg(w, h, body, label, wide=False):
    return (f'<svg class="hz-fig{" hz-wide" if wide else ""}" viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" '
            f'xmlns="http://www.w3.org/2000/svg" preserveAspectRatio="xMidYMid meet">{body}</svg>\n')


def path(xs, ys):
    return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))


def scale(lo, hi, a, b):
    return lambda v: a + (np.asarray(v, dtype=float) - lo) / (hi - lo or 1) * (b - a)


def hero_backtest(f: pd.DataFrame, train: pd.DataFrame, sid: str = "synthetic:van_der_pol#0001") -> str:
    """One periodic series, its last three cutoffs, the truth and two forecasts with the 80% band."""
    s = f[f["unique_id"] == sid]
    cuts = sorted(s["cutoff"].unique())[-3:]
    hist = train[(train["unique_id"] == sid) & (train["ds"] <= cuts[0])].tail(60)
    truth = pd.concat([hist[["ds", "y"]], s[s["cutoff"].isin(cuts)][["ds", "y"]]]).drop_duplicates("ds").sort_values("ds")
    W, H, L, R, T, B = 960, 360, 16, 16, 34, 40
    win = s[s["cutoff"].isin(cuts)]
    lo_all = min(truth["y"].min(), win["seasonal_naive-q-10"].min(), win["naive-q-10"].min())
    hi_all = max(truth["y"].max(), win["seasonal_naive-q-90"].max(), win["naive-q-90"].max())
    pad = 0.06 * (hi_all - lo_all)
    X = scale(truth["ds"].min(), truth["ds"].max(), L, W - R)
    Y = scale(lo_all - pad, hi_all + pad, H - B, T)
    out = []
    for i, c in enumerate(cuts):
        w = s[s["cutoff"] == c]
        x0, x1 = float(X(c)), float(X(w["ds"].max()))
        out.append(f'<rect class="hz-window" x="{x0:.1f}" y="{T - 6}" width="{x1 - x0:.1f}" height="{H - B - T + 6}" rx="6"/>')
        out.append(f'<line class="hz-cut" x1="{x0:.1f}" x2="{x0:.1f}" y1="{T - 10}" y2="{H - B}"/>')
        out.append(f'<text class="hz-small" x="{x0 + 6:.1f}" y="{T + 8}">cutoff {i + 1}</text>')
    for i, c in enumerate(cuts):
        w = s[s["cutoff"] == c].sort_values("ds")
        xs = X(w["ds"])
        band = path(xs, Y(w["seasonal_naive-q-90"])) + " L" + " L".join(
            f"{x:.1f},{y:.1f}" for x, y in zip(xs[::-1], Y(w["seasonal_naive-q-10"])[::-1])) + " Z"
        out.append(f'<path class="hz-band hz-s1" d="{band}" style="animation-delay:{0.5 + 0.5 * i:.1f}s"/>')
        out.append(f'<path class="hz-line hz-dash hz-s2 hz-draw" d="{path(xs, Y(w["naive"]))}" '
                   f'style="animation-delay:{0.5 + 0.5 * i:.1f}s"/>')
        out.append(f'<path class="hz-line hz-s1 hz-draw" d="{path(xs, Y(w["seasonal_naive"]))}" '
                   f'style="animation-delay:{0.6 + 0.5 * i:.1f}s"/>')
    out.insert(0, f'<path class="hz-truth" d="{path(X(truth["ds"]), Y(truth["y"]))}"/>')
    xh = float(X(hist["ds"].iloc[len(hist) // 3]))
    out.append(f'<text class="hz-small" x="{xh:.1f}" y="{H - B + 24}">history: what the forecaster may see</text>')
    out.append(f'<text class="hz-small" x="{float(X(cuts[0])) + 6:.1f}" y="{H - B + 24}">hidden, then revealed and scored</text>')
    legend = [("hz-truth-key", "what happened"), ("hz-s1", "seasonal naive, with its 80% band"), ("hz-s2 hz-dashkey", "naive")]
    lx = L
    for cls, text in legend:
        out.append(f'<line class="hz-key {cls}" x1="{lx}" x2="{lx + 22}" y1="12" y2="12"/>'
                   f'<text class="hz-small" x="{lx + 28}" y="16">{esc(text)}</text>')
        lx += 40 + 7.2 * len(text)
    return svg(W, H, "".join(out), wide=True, label="A periodic series with three backtest windows: history, cutoffs, forecasts and the truth")


def bars(rows, value_fmt, label, W=540, row_h=34, left=112, right=110):
    """Horizontal bars. rows: (name, value, slot, note)."""
    vals = np.array([r[1] for r in rows], dtype=float)
    lo, hi = min(0.0, vals.min()), max(0.0, vals.max())
    H = row_h * len(rows) + 16
    X = scale(lo, hi, left, W - right)
    x0 = float(X(0))
    out = [f'<line class="hz-axis" x1="{x0:.1f}" x2="{x0:.1f}" y1="4" y2="{H - 8}"/>']
    for i, (name, v, slot, note) in enumerate(rows):
        y = 8 + i * row_h
        x1 = float(X(v))
        a, b = sorted([x0, x1])
        out.append(f'<text class="hz-label" x="{left - 10}" y="{y + row_h / 2 + 1}" text-anchor="end">{esc(name)}</text>')
        out.append(f'<rect class="hz-bar hz-s{slot}" x="{a:.1f}" y="{y + 5}" width="{max(b - a, 1.5):.1f}" '
                   f'height="{row_h - 12}" rx="4" style="animation-delay:{0.08 * i:.2f}s"/>')
        tx, anchor = (b + 8, "start") if v >= 0 else (x0 + 8, "start")  # labels never cover the names
        out.append(f'<text class="hz-value" x="{tx:.1f}" y="{y + row_h / 2 + 1}" text-anchor="{anchor}">{esc(value_fmt(v))}'
                   + (f' <tspan class="hz-note">{esc(note)}</tspan>' if note else "") + "</text>")
    return svg(W, H, "".join(out), label)


def skill_chart(lb: pd.DataFrame) -> str:
    def note(v):
        if abs(v) < 0.005:
            return "same as naive"
        r = np.exp(abs(v))
        return f"{r:.1f}× {'less' if v > 0 else 'more'} squared error"
    rows = [(m, v, SLOT.get(m, 6), "reference" if m == "naive" else note(v))
            for m, v in zip(lb["model"], lb["mean_skill"])]
    return bars(rows, lambda v: f"{v:+.2f} nats", "Skill of each model over naive", right=200)


def pbest_chart(conf: pd.DataFrame, dataset: str) -> str:
    c = conf[conf["dataset"] == dataset].sort_values("p_best", ascending=False)
    rows = [(m, p, SLOT.get(m, 6), "") for m, p in zip(c["model"], c["p_best"])]
    return bars(rows, lambda v: f"{v:.0%}", f"Probability that each model is the best on {dataset}", right=60)


def horizon_chart(bh: pd.DataFrame, dataset: str) -> str:
    d = bh[bh["dataset"] == dataset]
    W, H, L, R, T, B = 720, 300, 52, 130, 14, 40
    X = scale(1, d["step"].max(), L, W - R)
    Y = scale(0, d["mase"].max() * 1.08, H - B, T)
    out = []
    for t in np.linspace(0, d["mase"].max(), 4):
        y = float(Y(t))
        out.append(f'<line class="hz-grid" x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}"/>'
                   f'<text class="hz-small" x="{L - 8}" y="{y + 4:.1f}" text-anchor="end">{t:.1f}</text>')
    for k in (1, 4, 8, 12, 16):
        if k <= d["step"].max():
            out.append(f'<text class="hz-small" x="{float(X(k)):.1f}" y="{H - B + 18}" text-anchor="middle">{k}</text>')
    out.append(f'<text class="hz-small" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">steps ahead</text>')
    ends = []
    for m, g in d.groupby("model"):
        g = g.sort_values("step")
        out.append(f'<path class="hz-line hz-s{SLOT.get(m, 6)} hz-draw" d="{path(X(g["step"]), Y(g["mase"]))}"/>')
        ends.append([float(Y(g["mase"].iloc[-1])), m])
    ends.sort()
    for i in range(1, len(ends)):  # keep end labels apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 14)
    for y, m in ends:
        out.append(f'<text class="hz-label hz-t{SLOT.get(m, 6)}" x="{W - R + 8}" y="{y + 4:.1f}">{esc(m)}</text>')
    return svg(W, H, "".join(out), f"Scaled error by forecast step on {dataset}", wide=True)


def ladder_chart(rungs: pd.DataFrame) -> str:
    r = rungs[rungs["group"] == "all"].copy()
    r = r[r["F_plus_L_pct"] > 0].sort_values("F_plus_L_pct", ascending=False)
    W, row_h, left, right = 540, 40, 112, 50
    H = row_h * len(r) + 40
    X = scale(0, 100, left, W - right)
    out = []
    for i, (_, x) in enumerate(r.iterrows()):
        y = 8 + i * row_h
        f_end, l_end = float(X(x["F_pct"])), float(X(x["F_plus_L_pct"]))
        out.append(f'<text class="hz-label" x="{left - 10}" y="{y + row_h / 2 + 1}" text-anchor="end">{esc(short(x["dataset"]))}</text>')
        out.append(f'<rect class="hz-track" x="{left}" y="{y + 6}" width="{W - right - left}" height="{row_h - 14}" rx="5"/>')
        out.append(f'<rect class="hz-bar hz-f" x="{left}" y="{y + 6}" width="{max(f_end - left, 1):.1f}" height="{row_h - 14}" rx="5"/>')
        out.append(f'<rect class="hz-bar hz-l" x="{f_end:.1f}" y="{y + 6}" width="{max(l_end - f_end, 0):.1f}" height="{row_h - 14}" rx="0"/>')
        out.append(f'<text class="hz-value" x="{W - right + 8}" y="{y + row_h / 2 + 1}">{x["F_plus_L_pct"]:.0f}%</text>')
    y = H - 14
    out.append(f'<rect class="hz-f" x="{left}" y="{y - 10}" width="14" height="12" rx="3"/>'
               f'<text class="hz-small" x="{left + 20}" y="{y}">free: best simple rule</text>'
               f'<rect class="hz-l" x="{left + 180}" y="{y - 10}" width="14" height="12" rx="3"/>'
               f'<text class="hz-small" x="{left + 200}" y="{y}">linear: what it adds</text>')
    return svg(W, H, "".join(out), "Share of squared error removed by simple rules and by a linear model, per dataset")


def explain_demo(run: Path) -> str:
    """The run's own plain-language answer, and the checker catching a number a draft made up."""
    ex = json.loads((run / "explanation.json").read_text())
    answer = next(a for a in ex["answer"] if a.startswith("(all)")).replace("(all): ", "")
    seg = next((a for a in ex["answer"] if "per segment" in a), "")
    draft = ("Seasonal naive is the best choice overall, with probability 1.00, and it has about 86% less squared "
             "error than repeating the last value. On the random walk, drift wins 73.5% of the time.")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write(draft)
    res = subprocess.run([sys.executable, "-m", "horizoneval.cli", "explain", str(run), "--check", fh.name],
                         capture_output=True, text=True, cwd=ROOT)
    Path(fh.name).unlink()
    check = json.loads(res.stdout.strip().splitlines()[-1])
    marked = esc(draft)
    for bad in check["unsupported"]:
        num = bad.replace(" %", "%")
        marked = marked.replace(esc(num), f'<mark class="hz-bad" title="not supported by the result">{esc(num)}</mark>')
    warn = next((c["message"] for c in ex["caveats"] if c["severity"] == "warning"), None)
    parts = [
        '<div class="hz-chat">',
        '<div class="hz-msg hz-user"><span class="hz-who">You</span>Which forecast should we use?</div>',
        f'<div class="hz-msg hz-bot"><span class="hz-who">horizoneval explain</span>{esc(answer)}'
        + (f'<p class="hz-aside">{esc(seg)}</p>' if seg else "") + "</div>",
        f'<div class="hz-msg hz-bot hz-check"><span class="hz-who">check_explanation on an assistant\'s draft</span>'
        f'<p>{marked}</p><p class="hz-verdict">{check["checked"]} numbers checked · '
        f'{len(check["unsupported"])} not supported by the result: '
        + ", ".join(f"<code>{esc(u)}</code>" for u in check["unsupported"]) + "</p></div>",
    ]
    if warn:
        parts.append(f'<div class="hz-msg hz-bot hz-warn"><span class="hz-who">a caveat it must mention</span>{esc(warn)}</div>')
    parts.append("</div>")
    return "\n".join(parts) + "\n"


def main(run=RUN):
    run = Path(run)
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(run / "forecasts.parquet")
    train = pd.read_parquet(run / "train.parquet")
    lb = pd.read_csv(run / "leaderboard.csv")
    conf = pd.read_csv(run / "confidence.csv")
    bh = pd.read_csv(run / "by_horizon.csv")
    rungs = pd.read_csv(run / "ladder_rungs.csv")
    files = {
        "hero-backtest.svg": hero_backtest(f, train),
        "skill.svg": skill_chart(lb),
        "pbest.svg": pbest_chart(conf, "synthetic:ar_differences"),
        "horizon.svg": horizon_chart(bh, "synthetic:seasonal_trend"),
        "ladder.svg": ladder_chart(rungs),
        "explain.html": explain_demo(run),
    }
    for name, text in files.items():
        (OUT / name).write_text(text)
        print("wrote", (OUT / name).relative_to(ROOT))


if __name__ == "__main__":
    main(*sys.argv[1:])
