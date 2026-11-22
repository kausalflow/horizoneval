"""horizoneval list | horizoneval run --models ... --data ... | horizoneval score forecasts.parquet --train train.parquet

Examples:
    uv run horizoneval list
    uv run horizoneval run --models seasonal_naive drift damped_drift --data synthetic
    uv run --extra tsfm horizoneval run --models chronos_bolt_small auto_ets damped_drift --data synthetic --out results/first-look
    uv run horizoneval index            # results/index.html over all runs
    uv run horizoneval score cv.parquet --train train.parquet --period 24 --reference SeasonalNaive --out results/x
"""

import argparse
import time

import pandas as pd

from horizoneval.data import SUITES
from horizoneval.data.synthetic import SYNTHETIC
from horizoneval.registry import available


def main(argv=None):
    p = argparse.ArgumentParser(prog="horizoneval")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list forecasters, datasets and suites")
    rep = sub.add_parser("report", help="rebuild report.html of a run folder")
    rep.add_argument("run_dir")
    idx = sub.add_parser("index", help="build an index.html over all runs")
    idx.add_argument("results_dir", nargs="?", default="results")
    rp = sub.add_parser("replicate", help="reproduce published results with horizoneval's scoring (needs --extra replicate)")
    rp.add_argument("benchmark", choices=["m4", "gift-eval"])
    rp.add_argument("--out", default=None, help="results folder (default: results/replication-<benchmark>)")
    ck = sub.add_parser("check", help="pre-flight checks on a forecast frame, without scoring it")
    ck.add_argument("forecasts")
    ck.add_argument("--train", default=None)
    ck.add_argument("--json", action="store_true", help="print the checks as JSON (horizoneval.tools 'check')")
    df = sub.add_parser("diff", help="say why two run folders disagree: every setting that differs, and the scores it moved")
    df.add_argument("run_a")
    df.add_argument("run_b")
    df.add_argument("--json", action="store_true", help="print the diff as JSON (horizoneval.tools 'diff')")
    sc = sub.add_parser("score", help="score a forecast frame made by any tool (Nixtla cross_validation layout)")
    sc.add_argument("forecasts", help="parquet or CSV with unique_id, ds, cutoff, y and one column per model")
    sc.add_argument("--train", default=None, help="history (unique_id, ds, y), needed for MASE and scaled CRPS")
    sc.add_argument("--series", default=None, help="optional table (unique_id, dataset, period)")
    sc.add_argument("--period", type=int, default=None, help="seasonal period of every series (default: 1)")
    sc.add_argument("--dataset", default="data", help="dataset name when --series gives none")
    sc.add_argument("--reference", default=None, help="model column for skill and paired tests (default: none)")
    sc.add_argument("--out", default=None, help="results folder (default: results/<timestamp>)")
    sc.add_argument("--cost", default=None, help="your own cost, e.g. linlin:0.9 (shortfalls cost 9x overshoots) or "
                    "pinball:0.95; it decides the choice")
    sc.add_argument("--smape-zero", default="zero", choices=["zero", "skip"],
                    help="sMAPE where truth and forecast are both 0: zero (M4, utilsforecast) or skip (gluonts)")
    sc.add_argument("--json", action="store_true", help="print the result document (result.json) instead of a table")
    sc.add_argument("--segment", default=None, help="column of --series whose groups may deserve different models")
    r = sub.add_parser("run", help="benchmark forecasters")
    r.add_argument("--models", nargs="+", required=True)
    r.add_argument("--data", nargs="+", default=["synthetic"], help="suites or synthetic:<name> / gift:<dataset>/<freq>")
    r.add_argument("--h", type=int, default=12, help="forecast horizon")
    r.add_argument("--windows", type=int, default=5, help="rolling windows per series")
    r.add_argument("--step", type=int, default=None, help="distance between cutoffs (default: h)")
    r.add_argument("--context", type=int, default=256, help="history length before the first cutoff")
    r.add_argument("--n-series", type=int, default=30)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--reference", default="naive", help="model the skill column is measured against")
    r.add_argument("--out", default=None, help="results folder (default: results/<timestamp>)")
    a = p.parse_args(argv)

    if a.cmd == "list":
        print("forecasters:", " ".join(available()))
        print("synthetic:  ", " ".join(f"synthetic:{k}" for k in SYNTHETIC))
        for name, items in SUITES.items():
            print(f"suite {name}:", " ".join(items))
        return

    if a.cmd == "report":
        from horizoneval.report import write_report

        print(write_report(a.run_dir))
        return
    if a.cmd == "index":
        from horizoneval.report import build_index

        print(build_index(a.results_dir))
        return

    if a.cmd == "check":
        from horizoneval.checks import preflight, summary
        from horizoneval.frame import read

        found = preflight(read(a.forecasts), read(a.train) if a.train else None)
        if a.json:
            from horizoneval import tools

            print(tools.call_json("check", {k: v for k, v in dict(forecasts=a.forecasts, train=a.train).items() if v}))
            raise SystemExit(1 if any(c["severity"] == "error" for c in found) else 0)
        print(summary(found))
        for c in found:
            print(f"- {c['severity'].upper()} [{c['check']}] {c['message']}")
        raise SystemExit(1 if any(c["severity"] == "error" for c in found) else 0)
    if a.cmd == "diff":
        from horizoneval.settings import diff

        d = diff(a.run_a, a.run_b)
        if a.json:
            from horizoneval import tools

            print(tools.call_json("diff", {"run_a": a.run_a, "run_b": a.run_b}))
            return
        print("\n".join(d["text"]))
        for c in d["causes"]:
            print(f"- {c['why']} ({c['setting']}, {c['where']}): {c['a']} -> {c['b']}")
        if d["moved"]:
            print(f"\n{len(d['moved'])} scores moved, e.g.:")
            for r in d["moved"][:10]:
                print(f"  {r['metric']} {r['model']} on {r['dataset']}: {r['a']:.6g} -> {r['b']:.6g}")
        return
    if a.cmd == "replicate":
        from horizoneval.replicate import gift_run, m4_run

        a.out = a.out or f"results/replication-{a.benchmark}"
        _, cmp = (m4_run if a.benchmark == "m4" else gift_run)(a.out)
        with pd.option_context("display.width", 160):
            print(cmp.to_string(index=False))
        print(f"\n{int(cmp['match'].sum())} of {len(cmp)} published numbers reproduced; results in {a.out}/")
        return
    if a.cmd == "score":
        from horizoneval import tools

        out = a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}"
        res = tools.call("score", {k: v for k, v in dict(
            forecasts=a.forecasts, train=a.train, series=a.series, period=a.period, dataset=a.dataset,
            reference=a.reference, cost=a.cost, segment=a.segment, smape_zero=a.smape_zero, out=out).items()
            if v is not None})
        if a.json:
            print(tools.dumps(res))
            return
        with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
            print(pd.DataFrame(res["leaderboard"]).to_string(index=False))
        print("\n" + "\n".join(res["verdict"]))
        if res["failures"]:
            print("not scored:", res["failures"])
        print(f"\nresults in {out}/ (open {out}/report.html)")
        return

    from horizoneval.benchmark import run

    out = a.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}"
    lb, _, _ = run(a.models, a.data, h=a.h, windows=a.windows, step=a.step, context=a.context,
                   n_series=a.n_series, seed=a.seed, out=out, reference=a.reference)
    with pd.option_context("display.width", 160, "display.float_format", "{:.3f}".format):
        print(lb.to_string(index=False))
    print(f"\nresults in {out}/ (open {out}/report.html)")


if __name__ == "__main__":
    main()
