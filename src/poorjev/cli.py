"""poorjev command line.

    poorjev eval   --set evalset/tasks.jsonl [--plots] [--verbose]
    poorjev ask    --state-file ticket.txt --questions questions.yaml   (M4+)
    poorjev calibrate --set evalset/tasks.jsonl                          (M4)

M3 ships ``eval``: run the set through the local backend and print accuracy +
ECE + Brier + AURC, overall and per task. ``--plots`` writes the reliability
diagram (the launch asset) once matplotlib is installed.
"""

from __future__ import annotations

import argparse
import sys


def _fmt(summary: dict) -> str:
    return (
        f"n={summary['n']:>4}  "
        f"acc={summary['accuracy']:.3f}  "
        f"ECE={summary['ece']:.3f}  "
        f"Brier={summary['brier']:.3f}  "
        f"AURC={summary['aurc']:.3f}"
    )


def cmd_eval(args) -> int:
    from .evaluate import evaluate, report
    from .client import Client

    print(f"Running eval set: {args.set}")
    print("(first run downloads a ~400MB model; then offline)\n")
    client = Client()
    records, _ = evaluate(args.set, client=client, verbose=args.verbose)
    rep = report(records)

    print("\n== overall ==")
    print("  " + _fmt(rep["overall"]))
    print("\n== by task ==")
    for task, s in rep["by_task"].items():
        print(f"  {task:<16} " + _fmt(s))

    if args.plots:
        try:
            from .plots import reliability_diagram
            out = reliability_diagram(records, title="poorjev, raw (uncalibrated)",
                                      path=args.plot_path)
            print(f"\nwrote reliability diagram -> {out}")
        except ImportError:
            print("\n[--plots needs the 'plots' extra: pip install 'poorjev[plots]']")
    return 0


def cmd_calibrate(args) -> int:
    import json
    from .evaluate import evaluate
    from .client import Client
    from .calibration import summarize_calibration, fit_global_temperature

    print(f"Calibrating on: {args.set}")
    print("(runs the eval set through the local model once)\n")
    raw, _ = evaluate(args.set, client=Client())

    summ = summarize_calibration(raw, k=args.folds, target_risk=args.target_risk)
    global_T = fit_global_temperature(raw)  # the deployable scalar

    print("== calibration (5-fold held-out) ==")
    print(f"  temperature (mean)   : {summ['temperature']:.3f}")
    print(f"  ECE  before -> after : {summ['ece_before']:.3f} -> {summ['ece_after']:.3f}")
    print(f"  selective @ risk<={summ['target_risk']:.2f}: "
          f"coverage={summ['coverage_at_target']:.2f} "
          f"(threshold {summ['abstain_threshold']:.3f})")

    saved = {
        "temperature": global_T,
        "target_risk": summ["target_risk"],
        "abstain_threshold": summ["abstain_threshold"],
        "ece_before": summ["ece_before"],
        "ece_after": summ["ece_after"],
    }
    with open(args.out, "w") as f:
        json.dump(saved, f, indent=2)
    print(f"\nsaved calibrator -> {args.out}")

    if args.plots:
        try:
            from .plots import reliability_pair, risk_coverage_plot
            from .calibration import cross_val_calibrate
            cal, _ = cross_val_calibrate(raw, k=args.folds)
            p1 = reliability_pair(raw, cal, path=args.plot_path)
            p2 = risk_coverage_plot(cal)
            print(f"wrote {p1}")
            print(f"wrote {p2}")
        except ImportError:
            print("[--plots needs the 'plots' extra: pip install 'poorjev[plots]']")
    return 0


def cmd_ask(args) -> int:
    import json
    from .client import Client

    temperature = 1.0
    if args.calibrator:
        try:
            with open(args.calibrator) as f:
                temperature = json.load(f).get("temperature", 1.0)
        except FileNotFoundError:
            print(f"[no calibrator at {args.calibrator}; using raw confidence]",
                  file=sys.stderr)

    if args.state_file:
        with open(args.state_file) as f:
            state = f.read().strip()
    else:
        state = args.state
    if not state:
        print("provide --state or --state-file", file=sys.stderr)
        return 1

    with open(args.questions) as f:
        specs = json.load(f)   # {name: {"type":..., ...}}

    from .evaluate import _build_primitive
    questions = {name: _build_primitive(spec) for name, spec in specs.items()}
    res = Client(temperature=temperature).ask(state, questions)

    print(json.dumps({
        name: {
            "value": ans.value,
            "confidence": round(ans.confidence, 4),
        } for name, ans in res.items()
    }, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="poorjev", description="the poor man's Jev")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="score the eval set (accuracy + calibration)")
    pe.add_argument("--set", default="evalset/tasks.jsonl", help="path to tasks.jsonl")
    pe.add_argument("--plots", action="store_true", help="write the reliability diagram")
    pe.add_argument("--plot-path", default="reliability.png")
    pe.add_argument("--verbose", action="store_true", help="print each item's answers")
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("ask", help="answer typed questions about one state")
    pa.add_argument("--state", default="", help="the state text")
    pa.add_argument("--state-file", help="read state from a file")
    pa.add_argument("--questions", required=True, help="JSON file: {name: spec}")
    pa.add_argument("--calibrator", default="calibration.json",
                    help="JSON from `poorjev calibrate` (applies the fitted temperature)")
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("calibrate", help="fit calibration on the eval set")
    pc.add_argument("--set", default="evalset/tasks.jsonl")
    pc.add_argument("--folds", type=int, default=5)
    pc.add_argument("--target-risk", type=float, default=0.1)
    pc.add_argument("--plots", action="store_true", help="write before/after diagrams")
    pc.add_argument("--plot-path", default="docs/reliability_before_after.png")
    pc.add_argument("--out", default="calibration.json")
    pc.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
