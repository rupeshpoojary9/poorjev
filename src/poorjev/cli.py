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


def cmd_ask(args) -> int:
    print("`poorjev ask` lands in M4 with calibration wired in.", file=sys.stderr)
    return 1


def cmd_calibrate(args) -> int:
    print("`poorjev calibrate` lands in M4.", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="poorjev", description="the poor man's Jev")
    sub = p.add_subparsers(dest="command", required=True)

    pe = sub.add_parser("eval", help="score the eval set (accuracy + calibration)")
    pe.add_argument("--set", default="evalset/tasks.jsonl", help="path to tasks.jsonl")
    pe.add_argument("--plots", action="store_true", help="write the reliability diagram")
    pe.add_argument("--plot-path", default="reliability.png")
    pe.add_argument("--verbose", action="store_true", help="print each item's answers")
    pe.set_defaults(func=cmd_eval)

    pa = sub.add_parser("ask", help="answer typed questions about one state (M4+)")
    pa.set_defaults(func=cmd_ask)

    pc = sub.add_parser("calibrate", help="fit calibration on the eval set (M4)")
    pc.set_defaults(func=cmd_calibrate)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
