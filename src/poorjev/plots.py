"""Plots, above all the reliability diagram: the one image that proves the pitch.

A reliability diagram bins decisions by confidence and plots mean confidence
against empirical accuracy. The diagonal is perfect calibration. Bars above the
diagonal mean under-confidence; below means over-confidence. M4 will render the
before/after pair, and the "after" hugging the diagonal is the launch asset.

matplotlib is an optional extra (`pip install 'poorjev[plots]'`) so the core
library stays dependency-free.
"""

from __future__ import annotations

from .metrics import DecisionRecord, reliability_bins, ece, accuracy


def reliability_diagram(records: list[DecisionRecord], title: str = "reliability",
                        path: str = "reliability.png", n_bins: int = 10) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'poorjev[plots]'") from e

    bins = reliability_bins(records, n_bins)
    width = 1.0 / n_bins
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["acc"] for b in bins]
    counts = [b["count"] for b in bins]

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1, label="perfect")
    ax.bar(centers, accs, width=width * 0.9, edgecolor="#1b3a5b",
           color="#4c93d6", alpha=0.85, label="accuracy")
    # gap markers: where the bar top sits vs its own mean confidence
    for b in bins:
        if b["count"]:
            ax.plot([b["conf"], b["conf"]], [0, b["acc"]], color="#d64c4c",
                    linewidth=0.8, alpha=0.6)
    e = ece(records, n_bins)
    a = accuracy(records)
    ax.set_title(f"{title}\nacc={a:.3f}  ECE={e:.3f}  n={len(records)}")
    ax.set_xlabel("confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
