"""Plots, above all the reliability diagram: the one image that proves the pitch.

A reliability diagram bins decisions by confidence and plots mean confidence
against empirical accuracy. The diagonal is perfect calibration. Bars above the
diagonal mean under-confidence; below means over-confidence. M4 will render the
before/after pair, and the "after" hugging the diagonal is the launch asset.

matplotlib is an optional extra (`pip install 'poorjev[plots]'`) so the core
library stays dependency-free.
"""

from __future__ import annotations

from .metrics import DecisionRecord, reliability_bins, ece, accuracy, risk_coverage


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


def _draw_reliability(ax, records, title, n_bins):
    bins = reliability_bins(records, n_bins)
    width = 1.0 / n_bins
    centers = [(b["lo"] + b["hi"]) / 2 for b in bins]
    accs = [b["acc"] for b in bins]
    ax.plot([0, 1], [0, 1], "--", color="#888", linewidth=1)
    ax.bar(centers, accs, width=width * 0.9, edgecolor="#1b3a5b",
           color="#4c93d6", alpha=0.85)
    for b in bins:
        if b["count"]:
            ax.plot([b["conf"], b["conf"]], [0, b["acc"]], color="#d64c4c",
                    linewidth=0.8, alpha=0.6)
    ax.set_title(f"{title}\nacc={accuracy(records):.3f}  ECE={ece(records, n_bins):.3f}")
    ax.set_xlabel("confidence")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")


def reliability_pair(raw, calibrated, path: str = "docs/reliability_before_after.png",
                     n_bins: int = 10) -> str:
    """The launch asset: raw (overconfident) vs calibrated (hugging the diagonal)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'poorjev[plots]'") from e

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.6, 5.0))
    _draw_reliability(ax1, raw, "raw (uncalibrated)", n_bins)
    _draw_reliability(ax2, calibrated, "calibrated (temperature scaled)", n_bins)
    ax1.set_ylabel("empirical accuracy")
    fig.suptitle("poorjev: confidence you can trust", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def risk_coverage_plot(records, path: str = "docs/risk_coverage.png") -> str:
    """Accuracy rises as we abstain on the least-confident decisions."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise ImportError("plots need matplotlib: pip install 'poorjev[plots]'") from e

    pts = risk_coverage(records)
    cov = [p["coverage"] for p in pts]
    risk = [p["risk"] for p in pts]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.plot(cov, risk, color="#4c93d6", linewidth=2)
    ax.fill_between(cov, risk, color="#4c93d6", alpha=0.12)
    ax.set_xlabel("coverage (fraction answered)")
    ax.set_ylabel("risk (error rate on answered)")
    ax.set_title("poorjev: abstain on the hard ones, risk drops")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(risk) * 1.1 if risk else 1)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
