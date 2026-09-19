"""Calibration and accuracy metrics.

Everything here works on a flat list of per-decision records: for each decision
the model made, its confidence in the value it picked, whether that value was
correct, and how much probability mass it put on the gold answer. From those we
compute accuracy, expected calibration error (ECE), top-label Brier, and the
risk-coverage curve that quantifies selective prediction.

These are the numbers that turn "trust me, it's calibrated" into a table anyone
can reproduce. M4's whole job is to move ECE down; this module is how we prove
it moved. Pure stdlib, no numpy, so it runs anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionRecord:
    """One decision the model made, scored against gold.

    ``dist`` and ``gold_index`` are optional and only used by calibration (M4):
    the full probability distribution over the decision space and the index of
    the gold class within it. Metrics ignore them.
    """
    confidence: float     # model's confidence in the value it picked, in [0, 1]
    correct: bool         # was the picked value the gold value
    prob_gold: float      # probability mass placed on the gold value, in [0, 1]
    n_classes: int        # size of the decision space (2 for Noul)
    task: str = ""
    question: str = ""
    dist: tuple = ()      # full distribution over classes, in class order
    gold_index: int = -1  # index of the gold class within dist


def accuracy(records: list[DecisionRecord]) -> float:
    if not records:
        return 0.0
    return sum(1 for r in records if r.correct) / len(records)


def brier_toplabel(records: list[DecisionRecord]) -> float:
    """Mean squared error between the confidence and the outcome (1 if correct).

    The top-label Brier score: 0 is perfect, lower is better. Sensitive to
    over/under-confidence, so it drops when calibration improves.
    """
    if not records:
        return 0.0
    return sum((r.confidence - (1.0 if r.correct else 0.0)) ** 2 for r in records) / len(records)


def reliability_bins(records: list[DecisionRecord], n_bins: int = 10):
    """Group decisions into equal-width confidence bins.

    Returns a list of dicts with the bin range, mean confidence, empirical
    accuracy, and count. This is exactly the data a reliability diagram plots:
    perfectly calibrated means mean-confidence == accuracy in every bin.
    """
    bins = []
    for b in range(n_bins):
        lo = b / n_bins
        hi = (b + 1) / n_bins
        in_bin = [
            r for r in records
            if (r.confidence > lo or (b == 0 and r.confidence >= lo)) and r.confidence <= hi
        ]
        count = len(in_bin)
        if count:
            conf = sum(r.confidence for r in in_bin) / count
            acc = sum(1 for r in in_bin if r.correct) / count
        else:
            conf = (lo + hi) / 2
            acc = 0.0
        bins.append({"lo": lo, "hi": hi, "conf": conf, "acc": acc, "count": count})
    return bins


def ece(records: list[DecisionRecord], n_bins: int = 10) -> float:
    """Expected Calibration Error: average gap between confidence and accuracy,
    weighted by how many decisions fall in each confidence bin. 0 is perfectly
    calibrated. This is the headline number M4 must reduce.
    """
    if not records:
        return 0.0
    n = len(records)
    total = 0.0
    for b in reliability_bins(records, n_bins):
        if b["count"]:
            total += (b["count"] / n) * abs(b["acc"] - b["conf"])
    return total


def risk_coverage(records: list[DecisionRecord]):
    """Selective-prediction curve.

    Sort decisions by confidence (high first); as we lower the abstention
    threshold we 'cover' more decisions. Returns points of (coverage, risk),
    where coverage is the fraction answered and risk is the error rate among the
    answered. A useful layer makes risk fall as coverage falls, you trade a few
    'I don't know's for higher accuracy on what remains.
    """
    if not records:
        return []
    ordered = sorted(records, key=lambda r: r.confidence, reverse=True)
    n = len(ordered)
    points = []
    errors = 0
    for i, r in enumerate(ordered, start=1):
        if not r.correct:
            errors += 1
        points.append({"coverage": i / n, "risk": errors / i, "threshold": r.confidence})
    return points


def aurc(records: list[DecisionRecord]) -> float:
    """Area under the risk-coverage curve (lower is better). A single number
    summarising how well confidence ranks correct answers above wrong ones."""
    pts = risk_coverage(records)
    if len(pts) < 2:
        return pts[0]["risk"] if pts else 0.0
    area = 0.0
    for a, b in zip(pts, pts[1:]):
        dx = b["coverage"] - a["coverage"]
        area += (a["risk"] + b["risk"]) / 2 * dx
    return area


def summarize(records: list[DecisionRecord], n_bins: int = 10) -> dict:
    return {
        "n": len(records),
        "accuracy": accuracy(records),
        "ece": ece(records, n_bins),
        "brier": brier_toplabel(records),
        "aurc": aurc(records),
    }
