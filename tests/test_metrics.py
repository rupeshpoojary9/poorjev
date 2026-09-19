"""Metrics correctness on hand-computed cases. If these are wrong, every number
poorjev reports is wrong, so they are pinned to values worked out by hand.
"""

import math

from poorjev.metrics import (
    DecisionRecord, accuracy, brier_toplabel, ece, reliability_bins,
    risk_coverage, aurc, summarize,
)


def rec(conf, correct, prob_gold=None, n=2):
    if prob_gold is None:
        prob_gold = conf if correct else 1 - conf
    return DecisionRecord(confidence=conf, correct=correct, prob_gold=prob_gold, n_classes=n)


def test_accuracy():
    assert accuracy([rec(0.9, True), rec(0.9, False), rec(0.9, True), rec(0.9, True)]) == 0.75
    assert accuracy([]) == 0.0


def test_perfect_calibration_has_zero_ece():
    # 10 decisions at confidence 1.0, all correct -> conf == acc -> ECE 0
    recs = [rec(1.0, True) for _ in range(10)]
    assert math.isclose(ece(recs), 0.0, abs_tol=1e-12)


def test_perfectly_calibrated_half_bin():
    # confidence 0.5, exactly half correct -> conf == acc == 0.5 -> ECE 0
    recs = [rec(0.5, True) for _ in range(5)] + [rec(0.5, False) for _ in range(5)]
    assert math.isclose(ece(recs), 0.0, abs_tol=1e-9)


def test_overconfident_has_positive_ece():
    # confidence 1.0 but only 50% correct -> gap of 0.5
    recs = [rec(1.0, True) for _ in range(5)] + [rec(1.0, False) for _ in range(5)]
    assert math.isclose(ece(recs), 0.5, abs_tol=1e-9)


def test_brier_bounds_and_direction():
    perfect = [rec(1.0, True), rec(1.0, True)]
    assert math.isclose(brier_toplabel(perfect), 0.0)
    worst = [rec(1.0, False), rec(1.0, False)]
    assert math.isclose(brier_toplabel(worst), 1.0)
    # a hedged wrong answer beats a confident wrong answer
    assert brier_toplabel([rec(0.6, False)]) < brier_toplabel([rec(0.9, False)])


def test_reliability_bins_partition_all_records():
    recs = [rec(c, True) for c in [0.05, 0.15, 0.25, 0.95]]
    bins = reliability_bins(recs, n_bins=10)
    assert sum(b["count"] for b in bins) == len(recs)


def test_risk_coverage_monotone_coverage_and_full_point():
    # high-confidence correct, low-confidence wrong -> risk grows as coverage grows
    recs = [rec(0.9, True), rec(0.8, True), rec(0.3, False), rec(0.2, False)]
    pts = risk_coverage(recs)
    covs = [p["coverage"] for p in pts]
    assert covs == sorted(covs)                      # coverage increases
    assert math.isclose(pts[-1]["coverage"], 1.0)    # ends at full coverage
    assert math.isclose(pts[-1]["risk"], 0.5)        # 2 of 4 wrong overall
    assert pts[0]["risk"] == 0.0                     # most-confident is correct


def test_aurc_better_when_confidence_ranks_correctness():
    good = [rec(0.9, True), rec(0.8, True), rec(0.3, False), rec(0.2, False)]
    bad = [rec(0.9, False), rec(0.8, False), rec(0.3, True), rec(0.2, True)]
    assert aurc(good) < aurc(bad)


def test_summarize_keys():
    s = summarize([rec(0.7, True), rec(0.4, False)])
    assert set(s) == {"n", "accuracy", "ece", "brier", "aurc"}
    assert s["n"] == 2
