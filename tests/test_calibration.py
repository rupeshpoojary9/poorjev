"""Calibration tests on synthetic data, so the moat's math is trustworthy
before we report any before/after number. No model involved.
"""

import math
import random

from poorjev.metrics import DecisionRecord, ece
from poorjev.calibration import (
    apply_temperature, fit_temperature, cross_val_calibrate,
    fit_abstention_threshold, summarize_calibration,
)


def test_apply_temperature_identity_and_bounds():
    d = [0.7, 0.2, 0.1]
    out = apply_temperature(d, 1.0)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(out, d))
    assert math.isclose(sum(out), 1.0)


def test_temperature_preserves_argmax():
    d = [0.5, 0.3, 0.2]
    for T in [0.3, 0.8, 1.0, 2.0, 5.0]:
        out = apply_temperature(d, T)
        assert out.index(max(out)) == 0          # winner never changes


def test_high_temperature_softens_confidence():
    d = [0.9, 0.05, 0.05]
    softened = apply_temperature(d, 3.0)
    assert max(softened) < 0.9                    # less confident
    sharpened = apply_temperature(d, 0.4)
    assert max(sharpened) > 0.9                    # more confident


def _overconfident_set(n=400, seed=1):
    """Build records that are systematically overconfident: the model reports a
    high prob but is right only ~65% of the time. A T>1 should fix the ECE.
    """
    rng = random.Random(seed)
    recs = []
    for _ in range(n):
        p = rng.uniform(0.8, 0.99)               # claims high confidence
        correct = rng.random() < 0.65            # but is right only 65% of the time
        gi = 0 if correct else 1
        dist = (p, 1 - p) if correct else (p, 1 - p)
        recs.append(DecisionRecord(
            confidence=p, correct=correct, prob_gold=(p if correct else 1 - p),
            n_classes=2, dist=dist, gold_index=gi,
        ))
    return recs


def test_fit_temperature_reduces_ece_on_overconfident_data():
    recs = _overconfident_set()
    before = ece(recs)
    calibrated, T = cross_val_calibrate(recs, k=5)
    after = ece(calibrated)
    assert T > 1.0                                # softening an overconfident model
    assert after < before                         # calibration helped
    assert after < 0.1                            # and helped a lot


def test_cross_val_preserves_correctness():
    recs = _overconfident_set(n=100)
    calibrated, _ = cross_val_calibrate(recs, k=5)
    assert sum(r.correct for r in calibrated) == sum(r.correct for r in recs)


def test_abstention_threshold_controls_risk():
    # confident+correct at the top, unsure+wrong at the bottom
    recs = (
        [DecisionRecord(0.95, True, 0.95, 2, dist=(0.95, 0.05), gold_index=0) for _ in range(20)]
        + [DecisionRecord(0.55, False, 0.45, 2, dist=(0.55, 0.45), gold_index=1) for _ in range(10)]
    )
    thr, cov, risk = fit_abstention_threshold(recs, target_risk=0.05)
    assert risk <= 0.05
    assert 0.0 < cov <= 1.0
    # answering only the confident ones should cover about the top 20/30
    assert cov >= 0.6


def test_summarize_calibration_keys():
    s = summarize_calibration(_overconfident_set(n=120), k=4)
    for key in ("temperature", "ece_before", "ece_after", "abstain_threshold",
                "coverage_at_target", "calibrated_records"):
        assert key in s
    assert s["ece_after"] <= s["ece_before"] + 1e-9
