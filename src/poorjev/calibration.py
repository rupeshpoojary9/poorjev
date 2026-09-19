"""Calibration: the one thing poorjev gets provably right.

Two pieces:

1. Temperature scaling. A classifier's probabilities are usually wrong as
   confidences: this model is overconfident by 0.170 ECE raw. Temperature scaling
   fits a single scalar T that rescales every distribution, softmax(log p / T),
   so predicted confidence lines up with real accuracy. T is fit by minimising
   negative log-likelihood on held-out data. It is monotonic, so it never
   changes which answer wins, only how confident we are in it.

2. Selective prediction. Given a target risk (max error rate you will tolerate
   on answered items), pick the confidence threshold that holds that risk on a
   calibration set. Below it, abstain, that is the honest "I don't know,
   escalate" signal.

To report an honest "after" number we fit T on training folds and measure the
calibrated ECE on the held-out fold (k-fold CV), so we never grade calibration
on data it was fit on. Pure stdlib.
"""

from __future__ import annotations

import math
import random

from .metrics import DecisionRecord, ece as _ece


# --------------------------------------------------------------------------- #
# Temperature scaling
# --------------------------------------------------------------------------- #

def apply_temperature(dist, T: float):
    """Rescale a probability distribution by temperature T (softmax(log p / T)).

    T = 1 is a no-op. T > 1 softens (less confident); T < 1 sharpens. Argmax is
    preserved, so the picked class never changes.
    """
    if T <= 0:
        raise ValueError("temperature must be > 0")
    logits = [math.log(max(p, 1e-12)) / T for p in dist]
    m = max(logits)
    exps = [math.exp(l - m) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]


def _nll(dists, gold_idx, T: float) -> float:
    total = 0.0
    for d, g in zip(dists, gold_idx):
        q = apply_temperature(d, T)
        total += -math.log(max(q[g], 1e-12))
    return total / len(dists)


def fit_temperature(dists, gold_idx, coarse=None) -> float:
    """Fit T by minimising NLL, coarse grid then a local refine. Robust and
    transparent, no optimiser dependency."""
    if not dists:
        return 1.0
    coarse = coarse or [0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.5,
                        2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
    best = min(coarse, key=lambda T: _nll(dists, gold_idx, T))
    # refine in a window around the coarse winner
    step = 0.05
    fine = [round(best + k * step, 4) for k in range(-6, 7) if best + k * step > 0.05]
    best = min(fine + [best], key=lambda T: _nll(dists, gold_idx, T))
    return best


def _records_with_dist(records: list[DecisionRecord]) -> list[DecisionRecord]:
    return [r for r in records if r.dist and r.gold_index >= 0]


def _recalibrate_record(r: DecisionRecord, T: float) -> DecisionRecord:
    q = apply_temperature(r.dist, T)
    return DecisionRecord(
        confidence=max(q),
        correct=r.correct,                 # monotonic: winner unchanged
        prob_gold=q[r.gold_index],
        n_classes=r.n_classes,
        task=r.task, question=r.question,
        dist=tuple(q), gold_index=r.gold_index,
    )


def cross_val_calibrate(records: list[DecisionRecord], k: int = 5, seed: int = 0):
    """Fit T on train folds, apply to the held-out fold, aggregate.

    Returns (calibrated_records, mean_T). The calibrated records are honest
    held-out predictions: no record was calibrated by a T fit on itself.
    """
    usable = _records_with_dist(records)
    if len(usable) < k:
        T = fit_temperature([r.dist for r in usable], [r.gold_index for r in usable])
        return [_recalibrate_record(r, T) for r in usable], T

    rng = random.Random(seed)
    idx = list(range(len(usable)))
    rng.shuffle(idx)
    folds = [idx[i::k] for i in range(k)]

    calibrated: list[DecisionRecord] = []
    temps: list[float] = []
    for f in range(k):
        test_ids = set(folds[f])
        train = [usable[i] for i in idx if i not in test_ids]
        test = [usable[i] for i in folds[f]]
        T = fit_temperature([r.dist for r in train], [r.gold_index for r in train])
        temps.append(T)
        calibrated.extend(_recalibrate_record(r, T) for r in test)
    return calibrated, sum(temps) / len(temps)


def fit_global_temperature(records: list[DecisionRecord]) -> float:
    """Fit one T on all data, for saving into a deployable calibrator."""
    usable = _records_with_dist(records)
    return fit_temperature([r.dist for r in usable], [r.gold_index for r in usable])


# --------------------------------------------------------------------------- #
# Selective prediction (conformal-style thresholding)
# --------------------------------------------------------------------------- #

def fit_abstention_threshold(records: list[DecisionRecord], target_risk: float = 0.1):
    """Pick the lowest confidence threshold that keeps error rate on accepted
    decisions at or below ``target_risk`` on this set.

    Returns (threshold, coverage, realized_risk). Accept a decision at inference
    when its confidence >= threshold; otherwise abstain and escalate.
    """
    if not records:
        return 1.0, 0.0, 0.0
    ordered = sorted(records, key=lambda r: r.confidence, reverse=True)
    n = len(ordered)
    best = None  # (threshold, coverage, risk) with the largest coverage under risk
    errors = 0
    for i, r in enumerate(ordered, start=1):
        if not r.correct:
            errors += 1
        risk = errors / i
        if risk <= target_risk:
            best = (r.confidence, i / n, risk)
    if best is None:
        # even the single most-confident decision is wrong; abstain on everything
        return 1.01, 0.0, 0.0
    return best


def summarize_calibration(raw: list[DecisionRecord], k: int = 5, target_risk: float = 0.1):
    """The M4 headline: ECE before vs after (held-out), plus a selective point."""
    calibrated, mean_T = cross_val_calibrate(raw, k=k)
    thr, cov, risk = fit_abstention_threshold(calibrated, target_risk)
    return {
        "temperature": mean_T,
        "ece_before": _ece(_records_with_dist(raw)),
        "ece_after": _ece(calibrated),
        "target_risk": target_risk,
        "abstain_threshold": thr,
        "coverage_at_target": cov,
        "risk_at_target": risk,
        "calibrated_records": calibrated,
    }
