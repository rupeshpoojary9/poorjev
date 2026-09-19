"""The three System One primitives and their typed answers.

This module is the contract. A backend's only job is to hand each primitive a
vector of raw scores; the primitive turns that into a typed answer whose
``value`` is *always* drawn from the declared set. Schema validity is therefore
structural: an out-of-set answer is impossible by construction, not by parsing
or by hope. That is the honest version of Jev's "0 type errors" claim, and the
invariant tests in ``tests/test_primitives.py`` assert it against adversarial
inputs (negatives, NaN, inf, wrong length, all-zero).

Nothing here imports a model or touches the network. Calibration (M4) slots in
by transforming the raw scores *before* ``decide``; it never has to police the
output, because the output cannot be invalid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


# --------------------------------------------------------------------------- #
# Numeric helpers (stdlib only, numerically stable, adversary tolerant)
# --------------------------------------------------------------------------- #

def _sanitize(xs: list[float]) -> list[float]:
    """Coerce anything to a list of finite floats.

    NaN and +/-inf are the enemy of "valid by construction": a NaN would make
    every comparison false and could yield an empty argmax. We map NaN -> 0.0,
    +inf -> a large finite number, -inf -> a large negative one, so softmax
    always produces a proper distribution.
    """
    out: list[float] = []
    for x in xs:
        try:
            v = float(x)
        except (TypeError, ValueError):
            v = 0.0
        if math.isnan(v):
            v = 0.0
        elif math.isinf(v):
            v = 1e30 if v > 0 else -1e30
        out.append(v)
    return out


def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax over a non-empty vector.

    Returns a valid probability distribution (non-negative, sums to 1) for any
    real input, including all-equal and extreme values.
    """
    xs = _sanitize(scores)
    if not xs:
        raise ValueError("softmax requires at least one score")
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    if total <= 0.0 or math.isnan(total):  # pragma: no cover - defensive
        n = len(xs)
        return [1.0 / n] * n
    return [e / total for e in exps]


def normalize_probs(scores: list[float]) -> list[float]:
    """Turn a vector of already-probability-like weights into a distribution.

    Negatives are clipped to 0. If everything is 0 (or was), fall back to
    uniform so the result is always a valid distribution.
    """
    xs = [max(0.0, x) for x in _sanitize(scores)]
    total = sum(xs)
    if total <= 0.0:
        n = len(xs) if xs else 1
        return [1.0 / n] * n
    return [x / total for x in xs]


def sigmoid(x: float) -> float:
    x = _sanitize([x])[0]
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _to_distribution(scores: list[float], kind: str) -> list[float]:
    if kind == "logits":
        return softmax(scores)
    if kind == "probs":
        return normalize_probs(scores)
    raise ValueError(f"kind must be 'logits' or 'probs', got {kind!r}")


# --------------------------------------------------------------------------- #
# Typed answers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChoiceAnswer:
    value: str                      # always one of the declared options
    probs: dict[str, float]         # option -> calibrated probability
    confidence: float               # == probs[value]
    abstained: bool = False         # True if confidence fell below the threshold

    def __repr__(self) -> str:
        tag = " (abstained)" if self.abstained else ""
        return f"ChoiceAnswer(value={self.value!r}, confidence={self.confidence:.3f}{tag})"


@dataclass(frozen=True)
class ScoreAnswer:
    value: str                      # always one of the declared, ordered levels
    score: float                    # continuous position on the level index scale
    distribution: dict[str, float]  # level -> probability
    confidence: float               # == distribution[value]
    abstained: bool = False

    def __repr__(self) -> str:
        tag = " (abstained)" if self.abstained else ""
        return (
            f"ScoreAnswer(value={self.value!r}, score={self.score:.3f}, "
            f"confidence={self.confidence:.3f}{tag})"
        )


@dataclass(frozen=True)
class NoulAnswer:
    value: bool                     # always a bool
    prob: float                     # calibrated P(true)
    confidence: float               # == max(prob, 1 - prob), confidence in the bool
    abstained: bool = False

    def __repr__(self) -> str:
        tag = " (abstained)" if self.abstained else ""
        return f"NoulAnswer(value={self.value}, prob={self.prob:.3f}{tag})"


# --------------------------------------------------------------------------- #
# Question primitives
# --------------------------------------------------------------------------- #

@dataclass
class Choice:
    """A discrete decision over a fixed set of options.

    >>> Choice(["billing", "technical", "other"]).decide([2.0, 0.1, 0.0]).value
    'billing'
    """

    options: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.options, (list, tuple)):
            raise TypeError("Choice options must be a list of strings")
        opts = list(self.options)
        if len(opts) < 2:
            raise ValueError("Choice needs at least 2 options")
        if any(not isinstance(o, str) or o == "" for o in opts):
            raise ValueError("Choice options must be non-empty strings")
        if len(set(opts)) != len(opts):
            raise ValueError("Choice options must be unique")
        self.options = opts

    @property
    def hypotheses(self) -> list[str]:
        """The strings a backend scores as entailment hypotheses (M2 hook)."""
        return list(self.options)

    def decide(self, scores, kind: str = "logits", threshold: float = 0.0) -> ChoiceAnswer:
        scores = list(scores)
        if len(scores) != len(self.options):
            raise ValueError(
                f"expected {len(self.options)} scores, got {len(scores)}"
            )
        probs = _to_distribution(scores, kind)
        best_i = max(range(len(probs)), key=lambda i: probs[i])
        value = self.options[best_i]
        confidence = probs[best_i]
        return ChoiceAnswer(
            value=value,
            probs={o: p for o, p in zip(self.options, probs)},
            confidence=confidence,
            abstained=confidence < threshold,
        )


@dataclass
class Score:
    """An ordinal rating over ordered levels (e.g. low < medium < high).

    Produces both the winning level and a continuous score = expected level
    index under the distribution, so downstream code can threshold on a number.
    """

    levels: list[str]

    def __post_init__(self) -> None:
        if not isinstance(self.levels, (list, tuple)):
            raise TypeError("Score levels must be a list of strings")
        lv = list(self.levels)
        if len(lv) < 2:
            raise ValueError("Score needs at least 2 ordered levels")
        if any(not isinstance(x, str) or x == "" for x in lv):
            raise ValueError("Score levels must be non-empty strings")
        if len(set(lv)) != len(lv):
            raise ValueError("Score levels must be unique")
        self.levels = lv

    @property
    def hypotheses(self) -> list[str]:
        return list(self.levels)

    def decide(self, scores, kind: str = "logits", threshold: float = 0.0) -> ScoreAnswer:
        scores = list(scores)
        if len(scores) != len(self.levels):
            raise ValueError(
                f"expected {len(self.levels)} scores, got {len(scores)}"
            )
        probs = _to_distribution(scores, kind)
        expected = sum(i * p for i, p in enumerate(probs))  # continuous score
        best_i = max(range(len(probs)), key=lambda i: probs[i])
        value = self.levels[best_i]
        confidence = probs[best_i]
        return ScoreAnswer(
            value=value,
            score=expected,
            distribution={lv: p for lv, p in zip(self.levels, probs)},
            confidence=confidence,
            abstained=confidence < threshold,
        )


@dataclass
class Noul:
    """A yes/no judgement about a statement, answered with a calibrated P(true).

    Named after Jev's ``Noul`` primitive. Backends supply a single raw score for
    "true"; ``decide`` turns it into P(true) and thresholds it to a bool.
    """

    statement: str

    def __post_init__(self) -> None:
        if not isinstance(self.statement, str) or self.statement.strip() == "":
            raise ValueError("Noul needs a non-empty statement")

    @property
    def hypotheses(self) -> list[str]:
        return [self.statement]

    def decide(
        self,
        score,
        kind: str = "logit",
        decision_threshold: float = 0.5,
        abstain_below: float = 0.0,
    ) -> NoulAnswer:
        """``score`` is a single number: a logit for "true" (kind='logit') or a
        probability already in [0, 1] (kind='prob')."""
        if isinstance(score, (list, tuple)):
            if len(score) != 1:
                raise ValueError("Noul expects a single score")
            score = score[0]
        if kind == "logit":
            prob = sigmoid(score)
        elif kind == "prob":
            prob = min(1.0, max(0.0, _sanitize([score])[0]))
        else:
            raise ValueError("kind must be 'logit' or 'prob'")
        value = prob >= decision_threshold
        confidence = prob if value else 1.0 - prob
        return NoulAnswer(
            value=value,
            prob=prob,
            confidence=confidence,
            abstained=confidence < abstain_below,
        )
