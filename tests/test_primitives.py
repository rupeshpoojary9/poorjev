"""M1 invariant tests: schema validity is structural.

The headline claim of poorjev is Jev's "0 type errors", made honest: no matter
what a backend feeds a primitive, the answer's ``value`` is always drawn from
the declared set and the probabilities always form a valid distribution. These
tests hammer ``decide`` with adversarial score vectors (NaN, inf, negatives,
all-zero, huge magnitudes) and assert the contract holds every time.
"""

import math
import random

import pytest

from poorjev import Choice, Score, Noul
from poorjev.primitives import softmax, normalize_probs, sigmoid


# --------------------------------------------------------------------------- #
# Adversarial score vectors reused across primitives
# --------------------------------------------------------------------------- #

def adversarial_vectors(n):
    """A pile of nasty length-n score vectors a backend might produce."""
    return [
        [0.0] * n,                                   # all zero
        [1.0] * n,                                   # all equal, non-zero
        [-5.0] * n,                                  # all negative equal
        [float("nan")] * n,                          # all NaN
        [float("inf")] + [0.0] * (n - 1),            # a +inf
        [float("-inf")] + [0.0] * (n - 1),           # a -inf
        [float("inf"), float("-inf")] + [0.0] * (n - 2) if n >= 2 else [float("inf")],
        [1e300] * n,                                 # overflow-prone
        [-1e300] * n,                                # underflow-prone
        [random.uniform(-1e6, 1e6) for _ in range(n)],  # wild range
        list(range(n)),                              # increasing ints
        [float(n - 1 - i) for i in range(n)],        # decreasing
    ]


def assert_valid_distribution(probs_list):
    assert all(0.0 <= p <= 1.0 for p in probs_list)
    assert not any(math.isnan(p) or math.isinf(p) for p in probs_list)
    assert math.isclose(sum(probs_list), 1.0, abs_tol=1e-9)


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #

def test_softmax_always_valid():
    for vec in adversarial_vectors(4):
        assert_valid_distribution(softmax(vec))


def test_normalize_probs_always_valid():
    for vec in adversarial_vectors(4):
        assert_valid_distribution(normalize_probs(vec))


def test_normalize_probs_clips_negatives():
    out = normalize_probs([-1.0, 0.0, 3.0])
    assert out[0] == 0.0
    assert math.isclose(out[2], 1.0)


def test_sigmoid_bounds():
    for x in [-1e9, -10, 0, 10, 1e9, float("nan"), float("inf"), float("-inf")]:
        p = sigmoid(x)
        assert 0.0 <= p <= 1.0


def test_softmax_empty_raises():
    with pytest.raises(ValueError):
        softmax([])


# --------------------------------------------------------------------------- #
# Choice
# --------------------------------------------------------------------------- #

def test_choice_value_always_in_options_under_adversarial_scores():
    opts = ["billing", "technical", "account", "other"]
    c = Choice(opts)
    for vec in adversarial_vectors(len(opts)):
        for kind in ("logits", "probs"):
            ans = c.decide(vec, kind=kind)
            assert ans.value in opts                      # THE invariant
            assert set(ans.probs) == set(opts)
            assert_valid_distribution(list(ans.probs.values()))
            assert math.isclose(ans.confidence, ans.probs[ans.value])
            # winner is the argmax
            assert ans.confidence == max(ans.probs.values())


def test_choice_picks_the_dominant_option():
    c = Choice(["a", "b", "c"])
    assert c.decide([10.0, 0.0, 0.0]).value == "a"
    assert c.decide([0.0, 10.0, 0.0]).value == "b"
    assert c.decide([0.1, 0.2, 0.9], kind="probs").value == "c"


def test_choice_wrong_length_raises():
    c = Choice(["a", "b"])
    with pytest.raises(ValueError):
        c.decide([1.0, 2.0, 3.0])


def test_choice_abstains_below_threshold():
    c = Choice(["a", "b", "c"])
    ans = c.decide([1.0, 1.0, 1.0], threshold=0.9)  # ~0.33 confidence
    assert ans.abstained is True
    ans2 = c.decide([10.0, 0.0, 0.0], threshold=0.9)
    assert ans2.abstained is False


@pytest.mark.parametrize("bad", [
    [],
    ["only"],
    ["a", "a"],
    ["a", ""],
    ["a", 3],
    "ab",
])
def test_choice_rejects_bad_option_sets(bad):
    with pytest.raises((ValueError, TypeError)):
        Choice(bad)


# --------------------------------------------------------------------------- #
# Score
# --------------------------------------------------------------------------- #

def test_score_value_always_a_level_and_score_in_range():
    levels = ["low", "medium", "high"]
    s = Score(levels)
    for vec in adversarial_vectors(len(levels)):
        ans = s.decide(vec)
        assert ans.value in levels                        # THE invariant
        assert 0.0 <= ans.score <= len(levels) - 1        # continuous, on-scale
        assert_valid_distribution(list(ans.distribution.values()))
        assert math.isclose(ans.confidence, ans.distribution[ans.value])


def test_score_monotonic_expected_index():
    s = Score(["low", "medium", "high"])
    low = s.decide([10.0, 0.0, 0.0]).score
    high = s.decide([0.0, 0.0, 10.0]).score
    assert low < high
    assert math.isclose(low, 0.0, abs_tol=1e-3)
    assert math.isclose(high, 2.0, abs_tol=1e-3)


def test_score_wrong_length_raises():
    with pytest.raises(ValueError):
        Score(["low", "high"]).decide([1.0])


# --------------------------------------------------------------------------- #
# Noul
# --------------------------------------------------------------------------- #

def test_noul_value_is_always_bool_and_prob_valid():
    n = Noul("The customer wants to cancel.")
    for x in [-1e9, -3.0, 0.0, 3.0, 1e9, float("nan"), float("inf"), float("-inf")]:
        ans = n.decide(x)
        assert isinstance(ans.value, bool)                # THE invariant
        assert 0.0 <= ans.prob <= 1.0
        assert 0.5 <= ans.confidence <= 1.0
        # confidence is confidence in the chosen bool
        expected_conf = ans.prob if ans.value else 1.0 - ans.prob
        assert math.isclose(ans.confidence, expected_conf)


def test_noul_threshold_and_prob_kind():
    n = Noul("It is urgent.")
    assert n.decide(0.8, kind="prob").value is True
    assert n.decide(0.2, kind="prob").value is False
    # a high decision_threshold makes it harder to say True
    assert n.decide(0.6, kind="prob", decision_threshold=0.7).value is False


def test_noul_prob_kind_clips_out_of_range():
    n = Noul("x")
    assert n.decide(5.0, kind="prob").prob == 1.0
    assert n.decide(-5.0, kind="prob").prob == 0.0


def test_noul_abstains_when_unsure():
    n = Noul("ambiguous")
    ans = n.decide(0.0)  # prob 0.5, confidence 0.5
    assert ans.abstained is False
    ans2 = n.decide(0.0, abstain_below=0.6)
    assert ans2.abstained is True


@pytest.mark.parametrize("bad", ["", "   ", 5, None])
def test_noul_rejects_bad_statement(bad):
    with pytest.raises((ValueError, TypeError)):
        Noul(bad)


# --------------------------------------------------------------------------- #
# Frozen answers (typed + immutable contract objects)
# --------------------------------------------------------------------------- #

def test_answers_are_frozen():
    ans = Choice(["a", "b"]).decide([1.0, 0.0])
    with pytest.raises(Exception):
        ans.value = "b"  # frozen dataclass
