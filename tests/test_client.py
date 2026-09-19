"""M2 orchestration tests, backend-mocked so they run in milliseconds with no
model. They pin the single-pass contract: one batched call for all questions,
correct slicing back to each primitive, and the right decode per primitive type.
"""

import math

import pytest

from poorjev import Client, Choice, Score, Noul


class FakeBackend:
    """Records the pairs it was asked about and returns scripted entailment
    probabilities, so we can assert exactly how the Client batches and slices."""

    def __init__(self, scores):
        self.scores = list(scores)
        self.calls = 0
        self.seen_pairs = None

    def entail_probs(self, pairs):
        self.calls += 1
        self.seen_pairs = list(pairs)
        assert len(self.scores) == len(pairs), "test wrote the wrong number of scores"
        return self.scores


def test_single_pass_one_call_for_all_questions():
    # topic(4) + frustration(3) + urgent(1) + cancel(1) = 9 pairs
    scores = [
        0.9, 0.1, 0.2, 0.05,   # topic -> billing dominates
        0.1, 0.3, 0.8,         # frustration -> high dominates
        0.92,                  # urgent -> true
        0.20,                  # cancel -> false
    ]
    be = FakeBackend(scores)
    client = Client(backend=be)

    res = client.ask(
        state="I was double charged and want to cancel.",
        questions={
            "topic": Choice(["billing", "technical", "account", "other"]),
            "frustration": Score(levels=["low", "medium", "high"]),
            "urgent": Noul("The customer needs a response today."),
            "cancel": Noul("The customer wants to cancel."),
        },
    )

    assert be.calls == 1                    # exactly one batched pass
    assert len(be.seen_pairs) == 9          # all pairs concatenated
    # every pair shares the one state as premise
    assert all(p[0] == "I was double charged and want to cancel." for p in be.seen_pairs)

    assert res["topic"].value == "billing"
    assert res["frustration"].value == "high"
    assert res["urgent"].value is True
    assert res["cancel"].value is False


def test_choice_normalises_entailment_probs_across_options():
    be = FakeBackend([0.6, 0.2, 0.2])
    res = Client(backend=be).ask("s", {"q": Choice(["a", "b", "c"])})
    probs = res["q"].probs
    assert math.isclose(sum(probs.values()), 1.0, abs_tol=1e-9)
    assert res["q"].value == "a"
    assert math.isclose(probs["a"], 0.6)   # 0.6 / (0.6+0.2+0.2)


def test_noul_uses_prob_directly():
    be = FakeBackend([0.87])
    res = Client(backend=be).ask("s", {"q": Noul("It is urgent.")})
    assert res["q"].value is True
    assert math.isclose(res["q"].prob, 0.87)


def test_choice_hypothesis_template_is_applied():
    be = FakeBackend([0.5, 0.5])
    client = Client(backend=be, hypothesis_template="This ticket is about {}.")
    client.ask("s", {"q": Choice(["billing", "sales"])})
    hyps = [h for _, h in be.seen_pairs]
    assert hyps == ["This ticket is about billing.", "This ticket is about sales."]


def test_noul_statement_used_verbatim_no_template():
    be = FakeBackend([0.5])
    client = Client(backend=be, hypothesis_template="This example is {}.")
    client.ask("s", {"q": Noul("The customer is angry.")})
    assert be.seen_pairs[0][1] == "The customer is angry."


def test_empty_state_rejected():
    with pytest.raises(ValueError):
        Client(backend=FakeBackend([])).ask("", {"q": Noul("x")})


def test_no_questions_rejected():
    with pytest.raises(ValueError):
        Client(backend=FakeBackend([])).ask("s", {})


def test_bad_question_type_rejected():
    with pytest.raises(TypeError):
        Client(backend=FakeBackend([])).ask("s", {"q": "not a primitive"})


def test_backend_wrong_score_count_raises():
    class BadBackend:
        def entail_probs(self, pairs):
            return [0.5]  # wrong length on purpose
    with pytest.raises(RuntimeError):
        Client(backend=BadBackend()).ask("s", {"q": Choice(["a", "b"])})
