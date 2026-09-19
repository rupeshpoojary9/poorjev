"""Eval pipeline tests, backend-mocked so they run instantly and deterministically.
They check that gold scoring is correct for each primitive type and that the
report slices by task.
"""

import json

from poorjev import Client
from poorjev.evaluate import evaluate, report


class ScriptedBackend:
    """Returns entailment probs in the exact order the Client builds pairs.

    evaluate() calls ask() once per item, so the backend is called once per
    item; this consumes ``len(pairs)`` scripted scores per call across the run.
    """
    def __init__(self, scores):
        self.scores = list(scores)
        self.i = 0
    def entail_probs(self, pairs):
        n = len(pairs)
        seg = self.scores[self.i:self.i + n]
        assert len(seg) == n, "not enough scripted scores for the pairs asked"
        self.i += n
        return seg


def _write(tmp_path, rows):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def test_scores_choice_and_noul_against_gold(tmp_path):
    rows = [{
        "id": "t1", "task": "demo", "state": "some ticket text",
        "questions": {
            "topic": {"type": "choice", "options": ["billing", "tech"], "gold": "billing"},
            "urgent": {"type": "noul", "statement": "It is urgent.", "gold": True},
        },
    }]
    path = _write(tmp_path, rows)
    # pairs order: topic(billing), topic(tech), urgent -> 3 scores
    be = ScriptedBackend([0.8, 0.2, 0.9])
    records, _ = evaluate(path, client=Client(backend=be))

    assert len(records) == 2
    topic = next(r for r in records if r.question == "topic")
    urgent = next(r for r in records if r.question == "urgent")
    assert topic.correct is True                 # billing wins, gold billing
    assert urgent.correct is True                 # prob 0.9 -> True, gold True
    assert topic.task == "demo"


def test_marks_incorrect_and_records_gold_prob(tmp_path):
    rows = [{
        "id": "t1", "task": "demo", "state": "x",
        "questions": {
            "topic": {"type": "choice", "options": ["billing", "tech"], "gold": "tech"},
        },
    }]
    path = _write(tmp_path, rows)
    be = ScriptedBackend([0.9, 0.1])   # picks billing, but gold is tech
    records, _ = evaluate(path, client=Client(backend=be))
    r = records[0]
    assert r.correct is False
    # gold 'tech' got the normalised 0.1/(0.9+0.1) = 0.1 mass
    assert abs(r.prob_gold - 0.1) < 1e-9


def test_report_has_overall_and_per_task(tmp_path):
    rows = [
        {"id": "a", "task": "alpha", "state": "x",
         "questions": {"q": {"type": "noul", "statement": "s", "gold": True}}},
        {"id": "b", "task": "beta", "state": "y",
         "questions": {"q": {"type": "noul", "statement": "s", "gold": False}}},
    ]
    path = _write(tmp_path, rows)
    be = ScriptedBackend([0.9, 0.9])  # both -> True
    records, _ = evaluate(path, client=Client(backend=be))
    rep = report(records)
    assert "overall" in rep
    assert set(rep["by_task"]) == {"alpha", "beta"}
    assert rep["by_task"]["alpha"]["accuracy"] == 1.0   # gold True, predicted True
    assert rep["by_task"]["beta"]["accuracy"] == 0.0    # gold False, predicted True
