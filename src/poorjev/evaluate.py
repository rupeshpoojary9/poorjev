"""Run the eval set through a backend and score it into DecisionRecords.

This is the bridge between the labelled data and the metrics. It builds the
right primitive for each question, asks the Client, then compares the picked
value to gold and records the confidence and the probability mass on the gold
answer. M4 will reuse the collected records (and their gold prob mass) to fit
and evaluate calibration.
"""

from __future__ import annotations

import json

from .primitives import Choice, Score, Noul
from .client import Client
from .metrics import DecisionRecord, summarize


def load_tasks(path: str) -> list[dict]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _build_primitive(spec: dict):
    t = spec["type"]
    if t == "choice":
        return Choice(spec["options"])
    if t == "score":
        return Score(spec["levels"])
    if t == "noul":
        return Noul(spec["statement"])
    raise ValueError(f"unknown question type {t!r}")


def _score_one(spec: dict, answer) -> DecisionRecord:
    t = spec["type"]
    gold = spec["gold"]
    if t == "noul":
        correct = bool(answer.value) == bool(gold)
        prob_gold = answer.prob if gold else (1.0 - answer.prob)
        # class order: index 0 = False, index 1 = True
        dist = (1.0 - answer.prob, answer.prob)
        gold_index = 1 if gold else 0
        return DecisionRecord(answer.confidence, correct, prob_gold, n_classes=2,
                              dist=dist, gold_index=gold_index)
    # choice / score share the same shape
    order = spec["options"] if t == "choice" else spec["levels"]
    probs = answer.probs if t == "choice" else answer.distribution
    correct = answer.value == gold
    prob_gold = probs.get(gold, 0.0)
    dist = tuple(probs[k] for k in order)
    gold_index = order.index(gold)
    return DecisionRecord(answer.confidence, correct, prob_gold, n_classes=len(probs),
                          dist=dist, gold_index=gold_index)


def evaluate(path: str, client: Client | None = None, verbose: bool = False):
    """Return (records, per_question_specs) after running the whole set.

    ``records`` is a flat list of DecisionRecord across every question of every
    item, tagged with task + question so callers can slice per task.
    """
    client = client or Client()
    items = load_tasks(path)
    records: list[DecisionRecord] = []

    for item in items:
        questions = {name: _build_primitive(spec) for name, spec in item["questions"].items()}
        answers = client.ask(item["state"], questions)
        for name, spec in item["questions"].items():
            r = _score_one(spec, answers[name])
            records.append(DecisionRecord(
                r.confidence, r.correct, r.prob_gold, r.n_classes,
                task=item["task"], question=name,
                dist=r.dist, gold_index=r.gold_index,
            ))
        if verbose:
            print(f"  {item['id']}: " + ", ".join(
                f"{n}={answers[n].value!r}" for n in item["questions"]
            ))
    return records, items


def report(records: list[DecisionRecord]) -> dict:
    """Overall summary plus a per-task breakdown."""
    out = {"overall": summarize(records), "by_task": {}}
    tasks = sorted({r.task for r in records})
    for task in tasks:
        out["by_task"][task] = summarize([r for r in records if r.task == task])
    return out
