"""
Shared, neutral benchmark: Banking77 (PolyAI, original GitHub source), a
public 77-way intent classification set. Not authored by poorjev, Laya, or
TypeSafe -- and it's specifically the high-cardinality-choice case Laya's own
README flags as Jev's strength. Stratified sample: 2 items x 77 classes = 154.

All three systems get exactly the same state text and exactly the same 77
bare (underscore-stripped) label names as their option set -- no per-option
descriptions for anyone. Scored with poorjev's own metrics.py so all three are
graded by identical code.
"""
import argparse
import json
import time

from poorjev.metrics import DecisionRecord, summarize

SAMPLE_PATH = "data/banking77_sample.jsonl"
LABELS_PATH = "data/banking77_labels.json"


def load_data():
    labels = json.load(open(LABELS_PATH))
    labels_norm = [l.replace("_", " ") for l in labels]
    items = [json.loads(l) for l in open(SAMPLE_PATH) if l.strip()]
    for it in items:
        it["gold"] = it["gold"].replace("_", " ")
    return items, labels_norm


def report(name, records, latencies, out_path):
    overall = summarize(records)
    print(f"\n== {name} on banking77_sample.jsonl (n={overall['n']}) ==")
    print(
        f"  acc={overall['accuracy']:.3f}  ECE={overall['ece']:.3f}  "
        f"Brier={overall['brier']:.3f}  AURC={overall['aurc']:.3f}"
    )
    lat = sorted(latencies)
    n = len(lat)
    p50 = lat[n // 2]
    p95 = lat[int(n * 0.95)] if n > 1 else lat[0]
    print(f"  latency (this CPU): p50={p50:.0f}ms p95={p95:.0f}ms")
    with open(out_path, "w") as f:
        json.dump({"overall": overall, "latency_ms": {"p50": p50, "p95": p95, "n": n}}, f, indent=2)
    return overall, {"p50": p50, "p95": p95}


def run_poorjev(items, labels):
    from poorjev.client import Client
    from poorjev.backends.local_nli import LocalNLIBackend
    from poorjev.primitives import Choice

    backend = LocalNLIBackend()
    # NOTE: temperature=1.0 (raw). T=2.71 was fit on poorjev's own 4-6 option
    # support-triage set and does not transfer to a 77-way task (see
    # banking77_poorjev_recalibrated.json: a Banking77-specific 5-fold CV
    # refit lands at T=1.01, i.e. ~no-op -- so raw is the honest number here,
    # not a stale cross-task temperature).
    client = Client(backend=backend, temperature=1.0)
    records, latencies = [], []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        result = client.ask(item["text"], {"intent": Choice(labels)})
        dt = (time.time() - t0) * 1000
        ans = result["intent"]
        records.append(
            DecisionRecord(
                confidence=ans.confidence,
                correct=(ans.value == item["gold"]),
                prob_gold=ans.probs.get(item["gold"], 0.0),
                n_classes=len(labels),
            )
        )
        latencies.append(dt)
        if i % 20 == 0:
            print(f"  poorjev {i}/{len(items)}")
    return records, latencies


def run_laya(items, labels):
    import laya

    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    # Laya's own docs: 50+ options need a bigger head budget than the 256-token default.
    agent.cfg["head_max_len"] = 1024
    agent.cfg["max_len"] = 2048
    criteria = {l: l for l in labels}
    records, latencies = [], []
    for i, item in enumerate(items, 1):
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the intent of this banking customer support message?",
                "criteria": criteria,
            }
        }
        t0 = time.time()
        result = agent.predict(item["text"], questions)
        dt = (time.time() - t0) * 1000
        ans = result["answers"]["intent"]
        records.append(
            DecisionRecord(
                confidence=ans["confidence"],
                correct=(ans["choice"] == item["gold"]),
                prob_gold=ans["probabilities"].get(item["gold"], 0.0),
                n_classes=len(labels),
            )
        )
        latencies.append(dt)
        if i % 20 == 0:
            print(f"  laya {i}/{len(items)}")
    return records, latencies


def run_von(items, labels):
    import von

    records, latencies = [], []
    for i, item in enumerate(items, 1):
        t0 = time.time()
        ans = von.decide(
            state=item["text"],
            choices=labels,
            instructions="What is the intent of this banking customer support message?",
        )
        dt = (time.time() - t0) * 1000
        records.append(
            DecisionRecord(
                confidence=ans.confidence,
                correct=(ans.choice == item["gold"]),
                prob_gold=ans.probabilities.get(item["gold"], 0.0),
                n_classes=len(labels),
            )
        )
        latencies.append(dt)
        if i % 20 == 0:
            print(f"  von {i}/{len(items)}")
    return records, latencies


def run_jev(items, labels):
    from jev_client import JevClient

    client = JevClient()
    criteria = {l: l for l in labels}
    records, latencies = [], []
    for i, item in enumerate(items, 1):
        questions = {
            "intent": {
                "type": "choice",
                "instructions": "What is the intent of this banking customer support message?",
                "criteria": criteria,
            }
        }
        t0 = time.time()
        result = client.predict(item["text"], questions)
        dt = (time.time() - t0) * 1000
        ans = result["answers"]["intent"]
        records.append(
            DecisionRecord(
                confidence=ans["confidence"],
                correct=(ans["choice"] == item["gold"]),
                prob_gold=ans["probabilities"].get(item["gold"], 0.0),
                n_classes=len(labels),
            )
        )
        latencies.append(dt)
        if i % 20 == 0:
            print(f"  jev {i}/{len(items)}")
    return records, latencies


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["poorjev", "laya", "jev", "von"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items, labels = load_data()
    if args.limit:
        items = items[: args.limit]
    print(f"Running {args.system} on {len(items)} items, {len(labels)} classes")

    fn = {"poorjev": run_poorjev, "laya": run_laya, "jev": run_jev, "von": run_von}[args.system]
    records, latencies = fn(items, labels)
    report(args.system, records, latencies, f"results/banking77_{args.system}_results.json")
