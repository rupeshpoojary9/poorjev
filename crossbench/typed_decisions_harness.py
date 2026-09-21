"""
Run ../evalset/tasks.jsonl (poorjev's own 55-item / 160-decision
multi-primitive set: Choice + Score + Noul) through laya, von, or jev, scored
with poorjev's own metrics.py so every system is graded by identical code.

For poorjev itself, just use its own CLI: `poorjev eval --set ../evalset/tasks.jsonl`
(pip install "poorjev[local]").

Run from inside crossbench/:
    python typed_decisions_harness.py --system laya
    python typed_decisions_harness.py --system von
    python typed_decisions_harness.py --system jev
"""
import argparse
import json
import time

from poorjev.metrics import DecisionRecord, summarize

DATA_PATH = "../evalset/tasks.jsonl"  # poorjev's own eval set, one level up from crossbench/


def load_tasks():
    with open(DATA_PATH) as f:
        return [json.loads(l) for l in f if l.strip()]


def score_and_append(records, task, key, spec, predicted, conf, n_classes):
    records.append(
        DecisionRecord(
            confidence=conf,
            correct=(predicted == spec["gold"]),
            prob_gold=0.0,
            n_classes=n_classes,
            task=task,
            question=key,
        )
    )


def run_laya(tasks):
    import laya

    agent = laya.load("convaiinnovations/laya", subfolder="typed-decisions")
    records, latencies = [], []
    for i, item in enumerate(tasks, 1):
        questions = {}
        for key, spec in item["questions"].items():
            if spec["type"] == "choice":
                questions[key] = {
                    "type": "choice",
                    "instructions": f"What is the {key.replace('_', ' ')}?",
                    "criteria": {o: o for o in spec["options"]},
                }
            elif spec["type"] == "score":
                questions[key] = {
                    "type": "score",
                    "instructions": f"What is the {key.replace('_', ' ')}?",
                    "criteria": list(spec["levels"]),
                }
            elif spec["type"] == "noul":
                questions[key] = {"type": "noul", "instructions": spec["statement"]}
        t0 = time.time()
        result = agent.predict(item["state"], questions)
        latencies.append((time.time() - t0) * 1000)
        for key, spec in item["questions"].items():
            ans = result["answers"][key]
            if spec["type"] == "choice":
                score_and_append(records, item["task"], key, spec, ans["choice"], ans["confidence"], len(spec["options"]))
            elif spec["type"] == "score":
                best = max(ans["probabilities"], key=lambda k: ans["probabilities"][k])
                score_and_append(records, item["task"], key, spec, ans["legend"][best], ans["confidence"], len(spec["levels"]))
            elif spec["type"] == "noul":
                predicted = ans["noul"] > 0.5
                score_and_append(records, item["task"], key, spec, predicted, ans["confidence"], 2)
        if i % 10 == 0:
            print(f"  laya {i}/{len(tasks)}")
    return records, latencies


def run_von(tasks):
    import von

    records, latencies = [], []
    for i, item in enumerate(tasks, 1):
        t0 = time.time()
        for key, spec in item["questions"].items():
            if spec["type"] == "choice":
                ans = von.decide(state=item["state"], choices=spec["options"], instructions=f"What is the {key.replace('_', ' ')}?")
                score_and_append(records, item["task"], key, spec, ans.choice, ans.confidence, len(spec["options"]))
            elif spec["type"] == "score":
                ans = von.rate(state=item["state"], criteria=spec["levels"], instructions=f"What is the {key.replace('_', ' ')}?")
                best = max(ans.probabilities, key=lambda k: ans.probabilities[k])
                score_and_append(records, item["task"], key, spec, ans.legend[best], ans.confidence, len(spec["levels"]))
            elif spec["type"] == "noul":
                p = von.judge(state=item["state"], instructions=spec["statement"])
                predicted = p > 0.5
                score_and_append(records, item["task"], key, spec, predicted, max(p, 1 - p), 2)
        latencies.append((time.time() - t0) * 1000)
        if i % 10 == 0:
            print(f"  von {i}/{len(tasks)}")
    return records, latencies


def run_jev(tasks):
    from jev_client import JevClient

    client = JevClient()
    records, latencies = [], []
    for i, item in enumerate(tasks, 1):
        questions = {}
        for key, spec in item["questions"].items():
            if spec["type"] == "choice":
                questions[key] = {"type": "choice", "instructions": f"What is the {key.replace('_', ' ')}?", "criteria": {o: o for o in spec["options"]}}
            elif spec["type"] == "score":
                questions[key] = {"type": "score", "instructions": f"What is the {key.replace('_', ' ')}?", "criteria": list(spec["levels"])}
            elif spec["type"] == "noul":
                questions[key] = {"type": "noul", "instructions": spec["statement"]}
        t0 = time.time()
        result = client.predict(item["state"], questions)
        latencies.append((time.time() - t0) * 1000)
        for key, spec in item["questions"].items():
            ans = result["answers"][key]
            if spec["type"] == "choice":
                score_and_append(records, item["task"], key, spec, ans["choice"], ans["confidence"], len(spec["options"]))
            elif spec["type"] == "score":
                best = max(ans["probabilities"], key=lambda k: ans["probabilities"][k])
                score_and_append(records, item["task"], key, spec, ans["legend"][best], ans["confidence"], len(spec["levels"]))
            elif spec["type"] == "noul":
                predicted = ans["noul"] > 0.5
                score_and_append(records, item["task"], key, spec, predicted, ans.get("confidence", max(ans["noul"], 1 - ans["noul"])), 2)
        if i % 10 == 0:
            print(f"  jev {i}/{len(tasks)}")
    return records, latencies


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["laya", "von", "jev"])
    args = ap.parse_args()

    tasks = load_tasks()
    print(f"Running {args.system} on {len(tasks)} items from {DATA_PATH}")
    fn = {"laya": run_laya, "von": run_von, "jev": run_jev}[args.system]
    records, latencies = fn(tasks)

    overall = summarize(records)
    print(f"\n== {args.system} on poorjev_evalset.jsonl (n={overall['n']}) ==")
    print(f"  acc={overall['accuracy']:.3f}  ECE={overall['ece']:.3f}  Brier={overall['brier']:.3f}  AURC={overall['aurc']:.3f}")
    lat = sorted(latencies)
    n = len(lat)
    print(f"  latency p50={lat[n // 2]:.0f}ms p95={lat[int(n * 0.95)]:.0f}ms")

    with open(f"results/typed_decisions_{args.system}_results.json", "w") as f:
        json.dump({"overall": overall, "latency_ms": {"p50": lat[n // 2], "p95": lat[int(n * 0.95)]}}, f, indent=2)
