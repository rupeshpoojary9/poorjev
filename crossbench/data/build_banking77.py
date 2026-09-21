"""
Builds data/banking77_sample.jsonl: a stratified, seeded sample of Banking77
(2 items x 77 classes = 154), pulled directly from the original PolyAI source
repo (CC-BY-4.0). Not authored by poorjev, Laya, or TypeSafe -- a neutral,
independently-sourced test set.

Source: https://github.com/PolyAI-LDN/task-specific-datasets
        banking_data/test.csv (3080 rows, 77 intents)

Re-run this to regenerate banking77_sample.jsonl (deterministic, seed=42).
"""
import csv
import io
import json
import random
import urllib.request

URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"
PER_CLASS = 2
SEED = 42


def main():
    raw = urllib.request.urlopen(URL).read().decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    labels = sorted({r["category"] for r in rows})

    by_label = {}
    for r in rows:
        by_label.setdefault(r["category"], []).append(r["text"])

    rng = random.Random(SEED)
    sample = []
    for lbl in labels:
        items = by_label[lbl][:]
        rng.shuffle(items)
        for text in items[:PER_CLASS]:
            sample.append({"text": text, "gold": lbl})
    rng.shuffle(sample)

    with open("banking77_labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    with open("banking77_sample.jsonl", "w") as f:
        for item in sample:
            f.write(json.dumps(item) + "\n")

    print(f"{len(rows)} source rows, {len(labels)} classes, wrote {len(sample)}-item sample")


if __name__ == "__main__":
    main()
