# crossbench

Independent, reproducible benchmark comparing `poorjev` against Jev (TypeSafe,
hosted), [Laya](https://huggingface.co/convaiinnovations/laya) (open weights),
and [von](https://github.com/wfzyx/von) (open weights) — same two test sets,
same metrics code (`poorjev`'s own `metrics.py`), for all four. The summary
table and the honest reading of it live in the main
[README's "Benchmarks" section](../README.md#benchmarks); this folder is the
full harness behind it.

**This exists because every one of these projects, including `poorjev`
before this, published numbers only against itself.** Jev's docs don't
publish an ECE. Laya's homepage claims "ECE 0.081 vs Jev's 0.246," but
Laya's own HuggingFace eval file reports a third, different number (0.030)
on its own in-house set. `von`'s README shows itself *losing* to Jev on its
primary accuracy metric (71.5% vs 96.6%) while leading only on a narrow
gaming benchmark. None of those numbers are comparable to each other. This
folder runs all four through the same inputs so the numbers mean something.

**Nobody sweeps.** Jev wins the multi-primitive set outright. `von` beats
Jev on Banking77 accuracy but loses on calibration. `poorjev` is the best
fully local/free option on accuracy in both sets but is not clearly better
calibrated than Laya at Banking77's 77-way cardinality. No result here was
cherry-picked, and the harness was not reshaped to force a particular winner
— see the [main README](../README.md#benchmarks) for the full table.

## Test sets

- `../evalset/tasks.jsonl` — poorjev's own 55-item / 160-decision set (already
  in this repo), reused here as a neutral cross-system set.
- `data/banking77_sample.jsonl` — 154 items (2 per class), stratified sample
  of [Banking77](https://github.com/PolyAI-LDN/task-specific-datasets)
  (CC-BY-4.0), a public 77-way intent classification set — the
  high-cardinality-choice case Laya's own README flags as Jev's strength,
  included on purpose. `data/build_banking77.py` regenerates it deterministically.

## Fairness notes

- Every system gets the same state text and the same bare option/level
  labels — no per-option descriptions for anyone.
- Laya and `von` were given an enlarged option-token budget for Banking77
  per each project's own documented guidance for 50+ option questions —
  applied identically, not a `poorjev`-only advantage.
- `poorjev` uses `temperature=1.0` (raw) on Banking77, not its own fitted
  `T=2.71` — that value was fit on the 4-6 option multi-primitive set and
  does not transfer to a 77-way task. `results/banking77_poorjev_recalibrated.json`
  shows a proper 5-fold CV refit *on Banking77 itself* lands at `T≈1.01`
  (essentially a no-op) — the miscalibration at this cardinality is
  structural to the small local NLI backend, not a fixable scalar.

## Reproducing

```bash
pip install "poorjev[local]" laya von-sdk

# poorjev, own CLI
poorjev eval --set ../evalset/tasks.jsonl

cd crossbench

# Banking77
python banking77_harness.py --system poorjev
python banking77_harness.py --system laya
python banking77_harness.py --system von
TYPESAFE_API_KEY=sk-... python banking77_harness.py --system jev

# Multi-primitive set, non-poorjev systems
python typed_decisions_harness.py --system laya
python typed_decisions_harness.py --system von
TYPESAFE_API_KEY=sk-... python typed_decisions_harness.py --system jev
```

Jev requires a `TYPESAFE_API_KEY` (paid, ~$0.042/1M input tokens — reproducing
this whole benchmark costs a few cents). No key is committed to this repo.

## Limits, stated up front

- 154 and 160 decisions respectively — small samples; treat differences under
  a few points as noise, not a ranking.
- The multi-primitive set is hand-labelled by one person (support-ticket
  flavored). Banking77 is the standard, public check.
- Bare labels only, no rich per-option descriptions — a level playing field,
  but not how you'd deploy any of these in production.
- Written by `poorjev`'s own author. The code and every result JSON in
  `results/` are here so anyone can check that for themselves — that's the
  actual point.
