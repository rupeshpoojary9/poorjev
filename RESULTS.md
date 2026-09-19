# Results

Every number here is produced by `poorjev eval` on the shipped eval set
(`evalset/tasks.jsonl`, 55 items / 160 labelled decisions). Reproduce with:

```bash
pip install -e ".[local]"
poorjev eval --set evalset/tasks.jsonl
```

The set is small and hand-labelled. It is enough to estimate accuracy and
calibration and to show the direction calibration moves them. It is not a large
benchmark, and we do not claim it is.

## Raw, uncalibrated (local NLI backend)

Backend: `MoritzLaurer/deberta-v3-base-zeroshot-v2.0`, keyless, offline.

| Slice | n | Accuracy | ECE | Brier | AURC |
|---|---:|---:|---:|---:|---:|
| **overall** | 160 | 0.781 | **0.170** | 0.184 | 0.084 |
| action_risk | 45 | 0.867 | 0.135 | 0.134 | 0.040 |
| intent | 15 | 0.933 | 0.143 | 0.095 | 0.007 |
| support_triage | 100 | 0.720 | 0.200 | 0.220 | 0.116 |

Reading this honestly: the model is right about 78% of the time, but its
confidence is off by 0.170 on average (ECE). That gap is the problem poorjev
exists to fix. The raw probabilities out of any classifier, local or LLM, are
not trustworthy as-is.

## Calibrated (temperature scaling + conformal abstention)

Reproduce with:

```bash
poorjev calibrate --set evalset/tasks.jsonl --plots
```

Temperature is fit by 5-fold cross-validation: on each fold T is fit on the
other four and the calibrated ECE is measured on the held-out fold, so the
"after" number is never graded on data it was fit on.

| Metric | Raw | Calibrated | |
|---|---:|---:|---|
| Accuracy | 0.781 | 0.781 | unchanged (temperature is monotonic) |
| **ECE** | **0.170** | **0.071** | **58% lower calibration error** |
| Temperature (mean) | 1.00 | 2.71 | > 1: the model was overconfident, softened |

![reliability before and after](docs/reliability_before_after.png)

Left: raw confidences scatter around the diagonal, mostly below it
(overconfident). Right: after temperature scaling the bars hug the diagonal,
predicted confidence now tracks real accuracy.

### Selective prediction

Set a risk budget and abstain below the matching confidence threshold:

| Target risk | Coverage | Threshold |
|---|---:|---:|
| <= 0.10 | 0.55 | 0.830 |

At a 10% error budget the model confidently answers 55% of decisions and
abstains on the rest (the honest "I don't know, escalate" signal). Full curve:

![risk coverage](docs/risk_coverage.png)

### Honest notes

- The after-ECE is 0.071, not below 0.05. That is the real cross-validated
  number on a small set; we report it as measured rather than tuning to a
  target. A per-task or per-question temperature (instead of one global scalar)
  would likely push it lower, and is a natural next step.
- Same eval set caveats as above: small, single-labeller, support flavoured.
