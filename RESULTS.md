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

Lands in M4. This section will show:

- ECE **before -> after** (target: a clear drop from 0.170 toward < 0.05),
- the reliability diagram, raw vs calibrated (the "after" hugging the diagonal),
- the risk-coverage curve: accuracy rising as the model abstains on its least
  confident decisions.

Numbers go here only once `poorjev eval` and `poorjev calibrate` produce them.
