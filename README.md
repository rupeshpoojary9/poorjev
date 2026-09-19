# poorjev

**the poor man's Jev.**

Poor in price. Rich in honesty. Typed decisions with confidence that is actually
calibrated, not vibes. Runs on your laptop. No API key. No waitlist.

```bash
pip install poorjev
```

```python
from poorjev import Choice, Score, Noul

topic = Choice(["billing", "technical", "account", "other"])
ans = topic.decide([2.4, 0.1, 0.3, 0.0])   # a backend supplies these scores

ans.value        # "billing"          (always one of your options, by construction)
ans.confidence   # 0.86               (calibrated once you fit the calibration layer)
ans.probs        # {"billing": 0.86, "account": 0.07, ...}
```

## Why

Most production AI work is fast structured decisions, route, classify, extract,
score, gate a tool call, not chat. Jev nailed that thesis but is a hosted
waitlist. poorjev gives you the same typed-question interface locally, and it is
the one that proves its confidence is real.

Everyone ships an LLM in JSON mode and calls the `0.9` a "confidence." It is a
vibe. poorjev is the only one built to prove its `0.9` actually means `0.9`
(reliability diagram and ECE before/after, landing in M4).

## Status

Early build. Shipped so far:

- **M1, the contract.** Three primitives (`Choice` / `Score` / `Noul`) and typed
  answers. Schema validity is structural: the returned `value` is always drawn
  from your declared set, proven against adversarial inputs (NaN, inf,
  negatives, all-zero) in the test suite. This is Jev's "0 type errors" claim,
  made honest.

Next: M2 local NLI backend (no key), M3 eval set + metrics, M4 calibration and
the reliability diagram, M5 optional LLM backend. See `PRD.md`.

## What this is not

Not a Jev clone weights-wise, not a hosted API, and no speed claims. It is
interface-compatible and honestly calibrated, on commodity models. The name is
playful; the numbers are not.

MIT licensed.
