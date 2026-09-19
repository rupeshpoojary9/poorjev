# PRD — `poorjev` (the poor man's Jev)

**An open-source, local-first "System One" decision layer.** Same developer interface as
Jev (typed questions over program state, answered in one pass with **calibrated**
probabilities) but backed by small local models or any LLM, and where the one thing we get
*provably* right is the exact thing Jev sells on: **honest confidence.**

- **Owner:** Rupesh Poojary
- **Date:** 2026-09-19
- **Package / repo:** `poorjev` → `pip install poorjev` → `github.com/rupeshpoojary9/poorjev`
  - *Name is free on PyPI + GitHub as of 2026-09-19. Backups: `poormansjev`, `poorjevai`.*
- **Status:** 🔴 not started — this PRD is the build spec
- **Lane:** LLM-evals authority (pairs with `rag-eval-benchmark`, `awesome-llm-evals`). Sister project to a future `system-one-bench`.

---

## 0. The viral wedge (read this first)

**Name:** `poorjev`. It carries the whole story in seven characters: open, cheap, Jev-adjacent, underdog. "Poor man's X" is a badge in dev culture, not an insult, it signals resourceful-hacker "I got 90% of the expensive thing for free." Nobody needs the joke explained; it lands in zero seconds.

**The one spicy, true claim we own:** *every* "System One" project on jevusecases.com (1350+ of them) quotes speed and cost. Almost none prove their **confidence is calibrated**. Jev sells calibrated confidence but gates it behind a waitlist. So the line that spreads is:

> Everyone ships an LLM in JSON mode and calls the `0.9` a "confidence." It's a vibe.
> `poorjev` is the poor man's Jev, and it's the only one that proves its `0.9` actually means `0.9`. Runs on your laptop. No waitlist.

**The one screenshot that sells it:** a **reliability diagram**, our predicted probability vs. actual accuracy, before and after calibration (ECE ~0.17 → <0.05). That single image is the launch asset. Build toward producing it (see §7, M4).

### Launch kit (draft copy, keep em-dash-free per house style)

**README hero (top of the repo):**
```
poorjev
the poor man's Jev.

Poor in price. Rich in honesty.
Typed decisions with confidence that is actually calibrated, not vibes.
Runs on your laptop. No API key. No waitlist.

pip install poorjev
```

**Launch tweet / X post:**
```
Jev is a $40M waitlist.

poorjev is the poor man's version: same typed-question interface, runs
local with no key, and it is the ONLY one that proves its confidence is
real (ECE 0.17 -> 0.04, reliability diagram below).

Your model's 0.9 is a vibe. Mine is a measurement.

pip install poorjev
[reliability-diagram.png]
```

**Hacker News title:**
```
Show HN: poorjev, a local "System One" decision layer that proves its confidence is calibrated
```

**One-line elevator (for GitHub About + resume):**
```
Open, local alternative to Jev's System One API: typed decisions with
provably calibrated confidence. 100% schema-valid by construction.
```

**Distribution plan (from [[github-followers-strategy]]: distribution, not more repos):**
1. Ship the repo with the reliability diagram in the README above the fold.
2. Submit to jevusecases.com (1350+ dir, the exact audience) as the calibration-first entry.
3. Show HN + one X thread built around the "your 0.9 is a vibe" line + the diagram.
4. Add to `awesome-llm-evals` under a new "calibration / selective prediction" section.
5. Reply-guy value: whenever someone benchmarks Jev on speed, add the calibration axis nobody else measures.

---

## 1. Why this exists

TypeSafe launched **Jev** (Sept 15 2026, $40M DCVC): a "System One model" that takes program
state + typed questions and returns **typed answers with calibrated probabilities in one
parallel pass**, 70-500ms, output tokens free, ~0% type errors. The category thesis is
correct: **most production AI work is fast structured decisions** (route, classify, extract,
score, moderate, gate a tool call), not chat. But Jev is **gated behind a waitlist**, closed,
and hosted-only.

The gap: developers want that *interface*, "ask N typed questions about this state, get typed
answers + trustworthy confidence, cheap and local", today, without a waitlist, and without
shipping their data to a new vendor. And nobody in the ecosystem is proving the confidence
half of the promise, which is the half that actually matters for routing and gating.

**What we do NOT claim:** we are not reproducing Jev's non-autoregressive parallel
architecture or its speed. We reproduce the **interface** and the **calibrated-confidence
guarantee**, on commodity models, and we prove the calibration with numbers. That honesty is
the whole pitch, see §9.

## 2. Goal / non-goals

**Goal.** A `pip install poorjev`-able Python library + CLI that:
1. exposes the three System One primitives (**Choice / Score / Noul**) over arbitrary state,
2. answers a batch of typed questions in a **single call** with typed, schema-valid outputs,
3. attaches a **calibrated** probability/confidence to every answer,
4. supports **abstention** (selective prediction) when confidence is low,
5. runs **fully local with no API key** by default, and swaps to a real LLM backend with **one env var**,
6. ships a **self-eval** that reports calibration + accuracy on a small labelled set, so the confidence claim is defensible,
7. produces the **reliability diagram** from §0 as a committed artifact.

**Non-goals (v1).**
- Not a hosted API / not a Jev clone weights-wise.
- No multimodal state (text / JSON / arrays only, matches Jev v1).
- No training a bespoke model from scratch (we adapt existing encoders + optional LLM).
- No sub-100ms latency promise. Latency is *reported*, not marketed.

## 3. Users

- App devs who want a cheap, local classify/route/score/gate layer with real confidence.
- Anyone building LLM-agent guardrails ("is this tool call risky?" as a Noul gate).
- **You**, in interviews: a working, measured artifact on the freshest model category, in your evals lane. Also a candidate decision layer for Vereno's live-demo loop.

## 4. The interface (the product surface)

Mirror the LangChain/TypeSafe shape closely enough to be a genuine drop-in mental model.

```python
from poorjev import Client, Choice, Score, Noul

client = Client()                       # local backend, no key, by default

result = client.ask(
    state="Customer: 'I've emailed three times and STILL been double-charged. Cancel my account.'",
    questions={
        "topic":      Choice(["billing", "technical", "account", "other"]),
        "frustration": Score(levels=["low", "medium", "high"]),
        "is_urgent":  Noul("The customer needs a response today."),
        "wants_cancel": Noul("The customer is asking to cancel."),
    },
)

result["topic"].value          # -> "billing"
result["topic"].probs          # -> {"billing": 0.86, "account": 0.09, ...}
result["topic"].confidence     # -> 0.86   (calibrated)
result["frustration"].value    # -> "high"
result["frustration"].score    # -> 1.72   (continuous, on the ordinal scale)
result["is_urgent"].value      # -> True
result["is_urgent"].prob       # -> 0.94   (calibrated P(true))
result["wants_cancel"].abstained  # -> False   (True if below the conformal threshold)
```

**Primitives (must match Jev semantics):**
| Primitive | Input | Output |
|---|---|---|
| `Choice(options)` | discrete options | winning option + per-option probs + confidence |
| `Score(levels=[...])` | ordered levels | continuous score on the scale + distribution + confidence |
| `Noul(statement)` | a yes/no statement | `P(true)` (a bool once thresholded) |

**Contract guarantees (enforced, tested):**
- Output is always schema-valid, the returned `value` is *always* one of the declared options / within the scale / a bool. No parsing, no "the model returned prose." (This is Jev's "0 type errors" claim, and for us it's true by construction, not by hope, see §5.)
- All questions in one `ask()` share one state encode and are answered in one pass.
- Every answer carries a probability; `--calibrate` makes those probabilities calibrated.

## 5. Architecture

```
                          ┌──────────── backends (pluggable) ────────────┐
state + questions ─▶ ask() ┤ local  (default): NLI / zero-shot encoder    ├─▶ raw scores
                          │ llm    (opt-in):  constrained decode + logprobs│
                          └───────────────────────────────────────────────┘
                                              │ raw scores
                          calibration layer ──┤ temperature scaling (fit on dev set)
                                              │ conformal prediction (abstention sets)
                                              ▼
                          typed, schema-valid answers + calibrated confidence
```

**5.1 Local backend (default, no key), the clever bit.**
Reuse an NLI / zero-shot model as a general decision engine. One small model does all three:
- **Noul** → natural-language inference: `P(entailment)` of the statement given the state = `P(true)`.
- **Choice** → zero-shot classification: score each option as an entailment hypothesis, softmax over options.
- **Score** → run the ordered levels as hypotheses, take the expected level index as the continuous score, argmax→winning level.

Default model: a small NLI checkpoint (e.g. `MoritzLaurer/deberta-v3-base` zero-shot or `all-MiniLM` + a cross-encoder). ~single-digit hundred MB, downloads once, then fully offline. **Schema validity is structural**, options/levels are fixed sets we softmax over, so an out-of-set answer is impossible by construction.

**5.2 LLM backend (opt-in, `POORJEV_BACKEND=llm` + key).**
Any chat LLM (Anthropic default; see `claude-api` for model IDs). Use **constrained/structured output** so the raw answer is always valid, and pull token **logprobs** for the probability. LLMs are notoriously *miscalibrated* (over-confident verbalized probs), so this backend leans hardest on §5.3 to fix that, which is itself a demonstrable result.

**5.3 Calibration layer (the differentiator, backend-agnostic).**
This is what makes us more than "an LLM in JSON mode":
- **Temperature scaling**: fit one scalar T per question-type on a small labelled dev split so predicted probs match empirical accuracy. Report **before/after ECE**.
- **Conformal prediction**: turn a target risk level (e.g. 90%) into a per-question confidence threshold; below it, `abstained=True`. Gives honest "I don't know → escalate to System Two" behaviour, the natural bridge to a router.

**5.4 Single-pass batching.** Encode state once; evaluate all questions against that encoding in one batched forward pass (local) or one structured request (LLM). No per-question round-trips.

## 6. Repo layout (matches your `rag-eval-benchmark` conventions)

```
poorjev/
├── README.md                 # the §0 hero + reliability diagram above the fold, quickstart, honest claims
├── RESULTS.md                # real calibration + accuracy tables, reliability diagrams
├── PRD.md                    # this file
├── LICENSE                   # MIT
├── pyproject.toml            # hatchling; extras: [llm], [plots], [dev]; script: poorjev=poorjev.cli:main
├── src/poorjev/
│   ├── __init__.py           # exports Client, Choice, Score, Noul
│   ├── primitives.py         # the three question types + typed Answer objects
│   ├── client.py             # ask(): orchestrates encode → backend → calibrate → type
│   ├── backends/
│   │   ├── local_nli.py      # default no-key backend
│   │   └── llm.py            # opt-in Anthropic/structured-output backend
│   ├── calibration.py        # temperature scaling + conformal prediction
│   ├── metrics.py            # ECE, Brier, accuracy, coverage/risk
│   ├── plots.py              # reliability diagrams, risk-coverage curves (THE launch asset)
│   └── cli.py                # poorjev ask / eval / calibrate
├── evalset/                  # small hand-labelled decision set (see §7)
│   ├── tasks.jsonl           # state + questions + gold answers
│   └── README.md             # how it was labelled
├── examples/
│   ├── ticket_router.py      # Choice+Score+Noul over support tickets
│   └── tool_gate.py          # Noul guardrail: "is this tool call risky?" (agent middleware)
└── tests/                    # schema-validity invariants, calibration math, primitives
```

**CLI:**
```bash
poorjev ask --state-file ticket.txt --questions questions.yaml     # one-off decision, JSON out
poorjev eval  --set evalset/tasks.jsonl --plots                    # accuracy + ECE + Brier + coverage + reliability.png
poorjev calibrate --set evalset/tasks.jsonl                        # fit + save temperature/conformal params
poorjev eval  --backend llm --plots                                # same, LLM backend (needs key)
```

## 7. Eval set (the part that makes it credible)

Hand-label a **small, honest** decision set, the same discipline that made `rag-eval-benchmark`
defensible. ~60-100 items across a few realistic decision tasks (support-ticket triage,
content moderation flags, intent detection, tool-risk gating). Each item = state + the typed
questions + **gold answers**. This is the hard, valuable part most write-ups skip; it's what
lets us report calibration honestly and it's the seed corpus for the future `system-one-bench`.

## 8. Success metrics / definition of done

v1 ships when, on the local backend over the eval set, `poorjev eval` reports real numbers and:

- 🎯 **Schema validity = 100%** by construction (invariant test: no `ask()` ever returns an out-of-set/wrong-type value, incl. adversarial states). This is our honest version of Jev's "0 type errors."
- 🎯 **Calibration improves measurably** post-temperature-scaling: report **ECE before → after** and **Brier**; target a clear ECE reduction (e.g. ~0.15 → <0.05), the exact number is whatever we honestly measure.
- 🎯 **The reliability diagram exists** and is committed to the repo + README (the §0 launch asset).
- 🎯 **Selective prediction works**: a risk-coverage curve showing accuracy rises as we abstain on low-confidence items.
- 🎯 **Accuracy is competitive** with an "LLM-in-JSON-mode" baseline while being local/free, and where it loses, we say so.
- 🎯 **Latency + cost reported** (not marketed) for local vs LLM backend.
- 🎯 Runs **offline, no key**, first-run downloads one small model; `pip install -e ".[dev]" && poorjev eval` reproduces the tables.
- 🎯 README + RESULTS.md written; tests green; MIT; pushed public.

## 9. Honest-claims guardrails (non-negotiable, your no-hype rule)

- Never claim to match Jev's speed or architecture. Frame explicitly as: *interface-compatible + honestly-calibrated, on commodity models.*
- The name is playful; the numbers are not. Every headline number in README/RESULTS is one `poorjev eval` produces on the shipped eval set. No cherry-picking, no vibes.
- "Poor man's" is positioning, never an excuse for a weak result. If a number is bad, we print it.
- State the eval set is small and name what that does/doesn't prove.
- No em-dashes in any rendered marketing/README/tweet copy (commas/colons). *(This PRD is internal, so dashes here are fine.)*

## 10. Milestones (suggested build order)

1. **M1 — Contract & primitives.** `primitives.py`, typed `Answer` objects, schema-validity invariant tests. *(Prove the "0 type errors" claim first.)*
2. **M2 — Local backend.** NLI/zero-shot engine for all three primitives; `ask()` single-pass. First end-to-end decision. *(Study `openjev-sglang` first, differentiate on calibration not speed.)*
3. **M3 — Eval set + metrics.** Hand-label `tasks.jsonl`; implement ECE/Brier/accuracy/coverage; `poorjev eval` raw (uncalibrated) numbers.
4. **M4 — Calibration + the money screenshot.** Temperature scaling + conformal abstention; before/after ECE; **reliability diagram + risk-coverage plot**. *(This is the differentiator AND the launch asset, don't skip to polish before it works.)*
5. **M5 — LLM backend.** Opt-in Anthropic/structured-output + logprobs; show it's miscalibrated raw and fixed by M4.
6. **M6 — Launch.** `ticket_router.py`, `tool_gate.py`, honest write-up, tests green, push public, then run the §0 distribution plan (jevusecases.com + Show HN + X thread + awesome-llm-evals).

## 11. Stretch / follow-ons

- **`system-one-bench`**: promote `evalset/` into the standalone benchmark. Score `poorjev`, an LLM baseline, and (when access lands) Jev, on accuracy + calibration + latency/cost. You'd own the yardstick for the category.
- **System-One/System-Two router** built on conformal abstention.
- **Vereno**: wire `poorjev` as the sub-500ms decision loop in the live-demo console.
- LangChain-shim so `poorjev` drops into the `TypeSafeClassifier` call site.
