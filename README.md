<h1 align="center">poorjev</h1>

<p align="center"><b>The poor man's Jev.</b> An open source, local-first "System One" decision layer for LLM apps: typed decisions with <b>provably calibrated confidence</b>. No API key. No waitlist.</p>

<p align="center">
  <a href="https://pypi.org/project/poorjev/"><img src="https://img.shields.io/pypi/v/poorjev" alt="PyPI version"></a>
  <img src="https://img.shields.io/badge/license-MIT-green" alt="MIT license">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <a href="https://github.com/rupeshpoojary9/poorjev/actions/workflows/tests.yml"><img src="https://github.com/rupeshpoojary9/poorjev/actions/workflows/tests.yml/badge.svg" alt="tests"></a>
  <img src="https://img.shields.io/badge/ECE-0.170%20%E2%86%92%200.071-orange" alt="ECE 0.170 to 0.071">
  <img src="https://img.shields.io/badge/API%20key-not%20required-blueviolet" alt="no API key required">
</p>

---

**Your model's `0.9` is a vibe. poorjev's `0.9` is a measurement.**

Every LLM-in-JSON-mode hands you a confidence score and hopes you don't check it. poorjev checks it. On the shipped eval set it cuts calibration error (ECE) from **0.170 to 0.071** with zero loss of accuracy, and it runs on your laptop with no API key.

<p align="center">
  <img src="https://raw.githubusercontent.com/rupeshpoojary9/poorjev/main/docs/reliability_before_after.png" alt="Reliability diagram: raw confidences are overconfident, calibrated confidences hug the diagonal" width="760">
</p>

<p align="center"><i>Left: raw confidences, overconfident. Right: calibrated, a stated 0.8 really is right about 80% of the time.</i></p>

## Quickstart

```bash
pip install "poorjev[local]"
```

```python
from poorjev import Client, Choice, Score, Noul

client = Client()  # local model, no key, offline after one download

result = client.ask(
    state="I've emailed three times and I'm STILL being double-charged. Cancel my account today.",
    questions={
        "topic":        Choice(["billing", "technical", "account", "shipping", "other"]),
        "frustration":  Score(levels=["low", "medium", "high"]),
        "is_urgent":    Noul("The customer needs a response today."),
        "wants_cancel": Noul("The customer wants to cancel their account."),
    },
)

result["topic"].value          # "billing"      always one of your options, by construction
result["topic"].confidence     # 0.86           calibrated, not a vibe
result["frustration"].value    # "high"
result["is_urgent"].value      # True
result["wants_cancel"].value   # True
```

One call, one model pass, four typed answers. No prompt engineering, no JSON parsing, no "the model returned prose."

## Use it in Claude Code (MCP server)

poorjev ships an MCP server, so a Claude Code (or Claude Desktop) agent can make
fast, local, calibrated decisions as tools, with no API key and no token cost.
The obvious use: gate a risky tool call before the agent runs it.

```bash
pip install "poorjev[local,mcp]"
claude mcp add poorjev -- poorjev serve
```

Or add it to `.mcp.json` by hand:

```json
{
  "mcpServers": {
    "poorjev": { "command": "poorjev", "args": ["serve"] }
  }
}
```

The agent then has these local tools:

| Tool | What it does |
|---|---|
| `gate(action)` | guardrail: should this action be blocked (moves money, deletes data)? |
| `judge(text, statement)` | a yes/no question, with calibrated `P(true)` |
| `classify(text, options)` | pick one option, with calibrated confidence |
| `rate(text, levels)` | an ordinal score (low / medium / high) |
| `decide(text, questions)` | several typed questions at once, one pass |

Why this beats asking an LLM to judge: it is local (private), free (no tokens),
fast, and the confidence is calibrated instead of made up.

## Why poorjev exists

Most production AI work is not chat. It is fast structured decisions: **route** a ticket, **classify** an intent, **score** a sentiment, **extract** a field, **gate** a tool call. TypeSafe's **Jev** named this category ("System One" models) and nailed the thesis — and, per the independent [cross-system benchmark](crossbench/) below, it currently backs its calibration claims up: it's the strongest model measured here. It's also closed, hosted, and behind a waitlist.

poorjev exists for the deployments where "call a hosted API" isn't the answer: private data, offline environments, zero marginal cost, no waitlist. It reproduces Jev's typed-decision interface on a small local model and proves its **own** calibration honestly (5-fold cross-validated, never graded on what it was fit on). Against the other open local alternatives it leads on the mixed decision-primitive benchmark below, but not on the high-cardinality one — see the real breakdown. It does not beat Jev. That's the honest trade for fully local and free.

## poorjev vs the field

Independently measured, not self-reported — see [`crossbench/`](crossbench/) for the full harness, data, and every raw result file.

<p align="center">
  <img src="https://raw.githubusercontent.com/rupeshpoojary9/poorjev/main/docs/vs_field_benchmark.png" alt="Bar chart comparing poorjev, Jev, von, and Laya on accuracy and ECE across Banking77 and the multi-primitive set. Jev leads the multi-primitive set on both metrics; von leads Banking77 on both accuracy and calibration among the open options; poorjev leads the open options on the multi-primitive set only." width="760">
</p>

<p align="center"><i>poorjev in red. Chart regenerates from <code>crossbench/results/</code> via <code>crossbench/plot_comparison.py</code> — same numbers as the table below.</i></p>

| | **Jev** (TypeSafe) | **von** | **Laya** | **poorjev** |
|---|---|---|---|---|
| Interface (typed questions, one pass) | yes | yes | yes | yes |
| Runs locally, no API key | no | yes | yes | **yes** |
| Your data stays in your environment | no | yes | yes | **yes** |
| Waitlist / signup | yes | no | no | **no** |
| Open source | no | yes (Apache-2.0) | yes (Apache-2.0) | **yes (MIT)** |
| Accuracy, Banking77 (77-way, n=154) | 0.812 | **0.838** | 0.519 | 0.656 |
| ECE, Banking77 (lower better) | **0.084** | 0.135 | 0.388 | 0.414 |
| Accuracy, multi-primitive set (n=160) | **0.906** | 0.775 | 0.775 | 0.781 |
| ECE, multi-primitive set (lower better) | **0.045** | 0.108 | 0.215 | 0.071 |

Read straight, because that's the point of doing this:

- **Jev wins the multi-primitive set outright** — best accuracy and best
  calibration, no caveats.
- **`von` wins Banking77** — best accuracy of *all four* systems (0.838,
  ahead of even Jev's 0.812), though Jev still calibrates better there
  (0.084 vs 0.135).
- **poorjev leads the open, local options on the multi-primitive set** — best
  accuracy and best calibration among Laya/`von`/poorjev there. That does
  **not** carry over to Banking77: `von` beats poorjev on accuracy by a wide
  margin (0.838 vs 0.656), and poorjev has the *worst* calibration of all four
  systems there (0.414 — even behind Laya's 0.388), not the best.

Nobody sweeps, and poorjev specifically does not sweep the open-source field —
it wins one benchmark and loses the other, to `von`, decisively. Full
methodology, fairness notes, and every raw result file are in
[`crossbench/`](crossbench/) — reproducible for a few cents of Jev API calls
and some CPU time.

poorjev is not a Jev clone and makes no claim to beat it, or to beat `von`
across the board. It reproduces the **interface**, proves its own calibration
with numbers instead of marketing copy, and is the strongest fully local
option on the mixed decision-primitive benchmark — not on high-cardinality
classification, where `von` currently leads.

## The three primitives

| Primitive | Use it for | Returns |
|---|---|---|
| `Choice(options)` | classification, routing | winning option, per-option probabilities, calibrated confidence |
| `Score(levels)` | ordinal rating, severity | winning level, a continuous score on the scale, confidence |
| `Noul(statement)` | yes/no gates, guardrails | `P(true)`, thresholded to a bool |

The returned `value` is **always** drawn from the set you declared. An invalid category is structurally impossible, not "usually avoided." This is tested against adversarial inputs (NaN, infinity, negatives, all-zero score vectors).

## How it works

```
state + typed questions
        |
        v
   one batched pass through a local zero-shot NLI model   (no API key)
        |
        v
   raw probabilities per option
        |
        v
   calibration: temperature scaling + conformal abstention
        |
        v
   typed, schema-valid answers + calibrated confidence
```

- **Local backend (default):** one small natural-language-inference model scores every option as an entailment hypothesis, in a single batched forward pass. Fully offline after a one-time ~400MB download. No key, no vendor, your text never leaves your machine.
- **Calibration (the moat):** temperature scaling fits one scalar so predicted confidence matches real accuracy; conformal thresholding turns a target risk budget into an "I don't know, escalate" signal.
- **LLM backend (optional, roadmap):** when you need more reasoning, point poorjev at an LLM and it makes that model's confidence honest too. That is the intelligence dial, not the default.

## Benchmarks

Reproduce everything with two commands:

```bash
poorjev eval       --set evalset/tasks.jsonl          # accuracy, ECE, Brier, risk-coverage
poorjev calibrate  --set evalset/tasks.jsonl --plots  # before/after ECE + the diagrams
```

On the shipped eval set (55 hand-labelled items, 160 decisions), local NLI backend, keyless:

| Metric | Raw | Calibrated |
|---|---:|---:|
| Accuracy | 0.781 | 0.781 |
| **ECE (calibration error)** | **0.170** | **0.071** |
| Brier | 0.184 | lower |
| Temperature | 1.00 | 2.71 |

Temperature is fit by 5-fold cross-validation, so the "after" number is measured on held-out data, never on data it was fit on. Full tables and the honest limitations are in [RESULTS.md](RESULTS.md).

**Against Jev, Laya, and von, on the same inputs, same metrics code:** see [poorjev vs the field](#poorjev-vs-the-field) above and the full harness in [`crossbench/`](crossbench/). Short version: poorjev leads the open options on this mixed decision-primitive benchmark, `von` leads on high-cardinality classification, Jev leads overall.

## Selective prediction: it knows when it doesn't know

Set a risk budget and poorjev abstains on its least confident decisions instead of guessing:

<p align="center">
  <img src="https://raw.githubusercontent.com/rupeshpoojary9/poorjev/main/docs/risk_coverage.png" alt="Risk-coverage curve: error rate drops as the model abstains on low-confidence decisions" width="440">
</p>

At a 10% error budget it confidently answers 55% of decisions and escalates the rest. That is the natural bridge from System One (fast automatic answer) to System Two (a human, or a bigger model).

## Real examples

```bash
python examples/ticket_router.py   # full triage on a support ticket
python examples/tool_gate.py       # gate a risky tool call before it runs
python examples/demo.py            # raw vs calibrated, side by side
```

The tool-gate example encodes a practical lesson: the local model is strong at **concrete** questions ("this action moves money", "this deletes data") and weak at **abstract** ones ("this is dangerous"). Ask concrete questions and let a one-line rule apply the policy.

## Honest limitations

No hype. Here is what this is not.

- **Not as fast as Jev.** Jev uses a custom model. poorjev uses commodity ones. We report latency, we do not market it.
- **The eval set is small** (tens of items, one labeller, English, support flavoured). Enough to show calibration direction and schema validity, not a leaderboard.
- **After-ECE is 0.071, not below 0.05.** That is the real cross-validated number, reported as measured. Per-question temperature would likely push it lower.
- **The local model is moderately intelligent.** It does real semantic entailment, not deep reasoning. Calibration and abstention are what make that safe.
- **Jev is currently ahead, measured, not assumed, and so is `von` on one axis.** The [cross-system benchmark](crossbench/) has Jev winning the multi-primitive set outright and leading Banking77 calibration; `von` beats both Jev and poorjev on Banking77 accuracy. poorjev's honest position is "best fully local/free option on the mixed decision-primitive benchmark," not "beats Jev" and not "beats every open alternative everywhere."
- **Temperature scaling doesn't fix everything.** At Banking77's 77-way cardinality, a proper cross-validated temperature refit barely moves ECE (0.414 → 0.416) — the miscalibration there is structural to the small NLI backend at high option counts, not a scalar you can fit away. See [`crossbench/results/banking77_poorjev_recalibrated.json`](crossbench/results/banking77_poorjev_recalibrated.json).

## FAQ

**Is this a Jev clone?** No. It reproduces Jev's developer interface and its calibrated-confidence guarantee on open, local models. It does not copy Jev's architecture or its speed.

**Can I run Jev locally?** Not Jev itself, it is closed and hosted. poorjev is the local, open-source alternative: it runs the same typed-decision interface on your own machine, offline, with no API key and no waitlist.

**Is there an open-source alternative to Jev?** Yes, this is one. poorjev is MIT-licensed, reproduces Jev's `Choice`/`Score`/`Noul` interface on commodity models, and proves its calibration with reproducible numbers.

**Do I need an API key or GPU?** No. The default backend runs on CPU, offline, after one model download.

**How is this different from an LLM in JSON mode?** Two ways. Output is schema-valid by construction, not by parsing. And the confidence is calibrated and proven, not a number the model made up.

**What is a "System One" model?** A model for fast, automatic, structured decisions (classify, route, score, gate), as opposed to slow, deliberative chat. The name is from Kahneman's System 1 / System 2.

**What is ECE?** Expected Calibration Error: the average gap between a model's confidence and its actual accuracy. Lower is better. poorjev's whole job is to shrink it.

**Can I use my own model?** Yes. Backends are pluggable; a backend only implements `entail_probs(pairs)`.

## Roadmap

- [x] Typed primitives, schema-valid by construction
- [x] Local NLI backend, single pass, keyless
- [x] Eval set + metrics (accuracy, ECE, Brier, risk-coverage)
- [x] Calibration: temperature scaling + conformal abstention
- [x] MCP server: use poorjev as local tools in Claude Code
- [x] Independent cross-system benchmark vs Jev, Laya, von ([`crossbench/`](crossbench/))
- [ ] Optional LLM backend (the intelligence dial)
- [ ] Close the Banking77 accuracy/calibration gap to Jev (bigger backend, per-class calibration)

## Contributing

Issues and PRs welcome, especially new labelled decision tasks for the eval set. If you find a case where the confidence is not honest, that is a bug worth filing.

## License

MIT. Use it, ship it, sell it.

---

<p align="center"><i>poorjev: poor in price, rich in honesty. If your model's confidence is a vibe, come check it.</i></p>
