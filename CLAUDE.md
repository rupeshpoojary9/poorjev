# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repo.

## What poorjev is

An open source, local-first "System One" decision layer: ask typed questions
about some state (Choice / Score / Noul), get typed answers with **calibrated**
confidence, in one pass, no API key by default. The differentiator is honest,
measured calibration. See `PRD.md` for the full spec and `README.md` for the pitch.

## Architecture (read in this order)

- `src/poorjev/primitives.py` — the contract. `Choice` / `Score` / `Noul` and
  their frozen typed answers. Schema validity is **structural**: a decided
  answer's `value` is always in the declared set, even for adversarial scores.
- `src/poorjev/client.py` — `Client.ask(state, questions)`. Builds one batched
  set of (state, hypothesis) pairs across all questions, calls the backend once,
  applies the fitted `temperature`, and decides. Backends are a Protocol.
- `src/poorjev/backends/local_nli.py` — default keyless backend. A zero-shot NLI
  model scores each hypothesis for entailment. Lazy torch/transformers import.
- `src/poorjev/calibration.py` — the moat. Temperature scaling (fit by k-fold
  CV) + conformal abstention.
- `src/poorjev/metrics.py` — accuracy, ECE, Brier, risk-coverage (stdlib only).
- `src/poorjev/evaluate.py` — run the eval set through a backend, score vs gold.
- `src/poorjev/plots.py` — reliability diagram + risk-coverage plot.
- `src/poorjev/mcp_server.py` — expose poorjev as MCP tools (gate/judge/classify/
  rate/decide). Logic is in plain `do_*` functions; MCP wrappers are thin.
- `src/poorjev/cli.py` — `poorjev eval | calibrate | ask | serve`.
- `evalset/source.py` -> `evalset/tasks.jsonl` — hand-labelled data.

## Commands

```bash
pip install -e ".[local,plots,mcp,dev]"   # full dev install
pytest -q                                  # fast suite, no model, no key
poorjev eval --set evalset/tasks.jsonl     # accuracy + ECE (downloads model)
poorjev calibrate --set evalset/tasks.jsonl --plots   # before/after ECE + diagrams
poorjev serve                              # run the MCP server (stdio)
python evalset/source.py                   # regenerate tasks.jsonl
```

## Conventions

- **Tests never need a model or a key.** Unit tests use a fake/scripted backend.
  Anything that loads the real model is an example or a manual run, not a test.
- **Honesty is the product.** Every headline number must be reproducible by
  `poorjev eval` / `poorjev calibrate` on the shipped set. No cherry-picking. If
  a number is bad, print it. Keep the honest-limitations sections truthful.
- **No em-dashes in README / marketing copy.** Use commas, colons, short
  sentences. (Source code and this file are exempt.)
- **Ask concrete questions.** The local NLI model handles "this moves money"
  well and "this is dangerous" poorly. Prefer concrete hypotheses; decompose
  abstract ones and apply a rule.
- **Commits:** author as the repo owner only; do not add co-author trailers.

## Good first contributions

- More labelled decision tasks in `evalset/` (raises the eval's credibility).
- Per-question temperature in `calibration.py` (likely lowers ECE further).
- A new backend implementing `entail_probs(pairs)` (e.g. the optional LLM one).
