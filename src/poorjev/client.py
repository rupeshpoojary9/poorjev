"""The Client: ask a batch of typed questions about one state, in a single pass.

The Client owns orchestration so backends stay simple. It:
  1. turns each question into (state, hypothesis) pairs,
  2. concatenates every pair across every question into ONE batched call,
  3. hands the raw entailment probabilities back to each primitive to decide.

A backend only has to implement ``entail_probs(pairs) -> list[float]``. The
default backend is the local NLI model (keyless, offline). Pass your own backend
(e.g. a fake in tests, or the future [llm] backend) to swap it out.
"""

from __future__ import annotations

from typing import Protocol

from .primitives import Choice, Score, Noul, ChoiceAnswer, ScoreAnswer, NoulAnswer

DEFAULT_TEMPLATE = "This example is {}."


class Backend(Protocol):
    def entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class Client:
    def __init__(self, backend: Backend | None = None,
                 hypothesis_template: str = DEFAULT_TEMPLATE):
        self._backend = backend
        self.hypothesis_template = hypothesis_template

    @property
    def backend(self) -> Backend:
        if self._backend is None:
            from .backends.local_nli import LocalNLIBackend
            self._backend = LocalNLIBackend()
        return self._backend

    def ask(self, state: str, questions: dict):
        """Answer every question about ``state`` in one batched pass.

        Returns a dict mapping each question name to its typed answer
        (ChoiceAnswer / ScoreAnswer / NoulAnswer).
        """
        if not isinstance(state, str) or state == "":
            raise ValueError("state must be a non-empty string")
        if not questions:
            raise ValueError("ask() needs at least one question")

        pairs: list[tuple[str, str]] = []
        plan: list[tuple[str, object, int, int, str]] = []  # name, prim, start, count, mode

        for name, prim in questions.items():
            if isinstance(prim, Noul):
                hyps = [prim.statement]
                mode = "noul"
            elif isinstance(prim, (Choice, Score)):
                hyps = [self.hypothesis_template.format(h) for h in prim.hypotheses]
                mode = "dist"
            else:
                raise TypeError(
                    f"question {name!r} must be a Choice, Score or Noul, got {type(prim).__name__}"
                )
            start = len(pairs)
            pairs.extend((state, h) for h in hyps)
            plan.append((name, prim, start, len(hyps), mode))

        probs = self.backend.entail_probs(pairs)  # the single pass
        if len(probs) != len(pairs):
            raise RuntimeError(
                f"backend returned {len(probs)} scores for {len(pairs)} pairs"
            )

        out: dict[str, object] = {}
        for name, prim, start, count, mode in plan:
            seg = probs[start:start + count]
            if mode == "noul":
                out[name] = prim.decide(seg[0], kind="prob")
            else:
                # entailment probs per option/level, normalised across the set
                out[name] = prim.decide(seg, kind="probs")
        return out
