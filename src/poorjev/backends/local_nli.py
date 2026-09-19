"""Local, keyless NLI backend, the hero.

One small natural-language-inference model does all three primitives by scoring
(state, hypothesis) pairs for entailment:

- Noul   -> P(entailment) of the statement given the state = P(true).
- Choice -> score each option as a hypothesis, softmax across options.
- Score  -> same, over the ordered levels.

Why NLI and not a sampled LLM: it is one forward pass (Jev's speed ballpark, not
seconds), it is fully local and offline after a one-time ~400MB download, and its
softmax output is a real probability we can calibrate directly, no sampling tax.
It is not the smartest option; that is the point of the optional [llm] backend.
Here, calibration + abstention are what make moderate intelligence safe: the
model knows when it is unsure and escalates.

The heavy imports (torch, transformers) are lazy so importing poorjev, and the
whole M1 contract, stays dependency-free.
"""

from __future__ import annotations

DEFAULT_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"


class LocalNLIBackend:
    """Scores (premise, hypothesis) pairs and returns P(entailment) for each.

    This is the only method the Client needs. Everything primitive-specific
    (templating options, normalising across a Choice) lives in the Client so a
    future backend can reuse the same orchestration.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 batch_size: int = 16, max_length: int = 512):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device = device
        self._tokenizer = None
        self._model = None
        self._entail_idx: int | None = None

    # -- lazy load -------------------------------------------------------- #
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "The local NLI backend needs the 'local' extra. "
                "Install it with:  pip install 'poorjev[local]'"
            ) from e

        if self._device is None:
            if torch.cuda.is_available():
                self._device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:  # deberta-v3 fast tokenizer can be finicky; fall back
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self._device)
        self._model.eval()
        self._entail_idx = self._find_entail_idx(self._model.config)

    @staticmethod
    def _find_entail_idx(config) -> int:
        id2label = getattr(config, "id2label", None) or {}
        for idx, label in id2label.items():
            name = str(label).lower()
            if "entail" in name and "not" not in name and "non" not in name:
                return int(idx)
        # Fallback: many NLI heads put entailment first or last; default to 0.
        return 0

    # -- the one method the Client calls ---------------------------------- #
    def entail_probs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Return P(entailment) in [0, 1] for each (premise, hypothesis) pair,
        computed in batched forward passes (the single-pass promise)."""
        if not pairs:
            return []
        self._ensure_loaded()
        import torch

        out: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]
            premises = [p for p, _ in chunk]
            hypotheses = [h for _, h in chunk]
            enc = self._tokenizer(
                premises, hypotheses,
                return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self._device)
            with torch.no_grad():
                logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, self._entail_idx]
            out.extend(probs.detach().cpu().tolist())
        return out
