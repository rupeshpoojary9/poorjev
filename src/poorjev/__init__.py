"""poorjev: the poor man's Jev.

A local-first "System One" decision layer. Ask typed questions about some state,
get typed answers with calibrated confidence, in one pass, with no API key by
default. The one thing we prove: the confidence is honest.

M1 ships the contract (the three primitives + typed answers). Backends (M2/M5)
and calibration (M4) build on top without ever being able to break schema
validity, which is structural. See PRD.md.
"""

from .primitives import (
    Choice,
    Score,
    Noul,
    ChoiceAnswer,
    ScoreAnswer,
    NoulAnswer,
)
from .client import Client

__version__ = "0.1.0"

__all__ = [
    "Client",
    "Choice",
    "Score",
    "Noul",
    "ChoiceAnswer",
    "ScoreAnswer",
    "NoulAnswer",
    "__version__",
]
