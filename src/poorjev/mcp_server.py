"""poorjev as an MCP server: a fast, local, calibrated decision layer that any
MCP client (Claude Code, Claude Desktop) can call as tools.

The point: an agent that wants to gate a tool call, route a request, or classify
an input should not pay an LLM round trip and token cost for a decision, and
should not trust an uncalibrated made-up confidence. poorjev answers locally,
with no API key, and its confidence is calibrated.

Tools exposed:
  - gate(action):     the agent guardrail. Is this action risky (money/data)?
  - judge(text, q):   a yes/no question with calibrated P(true).
  - classify(text, options): pick one option, with calibrated confidence.
  - rate(text, levels):      an ordinal score.
  - decide(text, questions): several typed questions at once, one pass.

The decision logic is factored into plain do_* functions so it is unit-testable
without a running server. `build_server()` wraps them as MCP tools.
"""

from __future__ import annotations

import json
import os

from .client import Client
from .primitives import Choice, Score, Noul


DEFAULT_GATE_CHECKS = {
    "moves_money": "This action moves, sends, or refunds money.",
    "deletes_data": "This action deletes or destroys data.",
}


def load_temperature(calibrator_path: str | None) -> float:
    """Read the fitted temperature from a calibrator json, or fall back to 1.0."""
    if not calibrator_path or not os.path.exists(calibrator_path):
        return 1.0
    try:
        with open(calibrator_path) as f:
            return float(json.load(f).get("temperature", 1.0))
    except (ValueError, OSError):
        return 1.0


# --------------------------------------------------------------------------- #
# Plain decision functions (unit-testable, take a Client)
# --------------------------------------------------------------------------- #

def do_classify(client: Client, text: str, options: list[str],
                hypothesis_template: str | None = None) -> dict:
    tmpl = hypothesis_template or client.hypothesis_template
    prev = client.hypothesis_template
    client.hypothesis_template = tmpl
    try:
        ans = client.ask(text, {"q": Choice(options)})["q"]
    finally:
        client.hypothesis_template = prev
    return {
        "value": ans.value,
        "confidence": round(ans.confidence, 4),
        "probs": {k: round(v, 4) for k, v in ans.probs.items()},
    }


def do_rate(client: Client, text: str, levels: list[str]) -> dict:
    ans = client.ask(text, {"q": Score(levels=levels)})["q"]
    return {
        "value": ans.value,
        "score": round(ans.score, 4),
        "confidence": round(ans.confidence, 4),
    }


def do_judge(client: Client, text: str, statement: str) -> dict:
    ans = client.ask(text, {"q": Noul(statement)})["q"]
    return {
        "value": bool(ans.value),
        "prob_true": round(ans.prob, 4),
        "confidence": round(ans.confidence, 4),
    }


def do_gate(client: Client, action: str, checks: dict | None = None) -> dict:
    """Agent guardrail: run concrete risk checks over an action and return a
    block/allow verdict with reasons. Concrete questions on purpose (see README):
    the local model judges "moves money" well and "is dangerous" poorly."""
    checks = checks or DEFAULT_GATE_CHECKS
    questions = {name: Noul(stmt) for name, stmt in checks.items()}
    answers = client.ask(action, questions)
    triggered = {name: round(a.prob, 4) for name, a in answers.items() if a.value}
    return {
        "block": bool(triggered),
        "verdict": "block, escalate to human" if triggered else "allow",
        "triggered": triggered,
        "details": {name: {"value": bool(a.value), "prob_true": round(a.prob, 4)}
                    for name, a in answers.items()},
    }


def do_decide(client: Client, text: str, questions: dict) -> dict:
    """Several typed questions at once. `questions` maps name -> spec, where spec
    is {"type": "choice"|"score"|"noul", ...}. One model pass for all of them."""
    from .evaluate import _build_primitive
    prims = {name: _build_primitive(spec) for name, spec in questions.items()}
    answers = client.ask(text, prims)
    out = {}
    for name, ans in answers.items():
        entry = {"value": ans.value, "confidence": round(ans.confidence, 4)}
        if hasattr(ans, "prob"):
            entry["prob_true"] = round(ans.prob, 4)
        out[name] = entry
    return out


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #

def _make_app(name: str):
    """Return an MCP server app across SDK versions.

    mcp 2.x renamed FastMCP -> MCPServer; both expose the same .tool() decorator
    and .run(transport='stdio'). Support whichever is installed.
    """
    try:  # mcp 2.x
        from mcp.server.mcpserver import MCPServer
        return MCPServer(name)
    except ImportError:
        pass
    try:  # mcp 1.x
        from mcp.server.fastmcp import FastMCP
        return FastMCP(name)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The MCP server needs the 'mcp' extra: pip install 'poorjev[local,mcp]'"
        ) from e


def build_server(calibrator_path: str | None = "calibration.json"):
    temperature = load_temperature(calibrator_path)
    client = Client(temperature=temperature)  # model loads lazily on first call
    server = _make_app("poorjev")

    @server.tool()
    def gate(action: str) -> dict:
        """Guardrail for an agent tool call. Given an action description, return
        whether it should be blocked and escalated to a human because it moves
        money or deletes data. Fast, local, no API key."""
        return do_gate(client, action)

    @server.tool()
    def judge(text: str, statement: str) -> dict:
        """Answer a yes/no question about text with a calibrated probability.
        `statement` is a claim; returns value (bool), prob_true, confidence."""
        return do_judge(client, text, statement)

    @server.tool()
    def classify(text: str, options: list[str]) -> dict:
        """Classify text into exactly one of `options`. Returns the chosen option
        (always one of yours), calibrated confidence, and per-option probs."""
        return do_classify(client, text, options)

    @server.tool()
    def rate(text: str, levels: list[str]) -> dict:
        """Rate text on an ordered scale (e.g. low, medium, high). Returns the
        winning level plus a continuous score and calibrated confidence."""
        return do_rate(client, text, levels)

    @server.tool()
    def decide(text: str, questions: dict) -> dict:
        """Answer several typed questions about text in one pass. `questions`
        maps a name to a spec: {"type":"choice","options":[...]} or
        {"type":"score","levels":[...]} or {"type":"noul","statement":"..."}."""
        return do_decide(client, text, questions)

    return server


def main(calibrator_path: str | None = "calibration.json") -> None:
    build_server(calibrator_path).run()  # stdio transport


if __name__ == "__main__":
    main()
