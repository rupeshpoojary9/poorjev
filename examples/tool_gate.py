"""Agent guardrail: gate a tool call before it runs, the honest way.

A lesson baked into this example: a local NLI model is strong at CONCRETE
entailment ("this moves money", "this deletes data") and weak at ABSTRACT value
judgements ("this is dangerous", "this needs approval"). So do not ask it the
abstract question. Ask it the concrete ones it answers well, then let a trivial
policy rule decide, that is the System One (fast perception) / System Two (the
rule, or a human) split. When perception is not confident, abstain and escalate.

    python examples/tool_gate.py
"""

from poorjev import Client, Noul

CANDIDATE_ACTIONS = [
    "Run `SELECT name, email FROM users LIMIT 10`.",
    "Run `DROP TABLE users;`.",
    "Send a $5 refund to the customer for order #4471.",
    "Wire $250,000 to a new external bank account.",
    "Reply to the customer thanking them for their patience.",
]


def main() -> None:
    client = Client()

    # Concrete perception questions the fast local model handles well.
    questions = {
        "moves_money": Noul("This action moves, sends, or refunds money."),
        "deletes_data": Noul("This action deletes or destroys data."),
    }

    print("Tool-call gate (local, keyless). Policy: money or data deletion -> human.\n")
    for action in CANDIDATE_ACTIONS:
        res = client.ask(state=action, questions=questions)
        money = res["moves_money"]
        data = res["deletes_data"]

        # System Two: a one-line policy over the perceptions.
        block = money.value or data.value
        verdict = "BLOCK -> human" if block else "allow"

        reasons = []
        if money.value:
            reasons.append(f"money P={money.prob:.2f}")
        if data.value:
            reasons.append(f"data P={data.prob:.2f}")
        why = ("  (" + ", ".join(reasons) + ")") if reasons else ""
        print(f"  {verdict:<16}{why:<28}|  {action}")


if __name__ == "__main__":
    main()
