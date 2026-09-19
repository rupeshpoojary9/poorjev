"""Route a support ticket with all three primitives in one pass.

Run it (after `pip install 'poorjev[local]'`):

    python examples/ticket_router.py

First run downloads a ~400MB model once, then it is fully offline and keyless.
"""

from poorjev import Client, Choice, Score, Noul


def main() -> None:
    client = Client()  # local NLI backend, no key

    ticket = (
        "I've emailed three times and I'm STILL being double-charged every month. "
        "This is ridiculous. Cancel my account today or I'm disputing with my bank."
    )

    result = client.ask(
        state=ticket,
        questions={
            "topic": Choice(["billing", "technical", "account", "sales", "other"]),
            "frustration": Score(levels=["low", "medium", "high"]),
            "is_urgent": Noul("The customer needs a response today."),
            "wants_cancel": Noul("The customer is asking to cancel their account."),
        },
    )

    print("Ticket:", ticket, "\n")
    print(f"  topic         : {result['topic'].value:<10} (conf {result['topic'].confidence:.2f})")
    print(f"  frustration   : {result['frustration'].value:<10} "
          f"(score {result['frustration'].score:.2f}, conf {result['frustration'].confidence:.2f})")
    print(f"  is_urgent     : {result['is_urgent'].value}  (P {result['is_urgent'].prob:.2f})")
    print(f"  wants_cancel  : {result['wants_cancel'].value}  (P {result['wants_cancel'].prob:.2f})")


if __name__ == "__main__":
    main()
