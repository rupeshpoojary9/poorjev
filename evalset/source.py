"""Hand-labelled evaluation set, the source of truth.

Every item below is a realistic decision-task input with gold answers assigned
by hand. Running this file regenerates ``tasks.jsonl``:

    python evalset/source.py

Honesty notes (see evalset/README.md): this set is small (tens of items, a few
hundred individual decisions). It is enough to estimate accuracy and calibration
and to show the direction M4 moves them; it is not a large benchmark and we do
not claim it is. Labels favour concrete, checkable questions, because that is
what the local backend can be fairly asked (see the M2 finding in the README).
A few items are deliberately ambiguous so the reliability curve is not trivial.
"""

from __future__ import annotations

import json
import os

# Shared question specs -------------------------------------------------------
TOPIC = ["billing", "technical", "account", "shipping", "other"]
INTENT = ["question", "complaint", "praise", "request"]
FRUSTRATION = ["low", "medium", "high"]

Q_URGENT = "The customer needs a response today."
Q_CANCEL = "The customer wants to cancel their account or subscription."
Q_MONEY = "This action moves, sends, or refunds money."
Q_DELETE = "This action deletes or destroys data."
Q_READONLY = "This action only reads data and changes nothing."


def support(state, topic, urgent, cancel, frustration):
    return {
        "task": "support_triage",
        "state": state,
        "questions": {
            "topic": {"type": "choice", "options": TOPIC, "gold": topic},
            "frustration": {"type": "score", "levels": FRUSTRATION, "gold": frustration},
            "is_urgent": {"type": "noul", "statement": Q_URGENT, "gold": urgent},
            "wants_cancel": {"type": "noul", "statement": Q_CANCEL, "gold": cancel},
        },
    }


def action(state, money, delete, readonly):
    return {
        "task": "action_risk",
        "state": state,
        "questions": {
            "moves_money": {"type": "noul", "statement": Q_MONEY, "gold": money},
            "deletes_data": {"type": "noul", "statement": Q_DELETE, "gold": delete},
            "is_read_only": {"type": "noul", "statement": Q_READONLY, "gold": readonly},
        },
    }


def intent(state, gold):
    return {
        "task": "intent",
        "state": state,
        "questions": {
            "intent": {"type": "choice", "options": INTENT, "gold": gold},
        },
    }


ITEMS = [
    # ---- support_triage ---------------------------------------------------
    support("I was charged twice for my subscription this month. Please refund the extra charge.",
            "billing", urgent=False, cancel=False, frustration="medium"),
    support("The app crashes every time I open the reports tab. I'm on the latest version.",
            "technical", urgent=False, cancel=False, frustration="medium"),
    support("I've emailed three times and I'm STILL being double-charged. Cancel my account today.",
            "billing", urgent=True, cancel=True, frustration="high"),
    support("Where is my order? It said delivered but nothing arrived. I need it before Friday.",
            "shipping", urgent=True, cancel=False, frustration="high"),
    support("Can you help me reset my password? I can't log in.",
            "account", urgent=False, cancel=False, frustration="low"),
    support("Loving the new dashboard, just wondering how to export to CSV?",
            "technical", urgent=False, cancel=False, frustration="low"),
    support("Please close my account. I no longer need the service. No rush.",
            "account", urgent=False, cancel=True, frustration="low"),
    support("Your last invoice has the wrong VAT number, we need a corrected one for our records.",
            "billing", urgent=False, cancel=False, frustration="low"),
    support("URGENT: production is down for all our users and it's pointing at your API. Call me now.",
            "technical", urgent=True, cancel=False, frustration="high"),
    support("My package arrived damaged. I'd like a replacement shipped out.",
            "shipping", urgent=False, cancel=False, frustration="medium"),
    support("I want to upgrade to the annual plan but the checkout keeps failing.",
            "billing", urgent=False, cancel=False, frustration="medium"),
    support("Hi, is there a student discount available?",
            "billing", urgent=False, cancel=False, frustration="low"),
    support("This is the third outage this week. If it happens again we are cancelling.",
            "technical", urgent=False, cancel=False, frustration="high"),
    support("I need to change the email address on my account.",
            "account", urgent=False, cancel=False, frustration="low"),
    support("Cancel everything and delete my data. I'm switching to a competitor today.",
            "account", urgent=True, cancel=True, frustration="high"),
    support("The tracking link you sent goes to a 404 page.",
            "shipping", urgent=False, cancel=False, frustration="medium"),
    support("Just wanted to say your support team was fantastic yesterday, thank you!",
            "other", urgent=False, cancel=False, frustration="low"),
    support("I was promised a refund two weeks ago and still haven't received it. This is unacceptable.",
            "billing", urgent=True, cancel=False, frustration="high"),
    support("How do I add a teammate to my workspace?",
            "account", urgent=False, cancel=False, frustration="low"),
    support("The mobile app won't sync with the web version anymore.",
            "technical", urgent=False, cancel=False, frustration="medium"),
    support("Do you ship to Canada, and how long does it take?",
            "shipping", urgent=False, cancel=False, frustration="low"),
    support("I've decided to keep my subscription after all, please ignore my earlier email.",
            "account", urgent=False, cancel=False, frustration="low"),
    support("There's a typo on my invoice but honestly it's fine, no need to fix it.",
            "billing", urgent=False, cancel=False, frustration="low"),
    support("Someone logged into my account from a device I don't recognise. Help.",
            "account", urgent=True, cancel=False, frustration="high"),
    support("Quick one: what's your refund policy window?",
            "billing", urgent=False, cancel=False, frustration="low"),

    # ---- action_risk ------------------------------------------------------
    action("Run `SELECT name, email FROM users LIMIT 10`.", money=False, delete=False, readonly=True),
    action("Run `DROP TABLE users;`.", money=False, delete=True, readonly=False),
    action("Send a $5 refund to the customer for order #4471.", money=True, delete=False, readonly=False),
    action("Wire $250,000 to a new external bank account.", money=True, delete=False, readonly=False),
    action("Reply to the customer thanking them for their patience.", money=False, delete=False, readonly=False),
    action("Delete all rows from the `sessions` table older than 30 days.", money=False, delete=True, readonly=False),
    action("Fetch the current weather for San Francisco.", money=False, delete=False, readonly=True),
    action("Issue a full refund of $1,299 to card ending 4242.", money=True, delete=False, readonly=False),
    action("Read the last 20 lines of the application log.", money=False, delete=False, readonly=True),
    action("Charge the customer's saved card $49 for the monthly plan.", money=True, delete=False, readonly=False),
    action("Remove the user's uploaded files from cloud storage permanently.", money=False, delete=True, readonly=False),
    action("List the open pull requests in the repository.", money=False, delete=False, readonly=True),
    action("Transfer 0.5 BTC to the address in the customer's message.", money=True, delete=False, readonly=False),
    action("Show the customer's current subscription tier.", money=False, delete=False, readonly=True),
    action("Truncate the `audit_log` table.", money=False, delete=True, readonly=False),

    # ---- intent -----------------------------------------------------------
    intent("How much does the premium plan cost?", "question"),
    intent("Your product is amazing, it saved my team hours every week.", "praise"),
    intent("This is the worst onboarding experience I've ever had.", "complaint"),
    intent("Please add a dark mode to the mobile app.", "request"),
    intent("What time zone are your support hours in?", "question"),
    intent("I've been on hold for 45 minutes and nobody has helped me.", "complaint"),
    intent("Could you send me the API documentation link?", "request"),
    intent("Honestly the best customer service I've dealt with all year.", "praise"),
    intent("Why did my invoice go up this month?", "question"),
    intent("Can you enable two-factor authentication on my account?", "request"),
    intent("The new update broke everything and I'm furious.", "complaint"),
    intent("Do you integrate with Salesforce?", "question"),
    intent("Thanks so much, the fix worked perfectly!", "praise"),
    intent("I'd like a callback from a human agent, not a bot.", "request"),
    intent("Your billing page is confusing and I hate using it.", "complaint"),
]


def build(path: str | None = None) -> str:
    path = path or os.path.join(os.path.dirname(__file__), "tasks.jsonl")
    with open(path, "w") as f:
        for i, item in enumerate(ITEMS, start=1):
            row = {"id": f"{item['task']}_{i:03d}", **item}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    p = build()
    n_items = len(ITEMS)
    n_decisions = sum(len(it["questions"]) for it in ITEMS)
    print(f"wrote {n_items} items ({n_decisions} labelled decisions) -> {p}")
