"""Live demo on fresh inputs (not in the eval set). Raw vs calibrated."""

import json
from poorjev import Client, Choice, Score, Noul

T = json.load(open("calibration.json"))["temperature"]
raw = Client(temperature=1.0)
cal = Client(temperature=T)

print("=" * 74)
print("1) Full triage on a brand-new ticket")
print("=" * 74)
ticket = ("Hi, my card was charged $240 but the order page still says 'pending'. "
          "I need this sorted before my trip tomorrow morning.")
q = {
    "topic": Choice(["billing", "technical", "account", "shipping", "other"]),
    "frustration": Score(levels=["low", "medium", "high"]),
    "is_urgent": Noul("The customer needs a response today."),
    "wants_cancel": Noul("The customer wants to cancel their account."),
}
r = cal.ask(ticket, q)
print(f"  ticket: {ticket}\n")
print(f"  topic        = {r['topic'].value:<10} conf {r['topic'].confidence:.2f}")
print(f"  frustration  = {r['frustration'].value:<10} conf {r['frustration'].confidence:.2f}")
print(f"  is_urgent    = {str(r['is_urgent'].value):<10} P {r['is_urgent'].prob:.2f}")
print(f"  wants_cancel = {str(r['wants_cancel'].value):<10} P {r['wants_cancel'].prob:.2f}")

print("\n" + "=" * 74)
print("2) Raw vs calibrated confidence (same answer, honest number)")
print("=" * 74)
cases = [
    ("Your app deleted all my saved drafts and I am LIVID.",
     Noul("The customer is angry.")),
    ("Could you point me to the API docs?",
     Noul("The customer needs a response today.")),
    ("I think I maybe want to keep my plan, not totally sure.",
     Noul("The customer wants to cancel their account.")),
]
print(f"  {'statement':<42}{'raw':>8}{'calibrated':>12}")
for state, noul in cases:
    a0 = raw.ask(state, {"q": noul})["q"]
    a1 = cal.ask(state, {"q": noul})["q"]
    label = f"{noul.statement[:30]} -> {a1.value}"
    print(f"  {label:<42}{a0.confidence:>8.2f}{a1.confidence:>12.2f}")

print("\n" + "=" * 74)
print("3) Abstention: when it is not sure, it says so")
print("=" * 74)
threshold = json.load(open("calibration.json"))["abstain_threshold"]
ambiguous = [
    "The product is fine I guess.",
    "This is absolutely the best tool I have ever used, hands down.",
    "meh.",
]
sent = Choice(["positive", "negative", "neutral"])
for state in ambiguous:
    a = cal.ask(state, {"sentiment": sent})["sentiment"]
    verdict = a.value if a.confidence >= threshold else f"ABSTAIN (would guess {a.value})"
    print(f"  conf {a.confidence:.2f}  {verdict:<34} <- {state}")
print(f"\n  (abstain threshold from calibration.json = {threshold:.2f})")
