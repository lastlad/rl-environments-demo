"""PREREQUISITE: the refund policy and the answer key.

The policy has 5 rules. Apply them in this sequence:
  1. customer is flagged for fraud        -> escalate to a human
  2. delivered more than 30 days ago      -> deny
  3. amount over $100                     -> escalate to a human
  4. the item was opened                  -> refund half the amount
  5. otherwise                            -> refund the full amount

`expected_case` gives the correct answer for a ticket. The reward functions
use it to check the agent. In this demo the answer is exact, because the rules
are exact. In a real project, the answer key is usually the most difficult
part to make: unit tests, a reference final state, a rubric or human labels.
"""
from __future__ import annotations

from dataclasses import dataclass

REFUND_WINDOW_DAYS = 30
AUTO_REFUND_LIMIT_CENTS = 100_00  # money is integer cents: float equality breaks graders
RESOLUTIONS = ("refunded", "denied", "escalated")
# The five kinds of ticket. The part before "_" is the right reply.
CASES = ("refunded_full", "refunded_half", "escalated_over_limit", "escalated_flagged", "denied")


@dataclass(frozen=True)
class Rules:
    """The two numbers in the policy. A change to them changes the correct
    answer to many tickets (step 8)."""
    window_days: int = REFUND_WINDOW_DAYS    # rule 2: deny if delivered more than this many days ago
    limit_cents: int = AUTO_REFUND_LIMIT_CENTS  # rule 3: escalate if the amount is over this


RULES = Rules()    # the normal rules: every trainee practised on these
CHANGED_RULES = Rules(window_days=14, limit_cents=50_00)   # step 8's exam


def expected_case(flagged: bool, days_since_delivery: int, amount_cents: int,
                  opened: bool, rules: Rules = RULES) -> str:
    """The policy, written once. The task generator is tested against it."""
    if flagged:
        return "escalated_flagged"
    if days_since_delivery > rules.window_days:
        return "denied"
    if amount_cents > rules.limit_cents:
        return "escalated_over_limit"
    return "refunded_half" if opened else "refunded_full"


def refund_due_cents(order: dict) -> int:
    """How much a refundable order should get back: half (rounded down) if opened."""
    return order["amount_cents"] // 2 if order["opened"] else order["amount_cents"]
