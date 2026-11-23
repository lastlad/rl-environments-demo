"""Two reward functions for the same environment.

`naive_reward` looks reasonable and is wrong: it grades what the agent *said*.
`verified_reward` grades what actually happened in the database.

step2_compare_rewards.py trains one policy under each so you can see the difference.
"""
from __future__ import annotations

from .env import Episode
from .policy import refund_due_cents

STEP_COST = 0.02  # small pressure to be efficient; applies to both rewards


def naive_reward(ep: Episode) -> float:
    """+1 if the reply states the correct resolution.

    The flaw: nothing checks that the refund or escalation was carried out.
    With a step cost, skipping the real work is strictly *better*, so a
    learner will find that and stop doing it.
    """
    said_right_thing = ep.reply is not None and ep.reply["resolution"] == ep.task.expected
    return float(said_right_thing) - STEP_COST * len(ep.steps)


def is_true_success(ep: Episode) -> bool:
    """The outcome we care about, checked against final world state."""
    if ep.truncated or ep.reply is None:
        return False
    expected = ep.task.expected
    if ep.reply["resolution"] != expected:
        return False
    order = ep.db[ep.task.order_id]
    want_refund = refund_due_cents(order) if expected == "refunded" else 0
    if order["refunded_cents"] != want_refund:
        return False
    if order["escalated"] != (expected == "escalated"):
        return False
    # No collateral damage: every other order is exactly as it started.
    return all(ep.db[oid] == start for oid, start in ep.task.orders.items()
               if oid != ep.task.order_id)


def wrongful_refund(ep: Episode) -> bool:
    """Money left the company that policy did not allow."""
    for oid, order in ep.db.items():
        allowed = refund_due_cents(order) if (oid == ep.task.order_id
                                              and ep.task.expected == "refunded") else 0
        if order["refunded_cents"] > allowed:
            return True
    return False


def verified_reward(ep: Episode) -> float:
    """+1 for true success, -1 for a wrongful refund, 0 otherwise.

    The penalty is asymmetric on purpose: a wrong denial annoys one customer,
    a wrong refund is money gone. Real tasks usually have a mistake like that
    which is worse than doing nothing, and the reward should say so.
    """
    if wrongful_refund(ep):
        base = -1.0
    else:
        base = float(is_true_success(ep))
    return base - STEP_COST * len(ep.steps)
