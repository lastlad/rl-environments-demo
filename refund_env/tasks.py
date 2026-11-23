"""PREREQUISITE: the tasks (the tickets and their order data).

This demo makes synthetic tickets with code. One seed number gives one ticket,
so the supply of tickets has no limit and costs nothing. In a real project,
tasks usually come from real data, for example old support tickets with the
private data removed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .policy import AUTO_REFUND_LIMIT_CENTS, CASES, REFUND_WINDOW_DAYS, RULES, Rules, expected_case

ITEMS = ("headphones", "kettle", "backpack", "desk lamp", "keyboard")
CLAIMS = (
    "It only arrived last week.",
    "I've had it for a while now.",
    "It's well within your return window.",
    "I never even opened it.",
    "",
)


@dataclass(frozen=True)
class Task:
    task_id: str
    customer: str
    order_id: str
    message: str
    orders: dict    # initial database; the env works on a copy
    flagged: bool   # is the customer flagged for fraud?
    case: str       # ground truth, one of CASES. Never shown to the agent.
    expected: str   # the right reply, one of RESOLUTIONS. Never shown either.


def make_task(seed: int, rules: Rules = RULES) -> Task:
    """Same seed -> same task, always. Uses its own RNG, never the global one.

    The customer, the orders and the message are always drawn the same way, so
    a seed gives the same ticket under any rules. Only the right answer
    (`case`, `expected`) depends on the rules.
    """
    rng = random.Random(seed)
    case = rng.choice(CASES)
    flagged = case == "escalated_flagged"
    opened = rng.random() < 0.5
    # randint is inclusive, so the boundaries (day 30, exactly $100) do occur.
    if case == "denied":
        days, amount = rng.randint(REFUND_WINDOW_DAYS + 1, 90), rng.randint(5_00, 300_00)
    elif case == "escalated_flagged":
        # Any order at all: a flagged customer is escalated even when the
        # order would otherwise be refunded or denied.
        days, amount = rng.randint(1, 90), rng.randint(5_00, 300_00)
    elif case == "escalated_over_limit":
        days, amount = rng.randint(1, REFUND_WINDOW_DAYS), rng.randint(AUTO_REFUND_LIMIT_CENTS + 1, 300_00)
    else:
        days, amount = rng.randint(1, REFUND_WINDOW_DAYS), rng.randint(5_00, AUTO_REFUND_LIMIT_CENTS)
        opened = case == "refunded_half"

    customer = f"C{rng.randint(100, 999)}"
    items = rng.sample(ITEMS, 3)
    ids = [f"A{n}" for n in rng.sample(range(1000, 9999), 3)]

    def order(item, amount_cents, days_since_delivery, opened):
        return {
            "customer": customer,
            "item": item,
            "amount_cents": amount_cents,
            "days_since_delivery": days_since_delivery,
            "opened": opened,
            "refunded_cents": 0,
            "escalated": False,
        }

    orders = {ids[0]: order(items[0], amount, days, opened)}
    # Two more orders on the same account. They are refundable, so an agent
    # that acts on the wrong order moves real (simulated) money.
    for oid, item in zip(ids[1:], items[1:]):
        orders[oid] = order(item, rng.randint(5_00, 100_00), rng.randint(1, 30), rng.random() < 0.5)
    orders = dict(sorted(orders.items()))  # so the right order is not always listed first
    # What the customer says is not correlated with the truth. The agent has to
    # check the records instead of trusting the ticket.
    claim = rng.choice(CLAIMS)
    message = f"I want a refund for my {items[0]}. {claim}".strip()
    real = orders[ids[0]]
    case = expected_case(flagged, real["days_since_delivery"], real["amount_cents"],
                         real["opened"], rules)
    expected = case.split("_")[0]
    return Task(f"refund-{seed}", customer, ids[0], message, orders, flagged, case, expected)
