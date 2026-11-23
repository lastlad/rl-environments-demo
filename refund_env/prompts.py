"""The text interface for a language model.

A scripted agent reads the observation dict directly. A language model needs
the rules and each observation as text. Its text output must then change back
into an action.
"""
from __future__ import annotations

import json

from .policy import RULES, Rules


def system_prompt(rules: Rules = RULES) -> str:
    """The rulebook pinned to the wall: what every model reads first."""
    window, limit = rules.window_days, rules.limit_cents
    return f"""You are a refund desk agent. Customers name the item, not the
order number: find their order with list_orders, then look it up.

Policy, checked in this order:
1. Customer flagged for fraud (see check_customer): escalate to a human.
2. Delivered more than {window} days ago: deny.
3. amount_cents over {limit} (${limit // 100}): escalate to a human.
4. The item was opened: refund half the amount, rounded down to whole cents.
5. Otherwise: refund the full amount.
All amounts are in cents: amount_cents {limit} means ${limit // 100}.
Customers can be wrong about dates and about whether they opened the item.
Check the records.

Answer with exactly one JSON tool call per turn:
{{"tool": "list_orders", "args": {{"customer": "..."}}}}
{{"tool": "lookup_order", "args": {{"order_id": "..."}}}}
{{"tool": "check_customer", "args": {{"customer": "..."}}}}
{{"tool": "issue_refund", "args": {{"order_id": "...", "amount_cents": 1234}}}}
{{"tool": "escalate", "args": {{"order_id": "..."}}}}
{{"tool": "reply", "args": {{"resolution": "refunded|denied|escalated", "message": "..."}}}}"""


SYSTEM_PROMPT = system_prompt(RULES)


def render_prompt(obs: dict) -> str:
    if obs["last_result"] is None:
        t = obs["ticket"]
        return f"Ticket from customer {t['customer']}: {t['message']}\n({obs['steps_left']} steps left)"
    return f"Tool result: {json.dumps(obs['last_result'])}\n({obs['steps_left']} steps left)"


def parse_action(text: str):
    """Model output -> action. Garbage is passed through so the env can
    penalise it as an invalid action instead of the harness crashing."""
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text
