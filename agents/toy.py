"""Policies: anything with reset() and act(obs) -> action.

A policy sees observations only. It never touches env.db, env.task or `info`.
"""
from __future__ import annotations

import random

from refund_env.policy import RULES, Rules

MACROS = ("list", "lookup", "check", "refund_full", "refund_half", "escalate",
          "reply_refunded", "reply_denied", "reply_escalated")


class Memory:
    """What the agent has learned so far this episode, built from observations."""

    def __init__(self, rules: Rules = RULES):
        self.rules = rules

    def reset(self):
        self.order_id = None   # found by list_orders: the order whose item the customer named
        self.item = None
        self.order = None      # found by lookup_order
        self.flagged = None    # found by check_customer
        self.refunded = False
        self.escalated = False

    def update(self, obs):
        r = obs["last_result"]
        if not r or not r["ok"]:
            return
        if r["tool"] == "list_orders":
            message = obs["ticket"]["message"]
            named = [o for o in r["orders"] if o["item"] in message]
            if named:
                self.order_id, self.item = named[0]["order_id"], named[0]["item"]
        elif r["tool"] == "lookup_order":
            self.order = r["order"]
        elif r["tool"] == "check_customer":
            self.flagged = r["flagged"]
        elif r["tool"] == "issue_refund":
            self.refunded = True
        elif r["tool"] == "escalate":
            self.escalated = True

    def state(self):
        """Compact state for the tabular policy. An LLM would read the raw
        observation and the policy text instead of these hand-made features."""
        looked = None
        if self.order is not None:
            looked = (self.order["days_since_delivery"] <= self.rules.window_days,
                      self.order["amount_cents"] > self.rules.limit_cents,
                      self.order["opened"])
        return (self.order_id is not None, looked, self.flagged, self.refunded, self.escalated)


def macro_to_action(macro: str, obs: dict, mem: Memory) -> dict:
    customer, oid = obs["ticket"]["customer"], mem.order_id
    if macro == "list":
        return {"tool": "list_orders", "args": {"customer": customer}}
    if macro == "lookup":
        # Looking up before listing means the order number is unknown -> the
        # env rejects it. The policy has to learn to search first.
        return {"tool": "lookup_order", "args": {"order_id": oid}}
    if macro == "check":
        return {"tool": "check_customer", "args": {"customer": customer}}
    if macro.startswith("refund"):
        amount = mem.order["amount_cents"] if mem.order else None
        if amount is not None and macro == "refund_half":
            amount //= 2
        return {"tool": "issue_refund", "args": {"order_id": oid, "amount_cents": amount}}
    if macro == "escalate":
        return {"tool": "escalate", "args": {"order_id": oid}}
    return {"tool": "reply", "args": {"resolution": macro.removeprefix("reply_"), "message": ""}}


class OraclePolicy:
    """Hand-written correct behaviour. If this can't score ~100%, the
    environment or the reward is broken, not the learner.

    After each act(), self.thought holds the expert's reasoning for that step,
    in plain words. Step 5 uses it to teach a model to think, not just to copy.
    """

    def __init__(self, rules: Rules = RULES):
        self.mem = Memory(rules)
        self.thought = ""

    def reset(self):
        self.mem.reset()
        self.thought = ""

    def act(self, obs):
        self.mem.update(obs)
        found, looked, flagged, refunded, escalated = self.mem.state()
        if not found:
            macro = "list"
        elif looked is None:
            macro = "lookup"
        elif flagged is None:
            macro = "check"
        else:
            in_window, over_limit, opened = looked
            if flagged or (in_window and over_limit):
                macro = "reply_escalated" if escalated else "escalate"
            elif not in_window:
                macro = "reply_denied"
            elif refunded:
                macro = "reply_refunded"
            else:
                macro = "refund_half" if opened else "refund_full"
        self.thought = self.explain(macro, obs)
        return macro_to_action(macro, obs, self.mem)

    def rules(self):
        """Walk through the policy for this order, one rule at a time."""
        o, flagged = self.mem.order, self.mem.flagged
        window, limit = self.mem.rules.window_days, self.mem.rules.limit_cents
        days, amount = o["days_since_delivery"], o["amount_cents"]
        if flagged:
            return "Rule 1: the customer is flagged for fraud, so I escalate to a human."
        lines = ["Rule 1: the customer is not flagged."]
        if days > window:
            lines.append(f"Rule 2: delivered {days} days ago, more than {window}, so I deny.")
            return " ".join(lines)
        lines.append(f"Rule 2: delivered {days} days ago, within {window}.")
        if amount > limit:
            lines.append(f"Rule 3: amount_cents {amount} is over {limit}, "
                         "so I escalate to a human.")
            return " ".join(lines)
        lines.append(f"Rule 3: amount_cents {amount} is not over {limit}.")
        if o["opened"]:
            lines.append(f"Rule 4: the item was opened, so I refund half: "
                         f"{amount} // 2 = {amount // 2} cents.")
        else:
            lines.append(f"Rule 4: the item was not opened. Rule 5: I refund the full {amount} cents.")
        return " ".join(lines)

    def explain(self, macro, obs):
        m, customer = self.mem, obs["ticket"]["customer"]
        if macro == "list":
            return (f"Customer {customer} names an item, not an order number. "
                    "First I list their orders to find the right one.")
        if macro == "lookup":
            return (f"The customer's {m.item} is order {m.order_id}. "
                    "I look it up to see the delivery date, the amount and whether it was opened.")
        if macro == "check":
            o = m.order
            return (f"Order {m.order_id}: delivered {o['days_since_delivery']} days ago, "
                    f"amount_cents {o['amount_cents']}, opened {str(o['opened']).lower()}. "
                    f"Rule 1 comes first, so I check whether customer {customer} is flagged for fraud.")
        if macro == "escalate":
            return f"{self.rules()} I press escalate on order {m.order_id}."
        if macro.startswith("refund"):
            return f"{self.rules()} I issue the refund on order {m.order_id}."
        if macro == "reply_denied":
            return f"{self.rules()} A denial needs no other action, so I reply now."
        if macro == "reply_escalated":
            return f"Order {m.order_id} has been escalated. Now I tell the customer."
        return f"The refund on order {m.order_id} went through. Now I tell the customer."


class RandomPolicy:
    """Floor baseline. If this scores well, the task is too easy or the reward leaks."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.mem = Memory()

    def reset(self):
        self.mem.reset()

    def act(self, obs):
        self.mem.update(obs)
        return macro_to_action(self.rng.choice(MACROS), obs, self.mem)


class TabularPolicy:
    """Tabular Q-learning with epsilon-greedy exploration.

    Stands in for the model being fine-tuned: small enough to train in seconds,
    but it chases whatever the reward function pays for, like a big one does.
    """

    def __init__(self, seed=0, explore=0.2):
        self.rng = random.Random(seed)
        self.q = {}
        self.mem = Memory()
        self.greedy = False
        self.explore = explore  # chance of a random action while training
        self.trace = []

    def reset(self):
        self.mem.reset()
        self.trace = []

    def values(self, state):
        return self.q.setdefault(state, [0.0] * len(MACROS))

    def act(self, obs):
        self.mem.update(obs)
        state = self.mem.state()
        if not self.greedy and self.rng.random() < self.explore:
            a = self.rng.randrange(len(MACROS))
        else:
            v = self.values(state)
            a = max(range(len(MACROS)), key=v.__getitem__)
        self.trace.append((state, a))
        return macro_to_action(MACROS[a], obs, self.mem)

    def update(self, reward, lr):
        """Reward arrives only at the end of the episode, so walk backwards:
        the last action is worth the reward, and each earlier action is worth
        the best value available from the state it led to."""
        target = reward
        for state, a in reversed(self.trace):
            v = self.values(state)
            v[a] += lr * (target - v[a])
            target = max(v)
