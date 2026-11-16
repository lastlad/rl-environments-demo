"""Policies: anything with reset() and act(obs) -> action.

A policy sees observations only. It never touches env.db, env.task or `info`.
"""
from __future__ import annotations

import random

from refund_env.env import AUTO_REFUND_LIMIT_CENTS, REFUND_WINDOW_DAYS

MACROS = ("lookup", "refund", "escalate", "reply_refunded", "reply_denied", "reply_escalated")


class Memory:
    """What the agent has learned so far this episode, built from observations."""

    def reset(self):
        self.order = None
        self.refunded = False
        self.escalated = False

    def update(self, obs):
        r = obs["last_result"]
        if not r or not r["ok"]:
            return
        if r["tool"] == "lookup_order":
            self.order = r["order"]
        elif r["tool"] == "issue_refund":
            self.refunded = True
        elif r["tool"] == "escalate":
            self.escalated = True

    def state(self):
        """Compact state for the tabular policy. An LLM would read the raw
        observation and the policy text instead of these hand-made features."""
        looked = None
        if self.order is not None:
            looked = (self.order["days_since_delivery"] <= REFUND_WINDOW_DAYS,
                      self.order["amount_cents"] > AUTO_REFUND_LIMIT_CENTS)
        return (looked, self.refunded, self.escalated)


def macro_to_action(macro: str, obs: dict, mem: Memory) -> dict:
    oid = obs["ticket"]["order_id"]
    if macro == "lookup":
        return {"tool": "lookup_order", "args": {"order_id": oid}}
    if macro == "refund":
        # Refunding before looking up means the amount is unknown -> the env
        # rejects it as an invalid action. The policy has to learn the order.
        amount = mem.order["amount_cents"] if mem.order else None
        return {"tool": "issue_refund", "args": {"order_id": oid, "amount_cents": amount}}
    if macro == "escalate":
        return {"tool": "escalate", "args": {"order_id": oid}}
    return {"tool": "reply", "args": {"resolution": macro.removeprefix("reply_"), "message": ""}}


class OraclePolicy:
    """Hand-written correct behaviour. If this can't score ~100%, the
    environment or the reward is broken, not the learner."""

    def __init__(self):
        self.mem = Memory()

    def reset(self):
        self.mem.reset()

    def act(self, obs):
        self.mem.update(obs)
        looked, refunded, escalated = self.mem.state()
        if looked is None:
            macro = "lookup"
        elif not looked[0]:
            macro = "reply_denied"
        elif looked[1]:
            macro = "reply_escalated" if escalated else "escalate"
        else:
            macro = "reply_refunded" if refunded else "refund"
        return macro_to_action(macro, obs, self.mem)


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
