"""THE ENVIRONMENT: the order database, the tools and the episode.

One episode = one support ticket. The agent calls tools against a simulated
order database, then ends the episode with a `reply`.

An RL environment needs some parts that the environment does not make itself 
and are part of prerequisities below.

  PREREQUISITES (made for this demo; in a real project they come from real data)
    tasks            tasks.py     make_task(seed): the tickets and their orders
    policy + answer  policy.py    the 5 rules and expected_case(): the correct answer

  THE ENVIRONMENT (this file)
    state            env.db       the simulated world, mutated by tools
    observation      env._obs()   the part of the world the agent may see
    actions          env._execute()  tool calls: {"tool": ..., "args": {...}}
    termination      env.step()   `reply` ends it; the step limit truncates it

  AROUND THE ENVIRONMENT
    reward           rewards.py   scores the finished episode
    episode loop     runner.py    connects one agent to the environment
    text interface   prompts.py   the rules and observations as text, for a language model
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

from .policy import RESOLUTIONS, RULES, Rules
from .tasks import Task, make_task

TOOL_ARGS = {   # every tool and the arguments it accepts; anything else is rejected
    "list_orders": ("customer",),
    "lookup_order": ("order_id",),
    "check_customer": ("customer",),
    "issue_refund": ("order_id", "amount_cents"),
    "escalate": ("order_id",),
    "reply": ("resolution", "message"),
}


@dataclass
class Episode:
    """Everything a reward function may look at once the episode is over."""
    task: Task
    db: dict              # final database state
    steps: list           # [{"action": ..., "result": ...}, ...]
    reply: dict | None    # {"resolution": ..., "message": ...}, None if never sent
    truncated: bool
    invalid_actions: int


class RefundDeskEnv:
    """reset(seed) -> obs;  step(action) -> (obs, reward, terminated, truncated, info)"""

    def __init__(self, reward_fn, max_steps: int = 8, rules: Rules = RULES):
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.rules = rules
        self.task = None
        self.done = True

    def reset(self, seed: int) -> dict:
        self.task = make_task(seed, self.rules)
        self.db = copy.deepcopy(self.task.orders)  # nothing survives from the last episode
        self.steps = []
        self.reply = None
        self.invalid_actions = 0
        self.done = False
        return self._obs(None)

    def step(self, action):
        if self.done:
            raise RuntimeError("episode is over (or never started): call reset() first")

        result = self._execute(action)
        logged = copy.deepcopy(action) if isinstance(action, dict) else repr(action)
        self.steps.append({"action": logged, "result": result})

        # terminated = the agent ended it. truncated = we cut it off.
        terminated = self.reply is not None
        truncated = not terminated and len(self.steps) >= self.max_steps
        self.done = terminated or truncated

        reward, info = 0.0, {}
        if self.done:
            from .rewards import is_true_success
            ep = self.episode(truncated)
            reward = self.reward_fn(ep)
            # `info` is for the training loop and for logging. A policy must
            # never read it: it contains the answer.
            info = {"true_success": is_true_success(ep), "expected": self.task.expected,
                    "case": self.task.case}
        return self._obs(result), reward, terminated, truncated, info

    def episode(self, truncated: bool = False) -> Episode:
        return Episode(self.task, copy.deepcopy(self.db), copy.deepcopy(self.steps),
                       copy.deepcopy(self.reply), truncated, self.invalid_actions)

    def _obs(self, last_result) -> dict:
        return {
            "ticket": {
                "customer": self.task.customer,
                "message": self.task.message,   # names the item, not the order
            },
            "last_result": copy.deepcopy(last_result),
            "steps_left": self.max_steps - len(self.steps),
        }

    # A bad action never raises. It costs a step and returns an error the agent
    # can read, exactly like a real API would.

    def _execute(self, action) -> dict:
        if not isinstance(action, dict) or not isinstance(action.get("tool"), str):
            return self._invalid(None, 'action must look like {"tool": "<name>", "args": {...}}')
        tool, args = action["tool"], action.get("args", {})
        if not isinstance(args, dict):
            return self._invalid(tool, "args must be an object")
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            return self._invalid(tool, f"unknown tool {tool!r}")
        # Strict like a real API: an argument the tool does not know is an
        # error, not something to ignore quietly.
        unknown = sorted(set(args) - set(TOOL_ARGS[tool]))
        if unknown:
            return self._invalid(tool, f"unknown argument(s) for {tool}: {unknown}")
        return handler(args)

    def _invalid(self, tool, error) -> dict:
        self.invalid_actions += 1
        return {"ok": False, "tool": tool, "error": error}

    def _order(self, tool, args):
        order = self.db.get(args.get("order_id")) if isinstance(args.get("order_id"), str) else None
        if order is None:
            return None, self._invalid(tool, f"no such order: {args.get('order_id')!r}")
        return order, None

    def _tool_list_orders(self, args):
        customer = args.get("customer")
        found = [{"order_id": oid, "item": o["item"]}
                 for oid, o in self.db.items() if o["customer"] == customer]
        if not found:
            return self._invalid("list_orders", f"no orders for customer {customer!r}")
        return {"ok": True, "tool": "list_orders", "orders": found}

    def _tool_check_customer(self, args):
        if args.get("customer") != self.task.customer:
            return self._invalid("check_customer", f"no such customer: {args.get('customer')!r}")
        return {"ok": True, "tool": "check_customer", "customer": self.task.customer,
                "flagged": self.task.flagged}

    def _tool_lookup_order(self, args):
        order, err = self._order("lookup_order", args)
        if err:
            return err
        return {"ok": True, "tool": "lookup_order", "order": {"order_id": args["order_id"], **order}}

    def _tool_issue_refund(self, args):
        # The tool checks what a payments API would check (amount, duplicates).
        # It does NOT check company policy. That is the agent's job, and the
        # reward function's job to verify.
        order, err = self._order("issue_refund", args)
        if err:
            return err
        amount = args.get("amount_cents")
        if type(amount) is not int or amount <= 0:
            return self._invalid("issue_refund", "amount_cents must be a positive integer")
        if order["refunded_cents"] > 0:
            return self._invalid("issue_refund", "order was already refunded")
        if amount > order["amount_cents"]:
            return self._invalid("issue_refund", "amount exceeds what was paid")
        order["refunded_cents"] = amount
        return {"ok": True, "tool": "issue_refund", "refunded_cents": amount}

    def _tool_escalate(self, args):
        order, err = self._order("escalate", args)
        if err:
            return err
        if order["escalated"]:
            return self._invalid("escalate", "order was already escalated")
        order["escalated"] = True
        return {"ok": True, "tool": "escalate"}

    def _tool_reply(self, args):
        if args.get("resolution") not in RESOLUTIONS:
            return self._invalid("reply", f"resolution must be one of {list(RESOLUTIONS)}")
        self.reply = {"resolution": args["resolution"], "message": str(args.get("message", ""))}
        return {"ok": True, "tool": "reply"}
