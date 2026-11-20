"""Refund desk: a small RL environment shaped like a real agent task.

One episode = one support ticket. The agent calls tools against a simulated
order database, then ends the episode with a `reply`.

The customer names the item, not the order number, so the agent has to find
the order first. Company policy, checked in this order:
  1. customer is flagged for fraud        -> escalate to a human
  2. delivered more than 30 days ago      -> deny
  3. amount over $100                     -> escalate to a human
  4. the item was opened                  -> refund half the amount
  5. otherwise                            -> refund the full amount
Money is always in cents (amount_cents 10000 = $100), and the prompt says so.

The pieces every environment has, and where they live here:
  task generator   make_task(seed)      what situation the agent is dropped into
  state            env.db               the simulated world, mutated by tools
  observation      env._obs()           the part of the world the agent may see
  actions          env._execute()       tool calls: {"tool": ..., "args": {...}}
  reward           rewards.py           scores the finished episode
  termination      env.step()           `reply` ends it; the step limit truncates it
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass

REFUND_WINDOW_DAYS = 30
AUTO_REFUND_LIMIT_CENTS = 100_00  # money is integer cents: float equality breaks graders
RESOLUTIONS = ("refunded", "denied", "escalated")
# The five kinds of ticket. The part before "_" is the right reply.
CASES = ("refunded_full", "refunded_half", "escalated_over_limit", "escalated_flagged", "denied")
TOOL_ARGS = {   # every tool and the arguments it accepts; anything else is rejected
    "list_orders": ("customer",),
    "lookup_order": ("order_id",),
    "check_customer": ("customer",),
    "issue_refund": ("order_id", "amount_cents"),
    "escalate": ("order_id",),
    "reply": ("resolution", "message"),
}
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


@dataclass
class Episode:
    """Everything a reward function may look at once the episode is over."""
    task: Task
    db: dict              # final database state
    steps: list           # [{"action": ..., "result": ...}, ...]
    reply: dict | None    # {"resolution": ..., "message": ...}, None if never sent
    truncated: bool
    invalid_actions: int


def expected_case(flagged: bool, days_since_delivery: int, amount_cents: int,
                  opened: bool) -> str:
    """The policy, written once. The task generator is tested against it."""
    if flagged:
        return "escalated_flagged"
    if days_since_delivery > REFUND_WINDOW_DAYS:
        return "denied"
    if amount_cents > AUTO_REFUND_LIMIT_CENTS:
        return "escalated_over_limit"
    return "refunded_half" if opened else "refunded_full"


def refund_due_cents(order: dict) -> int:
    """How much a refundable order should get back: half (rounded down) if opened."""
    return order["amount_cents"] // 2 if order["opened"] else order["amount_cents"]


def make_task(seed: int) -> Task:
    """Same seed -> same task, always. Uses its own RNG, never the global one."""
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
    expected = case.split("_")[0]
    return Task(f"refund-{seed}", customer, ids[0], message, orders, flagged, case, expected)


class RefundDeskEnv:
    """reset(seed) -> obs;  step(action) -> (obs, reward, terminated, truncated, info)"""

    def __init__(self, reward_fn, max_steps: int = 8):
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.task = None
        self.done = True

    def reset(self, seed: int) -> dict:
        self.task = make_task(seed)
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

    # ---- observation ------------------------------------------------------

    def _obs(self, last_result) -> dict:
        return {
            "ticket": {
                "customer": self.task.customer,
                "message": self.task.message,   # names the item, not the order
            },
            "last_result": copy.deepcopy(last_result),
            "steps_left": self.max_steps - len(self.steps),
        }

    # ---- actions ----------------------------------------------------------
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


# ---- adapters for a language model -----------------------------------------
# A scripted or tabular policy consumes the observation dict directly. An LLM
# needs it as text, and its text output needs to be turned back into an action.

SYSTEM_PROMPT = f"""You are a refund desk agent. Customers name the item, not the
order number: find their order with list_orders, then look it up.

Policy, checked in this order:
1. Customer flagged for fraud (see check_customer): escalate to a human.
2. Delivered more than {REFUND_WINDOW_DAYS} days ago: deny.
3. amount_cents over {AUTO_REFUND_LIMIT_CENTS} (${AUTO_REFUND_LIMIT_CENTS // 100}): escalate to a human.
4. The item was opened: refund half the amount, rounded down to whole cents.
5. Otherwise: refund the full amount.
All amounts are in cents: amount_cents {AUTO_REFUND_LIMIT_CENTS} means ${AUTO_REFUND_LIMIT_CENTS // 100}.
Customers can be wrong about dates and about whether they opened the item.
Check the records.

Answer with exactly one JSON tool call per turn:
{{"tool": "list_orders", "args": {{"customer": "..."}}}}
{{"tool": "lookup_order", "args": {{"order_id": "..."}}}}
{{"tool": "check_customer", "args": {{"customer": "..."}}}}
{{"tool": "issue_refund", "args": {{"order_id": "...", "amount_cents": 1234}}}}
{{"tool": "escalate", "args": {{"order_id": "..."}}}}
{{"tool": "reply", "args": {{"resolution": "refunded|denied|escalated", "message": "..."}}}}"""


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
