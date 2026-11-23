# 1. The environment

The code is in `refund_env/`. It has two groups of files:

- **Prerequisites:** the data and the answer key that an environment needs.
  This demo makes them with code. In a real project, they usually come from
  real data.
- **The environment:** the order database, the tools and the episode. This
  code is the same for synthetic data and for real data.

## Prerequisites and the environment

```mermaid
flowchart TB
    subgraph PRE["PREREQUISITES (synthetic in this demo)"]
        GEN["tasks.py<br/>make_task(seed):<br/>a customer, 3 orders,<br/>a refund request"]
        POL["policy.py<br/>the 5 rules and<br/>expected_case():<br/>the correct answer"]
    end

    subgraph ENV["THE ENVIRONMENT: env.py"]
        DB[("env.db<br/>the order database<br/>for this ticket")]
        TOOLS["6 tools<br/>list_orders, lookup_order,<br/>check_customer, issue_refund,<br/>escalate, reply"]
        TOOLS <-->|read and change| DB
    end

    subgraph AROUND["AROUND THE ENVIRONMENT"]
        REW["rewards.py<br/>naive_reward and<br/>verified_reward"]
        RUN["runner.py<br/>run_episode:<br/>one ticket from start to end"]
        TXT["prompts.py<br/>the rules and observations<br/>as text for a model"]
        EXAM["exam.py<br/>the same 100 tickets<br/>for every agent"]
    end

    GEN -->|the ticket and its orders| DB
    POL -->|the correct answer| REW
    RUN --> ENV
    RUN --> REW
    EXAM --> RUN
```

| Part | Location | Function | In a real project |
|---|---|---|---|
| Tasks | `make_task(seed)` in `tasks.py` | Makes one ticket and its orders from a seed number | Real tickets, for example old support tickets with the private data removed |
| Policy and answer key | `Rules` and `expected_case()` in `policy.py` | Give the 5 rules and the correct answer. The two numbers (30 days, 10000 cents) are settings. Step 8 changes them | A policy document, and a checker: unit tests, a reference final state, a rubric or human labels |
| Database | `env.db` in `env.py` | Holds the 3 orders of the customer | A test copy of the real system |
| Tools | `_tool_...` functions in `env.py` | Read or change the database | The real APIs, connected to the test copy |
| Reward functions | `rewards.py` | Give a score to a completed episode | The same idea, with the real checker |
| Episode loop | `run_episode` in `runner.py` | Runs one ticket | The same idea, often in a training library |
| Text interface | `prompts.py` | Changes the rules and the observations to text, and the model text to an action | The same idea |
| Exam | `exam.py` | Runs 100 fixed tickets and records the scores | A fixed evaluation set |

## The refund policy

The agent must apply these 5 rules in this sequence:

1. If the customer has a fraud flag, escalate the order to a human.
2. If the delivery was more than 30 days ago, deny the refund.
3. If the amount is more than 10000 cents ($100), escalate the order.
4. If the customer opened the item, refund half of the amount.
5. If no rule above applies, refund the full amount.

## Tickets come from seed numbers

The same seed number always makes the same ticket. For this reason, every agent can do
the same exam.

```mermaid
flowchart LR
    SEED["seed 1000014"] --> PICK["select a ticket type<br/>for example: escalated_flagged"]
    PICK --> FACTS["select facts for that type<br/>days, amount, opened, flagged"]
    FACTS --> ORDERS["3 orders for one customer<br/>the correct order and 2 other orders"]
    ORDERS --> MSG["the customer message<br/>gives the item name and a claim<br/>that can be false"]
    MSG --> TASK["Task<br/>the agent sees: the customer and the message<br/>the agent does not see: the type and the correct answer"]
```

There are five ticket types. Each type agrees with one rule:

| Ticket type | Correct answer | Rule |
|---|---|---|
| `escalated_flagged` | Escalate | Rule 1: the customer has a fraud flag |
| `denied` | Deny | Rule 2: the delivery was more than 30 days ago |
| `escalated_over_limit` | Escalate | Rule 3: the amount is more than 10000 cents |
| `refunded_half` | Refund half | Rule 4: the customer opened the item |
| `refunded_full` | Refund in full | Rule 5: no other rule applies |

Each script uses a different range of seeds:

```mermaid
flowchart LR
    A["seeds 0 to 399<br/>step 5 training"] ~~~ B["seeds 10000 and higher<br/>step 6 training"] ~~~ C["seeds 1000000 to 1000099<br/>the exam"]
```

The ranges do not overlap. For this reason, no agent trains on an exam ticket.

## One episode

The function `run_episode` in `runner.py` runs one episode. All scripts use
this function.

```mermaid
sequenceDiagram
    participant Run as runner.py
    participant Env as env.py
    participant A as agent
    participant Rew as reward function

    Run->>Env: reset(seed)
    Env-->>Run: the ticket: "C365: refund for my keyboard"
    Run->>A: reset()
    loop each step, 8 steps maximum
        Run->>A: act(observation)
        A-->>Run: one tool call
        Run->>Env: step(tool call)
        Note over Env: runs the tool,<br/>can change the database
        Env-->>Run: the tool result
    end
    Note over Run,Env: the loop stops when the agent<br/>calls reply, or after 8 steps
    Env->>Rew: the episode and the final database
    Rew-->>Run: a score, for example +0.90
```

At each step, the agent sees three items:

- The ticket.
- The result of the last tool call.
- The number of steps that remain.

The agent does not see the ticket type or the correct answer. It must use the
tools to find the facts.

## The tools

```mermaid
flowchart LR
    T["agent"] -->|list_orders C365| L["the orders of the customer<br/>keyboard = A5969, ..."]
    T -->|lookup_order A5969| O["days, amount,<br/>opened or not opened"]
    T -->|check_customer C365| F["fraud flag:<br/>true or false"]
    T -->|issue_refund A5969, cents| R["sends money<br/>changes the database"]
    T -->|escalate A5969| E["sends the order to a human<br/>changes the database"]
    T -->|reply| Y["sends the answer to the customer<br/>ends the episode"]
```

Only `issue_refund` and `escalate` change the database. The first three tools
only read data. The `reply` tool only sends text.

The tools operate like a real system:

- The environment rejects an incorrect tool call. Examples are an unknown tool
  name, an unknown order or an unknown argument. The agent gets an error
  message, and the step counts.
- `issue_refund` checks that the amount is a positive whole number. It also
  checks that the order has no previous refund. It does not check the refund
  policy. The agent must apply the policy, and the reward function checks the
  result.

## The two reward functions

At the end of an episode, a reward function gives a score. The two functions
in `rewards.py` examine different data:

```mermaid
flowchart TB
    END["completed episode"] --> N["naive_reward"]
    END --> V["verified_reward"]

    N --> NQ{"does the reply give<br/>the correct answer?"}
    NQ -->|yes| N1["+1"]
    NQ -->|no| N0["0"]

    V --> W{"did the agent send money<br/>that the policy does not allow?"}
    W -->|yes| VM["-1"]
    W -->|no| VQ{"is the reply correct, AND does the database<br/>show the correct refund or escalation,<br/>AND are the other orders unchanged?"}
    VQ -->|yes| V1["+1"]
    VQ -->|no| V0["0"]
```

Both functions then subtract 0.02 for each step. A correct episode with 5
steps gets 1 − 0.10 = **+0.90**.

An agent can get a high `naive_reward` without the correct actions. For
example, the reply can say "escalated", but the agent does not call
`escalate`. Step 2 shows an agent that learns this incorrect behaviour. The
exam and step 6 use `verified_reward` only.

## The exam

The function `run_exam` in `exam.py` gives the same 100 tickets to an agent.
Then it records the results in a score card:

```mermaid
flowchart LR
    P["an agent"] --> E["run_exam<br/>100 tickets<br/>seeds 1000000 to 1000099"]
    E --> D["run_episode<br/>for each ticket"]
    D --> G["verified_reward<br/>gives the score"]
    G --> C["score card<br/>correct answers for each ticket type,<br/>correct reply without the action,<br/>refunds that the policy does not allow"]
    C --> J[("results/name.json")]
```

The exam saves a score card only for a run of 100 tickets. For this reason, a short test
run cannot replace a full result.

Next: [the agents](2-agents.md).
