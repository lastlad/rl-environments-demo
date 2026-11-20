# 2. The trainees

A trainee is anything that can sit at the desk and work a ticket. They live in
`agents/`.

## One seat, many trainees

The desk (`runner.py`) asks a trainee only two things:

```python
trainee.reset()            # a new ticket is starting
action = trainee.act(obs)  # here is what you can see; what's your next tool call?
```

The answer is always a tool call like
`{"tool": "lookup_order", "args": {"order_id": "A5969"}}`.

Anything that answers those two questions fits the seat. The desk never asks
**how** the answer was made:

```mermaid
flowchart TB
    DESK(("the desk<br/>runner.py<br/>calls reset and act"))

    O["OraclePolicy<br/>agents/toy.py<br/>if/else rules"] --> DESK
    R["RandomPolicy<br/>agents/toy.py<br/>random moves"] --> DESK
    TB["TabularPolicy<br/>agents/toy.py<br/>a small table of scores"] --> DESK
    C["ClaudePolicy, ClaudeLoginPolicy<br/>agents/claude.py<br/>asks Claude"] --> DESK
    G["LocalPolicy<br/>agents/local.py<br/>runs Gemma on your Mac"] --> DESK
```

Because they all fit the same seat, the shop, the managers and the exam work
with every one of them, unchanged. The scores are directly comparable: the same
tickets, the same rules, the same manager.

## Same seat, different jobs

```mermaid
flowchart LR
    subgraph SEAT["all fit the same seat"]
        O["THE EXPERT<br/>OraclePolicy<br/>always right, never learns"]
        TOYS["PRACTICE DUMMIES<br/>RandomPolicy, TabularPolicy<br/>for step 2's demo"]
        C["THE EXPERIENCED HIRE<br/>Claude<br/>a strong model to compare with"]
        G["THE NEW TRAINEE<br/>Gemma<br/>the one we train"]
    end
    O -->|"its recorded work<br/>teaches (step 5)"| G
    C -.->|"the score to aim for"| G
```

| Trainee | Job | Used in |
|---|---|---|
| `OraclePolicy` | The expert. Shows the job can be done (scores 100%), and its recorded work is step 5's training data | steps 1, 2, 5 |
| `RandomPolicy` | The lowest possible score, for comparison | step 2 |
| `TabularPolicy` | A tiny learner that shows what each manager teaches | step 2 |
| Claude | A strong model; the benchmark | step 3 |
| Gemma | The small model on your Mac that we test and train | steps 4, 5, 6 |

## How a language model sits at the desk

Claude and Gemma read and write text, not Python dictionaries. Their trainee
classes translate in both directions:

```mermaid
sequenceDiagram
    participant Desk as runner.py
    participant P as LocalPolicy (Gemma)
    participant M as the Gemma model

    Desk->>P: act(obs)
    Note over P: render_prompt(obs) turns it into text,<br/>e.g. "Tool result: ... (6 steps left)"
    Note over P: adds it to the conversation:<br/>rules + ticket + every result so far
    P->>M: the whole conversation as one long text
    M-->>P: thinking, then a JSON tool call
    Note over P: keeps the full text in self.trace<br/>(step 6 uses it)
    Note over P: drops the thinking,<br/>parse_action reads the JSON
    P-->>Desk: the tool call
```

Claude works the same way: `ClaudePolicy` sends the conversation through the
API, and `ClaudeLoginPolicy` sends it through your Claude Code login.

If a model writes something that isn't valid JSON, `parse_action` passes it on
as-is and the shop rejects it as a bad action. The desk never crashes on a bad
answer. The trainee just loses a step.

## The expert, up close

The expert in `agents/toy.py` is plain Python. It keeps a small memory of what
the tool results said, then asks "what don't I know yet?":

```mermaid
flowchart TB
    START(["act(obs)"]) --> Q1{"know which order<br/>the item is?"}
    Q1 -->|no| A1["list_orders"]
    Q1 -->|yes| Q2{"looked up<br/>the order?"}
    Q2 -->|no| A2["lookup_order"]
    Q2 -->|yes| Q3{"checked<br/>the customer?"}
    Q3 -->|no| A3["check_customer"]
    Q3 -->|yes| R1{"Rule 1: flagged?"}
    R1 -->|yes| ESC["escalate, then reply escalated"]
    R1 -->|no| R2{"Rule 2: over 30 days?"}
    R2 -->|yes| DEN["reply denied"]
    R2 -->|no| R3{"Rule 3: over 10000 cents?"}
    R3 -->|yes| ESC
    R3 -->|no| R4{"Rule 4: opened?"}
    R4 -->|yes| HALF["issue_refund half, then reply refunded"]
    R4 -->|no| FULL["issue_refund full, then reply refunded"]
```

Alongside each move, the expert fills in a sentence that explains it, with the
real numbers dropped in, for example *"Rule 1: the customer is not flagged.
Rule 2: delivered 6 days ago, within 30. ..."*. That sentence is stored in
`self.thought`. Step 5 records it as the reasoning Gemma learns from.

Next: [the steps](3-the-steps.md).
