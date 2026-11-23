# 2. The agents

An agent is a program or a model that selects the tool calls for a ticket. The
agents are in `agents/`.

## One interface for all agents

The episode loop (`runner.py`) calls only two methods on an agent:

```python
agent.reset()            # a new ticket starts
action = agent.act(obs)  # the agent receives the observation and returns one tool call
```

The return value is always one tool call, for example
`{"tool": "lookup_order", "args": {"order_id": "A5969"}}`.

Each agent class has these two methods. The episode loop does not know how an
agent selects its tool call:

```mermaid
flowchart TB
    LOOP(("episode loop<br/>runner.py<br/>calls reset and act"))

    O["OraclePolicy<br/>agents/toy.py<br/>fixed if/else rules"] --> LOOP
    R["RandomPolicy<br/>agents/toy.py<br/>random tool calls"] --> LOOP
    TB["TabularPolicy<br/>agents/toy.py<br/>a small table of scores"] --> LOOP
    C["ClaudeCodePolicy<br/>agents/claude_code.py<br/>sends the text to Claude"] --> LOOP
    G["LocalPolicy<br/>agents/local.py<br/>runs Gemma on your Mac"] --> LOOP
```

Because of this interface, the environment, the reward functions and the exam
operate with every agent without changes. You can compare the scores
directly. Each agent gets the same tickets, the same rules and the same
reward function.

## The function of each agent

```mermaid
flowchart LR
    O["OraclePolicy<br/>always correct,<br/>does not learn"]
    TOYS["RandomPolicy, TabularPolicy<br/>used in step 2"]
    C["Claude<br/>a large model<br/>for comparison"]
    G["Gemma<br/>the model<br/>that we train"]
    O -->|"its recorded episodes<br/>are the training data (step 5)"| G
    C -.->|"the score to compare with"| G
```

| Agent | Function | Steps |
|---|---|---|
| `OraclePolicy` | Shows that a score of 100% is possible. Its recorded episodes are the training data for step 5 | 1, 2, 5 |
| `RandomPolicy` | Gives the lowest score, for comparison | 2 |
| `TabularPolicy` | A small program that learns from scores. It shows what each reward function teaches | 2 |
| `ClaudeCodePolicy` | Claude, a large model. Its score is the target | 3 |
| Gemma | The small model on your Mac. We test it and train it | 4, 5, 6, 8 |

## How a language model operates as an agent

Claude and Gemma read text and write text. They do not use Python
dictionaries. Their agent classes change the data in the two directions:

```mermaid
sequenceDiagram
    participant Run as runner.py
    participant P as LocalPolicy (Gemma)
    participant M as the Gemma model

    Run->>P: act(obs)
    Note over P: render_prompt(obs) changes it to text,<br/>for example "Tool result: ... (6 steps left)"
    Note over P: adds the text to the conversation:<br/>rules, ticket and all results
    P->>M: the full conversation as one text
    M-->>P: reasoning, then a JSON tool call
    Note over P: keeps the full text in self.trace<br/>(step 6 uses it)
    Note over P: removes the reasoning,<br/>parse_action reads the JSON
    P-->>Run: the tool call
```

Claude uses the same procedure. `ClaudeCodePolicy` sends the conversation
through the Claude Code command-line tool. The tool uses your login, so you do
not need an API key.

Sometimes a model writes text that is not valid JSON. Then `parse_action`
sends the text to the environment without changes. The environment rejects it
as an incorrect tool call. The episode loop does not stop. The agent only
loses one step.

## How the expert selects a tool call

`OraclePolicy` in `agents/toy.py` is plain Python code. It keeps a record of
the tool results. At each step, it finds the first fact that it does not know:

```mermaid
flowchart TB
    START(["act(obs)"]) --> Q1{"is the order<br/>for the item known?"}
    Q1 -->|no| A1["list_orders"]
    Q1 -->|yes| Q2{"is the order record<br/>known?"}
    Q2 -->|no| A2["lookup_order"]
    Q2 -->|yes| Q3{"is the fraud flag<br/>known?"}
    Q3 -->|no| A3["check_customer"]
    Q3 -->|yes| R1{"Rule 1: fraud flag?"}
    R1 -->|yes| ESC["escalate, then reply escalated"]
    R1 -->|no| R2{"Rule 2: more than 30 days?"}
    R2 -->|yes| DEN["reply denied"]
    R2 -->|no| R3{"Rule 3: more than 10000 cents?"}
    R3 -->|yes| ESC
    R3 -->|no| R4{"Rule 4: opened?"}
    R4 -->|yes| HALF["issue_refund half, then reply refunded"]
    R4 -->|no| FULL["issue_refund full, then reply refunded"]
```

For each tool call, the expert also writes one sentence of reasoning. The
sentence contains the real numbers from the order. An example is: *"Rule 1:
the customer is not flagged. Rule 2: delivered 6 days ago, within 30."* The
expert keeps the sentence in `self.thought`. Step 5 records it as the
reasoning that Gemma learns.

The expert is also a prerequisite. It is possible only because the rules
are exact. In a real project, the examples usually come from logs of human
agents or from a larger model.

Next: [the steps](3-steps.md).
