# 3. The steps

The project has eight scripts. Run them in sequence. Each script connects an
agent to the environment for one purpose: to show, to test, to record or to
train.

```mermaid
flowchart LR
    subgraph A["Check the environment"]
        S1["1 one ticket"] --> S2["2 two reward functions"]
    end
    subgraph B["Test before training"]
        S3["3 Claude exam"]
        S4["4 Gemma exam"]
    end
    subgraph C["Train Gemma"]
        S5["5 supervised"] --> S6["6 reinforcement"]
    end
    subgraph D["Compare"]
        S4b["4 again, trained Gemma"] --> S7["7 results table"]
    end
    subgraph E["Test what Gemma learned"]
        S8["8 changed rules"]
    end
    A --> B --> C --> D --> E
```

| Step | Agent | Reward function | Output |
|---|---|---|---|
| 1 | The expert | `verified_reward` | `out/trajectories.jsonl` |
| 2 | `RandomPolicy`, then `TabularPolicy` | `naive_reward`, then `verified_reward` | Printed comparison |
| 3 | Claude | `verified_reward` (exam) | `results/claude.json` |
| 4 | Gemma | `verified_reward` (exam) | `results/gemma_....json` |
| 5 | The expert (recorded), then Gemma trains on the records | `verified_reward` | `out/sft_data/`, `out/sft_adapter/` |
| 6 | Gemma, with training tickets | `verified_reward` | `out/sft_rft_data/`, `out/sft_rft_adapter/` |
| 7 | No agent | No reward function | The results table |
| 8 | The expert, then Gemma, with new rules | `verified_reward`, with the new rules | `results/changed_rules/` |

---

## Step 1: run one ticket

`python3 step1_watch_one_ticket.py`

**Purpose:** Step 1 shows the environment before you use a model. The expert
must get a high score. If it does not, the environment has a defect.

```mermaid
sequenceDiagram
    participant S as step1
    participant R as runner.py rollout
    participant X as expert (toy.py)
    participant E as env.py

    S->>R: rollout(seed 7, expert)
    R->>E: reset(7)
    loop until reply
        R->>X: act
        X-->>R: tool call
        R->>E: step
        E-->>R: tool result
        Note over R: records the step as one<br/>user and assistant message pair
    end
    R-->>S: the conversation and the score
    Note over S: shows it
    S->>R: the same for seeds 0 to 49
    R-->>S: 50 conversations
    Note over S: saves out/trajectories.jsonl
```

---

## Step 2: compare the two reward functions

`python3 step2_compare_rewards.py`

**Purpose:** Step 2 shows that the reward function is more important than the
method of the agent. One small agent (`TabularPolicy`) trains two times, with a
different reward function each time. Then `verified_reward` gives a score to
both results on new tickets.

```mermaid
flowchart TB
    subgraph BOUNDS["the lowest and the highest score"]
        OR["expert: 100%"]
        RA["RandomPolicy: 5%"]
    end

    subgraph RUN1["training run 1"]
        T1["TabularPolicy<br/>8000 training tickets"] -->|score from| N["naive_reward"]
        N -->|the score changes<br/>its table| T1
    end

    subgraph RUN2["training run 2"]
        T2["TabularPolicy<br/>8000 training tickets"] -->|score from| V["verified_reward"]
        V -->|the score changes<br/>its table| T2
    end

    T1 --> EV["both do 600 new tickets<br/>scored by verified_reward"]
    T2 --> EV
    EV --> R1["trained with naive_reward: 40%<br/>learned to write the answer<br/>without the action"]
    EV --> R2["trained with verified_reward: 100%"]
```

`naive_reward` examines only the reply text. Each step costs 0.02. For this reason, a
reply of "refunded" without a call to `issue_refund` gets a higher score.
It gets the same reward and has fewer step costs. `TabularPolicy` finds
this behaviour and repeats it.

---

## Step 3: Claude does the exam

`python3 step3_test_claude.py 100`

**Purpose:** Step 3 shows the score of a large model on this environment.

```mermaid
sequenceDiagram
    participant S as step3
    participant X as exam.py
    participant C as ClaudeCodePolicy
    participant API as Claude (through your login)

    S->>X: run_exam(Claude, 100 tickets)
    loop 100 exam tickets
        loop each step of the ticket
            X->>C: act(observation)
            C->>API: rules and conversation
            API-->>C: a JSON tool call
            C-->>X: the tool call
        end
        Note over X: verified_reward gives the score
    end
    X-->>S: score card: 100/100
    Note over S: saves results/claude.json
```

---

## Step 4: Gemma does the exam

`.venv/bin/python step4_test_local_model.py 100`

**Purpose:** Step 4 shows the score of the small model before training. After
training, use step 4 again to test the trained model. The exam is the same as
in step 3. Only the agent is different.

```mermaid
sequenceDiagram
    participant S as step4
    participant X as exam.py
    participant G as LocalPolicy
    participant M as Gemma on your Mac

    S->>G: load Gemma, and an adapter if you give one
    S->>X: run_exam(Gemma, 100 tickets)
    loop 100 exam tickets
        loop each step of the ticket
            X->>G: act(observation)
            G->>M: rules and conversation
            M-->>G: reasoning, then a JSON tool call
            G-->>X: the tool call only
        end
        Note over X: verified_reward gives the score
    end
    X-->>S: score card
    Note over S: saves results/gemma_untrained.json<br/>(or gemma_sft, gemma_sft_then_rft, ...)
```

The second argument selects the adapter:

```mermaid
flowchart LR
    BASE["Gemma as downloaded"] --> U["no adapter<br/>gemma_untrained: 78"]
    BASE --> SFT["+ out/sft_adapter<br/>gemma_sft: 99"]
    BASE --> SR["+ out/sft_rft_adapter<br/>gemma_sft_then_rft: 100"]
    BASE --> RFT["+ out/rft_adapter<br/>gemma_rft: 79"]
```

An adapter is a small file of weight changes. It loads on top of the original
Gemma weights. Training does not change the original model. Training writes
only an adapter.

**Result:** Untrained Gemma did not call `check_customer`. For this reason, it gave
refunds to customers with a fraud flag. Steps 5 and 6 correct this problem.

---

## Step 5: supervised fine-tuning

`.venv/bin/python step5_supervised_finetune.py`

**Purpose:** Gemma does not check the customer. In step 5, Gemma learns from
the expert. The expert checks the customer and writes the reason for each
action.

Step 5 has two parts:

1. Record the expert.
2. Train Gemma on the records.

### Part 1: record the expert

```mermaid
sequenceDiagram
    participant S as step5
    participant R as runner.py
    participant RE as RecordedExpert
    participant X as expert (toy.py)
    participant E as env.py

    loop seeds 0 to 399
        S->>R: run_episode(seed, RecordedExpert)
        R->>E: reset(seed)
        loop each step
            R->>RE: act(observation)
            Note over RE: PROMPT = the conversation,<br/>in the text format of Gemma
            RE->>X: act
            X-->>RE: tool call and reasoning sentence
            Note over RE: OUTPUT = reasoning and tool call
            Note over RE: saves the (PROMPT, OUTPUT) pair
            RE-->>R: the tool call
            R->>E: step
        end
    end
    Note over S: writes all pairs to<br/>out/sft_data/train.jsonl (1,921 lines)
```

Each step of a ticket gives one line. This table shows the ticket "C365: refund
for my keyboard". The customer for this ticket has a fraud flag.

| Line | PROMPT ends with | OUTPUT (the text that Gemma learns) |
|---|---|---|
| 1 | the ticket | *"Customer C365 names an item... I list their orders"* → `list_orders` |
| 2 | orders: keyboard = A5969 | *"The keyboard is order A5969. I look it up..."* → `lookup_order` |
| 3 | 54 days, 1826 cents, not opened | *"Rule 1 comes first, so I check whether C365 is flagged"* → `check_customer` |
| 4 | flagged: true | *"Rule 1: flagged for fraud, so I escalate"* → `escalate` |
| 5 | escalate ok | *"Escalated. Now I tell the customer."* → `reply escalated` |

### Part 2: train Gemma on the records

```mermaid
flowchart LR
    D[("out/sft_data/train.jsonl<br/>1,921 PROMPT and OUTPUT pairs")] --> T["agents/local_trainer.py<br/>800 training steps"]
    BASE["Gemma as downloaded"] --> T
    T --> A[("out/sft_adapter")]
    A --> EX["step 4 exam<br/>99/100"]
```

At each training step, `local_trainer.py` gives one pair to Gemma. The
training measures only the OUTPUT part. It measures the probability that Gemma
writes the reasoning and the tool call of the expert. Then the training
changes the weights to increase this probability.

**Why the reasoning is necessary:** An earlier version recorded only the tool
calls. Gemma copied the tool calls but stopped its reasoning. Its score was
68. With the reasoning in the records, its score was 99.

---

## Step 6: reinforcement fine-tuning

`.venv/bin/python step6_reinforcement_finetune.py --after-sft`

**Purpose:** Step 6 does not use the expert. Gemma works training tickets.
`verified_reward` selects the attempts that Gemma learns from. The option
`--after-sft` starts from the step 5 adapter. Without this option, step 6
starts from untrained Gemma.

```mermaid
flowchart TB
    START[("out/sft_adapter<br/>copied to out/sft_rft_adapter")] --> Q0["short exam: 30 tickets"]
    Q0 --> P

    subgraph ROUND["one round, 3 rounds in total"]
        P["Gemma works 60 new training tickets<br/>with random selection on"]
        P --> J{"verified_reward:<br/>correct, and<br/>no rejected tool calls?"}
        J -->|no, attempts remain| P
        J -->|no, 3 attempts used| SKIP["go to the next ticket"]
        J -->|yes| KEEP["keep the attempt:<br/>the PROMPT and OUTPUT pairs<br/>of Gemma, with reasoning"]
        KEEP --> DATA[("out/sft_rft_data/train.jsonl<br/>all kept attempts")]
        DATA --> T["local_trainer.py<br/>starts from the current adapter"]
        T --> W[("out/sft_rft_adapter")]
        W --> QE["short exam: 30 tickets,<br/>random selection off"]
    end

    QE -->|next round,<br/>new tickets| P
    W --> FIN["step 4 exam<br/>100/100"]
```

This diagram shows the procedure for one training ticket:

```mermaid
sequenceDiagram
    participant S as step6
    participant R as runner.py
    participant G as Gemma (LocalPolicy)
    participant E as env.py
    participant M as verified_reward

    loop 3 attempts maximum
        S->>R: run_episode(ticket, Gemma)
        loop each step
            R->>G: act
            Note over G: writes reasoning and tool call,<br/>keeps both in self.trace
            G-->>R: tool call
            R->>E: step
        end
        E->>M: the episode
        M-->>S: correct or not correct
        alt correct and no rejected tool calls
            Note over S: keep the trace of Gemma,<br/>go to the next ticket
        else
            Note over S: discard the trace,<br/>do the ticket again
        end
    end
```

Step 5 and step 6 are different in the source of the OUTPUT:

| | Step 5 | Step 6 |
|---|---|---|
| Source of the reasoning and tool calls | The expert | Gemma |
| Kept tickets | All tickets (the expert is always correct) | Only the tickets that `verified_reward` accepts |
| What Gemma learns from | The examples of the expert | The scores from `verified_reward` |

This table shows the results of each round, from the step 5 adapter:

| | Training tickets correct | Short exam |
|---|---|---|
| Start | | 29/30 |
| Round 1 | 59/60 | 29/30 |
| Round 2 | 60/60 | 29/30 |
| Round 3 | 60/60 | 30/30 |

**Why step 5 is necessary before step 6:** From untrained Gemma, step 6 gave a
score of 79. Untrained Gemma did not call `check_customer`. For this reason, no correct
attempt included this call, and step 6 had no good example to keep. Step 5
taught Gemma to check the customer. Then step 6 had good attempts to keep.

---

## Step 7: show the results table

`python3 step7_show_results.py`

```mermaid
flowchart LR
    C[("results/claude.json")] --> S7["step7<br/>reads all score cards"]
    U[("results/gemma_untrained.json")] --> S7
    N[("results/gemma_sft_no_reasoning.json")] --> S7
    F[("results/gemma_sft.json")] --> S7
    R[("results/gemma_rft.json")] --> S7
    SR[("results/gemma_sft_then_rft.json")] --> S7
    S7 --> T["one table:<br/>score, score for each ticket type,<br/>correct reply without the action,<br/>refunds that the policy does not allow"]
```

The table is in the main [README](../README.md#results).

---

## Step 8: change the rules

`.venv/bin/python step8_changed_rules.py 100 out/sft_adapter`

**Purpose:** Trained Gemma got a score of 100. But all its training tickets
and exam tickets used the same two numbers: deny after 30 days, and escalate
above 10000 cents. For this reason, the score cannot show the difference between two
possible causes:

- Gemma reads the rules.
- Gemma memorised the numbers 30 and 10000.

Step 8 changes the two numbers. Then it tests Gemma again.

The tickets do not change. Only the rules and the correct answers change:

```mermaid
flowchart LR
    SEED["exam seed"] --> T["the same ticket:<br/>same customer, orders and message"]
    T --> OLD["normal rules<br/>deny after 30 days<br/>escalate above 10000"]
    T --> NEW["changed rules<br/>deny after 14 days<br/>escalate above 5000"]
    OLD --> A1["correct answer,<br/>for example: refund in full"]
    NEW --> A2["correct answer,<br/>for example: deny"]
```

`make_task(seed, rules)` makes the ticket with the same procedure for all
rules. Then it calculates the correct answer from the given rules. The correct
answer changes on 36 of the 100 exam tickets.

```mermaid
sequenceDiagram
    participant S as step8
    participant G as Gemma (LocalPolicy)
    participant X as exam.py
    participant M as verified_reward

    S->>G: load the adapter,<br/>give it the NEW rules
    S->>X: run_exam(Gemma, rules = changed)
    loop 100 exam tickets
        X->>G: the same ticket as before
        G-->>X: tool calls
        X->>M: score by the NEW rules
        Note over X: on a changed ticket, also check:<br/>do the OLD rules accept it?
    end
    X-->>S: score card, with<br/>"answer changed: correct" and<br/>"followed the old rules"
```

Without arguments, step 8 shows the changes and the results of two experts.
One expert reads the new rules and gets 100. The other expert uses the old
numbers and gets 64. This expert does all 36 changed tickets with the old
rules. Its result shows the behaviour of a model that memorised the numbers.

**Result:** Two trained versions of Gemma did the exam: one from step 5, and
one from step 5 and step 6. Both got all 36 changed tickets correct. Neither
version used the old rules. The reasoning of Gemma contains the new number:
*"delivered 30 days ago, more than 14, so I deny."* This shows that Gemma reads the
rules.

---

## Summary of the Gemma results

```mermaid
flowchart LR
    U["untrained Gemma<br/>78<br/>does not check the customer"]
    U -->|"step 5 without reasoning"| N["68<br/>checks, but no reasoning"]
    U -->|"step 6 only"| R["79<br/>does not check"]
    U -->|"step 5 with reasoning"| S["99<br/>checks and gives reasons"]
    S -->|"step 6 --after-sft"| SR["100<br/>same score as Claude"]
    S -->|"step 8: new rule numbers"| G1["100<br/>reads the new rules"]
    SR -->|"step 8: new rule numbers"| G2["100<br/>reads the new rules"]
```

The training has two parts:

1. In step 5, Gemma learns from the expert. The expert gives a reason for each
   decision.
2. In step 6, Gemma works without the expert. `verified_reward` selects the
   decisions that Gemma learns from.

Back to the [start of the docs](README.md).
