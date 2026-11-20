# 3. The steps

Seven scripts, run in order. Each one puts a trainee at the desk for some
reason: to show the shop, to test, to record, or to train.

```mermaid
flowchart LR
    subgraph A["Build and check the shop"]
        S1["1 watch one ticket"] --> S2["2 compare managers"]
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
    A --> B --> C --> D
```

| Step | Who sits at the desk | Which manager | What comes out |
|---|---|---|---|
| 1 | The expert | careful | `out/trajectories.jsonl` |
| 2 | Random, then a tiny learner | lazy, then careful | Printed comparison |
| 3 | Claude | careful (exam) | `results/claude.json` |
| 4 | Gemma | careful (exam) | `results/gemma_....json` |
| 5 | The expert (recorded), then Gemma studies it | careful | `out/sft_data/`, `out/sft_adapter/` |
| 6 | Gemma, practising | careful | `out/sft_rft_data/`, `out/sft_rft_adapter/` |
| 7 | Nobody | none | The results table |

---

## Step 1: watch one ticket

`python3 step1_watch_one_ticket.py`

**Why:** to see the shop working before any AI is involved. If the expert
can't score well, the shop is broken, not the trainee.

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
        Note over R: records the step as a<br/>user / assistant message pair
    end
    R-->>S: the conversation and the score
    Note over S: prints it
    S->>R: the same for seeds 0 to 49
    R-->>S: 50 conversations
    Note over S: saves out/trajectories.jsonl
```

---

## Step 2: compare the two managers

`python3 step2_compare_rewards.py`

**Why:** to show the reward function matters more than the learner. The same
tiny learner is trained twice, once by each manager, then both are graded
fairly on unseen tickets.

```mermaid
flowchart TB
    subgraph BOUNDS["first, the floor and ceiling"]
        OR["expert: 100%"]
        RA["random clicker: 5%"]
    end

    subgraph RUN1["training run 1"]
        T1["TabularPolicy<br/>8000 practice tickets"] -->|scored by| N["LAZY MANAGER"]
        N -->|reward nudges<br/>its table| T1
    end

    subgraph RUN2["training run 2"]
        T2["TabularPolicy<br/>8000 practice tickets"] -->|scored by| V["CAREFUL MANAGER"]
        V -->|reward nudges<br/>its table| T2
    end

    T1 --> EV["both sit 600 unseen tickets<br/>graded by the CAREFUL manager"]
    T2 --> EV
    EV --> R1["lazy-trained: 40%<br/>learned to SAY the answer<br/>without doing the work"]
    EV --> R2["careful-trained: 100%"]
```

The lazy manager pays for words, and every step costs 0.02. So replying
"refunded" without issuing the refund, or "escalated" without pressing
escalate, is literally rewarded: same score, fewer steps. The learner finds
that and keeps doing it.

---

## Step 3: Claude sits the exam

`.venv/bin/python step3_test_claude.py 100`

**Why:** a strong model shows what a good score looks like on this shop.

```mermaid
sequenceDiagram
    participant S as step3
    participant X as exam.py
    participant C as ClaudeLoginPolicy
    participant API as Claude (via your login)

    S->>X: run_exam(Claude, 100 tickets)
    loop 100 exam tickets
        loop each step of the ticket
            X->>C: act(what you can see)
            C->>API: rules + conversation so far
            API-->>C: a JSON tool call
            C-->>X: the tool call
        end
        Note over X: careful manager grades the ticket
    end
    X-->>S: score card: 100/100
    Note over S: saves results/claude.json
```

---

## Step 4: Gemma sits the exam

`.venv/bin/python step4_test_local_model.py 100`

**Why:** to see where the small model starts, and later to test it after
training. It's the same exam as step 3; only the trainee changes.

```mermaid
sequenceDiagram
    participant S as step4
    participant X as exam.py
    participant G as LocalPolicy
    participant M as Gemma on your Mac

    S->>G: load Gemma, plus trained weights if given
    S->>X: run_exam(Gemma, 100 tickets)
    loop 100 exam tickets
        loop each step of the ticket
            X->>G: act(what you can see)
            G->>M: rules + conversation so far
            M-->>G: thinking, then a JSON tool call
            G-->>X: the tool call only
        end
        Note over X: careful manager grades the ticket
    end
    X-->>S: score card
    Note over S: saves results/gemma_untrained.json<br/>(or gemma_sft, gemma_sft_then_rft, ...)
```

The second argument picks which weights to load:

```mermaid
flowchart LR
    BASE["Gemma as downloaded"] --> U["no folder<br/>gemma_untrained: 78"]
    BASE --> SFT["+ out/sft_adapter<br/>gemma_sft: 99"]
    BASE --> SR["+ out/sft_rft_adapter<br/>gemma_sft_then_rft: 100"]
    BASE --> RFT["+ out/rft_adapter<br/>gemma_rft: 79"]
```

An adapter is a small file of changes laid on top of Gemma's original weights.
Training never changes the original model; it only writes an adapter.

**What the untrained exam showed:** Gemma never called `check_customer`, so it
refunded customers who were flagged for fraud. That's the problem steps 5 and 6
set out to fix.

---

## Step 5: supervised fine-tuning (learn from the expert)

`.venv/bin/python step5_supervised_finetune.py`

**Why:** Gemma never checks the customer, so we show it someone who does, and
who explains why.

Step 5 has two parts: **record** the expert, then **train** on the recording.

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
            R->>RE: act(what you can see)
            Note over RE: PROMPT = the conversation so far,<br/>in Gemma's exact text format
            RE->>X: act
            X-->>RE: tool call + thought sentence
            Note over RE: OUTPUT = thinking + tool call
            Note over RE: saves the (PROMPT, OUTPUT) pair
            RE-->>R: the tool call
            R->>E: step
        end
    end
    Note over S: writes all pairs to<br/>out/sft_data/train.jsonl (1,921 lines)
```

One ticket gives one line per step. For the ticket "C365: refund for my
keyboard", whose customer turns out to be flagged:

| Line | PROMPT ends with | OUTPUT (what Gemma learns to write) |
|---|---|---|
| 1 | the ticket | *"Customer C365 names an item... I list their orders"* → `list_orders` |
| 2 | orders: keyboard = A5969 | *"The keyboard is order A5969. I look it up..."* → `lookup_order` |
| 3 | 54 days, 1826 cents, not opened | *"Rule 1 comes first, so I check whether C365 is flagged"* → `check_customer` |
| 4 | flagged: true | *"Rule 1: flagged for fraud, so I escalate"* → `escalate` |
| 5 | escalate ok | *"Escalated. Now I tell the customer."* → `reply escalated` |

### Part 2: train on the recording

```mermaid
flowchart LR
    D[("out/sft_data/train.jsonl<br/>1,921 PROMPT / OUTPUT pairs")] --> T["agents/local_trainer.py<br/>800 training steps"]
    BASE["Gemma as downloaded"] --> T
    T --> A[("out/sft_adapter")]
    A --> EX["step 4 exam<br/>99/100"]
```

At each training step, `local_trainer.py` shows Gemma one pair. Gemma is graded
**only on the OUTPUT part**: how likely it was to write the expert's thinking
and tool call after that prompt. Its weights are nudged to make that more
likely next time.

**Why the reasoning matters:** an earlier version recorded only the tool calls.
Gemma copied the steps but stopped thinking, and scored 68. With the reasoning
included, it scored 99.

---

## Step 6: reinforcement fine-tuning (learn from its own good attempts)

`.venv/bin/python step6_reinforcement_finetune.py --after-sft`

**Why:** no expert this time. Gemma practises, and the careful manager decides
which attempts are worth learning from. `--after-sft` starts from step 5's
weights; without it, step 6 starts from untrained Gemma.

```mermaid
flowchart TB
    START[("out/sft_adapter<br/>copied to out/sft_rft_adapter")] --> Q0["quick exam: 30 tickets"]
    Q0 --> P

    subgraph ROUND["one round, done 3 times"]
        P["Gemma tries 60 new practice tickets<br/>with some randomness switched on"]
        P --> J{"careful manager:<br/>really done right,<br/>no rejected calls?"}
        J -->|no, and tries left| P
        J -->|no, 3 tries used| SKIP["give up on this ticket"]
        J -->|yes| KEEP["keep the attempt:<br/>Gemma's own PROMPT / OUTPUT<br/>pairs, thinking included"]
        KEEP --> DATA[("out/sft_rft_data/train.jsonl<br/>all kept attempts so far")]
        DATA --> T["local_trainer.py<br/>continues from the current weights"]
        T --> W[("out/sft_rft_adapter")]
        W --> QE["quick exam: 30 tickets,<br/>randomness off"]
    end

    QE -->|next round,<br/>new tickets| P
    W --> FIN["step 4 exam<br/>100/100"]
```

How one practice ticket is handled:

```mermaid
sequenceDiagram
    participant S as step6
    participant R as runner.py
    participant G as Gemma (LocalPolicy)
    participant E as env.py
    participant M as careful manager

    loop up to 3 tries
        S->>R: run_episode(ticket, Gemma)
        loop each step
            R->>G: act
            Note over G: writes thinking + tool call,<br/>saves both in self.trace
            G-->>R: tool call
            R->>E: step
        end
        E->>M: grade the ticket
        M-->>S: done right or not
        alt done right and no rejected calls
            Note over S: keep G's trace, stop trying
        else
            Note over S: throw it away, try again
        end
    end
```

The difference from step 5 is **who wrote the OUTPUT**:

| | Step 5 | Step 6 |
|---|---|---|
| Who wrote the thinking and tool calls | The expert | Gemma itself |
| Which tickets are kept | All of them (the expert is always right) | Only the ones the careful manager approves |
| Teacher | The expert's examples | The manager's grades |

What the rounds showed, starting from step 5's weights:

| | Practice tickets solved | Quick exam |
|---|---|---|
| start | | 29/30 |
| round 1 | 59/60 | 29/30 |
| round 2 | 60/60 | 29/30 |
| round 3 | 60/60 | 30/30 |

**Why step 5 had to come first:** run from untrained Gemma, step 6 scored 79.
Untrained Gemma never checked the customer, so none of its successful attempts
included a check, and there was nothing good to keep. Step 5 put the check into
Gemma; step 6 then had good attempts to reinforce.

---

## Step 7: the results table

`python3 step7_show_results.py`

```mermaid
flowchart LR
    C[("results/claude.json")] --> S7["step7<br/>reads every score card"]
    U[("results/gemma_untrained.json")] --> S7
    N[("results/gemma_sft_no_reasoning.json")] --> S7
    F[("results/gemma_sft.json")] --> S7
    R[("results/gemma_rft.json")] --> S7
    SR[("results/gemma_sft_then_rft.json")] --> S7
    S7 --> T["one table:<br/>score, score per ticket type,<br/>'said it, did not do it',<br/>wrongful refunds"]
```

The table is in the main [README](../README.md#results).

---

## The whole journey of Gemma

```mermaid
flowchart LR
    U["untrained Gemma<br/>78<br/>never checks the customer"]
    U -->|"step 5 without reasoning"| N["68<br/>checks, but stops thinking"]
    U -->|"step 6 alone"| R["79<br/>still never checks"]
    U -->|"step 5 with reasoning"| S["99<br/>checks and thinks"]
    S -->|"step 6 --after-sft"| SR["100<br/>matches Claude"]
```

In shop terms: first the trainee shadows an expert who explains every decision
out loud (step 5). Then the trainee works the desk alone, and the manager's
grades tell it which of its own decisions to keep doing (step 6).

Back to the [start of the docs](README.md).
