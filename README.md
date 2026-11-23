# Refund desk: a demo RL environment

This project shows how to make a reinforcement learning (RL) environment. Then
it uses the environment to train a language model.

In each episode, an agent works one customer ticket. The agent uses tools on a
simulated order database. A reward function gives a score to the result.

The project uses these terms:

- **Environment:** the tickets, the order database and the tools.
- **Agent:** a small Python program, Claude, or a small model on your Mac.
- **Reward function:** the function that gives a score to each completed
  ticket.

For definitions of all terms and diagrams of each part, read
[docs/](docs/README.md).

## The task

The customer gives the name of an item, for example "I want a refund for my
kettle". The customer does not give the order number. The agent must do these
tasks:

1. Find the order.
2. Read the order record.
3. Check the customer.
4. Apply the refund policy.

The refund policy has 5 rules. Apply them in this sequence:

1. If the customer has a fraud flag, escalate the order to a human.
2. If the delivery was more than 30 days ago, deny the refund.
3. If the amount is more than $100, escalate the order to a human.
4. If the customer opened the item, refund half of the amount.
5. If no rule above applies, refund the full amount.

The tools are `list_orders`, `lookup_order`, `check_customer`, `issue_refund`,
`escalate` and `reply`. The `reply` tool ends the ticket.

The customer message can contain false statements about dates or about the
item. Only the records are correct.

## Setup

Steps 1, 2, 3 and 7 need only Python 3.10 or a later version. You do not
have to install packages.

Steps 4, 5, 6 and 8 need a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

| Step | Also necessary |
|---|---|
| 3 (optional) | The Claude Code command-line tool, with a login |
| 4, 5, 6, 8 | An Apple Silicon Mac with 16 GB of memory. The first run downloads the Gemma model (3.6 GB) |

## Files

```
refund_env/    THE ENVIRONMENT
  tasks.py       prerequisite: the synthetic tickets and their orders
  policy.py      prerequisite: the 5 rules and the correct answer for a ticket
  env.py         the order database, the tools, the end of an episode
  prompts.py     the rules and the observations as text, for a language model
  rewards.py     the two reward functions: naive_reward and verified_reward
  runner.py      the episode loop for one ticket; all scripts use it
  exam.py        the exam: the same 100 new tickets for each agent

agents/        THE AGENTS
  toy.py           an expert, a random agent and a small agent that learns (no model)
  claude_code.py   Claude, through the Claude Code command-line tool
  local.py         Gemma, a small model that runs on your Mac
  local_trainer.py the fine-tuning function for steps 5 and 6

step1 ... step8    the scripts, in the sequence to run them
tests/             one test for each type of environment defect
results/           the saved exam scores, one file for each agent
out/               files that the scripts make (not in git; you can delete them)
LESSONS.md         the problems that occurred during the work, and their causes
docs/              diagrams of all parts and how they connect
```

## The steps

| Step | Command | Purpose | Time |
|---|---|---|---|
| 1 | `python3 step1_watch_one_ticket.py` | Shows the expert on one ticket, from start to end | seconds |
| 2 | `python3 step2_compare_rewards.py` | Shows an agent that gets a high `naive_reward` without the correct actions. `verified_reward` prevents this | seconds |
| 3 | `python3 step3_test_claude.py` | Tests Claude | approximately 1 minute |
| 4 | `.venv/bin/python step4_test_local_model.py` | Tests untrained Gemma | approximately 1 minute |
| 5 | `.venv/bin/python step5_supervised_finetune.py` | Trains Gemma on the reasoning and the tool calls of the expert | approximately 25 minutes |
| 6 | `.venv/bin/python step6_reinforcement_finetune.py` | Trains Gemma with only the `verified_reward` score | approximately 2 hours |
| 6b | `.venv/bin/python step6_reinforcement_finetune.py --after-sft` | Does step 6, but starts from the step 5 weights | approximately 70 minutes |
| 7 | `python3 step7_show_results.py` | Shows all saved exam scores in one table | seconds |
| 8 | `.venv/bin/python step8_changed_rules.py 100 out/sft_adapter` | Does the exam with new rule numbers. It shows if Gemma learned the rules or memorised them | approximately 12 minutes for each agent |

By default, steps 3 and 4 do 6 tickets as a short test. Add `100` to do the
full exam. The full exam saves the score in `results/`:

```bash
.venv/bin/python step4_test_local_model.py 100                    # untrained
.venv/bin/python step4_test_local_model.py 100 out/sft_adapter    # after step 5
.venv/bin/python step4_test_local_model.py 100 out/rft_adapter    # after step 6
.venv/bin/python step4_test_local_model.py 100 out/sft_rft_adapter   # after step 6b
```

To run the tests, use this command:

```bash
python3 -m unittest discover tests -v
```

## Reading sequence

0. [docs/](docs/README.md): the diagrams. Read these first for an overview.
1. `refund_env/env.py`: the top of the file gives the parts of an
   environment, and the file for each part.
2. `refund_env/rewards.py`: the two reward functions, one after the other.
3. `refund_env/runner.py`: the episode loop, 15 lines.
4. `step2_compare_rewards.py`: the same agent, trained with each reward
   function.
5. `tests/test_env.py`: each test name gives one type of environment defect.
6. `step6_reinforcement_finetune.py`: the training loop for a real model.
7. `LESSONS.md`: the failed runs and what they showed.

## Results

All agents did the same 100 new tickets. The output of
`python3 step7_show_results.py` gives these tables. Each result is from one
run. For this reason, a difference of a small number of tickets is not important.

### The current environment

| Trainee | Done right | refunded_full | refunded_half | escalated_over_limit | escalated_flagged | denied | Said it, did not do it | Wrongful refunds |
|---|---|---|---|---|---|---|---|---|
| claude | 100/100 | 17/17 | 13/13 | 24/24 | 18/18 | 28/28 | 0 | 0 |
| gemma_untrained | 78/100 | 17/17 | 13/13 | 19/24 | 10/18 | 19/28 | 0 | 9 |
| gemma_sft_no_reasoning | 68/100 | 15/17 | 0/13 | 12/24 | 18/18 | 23/28 | 0 | 12 |
| gemma_sft | 99/100 | 17/17 | 13/13 | 24/24 | 18/18 | 27/28 | 0 | 0 |
| gemma_rft | 79/100 | 17/17 | 12/13 | 22/24 | 11/18 | 17/28 | 0 | 9 |
| gemma_sft_then_rft | 100/100 | 17/17 | 13/13 | 24/24 | 18/18 | 28/28 | 0 | 0 |

The column names are the output of step 7. "Said it, did not do it" counts
tickets with the correct reply but without the correct action. "Wrongful
refunds" counts refunds that the policy does not allow.

- **Claude:** All tickets are correct. Claude checks each customer.
- **Untrained Gemma:** Gemma does not call `check_customer`. For this reason, it gives
  refunds to customers with a fraud flag. It sends 9 refunds that the policy
  does not allow.
- **Supervised training without reasoning** (an earlier version of step 5):
  The examples of the expert contained only tool calls. Gemma copied the
  steps, so it checks each customer. But it stopped its reasoning, and its
  decisions became worse. For example, it escalated all opened items. The
  correct action is to refund half.
- **Supervised training with reasoning** (step 5 now): Each example has the
  reasoning of the expert before the tool call. An example is "Rule 1: not
  flagged. Rule 2: 26 days, within 30...". Gemma checks each customer and
  applies the rules. Its score is 99 of 100.
- **Reinforcement training from untrained Gemma** (step 6): The score changes
  only from 78 to 79. Gemma did not call `check_customer` during training.
  For this reason, there was no good example of this call to keep. Refer to lesson 10 in
  [LESSONS.md](LESSONS.md).
- **Supervised training, then reinforcement training** (step 6 with
  `--after-sft`): The score is 100 of 100. Supervised training taught the
  step that Gemma did not do. Then reinforcement training had good attempts to keep. Refer
  to lesson 12.

### Changed rules: did Gemma learn or memorise? (step 8)

All training tickets used the same two numbers: deny after 30 days, and
escalate above 10000 cents. Step 8 uses the same 100 exam tickets with new
rules: deny after **14** days, and escalate above **5000** cents. Step 8 does
not train Gemma again. The new rules change the correct answer on 36 tickets.
The results are in `results/changed_rules/`:

| Trainee | Done right | refunded_full | refunded_half | escalated_over_limit | escalated_flagged | denied | Said it, did not do it | Wrongful refunds | Answer changed: done right | Followed the old rules |
|---|---|---|---|---|---|---|---|---|---|---|
| gemma_untrained | 60/100 | 2/3 | 3/3 | 19/23 | 10/18 | 26/53 | 0 | 10 | 23/36 | 9/36 |
| gemma_sft | 100/100 | 3/3 | 3/3 | 23/23 | 18/18 | 53/53 | 0 | 0 | 36/36 | 0/36 |
| gemma_sft_then_rft | 100/100 | 3/3 | 3/3 | 23/23 | 18/18 | 53/53 | 0 | 0 | 36/36 | 0/36 |

The last two columns are for the 36 changed tickets. They count the tickets
with the correct answer under the new rules, and the tickets done with the old
rules.

- **Trained Gemma reads the new rules.** All 36 changed tickets are correct.
  No ticket uses the old rules. The reasoning of Gemma shows the cause:
  *"delivered 30 days ago, more than 14, so I deny."* Gemma takes the number
  from the rules. It does not use a memorised number.
- **Untrained Gemma:** Gemma does not check the customer. Its score decreases
  to 60, because more than half of the tickets are now denials. Gemma does
  not do denials well.
- **For comparison:** An expert that uses the old numbers gets 64. It does all
  36 changed tickets with the old rules. To see this result, run
  `python3 step8_changed_rules.py`.

This test changes only one type of item: the numbers in the same rules. A new
rule, or the same rules in a different sequence, is a more difficult test.

### The first, easier environment

In this version, the customer gave the order number. There were no fraud flags
and no opened items. The results are in `results/v1_easy/`:

| Trainee | Done right | refunded | denied | escalated | Said it, did not do it | Wrongful refunds |
|---|---|---|---|---|---|---|
| claude | 100/100 | 22/22 | 40/40 | 38/38 | 0 | 0 |
| gemma_untrained | 75/100 | 22/22 | 40/40 | 13/38 | 25 | 0 |
| gemma_sft | 96/100 | 20/22 | 40/40 | 36/38 | 0 | 1 |
| gemma_rft | 100/100 | 22/22 | 40/40 | 38/38 | 0 | 0 |

On this environment, reinforcement fine-tuning increased the score of Gemma
from 75 to 100.

### Time

These times are for an M4 Mac with 16 GB of memory:

| Task | Time |
|---|---|
| One full Gemma exam | 7 to 20 minutes. The time increases when Gemma writes more reasoning |
| Step 5 | Approximately 25 minutes |
| Step 6 | Approximately 2 hours |
| Step 6b | Approximately 70 minutes. Gemma completes most tickets on the first attempt, so training is shorter |

## Contents of out/

| Folder | Made by | Contents |
|---|---|---|
| `trajectories.jsonl` | step 1 | 50 episodes of the expert |
| `sft_data/` | step 5 | the examples of the expert for Gemma |
| `sft_adapter/` | step 5 | the Gemma weights after supervised fine-tuning |
| `rft_data/` | step 6 | the correct attempts of Gemma |
| `rft_adapter/` | step 6 | the Gemma weights after reinforcement fine-tuning |
| `sft_rft_data/` | step 6b | the same data, when step 6 starts from the step 5 weights |
| `sft_rft_adapter/` | step 6b | the Gemma weights after step 5 and then step 6 |

## Tests to try

- In `refund_env/policy.py`, use other numbers in `CHANGED_RULES`. Or change the
  sequence of rules 1 and 2, so that old orders get a denial before the fraud
  check. Update `expected_case`, the expert and the rules text. Then run step
  8 again.
- In `refund_env/rewards.py`, set `STEP_COST = 0`. Then run step 2 again. Does
  an agent still get a high `naive_reward` without the correct actions?
- Change the penalty for a wrongful refund from `-1.0` to `0.0`. What does the
  agent do with refunds during training?
- Make the environment more difficult. Give a customer two orders for the same
  item, so that the agent must use the date to select the order. Update the
  generator, the expert, the reward function and the tests. Then run the exams
  again.
