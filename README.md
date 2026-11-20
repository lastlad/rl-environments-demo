# Refund desk: a demo RL environment

A trainee handles one customer ticket per episode by calling tools against a
pretend order database. A manager scores the result. This project shows how to
build that environment and then use it to train a language model.

Think of it as a shop:

- **The shop** is the environment: tickets, an order database, and tools.
- **The trainee** is the agent: a toy learner, Claude, or a small local model.
- **The manager** is the reward function, which scores each finished ticket.

## The task

The customer names an item ("I want a refund for my kettle"), not an order
number. The trainee has to find the order, look it up, check the customer, and
then apply the shop's policy, in this order:

1. Customer flagged for fraud: escalate to a human.
2. Delivered more than 30 days ago: deny.
3. Amount over $100: escalate to a human.
4. Item was opened: refund half.
5. Otherwise: refund in full.

Tools: `list_orders`, `lookup_order`, `check_customer`, `issue_refund`,
`escalate`, and `reply`, which ends the ticket. What the customer says about
dates or opening the item is not reliable; only the records are.

## Setup

Steps 1 and 2 need only Python 3.10 or newer. Nothing to install.

Steps 3 to 6 need a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

| Step | Also needs |
|---|---|
| 3 (optional) | The Claude Code command-line tool, logged in, or an `ANTHROPIC_API_KEY` |
| 4, 5, 6 | An Apple Silicon Mac with 16 GB of memory. The first run downloads the Gemma model (3.6 GB) |

## What is where

```
refund_env/    THE SHOP: the environment
  env.py         tickets, the order database, the tools, when an episode ends
  rewards.py     the two managers: naive_reward (lazy) and verified_reward (careful)
  runner.py      the loop that works one ticket; every script uses it
  exam.py        the exam: the same 100 unseen tickets for every trainee

agents/        THE TRAINEES: who can sit at the desk
  toy.py           an expert, a random clicker, and a tiny learner (no AI model)
  claude.py        Claude
  local.py         Gemma, a small model that runs on your Mac
  local_trainer.py the fine-tuning helper used by step 6

step1 ... step7    scripts to run, in learning order
tests/             one test per way an environment can quietly go wrong
results/           saved exam scores, one file per trainee
out/               files the scripts create (not in git; safe to delete)
LESSONS.md         what went wrong while building this, and why
```

## The steps, in order

| Step | Command | What it shows | Time |
|---|---|---|---|
| 1 | `python3 step1_watch_one_ticket.py` | One ticket handled by the expert, start to finish | seconds |
| 2 | `python3 step2_compare_rewards.py` | The lazy manager gets cheated; the careful one does not | seconds |
| 3 | `.venv/bin/python step3_test_claude.py` | Claude at the desk | about a minute |
| 4 | `.venv/bin/python step4_test_local_model.py` | Untrained Gemma at the desk | about a minute |
| 5 | `.venv/bin/python step5_supervised_finetune.py` | Teach Gemma by showing it the expert's reasoning and answers | about 25 minutes |
| 6 | `.venv/bin/python step6_reinforcement_finetune.py` | Teach Gemma using only the careful manager's score | about 2 hours |
| 6b | `.venv/bin/python step6_reinforcement_finetune.py --after-sft` | The same, starting from step 5's weights | about 70 minutes |
| 7 | `python3 step7_show_results.py` | Every saved exam score in one table | seconds |

Steps 3 and 4 work 6 tickets by default, as a quick look. Add `100` to sit the
full exam, which saves the score to `results/`:

```bash
.venv/bin/python step4_test_local_model.py 100                    # untrained
.venv/bin/python step4_test_local_model.py 100 out/sft_adapter    # after step 5
.venv/bin/python step4_test_local_model.py 100 out/rft_adapter    # after step 6
.venv/bin/python step4_test_local_model.py 100 out/sft_rft_adapter   # after step 6b
```

Run the tests with:

```bash
python3 -m unittest discover tests -v
```

## Reading order

1. `refund_env/env.py`: the top of the file lists the parts every environment has.
2. `refund_env/rewards.py`: the two managers, side by side.
3. `refund_env/runner.py`: the episode loop, 15 lines.
4. `step2_compare_rewards.py`: the same learner trained under each manager.
5. `tests/test_env.py`: each test names one way an environment can go wrong.
6. `step6_reinforcement_finetune.py`: the training loop for a real model.
7. `LESSONS.md`: the failed runs and what they taught.

## Results

Every trainee sat the same 100 unseen tickets. These tables are the output of
`python3 step7_show_results.py`. Each was a single run, so treat a gap of a few
tickets as noise.

### The current shop

| Trainee | Done right | refunded_full | refunded_half | escalated_over_limit | escalated_flagged | denied | Said it, did not do it | Wrongful refunds |
|---|---|---|---|---|---|---|---|---|
| claude | 100/100 | 17/17 | 13/13 | 24/24 | 18/18 | 28/28 | 0 | 0 |
| gemma_untrained | 78/100 | 17/17 | 13/13 | 19/24 | 10/18 | 19/28 | 0 | 9 |
| gemma_sft_no_reasoning | 68/100 | 15/17 | 0/13 | 12/24 | 18/18 | 23/28 | 0 | 12 |
| gemma_sft | 99/100 | 17/17 | 13/13 | 24/24 | 18/18 | 27/28 | 0 | 0 |
| gemma_rft | 79/100 | 17/17 | 12/13 | 22/24 | 11/18 | 17/28 | 0 | 9 |
| gemma_sft_then_rft | 100/100 | 17/17 | 13/13 | 24/24 | 18/18 | 28/28 | 0 | 0 |

How to read it:

- **Claude** gets everything right and checks every customer.
- **Untrained Gemma** never once calls `check_customer`. Flagged customers get
  refunded, and 9 refunds go out that the policy did not allow.
- **Supervised, no reasoning** (an earlier version of step 5): the expert's
  examples had only tool calls. Gemma copied the steps, so it checks every
  customer, but it stopped thinking and its decisions got worse: it escalated
  every opened item instead of refunding half.
- **Supervised, with reasoning** (step 5 now): each example has the expert's
  reasoning ("Rule 1: not flagged. Rule 2: 26 days, within 30...") before the
  tool call. Gemma checks every customer *and* applies the rules: 99 of 100.
- **Reinforcement from untrained Gemma** (step 6) barely moves the score (78 to
  79). Gemma never called `check_customer` in practice, so there was no good
  example of checking to keep. See lesson 10 in [LESSONS.md](LESSONS.md).
- **Supervised, then reinforcement** (step 6 with `--after-sft`) fixes the last
  ticket: 100 of 100. Supervised training taught the step that was missing;
  reinforcement then had good attempts to keep. See lesson 12.

### The first, easier shop

The customer gave the order number, and there were no fraud flags or opened
items. Saved in `results/v1_easy/`:

| Trainee | Done right | refunded | denied | escalated | Said it, did not do it | Wrongful refunds |
|---|---|---|---|---|---|---|
| claude | 100/100 | 22/22 | 40/40 | 38/38 | 0 | 0 |
| gemma_untrained | 75/100 | 22/22 | 40/40 | 13/38 | 25 | 0 |
| gemma_sft | 96/100 | 20/22 | 40/40 | 36/38 | 0 | 1 |
| gemma_rft | 100/100 | 22/22 | 40/40 | 38/38 | 0 | 0 |

On that shop, reinforcement fine-tuning took Gemma from 75 to 100.

### Time taken

On an M4 Mac with 16 GB: a full Gemma exam took 7 to 20 minutes (longer when
Gemma thinks before each step), step 5 took about 25 minutes, step 6 about 2
hours, and step 6b about 70 minutes (it solves most tickets on the first try,
so practice is quicker).

## What is in out/

| Folder | Made by | Contents |
|---|---|---|
| `trajectories.jsonl` | step 1 | 50 expert episodes |
| `sft_data/` | step 5 | the expert examples Gemma studied |
| `sft_adapter/` | step 5 | Gemma's weights after supervised fine-tuning |
| `rft_data/` | step 6 | Gemma's own successful attempts |
| `rft_adapter/` | step 6 | Gemma's weights after reinforcement fine-tuning |
| `sft_rft_data/` | step 6b | the same, when starting from step 5's weights |
| `sft_rft_adapter/` | step 6b | Gemma's weights after step 5 and then step 6 |

## Things to try

- In `refund_env/rewards.py`, set `STEP_COST = 0` and rerun step 2. Does the
  naive reward still get exploited?
- Change the wrongful-refund penalty from `-1.0` to `0.0`. What does the
  learner do with refunds while exploring?
- Make the shop harder again: give a customer two orders of the same item, so
  the trainee has to pick by date. Update the generator, expert, reward and
  tests, then re-run the exams.
