# Refund desk: a demo RL environment

A trainee handles one customer ticket per episode by calling tools against a
pretend order database. A manager scores the result. This project shows how to
build that environment and then use it to train a language model.

Think of it as a shop:

- **The shop** is the environment: tickets, an order database, and tools.
- **The trainee** is the agent: a toy learner, Claude, or a small local model.
- **The manager** is the reward function, which scores each finished ticket.

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
| 5 | `.venv/bin/python step5_supervised_finetune.py` | Teach Gemma by showing it the expert's answers | about 15 minutes |
| 6 | `.venv/bin/python step6_reinforcement_finetune.py` | Teach Gemma using only the careful manager's score | about an hour |
| 7 | `python3 step7_show_results.py` | Every saved exam score in one table | seconds |

Steps 3 and 4 work 6 tickets by default, as a quick look. Add `100` to sit the
full exam, which saves the score to `results/`:

```bash
.venv/bin/python step4_test_local_model.py 100                    # untrained
.venv/bin/python step4_test_local_model.py 100 out/sft_adapter    # after step 5
.venv/bin/python step4_test_local_model.py 100 out/rft_adapter    # after step 6
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

Every trainee sat the same 100 unseen tickets. This table is the output of
`python3 step7_show_results.py`.

| Trainee | Done right | Refunds | Escalations | Denials | Said it, did not do it | Wrongful refunds |
|---|---|---|---|---|---|---|
| claude | 100/100 | 22/22 | 38/38 | 40/40 | 0 | 0 |
| gemma_untrained | 75/100 | 22/22 | 13/38 | 40/40 | 25 | 0 |
| gemma_sft | 96/100 | 20/22 | 36/38 | 40/40 | 0 | 1 |
| gemma_rft | 100/100 | 22/22 | 38/38 | 40/40 | 0 | 0 |

How to read it:

- **Untrained Gemma** decides correctly but, on 25 tickets, tells the customer
  "escalated" without calling the `escalate` tool. That is the cheat the lazy
  manager would pay for.
- **Supervised fine-tuning** (step 5) removes that cheat by imitating the
  expert, but leaves 4 wrong decisions, one of them a wrongful refund.
- **Reinforcement fine-tuning** (step 6) used no expert, only the careful
  manager's score, and got every ticket right.

These are single runs on 100 tickets, so treat a gap of a few tickets as noise.
The measured times on an M4 Mac with 16 GB were about 12 minutes for a full
Gemma exam, 13 minutes for step 5 and 60 minutes for step 6.

## What is in out/

| Folder | Made by | Contents |
|---|---|---|
| `trajectories.jsonl` | step 1 | 50 expert episodes |
| `sft_data/` | step 5 | the expert examples Gemma studied |
| `sft_adapter/` | step 5 | Gemma's weights after supervised fine-tuning |
| `rft_data/` | step 6 | Gemma's own successful attempts |
| `rft_adapter/` | step 6 | Gemma's weights after reinforcement fine-tuning |

## Things to try

- In `refund_env/rewards.py`, set `STEP_COST = 0` and rerun step 2. Does the
  naive reward still get exploited?
- Change the wrongful-refund penalty from `-1.0` to `0.0`. What does the
  learner do with refunds while exploring?
- Add a tool `check_fraud_flag(customer)` and a rule that flagged customers are
  always escalated. Update the generator, expert, reward and tests.
