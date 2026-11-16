# Refund desk: a demo RL environment

A trainee handles one customer ticket per episode by calling tools against a
pretend order database. A manager scores the result. This project shows how to
build that environment and then use it to train a language model.

## What is where

```
refund_env/    THE SHOP: the environment itself
  env.py         tickets, the order database, the tools, when an episode ends
  rewards.py     the two managers: naive_reward (lazy) and verified_reward (careful)

agents/        THE TRAINEES: who can sit at the desk
  toy.py           an expert, a random clicker, and a tiny learner (no AI model)
  claude.py        Claude
  local.py         Gemma, a small model that runs on this Mac
  local_trainer.py the fine-tuning helper used by step 6

step1 ... step6    scripts to run, in learning order (see below)
tests/             one test per way an environment can quietly go wrong
out/               files the scripts create (safe to delete; they get rebuilt)
```

## The steps, in order

| Step | Command | What it shows | Time |
|---|---|---|---|
| 1 | `python3 step1_watch_one_ticket.py` | One ticket handled by the expert, start to finish | seconds |
| 2 | `python3 step2_compare_rewards.py` | The lazy manager gets cheated; the careful one does not | seconds |
| 3 | `.venv/bin/python step3_test_claude.py` | Claude at the desk (uses your Claude login) | 1-2 min |
| 4 | `.venv/bin/python step4_test_local_model.py 30` | Untrained Gemma at the desk | ~5 min |
| 5 | `.venv/bin/python step5_supervised_finetune.py` | Teach Gemma by showing it the expert's answers | ~15 min |
| 6 | `.venv/bin/python step6_reinforcement_finetune.py` | Teach Gemma using only the careful manager's score | ~1 hour |

To test a trained model, give step 4 the folder holding its weights:

```bash
.venv/bin/python step4_test_local_model.py 30 out/sft_adapter   # after step 5
.venv/bin/python step4_test_local_model.py 30 out/rft_adapter   # after step 6
```

Run the tests with:

```bash
python3 -m unittest discover tests -v
```

## Results so far (Gemma 4 E2B, unseen tickets)

| Model | Done right |
|---|---|
| Untrained | 23 of 30 |
| After step 5 (supervised) | 42 of 50 |
| After step 6 (reinforcement) | 59 of 60 |

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
