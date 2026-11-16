"""Reinforcement fine-tuning: step 2's idea, with a language model as the trainee.

    .venv/bin/python step6_reinforcement_finetune.py          # 3 rounds (about 20 minutes each)
    .venv/bin/python step6_reinforcement_finetune.py 1        # 1 round
Then test it:  .venv/bin/python step4_test_local_model.py 100 out/rft_adapter

Starts from the untrained model and saves the improved weights to
out/rft_adapter. No expert is used: the only teacher is the careful manager's
score. The model keeps its thinking step, and the thinking is part of what it
is trained on.

This is the simplest form of reinforcement fine-tuning: keep the attempts that
scored well and train the model to imitate them ("filtered behaviour cloning").
Production systems use more refined methods (for example, weighting every
attempt by its score), but the loop below is the core idea.

Each round:
  1. The model tries each ticket up to 3 times, with some randomness.
  2. verified_reward scores every attempt.
  3. The first attempt that really did the job is kept; failures are thrown away.
  4. The model is fine-tuned on every kept attempt so far.
  5. It sits a quick exam (the first 30 exam tickets), with randomness switched off.
"""
import collections
import json
import pathlib
import shutil
import subprocess
import sys

from mlx_lm.sample_utils import make_sampler

from agents.local import LocalPolicy
from refund_env import RefundDeskEnv, run_episode, verified_reward
from refund_env.exam import run_exam

HERE = pathlib.Path(__file__).parent
ADAPTER = HERE / "out" / "rft_adapter"  # where this script saves the weights
DATA = HERE / "out" / "rft_data"

TICKETS_PER_ROUND = 60
ATTEMPTS = 3
EXPLORE_TEMP = 1.0             # randomness while practising; 0 during the exam
QUICK_EXAM_TICKETS = 30        # a progress check each round; step 4 runs the full 100
TRAIN_SEED_START = 10_000      # practice tickets; never overlap the exam's
END_OF_TURN = "<turn|>"        # the token Gemma writes when it has finished speaking


def exam(policy):
    policy.sampler = make_sampler(temp=0.0)
    card = run_exam(policy, "quick_check", QUICK_EXAM_TICKETS, show=False)
    return f"{card['right']}/{QUICK_EXAM_TICKETS}"


def practise(policy, seeds):
    """Steps 1-3: retry each ticket until an attempt is good, and keep that one."""
    env = RefundDeskEnv(verified_reward)
    policy.sampler = make_sampler(temp=EXPLORE_TEMP)
    kept = []
    for seed in seeds:
        for _ in range(ATTEMPTS):
            reward, info = run_episode(env, policy, seed)
            clean = all(s["result"]["ok"] for s in env.steps)  # no rejected actions
            if info["true_success"] and clean:
                # policy.trace = what the model saw and wrote (thinking included)
                kept.append((info["expected"], list(policy.trace)))
                break
    return kept


def fine_tune(episodes):
    """Step 4: one example per decision: what the model saw, then what it wrote
    (thinking included). It is graded only on what it wrote."""
    DATA.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"prompt": prompt, "output": output + END_OF_TURN})
             for _, trace in episodes for prompt, output in trace]
    (DATA / "train.jsonl").write_text("\n".join(lines) + "\n")
    iters = min(len(lines), 200)   # about one pass, capped to keep rounds short
    subprocess.run([sys.executable, "-m", "agents.local_trainer",
                    str(DATA / "train.jsonl"), str(ADAPTER), str(iters)],
                   check=True, capture_output=True, cwd=HERE)


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    shutil.rmtree(ADAPTER, ignore_errors=True)

    policy = LocalPolicy()
    print(f"start     unseen tickets done right {exam(policy)}", flush=True)

    good_episodes = []  # every good attempt so far; each round trains on all of them
    for r in range(rounds):
        first = TRAIN_SEED_START + r * TICKETS_PER_ROUND
        kept = practise(policy, range(first, first + TICKETS_PER_ROUND))
        good_episodes += kept
        kinds = collections.Counter(kind for kind, _ in kept)
        del policy  # free memory while the fine-tuning process runs
        fine_tune(good_episodes)
        policy = LocalPolicy(adapter_path=str(ADAPTER))
        print(f"round {r + 1}   solved {len(kept)}/{TICKETS_PER_ROUND} practice tickets "
              f"(refunded {kinds['refunded']}, escalated {kinds['escalated']}, "
              f"denied {kinds['denied']})   "
              f"unseen tickets done right {exam(policy)}", flush=True)

    print("\nsaved fine-tuned weights to out/rft_adapter")
