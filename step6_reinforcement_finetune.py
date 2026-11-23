"""Reinforcement fine-tuning: step 2's idea, with a language model as the trainee.

    .venv/bin/python step6_reinforcement_finetune.py           # 3 rounds, from the untrained model
    .venv/bin/python step6_reinforcement_finetune.py 1         # 1 round
    .venv/bin/python step6_reinforcement_finetune.py --after-sft   # 3 rounds, starting from step 5's weights

Each round takes 30-40 minutes. Then test it:
    .venv/bin/python step4_test_local_model.py 100 out/rft_adapter       # from untrained
    .venv/bin/python step4_test_local_model.py 100 out/sft_rft_adapter   # after step 5

No expert is used here: the only teacher is the careful manager's score. The
model keeps its thinking step, and the thinking is part of what it is trained on.

Reinforcement can only strengthen what the model already does some of the
time. If it never takes a step (say, checking the customer), run step 5 first
to teach that step, then use --after-sft.

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

from agents.local import END_OF_TURN, LocalPolicy
from refund_env import RefundDeskEnv, run_episode, verified_reward
from refund_env.exam import run_exam

HERE = pathlib.Path(__file__).parent
AFTER_SFT = "--after-sft" in sys.argv
SFT_ADAPTER = HERE / "out" / "sft_adapter"
ADAPTER = HERE / "out" / ("sft_rft_adapter" if AFTER_SFT else "rft_adapter")
DATA = HERE / "out" / ("sft_rft_data" if AFTER_SFT else "rft_data")

TICKETS_PER_ROUND = 60
ATTEMPTS = 3
EXPLORE_TEMP = 1.0             # randomness while practising; 0 during the exam
QUICK_EXAM_TICKETS = 30        # a progress check each round; step 4 runs the full 100
TRAIN_SEED_START = 10_000      # practice tickets; never overlap the exam's


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
            clean = all(s["result"]["ok"] for s in env.steps)
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
    iters = min(len(lines), 200)
    subprocess.run([sys.executable, "-m", "agents.local_trainer",
                    str(DATA / "train.jsonl"), str(ADAPTER), str(iters)],
                   check=True, capture_output=True, cwd=HERE)


if __name__ == "__main__":
    numbers = [a for a in sys.argv[1:] if a.isdigit()]
    rounds = int(numbers[0]) if numbers else 3
    shutil.rmtree(ADAPTER, ignore_errors=True)
    if AFTER_SFT:
        if not SFT_ADAPTER.exists():
            sys.exit("No step 5 weights in out/sft_adapter. Run step 5 first.")
        shutil.copytree(SFT_ADAPTER, ADAPTER)
        policy = LocalPolicy(adapter_path=str(ADAPTER))
    else:
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

    print(f"\nsaved fine-tuned weights to {ADAPTER.relative_to(HERE)}")
