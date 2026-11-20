"""Supervised fine-tuning: teach Gemma by showing it the expert at work.

    .venv/bin/python step5_supervised_finetune.py
Then test it:  .venv/bin/python step4_test_local_model.py 100 out/sft_adapter

Each training example is one step of one ticket: exactly what Gemma would see
at that point, then what the expert wrote. The expert writes its reasoning
first, in Gemma's own thinking format, then the tool call. So Gemma learns to
think the rules through, not only which tool comes next.

(An earlier version showed only the expert's tool calls. Gemma copied the
steps but stopped thinking, and its decisions got worse. See LESSONS.md.)
"""
import json
import pathlib
import shutil
import subprocess
import sys

from mlx_lm import load

from agents.local import END_OF_TURN, MODEL, THINK_END, THINK_START
from agents.toy import OraclePolicy
from refund_env import RefundDeskEnv, render_prompt, run_episode, verified_reward
from refund_env.env import SYSTEM_PROMPT

HERE = pathlib.Path(__file__).parent
DATA = HERE / "out" / "sft_data"
ADAPTER = HERE / "out" / "sft_adapter"
ITERS = 800               # training steps, one example each
LEARNING_RATE = 5e-5


class RecordedExpert:
    """The expert at the desk, recorded the way Gemma sees and answers each step."""

    def __init__(self, tokenizer):
        self.expert = OraclePolicy()
        self.tokenizer = tokenizer

    def reset(self):
        self.expert.reset()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.examples = []

    def act(self, obs):
        self.messages.append({"role": "user", "content": render_prompt(obs)})
        # The same prompt text Gemma gets in step 4 (agents/local.py).
        prompt = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=False)
        action = self.expert.act(obs)
        answer = json.dumps(action)
        output = f"{THINK_START}{self.expert.thought}{THINK_END}{answer}{END_OF_TURN}"
        self.examples.append({"prompt": prompt, "output": output})
        self.messages.append({"role": "assistant", "content": answer})
        return action


def write_examples(path, seeds, tokenizer):
    env = RefundDeskEnv(verified_reward)
    expert = RecordedExpert(tokenizer)
    count = 0
    with path.open("w") as f:
        for seed in seeds:
            run_episode(env, expert, seed)
            for example in expert.examples:
                f.write(json.dumps(example) + "\n")
                count += 1
    return count


if __name__ == "__main__":
    shutil.rmtree(DATA, ignore_errors=True)
    shutil.rmtree(ADAPTER, ignore_errors=True)   # start from the untrained model
    DATA.mkdir(parents=True)
    _, tokenizer = load(MODEL)
    # Training tickets are seeds 0-399. The exam uses seeds from 1,000,000, so
    # the model is never tested on a ticket it studied.
    n = write_examples(DATA / "train.jsonl", range(0, 400), tokenizer)
    print(f"wrote {n} training examples to out/sft_data/train.jsonl")
    print("one of them, as Gemma would write it:\n")
    print(json.loads((DATA / "train.jsonl").read_text().splitlines()[3])["output"], "\n")

    print(f"training for {ITERS} steps...", flush=True)
    subprocess.run([sys.executable, "-m", "agents.local_trainer",
                    str(DATA / "train.jsonl"), str(ADAPTER), str(ITERS), str(LEARNING_RATE)],
                   check=True, cwd=HERE)
    print("\nsaved fine-tuned weights to out/sft_adapter")
