"""Supervised fine-tuning: teach the small model by showing it the expert.

    .venv/bin/python step5_supervised_finetune.py

First it writes the expert's episodes as training examples.
Then it fine-tunes the model on them and saves the result to out/sft_adapter.
Then test it:  .venv/bin/python step4_test_local_model.py 100 out/sft_adapter
"""
import json
import pathlib
import subprocess
import sys

from agents.local import MODEL
from agents.toy import OraclePolicy
from refund_env import RefundDeskEnv, rollout, verified_reward

HERE = pathlib.Path(__file__).parent
DATA = HERE / "out" / "sft_data"
ADAPTER = HERE / "out" / "sft_adapter"


def write_examples(path, seeds):
    """One example per decision: the conversation so far -> the expert's next tool call."""
    env = RefundDeskEnv(verified_reward)
    count = 0
    with path.open("w") as f:
        for seed in seeds:
            messages = rollout(env, OraclePolicy(), seed)["messages"]
            for i, m in enumerate(messages):
                if m["role"] == "assistant":
                    f.write(json.dumps({"messages": messages[:i + 1]}) + "\n")
                    count += 1
    return count


if __name__ == "__main__":
    DATA.mkdir(parents=True, exist_ok=True)
    # Training tickets are seeds 0-449. The exam uses seeds from 1,000,000, so
    # the model is never tested on a ticket it studied.
    n_train = write_examples(DATA / "train.jsonl", range(0, 400))
    n_valid = write_examples(DATA / "valid.jsonl", range(400, 450))
    print(f"wrote {n_train} training examples and {n_valid} validation examples to out/sft_data\n")

    subprocess.run([
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--train",
        "--data", str(DATA),
        "--adapter-path", str(ADAPTER),
        "--mask-prompt",            # only grade the model on the tool call, not the prompt
        # Small settings so a 2-billion-parameter model fits in 16 GB of memory.
        "--batch-size", "1",
        "--num-layers", "8",
        "--grad-checkpoint",
        "--iters", "600",
        "--learning-rate", "5e-5",
        "--steps-per-report", "100",
        "--steps-per-eval", "200",
        "--val-batches", "20",
        "--save-every", "600",      # keep only the final weights, to save disk
        "--max-seq-length", "1024",
    ], check=True)
    print("\nsaved fine-tuned weights to out/sft_adapter")
