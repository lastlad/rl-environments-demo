"""Fine-tune on (prompt, output) pairs, grading the model only on the output.

    python -m agents.local_trainer DATA.jsonl ADAPTER_DIR ITERS

Each line of DATA.jsonl is {"prompt": ..., "output": ...}, the exact text the
model saw and the exact text it wrote, thinking included. The prompt is masked:
the model is not trained to reproduce it, only to write the output after it.

MLX's built-in trainer cannot do this for Gemma, because Gemma's chat template
deletes the thinking from past turns. So this file hands MLX the text directly.
"""
import json
import pathlib
import sys
import types

from mlx_lm import load
from mlx_lm.lora import CONFIG_DEFAULTS, train_model

from .local import MODEL


class PromptOutputDataset:
    def __init__(self, rows, tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]

    def process(self, row):
        tokens = self.tokenizer.encode(row["prompt"] + row["output"])
        prompt_length = len(self.tokenizer.encode(row["prompt"]))
        return tokens, prompt_length  # everything before prompt_length is not graded


if __name__ == "__main__":
    data, adapter, iters = sys.argv[1], pathlib.Path(sys.argv[2]), int(sys.argv[3])
    rows = [json.loads(line) for line in open(data)]
    model, tokenizer = load(MODEL)
    resume = adapter / "adapters.safetensors"
    args = types.SimpleNamespace(**{
        **CONFIG_DEFAULTS,
        "model": MODEL,
        "train": True,
        "adapter_path": str(adapter),
        "resume_adapter_file": str(resume) if resume.exists() else None,
        "mask_prompt": True,
        # Small settings so the model fits in 16 GB of memory (same as step5_supervised_finetune.py).
        "batch_size": 1,
        "num_layers": 8,
        "grad_checkpoint": True,
        "iters": iters,
        "learning_rate": 2e-5,     # small steps: big ones made the model lurch
        "val_batches": 1,
        "steps_per_eval": 100000,
        "save_every": 100000,
        "max_seq_length": 1536,
    })
    train_model(args, model,
                PromptOutputDataset(rows, tokenizer),
                PromptOutputDataset(rows[:2], tokenizer))
