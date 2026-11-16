"""A small open-weight model in the trainee's seat, running on this Mac.

Same interface as the other policies (reset / act). Unlike Claude, this model's
weights are on your disk, so it is one you can fine-tune.
"""
from __future__ import annotations

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from refund_env.env import SYSTEM_PROMPT, parse_action, render_prompt

MODEL = "mlx-community/gemma-4-e2b-it-4bit"


class LocalPolicy:
    def __init__(self, model: str = MODEL, adapter_path: str | None = None, temp: float = 0.0):
        # adapter_path: fine-tuned weights to load on top of the base model.
        # temp 0 = always pick the most likely word (use this for testing).
        self.model_name = model
        self.model, self.tokenizer = load(model, adapter_path=adapter_path)
        self.sampler = make_sampler(temp=temp)
        self.messages = []
        self.trace = []  # (prompt, full output including thinking) for each step

    def reset(self):
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.trace = []

    def act(self, obs):
        self.messages.append({"role": "user", "content": render_prompt(obs)})
        prompt = self.tokenizer.apply_chat_template(
            self.messages, add_generation_prompt=True, tokenize=False)
        text = generate(self.model, self.tokenizer, prompt=prompt,
                        max_tokens=600, sampler=self.sampler)
        self.trace.append((prompt, text))
        # Gemma may think out loud first, inside <|channel>thought ... <channel|>.
        # That is its scratch paper; only what comes after it is the answer.
        text = text.split("<channel|>")[-1].strip()
        self.messages.append({"role": "assistant", "content": text})
        return parse_action(text)
