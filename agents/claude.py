"""A real language model in the trainee's seat.

Same interface as the other policies (reset / act), so the environment and the
reward functions do not change at all. The model sees only what the env shows:
the system prompt, then the ticket, then each tool result, as text.
"""
from __future__ import annotations

import anthropic

from refund_env.env import SYSTEM_PROMPT, parse_action, render_prompt

MODEL = "claude-haiku-5-5"


class ClaudePolicy:
    def __init__(self, client: anthropic.Anthropic | None = None, model: str = MODEL,
                 system_prompt: str = SYSTEM_PROMPT):
        # Anthropic() reads ANTHROPIC_API_KEY from the environment.
        self.client = client or anthropic.Anthropic()
        self.model = model
        self.system_prompt = system_prompt   # the rulebook
        self.messages = []
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self):
        self.messages = []  # each ticket is a fresh conversation

    def act(self, obs):
        self.messages.append({"role": "user", "content": render_prompt(obs)})
        response = self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=self.system_prompt,
            output_config={"effort": "low"},  # the task is simple; keeps cost down
            # If the model declines a request, the API retries it on a fallback model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=self.messages,
        )
        self.input_tokens += response.usage.input_tokens
        self.output_tokens += response.usage.output_tokens

        # Keep the whole reply (including any thinking blocks) in the history.
        self.messages.append({"role": "assistant", "content": response.content})

        text = "".join(b.text for b in response.content if b.type == "text").strip()
        # Whatever the model wrote goes to the env as-is. If it is not a valid
        # tool call (prose, a code fence, a refusal), the env counts it as an
        # invalid action. That is deliberate: a real model can be wrong in ways
        # the toy trainee could not.
        return parse_action(text)


class ClaudeLoginPolicy:
    """Same trainee, but paid for by your Claude subscription instead of API credit.

    It runs the Claude Code command-line tool (`claude -p`) once per step, which
    uses the account you are logged in to. No API key is needed. All of Claude
    Code's own tools and settings are switched off, so the model can only answer
    with text, exactly like ClaudePolicy.
    """

    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT):
        import os
        import shutil
        self.cli = shutil.which("claude") or os.path.expanduser("~/.claude/local/claude")
        if not os.path.exists(self.cli):
            raise RuntimeError("Claude Code CLI not found. Install it and run `claude` once to log in.")
        self.model = model
        self.system_prompt = system_prompt   # the rulebook
        self.turns = []
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self):
        self.turns = []

    def act(self, obs):
        import json
        import subprocess
        self.turns.append(render_prompt(obs))
        # Each call is a fresh process with no memory, so send the whole
        # episode so far every time.
        prompt = "\n\n".join(self.turns) + "\n\nReply with your next tool call."
        proc = subprocess.run(
            [self.cli, "-p", prompt,
             "--system-prompt", self.system_prompt,
             "--model", self.model,
             "--tools", "",                 # no Claude Code tools: text only
             "--strict-mcp-config",         # no connectors
             "--setting-sources", "",       # ignore user/project settings
             "--no-session-persistence",
             "--output-format", "json"],
            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)
        try:
            out = json.loads(proc.stdout)
        except ValueError:
            raise RuntimeError(f"claude CLI failed: {(proc.stderr or proc.stdout).strip()[:500]}")
        if out.get("is_error"):
            raise RuntimeError(f"claude CLI error: {out.get('result')}")
        usage = out.get("usage", {})
        self.input_tokens += (usage.get("input_tokens", 0) + usage.get("cache_read_input_tokens", 0)
                              + usage.get("cache_creation_input_tokens", 0))
        self.output_tokens += usage.get("output_tokens", 0)
        text = (out.get("result") or "").strip()
        self.turns.append(f"Your tool call: {text}")
        return parse_action(text)
