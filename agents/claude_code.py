"""Claude as an agent, through the Claude Code command-line tool.

The agent runs `claude -p` once for each step. The tool uses the account that
you logged in to, so you do not need an API key.

The model sees only what the environment shows: the rules, the ticket and each
tool result, as text.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess

from refund_env.prompts import SYSTEM_PROMPT, parse_action, render_prompt

MODEL = "claude-haiku-5-5"


class ClaudeCodePolicy:
    def __init__(self, model: str = MODEL, system_prompt: str = SYSTEM_PROMPT):
        self.cli = shutil.which("claude") or os.path.expanduser("~/.claude/local/claude")
        if not os.path.exists(self.cli):
            raise RuntimeError("Claude Code CLI not found. Install it and run `claude` once to log in.")
        self.model = model
        self.system_prompt = system_prompt
        self.turns = []
        self.input_tokens = 0
        self.output_tokens = 0

    def reset(self):
        self.turns = []

    def act(self, obs):
        self.turns.append(render_prompt(obs))
        prompt = "\n\n".join(self.turns) + "\n\nReply with your next tool call."
        proc = subprocess.run(
            [self.cli, "-p", prompt,
             "--system-prompt", self.system_prompt,
             "--model", self.model,
             "--tools", "",
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
        # Text that is not a valid tool call (prose, a code fence, a refusal)
        # goes to the environment unchanged, which rejects it and counts the
        # step. A real model can make mistakes that the scripted agents cannot.
        return parse_action(text)
