"""The loop that every script in this project uses to work one ticket.

Whoever the trainee is (the toy learner, Claude, Gemma), an episode is always:
reset the environment, then act and step until the episode ends.
"""
from __future__ import annotations

import json

from .prompts import SYSTEM_PROMPT, render_prompt


def run_episode(env, policy, seed, on_step=None):
    """Work one ticket from start to finish. Returns (reward, info).

    on_step, if given, is called after every action as
    on_step(what the trainee saw, what it did, what the tool answered).
    """
    obs = env.reset(seed)
    policy.reset()
    while True:
        action = policy.act(obs)
        seen = obs
        obs, reward, terminated, truncated, info = env.step(action)
        if on_step:
            on_step(seen, action, obs["last_result"])
        if terminated or truncated:
            return reward, info


def rollout(env, policy, seed):
    """Work one ticket and return it as chat messages plus its score.

    This is the record you keep for fine-tuning: each "user" message is what
    the trainee saw, each "assistant" message is the tool call it made.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def record(seen, action, result):
        messages.append({"role": "user", "content": render_prompt(seen)})
        messages.append({"role": "assistant", "content": json.dumps(action)})

    reward, info = run_episode(env, policy, seed, on_step=record)
    return {"task_id": env.task.task_id, "messages": messages, "reward": reward, **info}
