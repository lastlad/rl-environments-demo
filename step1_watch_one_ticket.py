"""Walk through one episode, then save trajectories for fine-tuning.

    python3 step1_watch_one_ticket.py
"""
import json
import pathlib

from refund_env import RefundDeskEnv, render_prompt, verified_reward
from refund_env.env import SYSTEM_PROMPT
from agents.toy import OraclePolicy


def rollout(env, policy, seed):
    """Play one episode and return it as chat messages plus its score.

    This is the record you would keep for fine-tuning: filter to high-reward
    episodes for supervised fine-tuning, or keep every episode with its reward
    for a policy-gradient method.
    """
    obs = env.reset(seed)
    policy.reset()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    while True:
        messages.append({"role": "user", "content": render_prompt(obs)})
        action = policy.act(obs)
        messages.append({"role": "assistant", "content": json.dumps(action)})
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return {"task_id": env.task.task_id, "messages": messages, "reward": reward,
                    "truncated": truncated, **info}


if __name__ == "__main__":
    env = RefundDeskEnv(verified_reward)
    traj = rollout(env, OraclePolicy(), seed=7)
    for m in traj["messages"][1:]:
        print(f"[{m['role']:9s}] {m['content']}")
    print(f"\nreward {traj['reward']:+.2f}   true_success {traj['true_success']}   "
          f"expected {traj['expected']}")

    out = pathlib.Path(__file__).parent / "out" / "trajectories.jsonl"
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for seed in range(50):
            f.write(json.dumps(rollout(env, OraclePolicy(), seed)) + "\n")
    print(f"\nwrote 50 trajectories to {out.relative_to(pathlib.Path(__file__).parent)}")
