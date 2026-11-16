"""Watch the expert handle one ticket, then save 50 of its episodes.

    python3 step1_watch_one_ticket.py
"""
import json
import pathlib

from agents.toy import OraclePolicy
from refund_env import RefundDeskEnv, rollout, verified_reward

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
