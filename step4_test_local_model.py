"""Let a small open model work the refund desk, graded by the careful manager.

    .venv/bin/python step4_test_local_model.py        # 6 tickets
    .venv/bin/python step4_test_local_model.py 50     # 50 tickets
    .venv/bin/python step4_test_local_model.py 50 out/sft_adapter   # the fine-tuned model

Runs entirely on this Mac. No API key, no usage limits.
"""
import json
import sys

from refund_env import RefundDeskEnv, verified_reward
from agents.local import LocalPolicy

EVAL_SEED_START = 1_000_000  # the same unseen tickets step2_compare_rewards.py and step3_test_claude.py use


def show(action):
    return json.dumps(action) if isinstance(action, dict) else f"NOT A TOOL CALL: {action!r}"


def main(n_tickets, adapter=None):
    env = RefundDeskEnv(verified_reward)
    policy = LocalPolicy(adapter_path=adapter)
    wins = 0
    for i in range(n_tickets):
        obs = env.reset(EVAL_SEED_START + i)
        policy.reset()
        print(f"\nTicket {i + 1}: {obs['ticket']['message']}")
        while True:
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            result = obs["last_result"]
            note = "" if result["ok"] else f"   <- rejected: {result['error']}"
            print(f"  {show(action)}{note}")
            if terminated or truncated:
                break
        wins += info["true_success"]
        verdict = "ok" if info["true_success"] else f"FAIL (should be {info['expected']})"
        print(f"  => {verdict}   reward {reward:+.2f}")

    name = policy.model_name + (f" + {adapter}" if adapter else "")
    print(f"\n{name}: {wins}/{n_tickets} jobs really done right")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         sys.argv[2] if len(sys.argv) > 2 else None)
