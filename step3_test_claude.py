"""Let Claude work the refund desk and grade it with the careful manager.

    .venv/bin/python step3_test_claude.py        # 6 tickets
    .venv/bin/python step3_test_claude.py 20     # 20 tickets

With no API key set, this uses your Claude Code login (your subscription).
If ANTHROPIC_API_KEY is set, it uses the API and bills API credit instead.
Each ticket is 2-3 model calls either way.
"""
import json
import os
import sys

import anthropic

from refund_env import RefundDeskEnv, verified_reward
from agents.claude import ClaudeLoginPolicy, ClaudePolicy

EVAL_SEED_START = 1_000_000  # the same unseen tickets step2_compare_rewards.py uses for its exam


def show(action):
    return json.dumps(action) if isinstance(action, dict) else f"NOT A TOOL CALL: {action!r}"


def main(n_tickets):
    env = RefundDeskEnv(verified_reward)
    use_api = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    policy = ClaudePolicy() if use_api else ClaudeLoginPolicy()
    print("Using", "the Anthropic API (API credit)" if use_api else "your Claude Code login (subscription)")
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

    print(f"\n{policy.model}: {wins}/{n_tickets} jobs really done right")
    print(f"tokens used: {policy.input_tokens:,} in, {policy.output_tokens:,} out")


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    except RuntimeError as e:
        sys.exit(str(e))
    except anthropic.AuthenticationError:
        sys.exit("No valid API key. Set ANTHROPIC_API_KEY and try again.")
    except anthropic.RateLimitError:
        sys.exit("Rate limited by the API. Wait a minute and try again.")
    except anthropic.APIStatusError as e:
        sys.exit(f"API error {e.status_code}: {e.message}")
    except anthropic.APIConnectionError:
        sys.exit("Could not reach the API. Check your internet connection.")
