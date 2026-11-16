"""Let Claude work the refund desk and grade it with the careful manager.

    .venv/bin/python step3_test_claude.py        # a quick look: 6 tickets
    .venv/bin/python step3_test_claude.py 100    # the full exam (saved to results/)

With no API key set, this uses your Claude Code login (your subscription).
If ANTHROPIC_API_KEY is set, it uses the API and bills API credit instead.
Each ticket is 2-3 model calls either way.
"""
import os
import sys

import anthropic

from agents.claude import ClaudeLoginPolicy, ClaudePolicy
from refund_env.exam import print_card, run_exam


def main(n_tickets):
    use_api = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
    policy = ClaudePolicy() if use_api else ClaudeLoginPolicy()
    print("Using", "the Anthropic API (API credit)" if use_api else "your Claude Code login (subscription)")
    print_card(run_exam(policy, "claude", n_tickets))
    print(f"model: {policy.model}   tokens used: {policy.input_tokens:,} in, {policy.output_tokens:,} out")


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
