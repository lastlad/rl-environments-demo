"""Claude works the refund tickets, and verified_reward gives the scores.

    python3 step3_test_claude.py        # a short test: 6 tickets
    python3 step3_test_claude.py 100    # the full exam (saved to results/)

This script uses the Claude Code command-line tool and your login. You do not
need an API key. Each ticket needs 2 or 3 model calls.
"""
import sys

from agents.claude_code import ClaudeCodePolicy
from refund_env.exam import print_card, run_exam


def main(n_tickets):
    policy = ClaudeCodePolicy()
    print_card(run_exam(policy, "claude", n_tickets))
    print(f"model: {policy.model}   tokens used: {policy.input_tokens:,} in, {policy.output_tokens:,} out")


if __name__ == "__main__":
    try:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
    except RuntimeError as e:
        sys.exit(str(e))
