"""Change the shop's rules and see who notices.

    python3 step8_changed_rules.py                                    # what changes, and two experts
    .venv/bin/python step8_changed_rules.py 100                       # untrained Gemma
    .venv/bin/python step8_changed_rules.py 100 out/sft_adapter       # after step 5
    .venv/bin/python step8_changed_rules.py 100 out/sft_rft_adapter   # after step 5, then step 6
    .venv/bin/python step8_changed_rules.py 100 claude                # Claude (uses your login)
Then compare them:
    python3 step7_show_results.py results/changed_rules

Gemma scored 99-100 after training. But every ticket it practised on, and every
exam ticket, used the same two numbers: deny after 30 days, escalate over 10000
cents. So the score cannot tell these apart:
  - Gemma learned to read the rulebook and apply it.
  - Gemma memorised "30" and "10000" as habits.

This exam uses the same 100 tickets with a new rulebook: deny after 14 days,
escalate over 5000 cents. Everything else is unchanged. The rulebook in the
prompt says the new numbers, and the careful manager grades by them. No
training happens here; it only tests.

The number to watch is on the tickets where the new rules change the answer:
did the trainee follow the new rules, or do what the old rules would have said?
"""
import collections
import sys

from agents.toy import OraclePolicy
from refund_env import make_task
from refund_env.env import CHANGED_RULES, RULES, system_prompt
from refund_env.exam import EXAM_SEED_START, EXAM_TICKETS, print_card, run_exam

NEW_PROMPT = system_prompt(CHANGED_RULES)


def show_what_changes():
    old, new = RULES, CHANGED_RULES
    print("The rulebook changes in two places:")
    print(f"  rule 2: deny if delivered more than {old.window_days} days ago "
          f"-> more than {new.window_days} days ago")
    print(f"  rule 3: escalate if amount_cents over {old.limit_cents} "
          f"-> over {new.limit_cents}")

    moves = collections.Counter()
    for i in range(EXAM_TICKETS):
        before = make_task(EXAM_SEED_START + i).case
        after = make_task(EXAM_SEED_START + i, CHANGED_RULES).case
        if before != after:
            moves[before, after] += 1
    print(f"\nOn the same {EXAM_TICKETS} exam tickets, the right answer changes on "
          f"{sum(moves.values())}:")
    for (before, after), n in moves.most_common():
        print(f"  {n:3d}  {before:22s} -> {after}")

    # Two experts as a yardstick: one reads the new rulebook, one keeps the old habits.
    print("\nTwo hand-written experts sit the changed-rules exam:")
    for label, rules in (("follows the new rules", CHANGED_RULES),
                         ("still uses the old numbers", RULES)):
        card = run_exam(OraclePolicy(rules), "expert", EXAM_TICKETS, show=False,
                        rules=CHANGED_RULES, save=False)
        c = card["rules_changed_the_answer"]
        print(f"  expert that {label:27s} {card['right']:3d}/{EXAM_TICKETS}   "
              f"on the {c['tickets']} changed tickets: {c['right']} right, "
              f"{c['followed_old_rules']} followed the old rules")
    print("\nA trainee that memorised the numbers will look like the second expert.")


def make_trainee(which):
    if which == "claude":
        import os
        from agents.claude import ClaudeLoginPolicy, ClaudePolicy
        use_api = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        cls = ClaudePolicy if use_api else ClaudeLoginPolicy
        return "claude", cls(system_prompt=NEW_PROMPT)
    import pathlib
    from agents.local import LocalPolicy
    from step4_test_local_model import NAMES
    folder = pathlib.Path(which).name if which else None
    name = NAMES.get(folder, f"gemma_{folder}")
    return name, LocalPolicy(adapter_path=which, system_prompt=NEW_PROMPT)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_what_changes()
    else:
        name, trainee = make_trainee(sys.argv[2] if len(sys.argv) > 2 else None)
        print_card(run_exam(trainee, name, int(sys.argv[1]), rules=CHANGED_RULES))
