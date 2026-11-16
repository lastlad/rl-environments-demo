"""Print every saved exam result as one table.

    python3 step7_show_results.py

Each row comes from a file in results/, written by step 3 or step 4 when you
run the full 100-ticket exam.
"""
import json
import pathlib

RESULTS = pathlib.Path(__file__).parent / "results"
ORDER = ["claude", "gemma_untrained", "gemma_sft", "gemma_rft"]

if __name__ == "__main__":
    cards = {p.stem: json.loads(p.read_text()) for p in RESULTS.glob("*.json")}
    names = [n for n in ORDER if n in cards] + sorted(set(cards) - set(ORDER))
    print("| Trainee | Done right | Refunds | Escalations | Denials | Said it, did not do it | Wrongful refunds |")
    print("|---|---|---|---|---|---|---|")
    for name in names:
        c, t = cards[name], cards[name]["by_type"]
        cell = lambda k: f"{t[k]['right']}/{t[k]['tickets']}"
        print(f"| {name} | {c['right']}/{c['tickets']} | {cell('refunded')} | {cell('escalated')} | "
              f"{cell('denied')} | {c['said_it_without_doing_it']} | {c['wrongful_refunds']} |")
