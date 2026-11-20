"""Print every saved exam result as one table.

    python3 step7_show_results.py                    # current results
    python3 step7_show_results.py results/v1_easy    # results from the easier first version

Each row comes from a file in the folder, written by step 3 or step 4 when you
run the full 100-ticket exam.
"""
import json
import pathlib
import sys

ORDER = ["claude", "gemma_untrained", "gemma_sft_no_reasoning", "gemma_sft", "gemma_rft",
         "gemma_sft_then_rft"]

if __name__ == "__main__":
    folder = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "results")
    cards = {p.stem: json.loads(p.read_text()) for p in folder.glob("*.json")}
    names = [n for n in ORDER if n in cards] + sorted(set(cards) - set(ORDER))
    kinds = list(cards[names[0]]["by_type"])
    print("| Trainee | Done right | " + " | ".join(kinds) + " | Said it, did not do it | Wrongful refunds |")
    print("|---" * (len(kinds) + 4) + "|")
    for name in names:
        c, t = cards[name], cards[name]["by_type"]
        cells = " | ".join(f"{t[k]['right']}/{t[k]['tickets']}" for k in kinds)
        print(f"| {name} | {c['right']}/{c['tickets']} | {cells} | "
              f"{c['said_it_without_doing_it']} | {c['wrongful_refunds']} |")
