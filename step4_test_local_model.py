"""Let Gemma, a small open model, work the refund desk on this Mac.

    .venv/bin/python step4_test_local_model.py                      # a quick look: 6 tickets
    .venv/bin/python step4_test_local_model.py 100                  # the full exam, untrained
    .venv/bin/python step4_test_local_model.py 100 out/sft_adapter  # after step 5
    .venv/bin/python step4_test_local_model.py 100 out/rft_adapter  # after step 6
    .venv/bin/python step4_test_local_model.py 100 out/sft_rft_adapter  # after step 5, then step 6 --after-sft

The full exam (100 tickets) is saved to results/. No API key, no usage limits.
"""
import pathlib
import sys

from agents.local import LocalPolicy
from refund_env.exam import print_card, run_exam

NAMES = {None: "gemma_untrained", "sft_adapter": "gemma_sft", "rft_adapter": "gemma_rft",
         "sft_rft_adapter": "gemma_sft_then_rft"}


def main(n_tickets, adapter=None):
    folder = pathlib.Path(adapter).name if adapter else None
    name = NAMES.get(folder, f"gemma_{folder}")
    print_card(run_exam(LocalPolicy(adapter_path=adapter), name, n_tickets))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6,
         sys.argv[2] if len(sys.argv) > 2 else None)
