# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A teaching repo for beginners: a small RL environment (a refund desk) and eight numbered scripts that test and train agents on it, including a local Gemma model. The scope is steps 1–8. The audience is new to RL, so clarity of code and docs matters more than features.

## Commands

Run everything from the repository root.

```bash
python3 -m unittest discover tests -v                                  # all tests (no install needed)
python3 -m unittest tests.test_env.BadActions.test_double_refund_is_rejected   # one test
python3 step1_watch_one_ticket.py                                      # steps 1, 2, 3, 7: plain python3
.venv/bin/python step4_test_local_model.py 100 out/sft_adapter         # steps 4, 5, 6, 8: need .venv (mlx-lm)
```

- Steps 3 and 4 do 6 tickets by default; pass `100` for the full exam. Only a 100-ticket run saves a score card to `results/`.
- `python3 step7_show_results.py [results/changed_rules]` prints the results tables that are pasted into README.md.
- There is no linter or build step. `requirements.txt` holds only `mlx-lm`.

## Architecture

`refund_env/` separates the **prerequisites** from the **environment**:

- `tasks.py` and `policy.py` are the prerequisites. `make_task(seed, rules)` makes a synthetic ticket and its orders. `expected_case()` in `policy.py` is the answer key. In a real project, these would come from real data and a real checker.
- `env.py` holds only the database, the six tools and `reset`/`step`. A bad action never raises: it costs a step and returns an error.
- `rewards.py` holds `naive_reward` (grades the reply text and can be gamed) and `verified_reward` (grades the final database). Every exam and all of step 6 use `verified_reward`.
- `runner.py` (`run_episode`, `rollout`) is the one episode loop that every script uses. `prompts.py` turns the rules and observations into text for a model, and parses its output with `parse_action`.
- `exam.py` (`run_exam`) runs the fixed 100-ticket exam and writes `results/<name>.json`.

Agents in `agents/` share one interface, `reset()` and `act(obs) -> {"tool": ..., "args": ...}`:

- `toy.py`: `OraclePolicy` (the expert; its `self.thought` reasoning is step 5's training data), `RandomPolicy` and `TabularPolicy` (used by step 2 and the tests).
- `claude_code.py`: `ClaudeCodePolicy`, which calls the `claude -p` CLI with the user's login. No API key and no `anthropic` package.
- `local.py`: `LocalPolicy`, which runs Gemma (`mlx-community/gemma-4-e2b-it-4bit`) through MLX.
- `local_trainer.py`: LoRA fine-tuning, run as a subprocess by steps 5 and 6.

### Things that must stay true

- **Seed ranges never overlap.** Step 5 trains on seeds 0–399, step 6 on 10000 and up, and the exam uses 1,000,000–1,000,099. The same seed must always give the same task: `make_task` uses its own `random.Random(seed)`.
- **The environment must not change silently.** Saved results and adapters depend on exact task generation and prompt text. Before and after a refactor of `refund_env/`, compare a hash of `make_task` output for those seed ranges, `SYSTEM_PROMPT`, and some `rollout()` outputs.
- **Gemma's reasoning format.** Training targets use `<|channel>thought\n...<channel|>{json}<turn|>`. MLX's `TokenizerWrapper.apply_chat_template` quietly sets `enable_thinking=True`, so prompts contain `<|think|>`. A plain Hugging Face tokenizer does not, so pass it explicitly when porting.
- `info` from `env.step()` contains the answer. Agents must never read it.

## Constraints

- Target machine: Apple M4 Mac with 16 GB. Gemma runs and trains only through MLX.
- Use the Claude Code login for Claude, not the Anthropic API.
- Steps 5 and 6 take about 25 minutes and 1–2 hours. A full Gemma exam takes 7–20 minutes. Run them in the background.

## Writing rules

- `docs/`, `README.md` and `LESSONS.md` follow ASD-STE100 (Simplified Technical English): sentences of at most 25 words, active voice, simple tenses, no idioms or contractions, and procedures as numbered commands. Use one term per concept, as defined in the glossary in `docs/README.md` (environment, agent, reward function, `naive_reward`, `verified_reward`, exam). The shop analogy words appear only in that glossary.
- Mermaid diagrams in `docs/` must parse. Do not name a sequence-diagram participant `Loop`, because `loop` is a keyword.
- Code comments explain *why*. Do not add comments that repeat the code. Do not add comments that excuse a shortcut: fix the shortcut, or file a GitHub issue for it. Open issues #1–#5 describe known shortcuts.
