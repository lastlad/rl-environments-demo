"""Train the same policy under two reward functions and compare.

    python3 step2_compare_rewards.py

Both runs are scored afterwards on held-out tasks by the *true* success check,
whatever reward they were trained on.
"""
from agents.toy import OraclePolicy, RandomPolicy, TabularPolicy
from refund_env import RefundDeskEnv, make_task, naive_reward, run_episode, verified_reward
from refund_env.env import CASES
from refund_env.exam import EXAM_SEED_START as EVAL_SEED_START  # training uses seeds 0..N


def train(reward_fn, episodes=8000, lr=0.1, seed=0):
    env = RefundDeskEnv(reward_fn)
    policy = TabularPolicy(seed)
    window = []
    for ep in range(episodes):
        reward, info = run_episode(env, policy, seed=ep)
        policy.update(reward, lr)
        window.append((reward, info["true_success"]))
        if (ep + 1) % 2000 == 0:
            r = sum(w[0] for w in window) / len(window)
            s = sum(w[1] for w in window) / len(window)
            print(f"  episode {ep + 1:5d}   training reward {r:+.2f}   true success {s:4.0%}")
            window = []
    policy.greedy = True
    return policy


def evaluate(policy, n=600):
    env = RefundDeskEnv(verified_reward)
    wins = sum(run_episode(env, policy, EVAL_SEED_START + i)[1]["true_success"] for i in range(n))
    return wins / n


def show_behaviour(policy):
    """One held-out example of each case, so you can read what was learned."""
    env = RefundDeskEnv(verified_reward)
    seen, seed = set(), EVAL_SEED_START
    while len(seen) < len(CASES):
        seed += 1
        case = make_task(seed).case
        if case in seen:
            continue
        seen.add(case)
        _, info = run_episode(env, policy, seed)
        calls = []
        for s in env.steps:
            a = s["action"]
            name = a["tool"] + (f"({a['args']['resolution']})" if a["tool"] == "reply" else "")
            calls.append(name if s["result"]["ok"] else name + "[rejected]")
        mark = "ok  " if info["true_success"] else "FAIL"
        print(f"  {mark} should be {case:20s}: {' -> '.join(calls)}")


if __name__ == "__main__":
    print(f"oracle policy   true success {evaluate(OraclePolicy()):4.0%}   (ceiling)")
    print(f"random policy   true success {evaluate(RandomPolicy()):4.0%}   (floor)")
    for name, fn in (("naive_reward", naive_reward), ("verified_reward", verified_reward)):
        print(f"\ntraining with {name}")
        policy = train(fn)
        print(f"  held-out true success: {evaluate(policy):4.0%}")
        show_behaviour(policy)
