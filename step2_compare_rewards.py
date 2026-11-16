"""Train the same policy under two reward functions and compare.

    python3 step2_compare_rewards.py

Both runs are scored afterwards on held-out tasks by the *true* success check,
whatever reward they were trained on.
"""
from refund_env import RefundDeskEnv, make_task, naive_reward, verified_reward
from agents.toy import OraclePolicy, RandomPolicy, TabularPolicy

EVAL_SEED_START = 1_000_000  # training uses seeds 0..N; eval never overlaps


def run_episode(env, policy, seed):
    obs = env.reset(seed)
    policy.reset()
    while True:
        obs, reward, terminated, truncated, info = env.step(policy.act(obs))
        if terminated or truncated:
            return reward, info


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
    while len(seen) < 3:
        seed += 1
        expected = make_task(seed).expected
        if expected in seen:
            continue
        seen.add(expected)
        _, info = run_episode(env, policy, seed)
        calls = []
        for s in env.steps:
            a = s["action"]
            name = a["tool"] + (f"({a['args']['resolution']})" if a["tool"] == "reply" else "")
            calls.append(name if s["result"]["ok"] else name + "[rejected]")
        mark = "ok  " if info["true_success"] else "FAIL"
        print(f"  {mark} should be {expected:9s}: {' -> '.join(calls)}")


if __name__ == "__main__":
    print(f"oracle policy   true success {evaluate(OraclePolicy()):4.0%}   (ceiling)")
    print(f"random policy   true success {evaluate(RandomPolicy()):4.0%}   (floor)")
    for name, fn in (("naive_reward", naive_reward), ("verified_reward", verified_reward)):
        print(f"\ntraining with {name}")
        policy = train(fn)
        print(f"  held-out true success: {evaluate(policy):4.0%}")
        show_behaviour(policy)
