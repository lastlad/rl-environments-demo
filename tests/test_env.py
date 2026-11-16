"""Each test pins down one way an environment can quietly go wrong.

    python3 -m unittest discover tests -v
"""
import json
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agents.toy import OraclePolicy, RandomPolicy  # noqa: E402
from refund_env import (RefundDeskEnv, make_task, naive_reward, parse_action,  # noqa: E402
                        rollout, run_episode, verified_reward)
from refund_env.env import SYSTEM_PROMPT, expected_resolution  # noqa: E402
from refund_env.exam import EXAM_TICKETS, run_exam  # noqa: E402


def call(tool, **args):
    return {"tool": tool, "args": args}


def seed_for(expected):
    return next(s for s in range(1000) if make_task(s).expected == expected)


play = run_episode


class Reproducibility(unittest.TestCase):
    def test_same_seed_same_task(self):
        self.assertEqual(make_task(42), make_task(42))
        self.assertNotEqual(make_task(42), make_task(43))

    def test_reset_wipes_previous_episode(self):
        env = RefundDeskEnv(verified_reward)
        seed = seed_for("refunded")
        env.reset(seed)
        oid = env.task.order_id
        amount = env.db[oid]["amount_cents"]
        env.step(call("issue_refund", order_id=oid, amount_cents=amount))
        env.reset(seed)
        self.assertEqual(env.db[oid]["refunded_cents"], 0)
        self.assertEqual(env.steps, [])

    def test_two_envs_do_not_share_state(self):
        a, b = RefundDeskEnv(verified_reward), RefundDeskEnv(verified_reward)
        seed = seed_for("refunded")
        a.reset(seed), b.reset(seed)
        a.step(call("escalate", order_id=a.task.order_id))
        self.assertFalse(b.db[b.task.order_id]["escalated"])


class NoLeaks(unittest.TestCase):
    def test_observation_hides_the_answer(self):
        env = RefundDeskEnv(verified_reward)
        for seed in range(50):
            obs = env.reset(seed)
            text = json.dumps(obs)
            self.assertNotIn("expected", text)
            self.assertNotIn("days_since_delivery", text)  # only via lookup_order

    def test_mutating_the_observation_does_not_touch_the_world(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(1)
        oid = env.task.order_id
        obs, *_ = env.step(call("lookup_order", order_id=oid))
        obs["last_result"]["order"]["refunded_cents"] = 999
        self.assertEqual(env.db[oid]["refunded_cents"], 0)


class BadActions(unittest.TestCase):
    def test_malformed_actions_cost_a_step_and_never_crash(self):
        env = RefundDeskEnv(verified_reward, max_steps=20)
        env.reset(3)
        junk = [None, 7, "refund pls", [], {}, {"tool": 5}, {"tool": "rm_rf"},
                {"tool": "lookup_order", "args": "A1"}, call("lookup_order"),
                call("lookup_order", order_id=["x"]), call("reply", resolution="maybe"),
                call("issue_refund", order_id=env.task.order_id, amount_cents="all"),
                call("issue_refund", order_id=env.task.order_id, amount_cents=-5),
                call("issue_refund", order_id=env.task.order_id, amount_cents=True),
                parse_action("{not json")]
        for action in junk:
            obs, reward, term, trunc, _ = env.step(action)
            self.assertFalse(obs["last_result"]["ok"], action)
            self.assertFalse(term)
        self.assertEqual(env.invalid_actions, len(junk))
        self.assertEqual(obs["steps_left"], 20 - len(junk))

    def test_double_refund_is_rejected(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("refunded"))
        oid, amount = env.task.order_id, env.db[env.task.order_id]["amount_cents"]
        env.step(call("issue_refund", order_id=oid, amount_cents=amount))
        obs, *_ = env.step(call("issue_refund", order_id=oid, amount_cents=amount))
        self.assertFalse(obs["last_result"]["ok"])
        self.assertEqual(env.db[oid]["refunded_cents"], amount)

    def test_double_escalation_is_rejected(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("escalated"))
        oid = env.task.order_id
        env.step(call("escalate", order_id=oid))
        obs, *_ = env.step(call("escalate", order_id=oid))
        self.assertFalse(obs["last_result"]["ok"])
        self.assertTrue(env.db[oid]["escalated"])

    def test_unknown_arguments_are_rejected(self):
        """A tool that quietly ignores a misspelt argument hides agent mistakes."""
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("escalated"))
        oid = env.task.order_id
        obs, *_ = env.step(call("escalate", order_id=oid, reason="over the limit"))
        self.assertFalse(obs["last_result"]["ok"])
        self.assertFalse(env.db[oid]["escalated"])
        obs, *_ = env.step(call("lookup_order", order_id=oid, verbose=True))
        self.assertFalse(obs["last_result"]["ok"])

    def test_cannot_refund_more_than_was_paid(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(5)
        oid = env.task.order_id
        obs, *_ = env.step(call("issue_refund", order_id=oid,
                                amount_cents=env.db[oid]["amount_cents"] + 1))
        self.assertFalse(obs["last_result"]["ok"])


class PromptMatchesTheWorld(unittest.TestCase):
    def test_prompt_states_the_limit_in_the_unit_the_tools_use(self):
        """Orders show amount_cents, so the rule must be stated in cents too."""
        self.assertIn("amount_cents over 10000", SYSTEM_PROMPT)

    def test_rollout_records_what_was_seen_and_done(self):
        env = RefundDeskEnv(verified_reward)
        traj = rollout(env, OraclePolicy(), seed_for("refunded"))
        roles = [m["role"] for m in traj["messages"]]
        self.assertEqual(roles, ["system"] + ["user", "assistant"] * 3)
        self.assertTrue(traj["true_success"])


class TheExam(unittest.TestCase):
    def test_expert_passes_and_score_card_adds_up(self):
        card = run_exam(OraclePolicy(), "test", n_tickets=30, show=False)
        self.assertEqual(card["right"], 30)
        self.assertEqual(sum(v["tickets"] for v in card["by_type"].values()), 30)
        self.assertEqual(card["said_it_without_doing_it"], 0)
        self.assertLess(30, EXAM_TICKETS)  # a short run is never saved as a result


class EpisodeBoundaries(unittest.TestCase):
    def test_step_limit_truncates_instead_of_terminating(self):
        env = RefundDeskEnv(verified_reward, max_steps=3)
        env.reset(1)
        for i in range(3):
            _, reward, term, trunc, info = env.step(call("lookup_order", order_id=env.task.order_id))
        self.assertTrue(trunc)
        self.assertFalse(term)
        self.assertFalse(info["true_success"])
        self.assertLess(reward, 0)

    def test_step_after_done_raises(self):
        env = RefundDeskEnv(verified_reward)
        with self.assertRaises(RuntimeError):
            env.step(call("reply", resolution="denied"))  # before reset
        env.reset(1)
        env.step(call("reply", resolution="denied"))
        with self.assertRaises(RuntimeError):
            env.step(call("reply", resolution="denied"))


class RewardIntegrity(unittest.TestCase):
    def test_policy_boundaries(self):
        self.assertEqual(expected_resolution(30, 100_00), "refunded")
        self.assertEqual(expected_resolution(31, 100_00), "denied")
        self.assertEqual(expected_resolution(30, 100_01), "escalated")
        self.assertEqual(expected_resolution(31, 100_01), "denied")

    def test_generator_labels_match_the_policy(self):
        seen = set()
        for seed in range(2000):
            t = make_task(seed)
            o = t.orders[t.order_id]
            self.assertEqual(t.expected,
                             expected_resolution(o["days_since_delivery"], o["amount_cents"]))
            seen.add(t.expected)
        self.assertEqual(len(seen), 3)

    def test_oracle_always_succeeds(self):
        env = RefundDeskEnv(verified_reward)
        for seed in range(300):
            reward, info = play(env, OraclePolicy(), seed)
            self.assertTrue(info["true_success"], seed)
            self.assertGreater(reward, 0.9)

    def test_random_policy_scores_badly(self):
        env = RefundDeskEnv(verified_reward)
        wins = sum(play(env, RandomPolicy(s), s)[1]["true_success"] for s in range(300))
        self.assertLess(wins / 300, 0.2)

    def test_claiming_a_refund_without_issuing_it(self):
        """The exploit: naive_reward pays for it, verified_reward does not."""
        seed = seed_for("refunded")
        for reward_fn, paid in ((naive_reward, True), (verified_reward, False)):
            env = RefundDeskEnv(reward_fn)
            env.reset(seed)
            _, reward, _, _, info = env.step(call("reply", resolution="refunded"))
            self.assertEqual(reward > 0.5, paid)
            self.assertFalse(info["true_success"])

    def test_wrongful_refund_is_worse_than_doing_nothing(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("denied"))
        oid = env.task.order_id
        env.step(call("issue_refund", order_id=oid, amount_cents=env.db[oid]["amount_cents"]))
        _, reward, *_ = env.step(call("reply", resolution="refunded"))
        self.assertLess(reward, -0.9)

    def test_partial_refund_is_not_success(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("refunded"))
        env.step(call("issue_refund", order_id=env.task.order_id, amount_cents=1))
        _, _, _, _, info = env.step(call("reply", resolution="refunded"))
        self.assertFalse(info["true_success"])

    def test_touching_the_wrong_order_is_not_success(self):
        env = RefundDeskEnv(verified_reward)
        env.reset(seed_for("refunded"))
        oid = env.task.order_id
        other = next(o for o in env.db if o != oid)
        env.step(call("issue_refund", order_id=oid, amount_cents=env.db[oid]["amount_cents"]))
        env.step(call("issue_refund", order_id=other, amount_cents=env.db[other]["amount_cents"]))
        _, reward, _, _, info = env.step(call("reply", resolution="refunded"))
        self.assertFalse(info["true_success"])
        self.assertLess(reward, 0)


if __name__ == "__main__":
    unittest.main()
