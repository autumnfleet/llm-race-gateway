"""The routing policy itself: RouteRanker and pick(). Standard library only."""

import random
import unittest

from tests import _env  # noqa: F401  (puts the package on sys.path)
from llm_race_gateway.ranker import DEFAULT_FAILURE_PENALTY_WINDOWS as PENALTY
from llm_race_gateway.ranker import RouteRanker, pick


class RankerTest(unittest.TestCase):
    def test_before_any_evidence_the_configured_order_is_the_ranking(self):
        self.assertEqual(RouteRanker(["a", "b", "c"], 0.3, 2.5).ranked(), ["a", "b", "c"])

    def test_the_lowest_moving_average_ranks_first(self):
        r = RouteRanker(["a", "b"], 0.3, 2.5)
        r.observe("a", 1.0)
        r.observe("b", 0.2)
        self.assertEqual(r.ranked(), ["b", "a"])
        r.observe("b", 2.0)  # 0.3*2.0 + 0.7*0.2 = 0.74, still ahead of 1.0
        self.assertEqual(r.ranked(), ["b", "a"])
        self.assertAlmostEqual(r.snapshot()["b"]["ema_s"], 0.74)

    def test_an_untried_route_ranks_behind_answering_routes_but_ahead_of_failed_ones(self):
        r = RouteRanker(["a", "b", "c"], 0.3, 2.5)
        r.observe("b", 0.4)
        r.fail("a", PENALTY * 2.5)
        self.assertEqual(r.ranked(), ["b", "c", "a"])

    def test_one_failure_demotes_the_default_at_once(self):
        r = RouteRanker(["a", "b"], 0.3, 2.5)
        for _ in range(5):
            r.observe("a", 0.1)
        r.observe("b", 2.0)
        r.fail("a", PENALTY * 2.5)
        self.assertEqual(r.ranked()[0], "b")
        self.assertEqual(r.snapshot()["a"]["ema_s"], 5.0)  # the floor, not 0.3*5 + 0.7*0.1

    def test_a_default_change_is_logged_with_the_ranking(self):
        r = RouteRanker(["a", "b"], 0.3, 2.5)
        with self.assertLogs("llm_race_gateway", "WARNING") as cm:
            r.observe("b", 0.1)
        self.assertIn("default is now 'b' (was 'a')", cm.output[0])

    def test_an_unknown_route_is_ignored(self):
        r = RouteRanker(["a"], 0.3, 2.5)
        r.observe("zzz", 0.1)
        self.assertEqual(r.snapshot(), {"a": {"ema_s": None, "samples": 0}})

    def test_bad_construction_is_refused(self):
        for names, alpha in (([], 0.3), (["a", "a"], 0.3), (["a"], 0.0), (["a"], 1.5)):
            with self.assertRaises(ValueError):
                RouteRanker(names, alpha, 2.5)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class RecoveryTest(unittest.TestCase):
    def test_the_first_sample_after_a_failure_replaces_the_penalty(self):
        clock = _Clock()
        r = RouteRanker(["a", "b"], 0.3, 2.5, clock=clock)
        r.observe("a", 0.2)
        r.observe("b", 0.8)
        r.fail("a", PENALTY * 2.5)
        self.assertEqual(r.ranked()[0], "b")
        clock.now += 600
        r.observe("a", 0.3)  # smoothed, this would be 0.3*0.3 + 0.7*5.0 = 3.59 and stay behind b
        self.assertAlmostEqual(r.snapshot()["a"]["ema_s"], 0.3)
        self.assertEqual(r.ranked()[0], "a")

    def test_later_samples_are_smoothed_again(self):
        clock = _Clock()
        r = RouteRanker(["a"], 0.3, 2.5, clock=clock)
        r.fail("a", 5.0)
        clock.now += 600
        r.observe("a", 1.0)
        r.observe("a", 2.0)
        self.assertAlmostEqual(r.snapshot()["a"]["ema_s"], 0.3 * 2.0 + 0.7 * 1.0)

    def test_a_success_from_before_the_failure_does_not_replace_the_penalty(self):
        clock = _Clock()
        r = RouteRanker(["a", "b"], 0.3, 2.5, clock=clock)
        r.observe("a", 0.1)
        r.observe("b", 0.8)
        clock.now += 10
        r.fail("a", 5.0)  # a newer request to a stalled
        clock.now += 1
        r.observe("a", 0.05, started_ago=20)  # an older stream of a, reported only as it ends
        self.assertAlmostEqual(r.snapshot()["a"]["ema_s"], 0.3 * 0.05 + 0.7 * 5.0)
        self.assertEqual(r.ranked()[0], "b")
        r.observe("a", 0.2, started_ago=0.2)  # a new attempt: this one does show recovery
        self.assertAlmostEqual(r.snapshot()["a"]["ema_s"], 0.2)

    def test_repeated_failures_stay_at_least_the_penalty(self):
        r = RouteRanker(["a"], 0.3, 2.5)
        r.fail("a", 5.0)
        r.fail("a", 5.0)
        self.assertEqual(r.snapshot()["a"]["ema_s"], 5.0)


class RecheckTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.r = RouteRanker(["a", "b", "c"], 0.3, 2.5, clock=self.clock)

    def test_nothing_is_due_before_the_age_is_reached(self):
        self.r.observe("c", 0.1)  # a and b demoted, so only the age keeps them from being due
        self.clock.now += 599
        self.assertIsNone(self.r.claim_recheck(600))

    def test_a_demoted_route_is_due_once_unmeasured_for_the_age(self):
        r = RouteRanker(["a", "b"], 0.3, 2.5, clock=self.clock)
        r.fail("a", PENALTY * 2.5)
        r.observe("b", 0.5)
        self.clock.now += 300
        r.observe("b", 0.5)  # b keeps answering; that does not reset a's clock
        self.clock.now += 300
        self.assertEqual(r.claim_recheck(600), "a")

    def test_a_route_holding_its_configured_place_is_never_due(self):
        self.r.observe("a", 0.1)
        self.r.observe("b", 0.5)
        self.clock.now += 10000
        self.assertIsNone(self.r.claim_recheck(600))  # c is last but ranked where it is configured

    def test_claiming_restarts_the_clock_so_one_request_races_it(self):
        self.r.observe("c", 0.1)  # c leads, so a and b are demoted
        self.clock.now += 600
        self.assertEqual(self.r.claim_recheck(600), "a")
        self.assertEqual(self.r.claim_recheck(600), "b")
        self.assertIsNone(self.r.claim_recheck(600))
        self.clock.now += 600
        self.assertEqual(self.r.claim_recheck(600), "a")

    def test_the_longest_unmeasured_route_goes_first(self):
        self.r.observe("c", 0.1)
        self.clock.now += 100
        self.r.observe("a", 0.4)  # still behind c; b never measured
        self.clock.now += 650
        self.assertEqual(self.r.claim_recheck(600), "b")
        self.assertEqual(self.r.claim_recheck(600), "a")

    def test_a_sample_or_a_failure_restarts_the_clock(self):
        self.r.observe("c", 0.1)
        self.clock.now += 600
        self.r.observe("a", 1.0)
        self.r.fail("b", PENALTY * 2.5)
        self.assertIsNone(self.r.claim_recheck(600))

    def test_zero_disables_rechecks(self):
        self.r.observe("c", 0.1)
        self.clock.now += 10000
        self.assertEqual(self.r.claim_recheck(1), "a")  # due, so zero alone is what refuses below
        self.assertIsNone(self.r.claim_recheck(0))


class PickTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(1)

    def test_the_best_route_answers(self):
        self.assertEqual(pick(["b", "a"], ["a", "b"], (), False, self.rng), ("b", "best"))

    def test_a_retry_never_repeats_a_route_this_request_already_tried(self):
        """Independent of whether the failure was recorded yet: here it was not, so the failed
        route is still the best by its average."""
        self.assertEqual(pick(["a", "b", "c"], ["a", "b", "c"], {"a"}, False, self.rng), ("b", "retry"))
        self.assertEqual(pick(["a", "b", "c"], ["a", "b", "c"], {"a", "b"}, False, self.rng), ("c", "retry"))

    def test_when_every_route_was_tried_the_best_is_asked_again(self):
        self.assertEqual(pick(["a", "b"], ["a", "b"], {"a", "b"}, False, self.rng), ("a", "retry"))

    def test_a_probe_never_gets_the_best_route(self):
        picks = {pick(["c", "a", "b"], ["a", "b", "c"], (), True, self.rng)[0] for _ in range(40)}
        self.assertEqual(picks, {"a", "b"})

    def test_a_probe_naming_a_route_gets_that_route(self):
        self.assertEqual(pick(["a", "b", "c"], ["a", "b", "c"], (), "c", self.rng), ("c", "recheck"))

    def test_a_probe_naming_the_best_or_an_unavailable_route_is_an_ordinary_probe(self):
        for named in ("a", "zzz"):
            route, why = pick(["a", "b", "c"], ["a", "b", "c"], (), named, self.rng)
            self.assertEqual(why, "probe")
            self.assertIn(route, ("b", "c"))

    def test_a_probe_with_one_route_gets_that_route(self):
        self.assertEqual(pick(["a"], ["a"], (), True, self.rng), ("a", "best"))

    def test_a_retried_probe_is_an_ordinary_retry(self):
        self.assertEqual(pick(["a", "b", "c"], ["a", "b", "c"], {"b"}, True, self.rng), ("a", "retry"))

    def test_a_route_the_ranker_does_not_know_goes_last(self):
        self.assertEqual(pick(["a"], ["x", "a"], {"a"}, False, self.rng), ("x", "retry"))

    def test_no_route_is_an_error(self):
        with self.assertRaises(ValueError):
            pick(["a"], [], (), False, self.rng)


if __name__ == "__main__":
    unittest.main()
