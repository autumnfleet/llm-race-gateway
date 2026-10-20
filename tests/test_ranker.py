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
