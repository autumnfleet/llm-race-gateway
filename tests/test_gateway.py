"""The real LiteLLM proxy, launched by `python3 -m llm_race_gateway serve`, against a fake upstream
on 127.0.0.1. Skipped without litellm[proxy] installed.

Each test gets its own gateway, because the ranking is process state and the tests are about how it
moves. The facts in `llm_race_gateway.routing`'s docstring are what these pin down.
"""

import json
import os
import random
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests import _env

WINDOW = 0.6
ROUTE_IN_TEXT = re.compile(r"from (\w+)\.")
DECISION = re.compile(r"llm_race: race-routes -> (\w+) \((\w+); tried=\[[^]]*\]; (\w+)=")
TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "weather",
                                           "parameters": {"type": "object", "properties": {}}}}]


def _text(chunks):
    return "".join((c.get("delta") or {}).get("content") or "" for ch in chunks for c in ch.get("choices") or ())


@unittest.skipUnless(_env.HAVE_LITELLM, _env.SKIP_REASON)
class GatewayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _env.require_proxy()

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.upstream = _env.Upstream()
        self.procs = []

    def tearDown(self):
        for p in self.procs:
            p.terminate()
            try:
                p.wait(10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()
            p.stdout_file.close()
        self.upstream.stop()
        self._tmp.cleanup()

    def start(self, names, extra_body=None, window=WINDOW, share=0.0, recheck=0.0):
        routes = []
        for n in names:
            key = self.dir / ("%s.key" % n)
            key.write_text("upstream-key-" + n)
            routes.append({"name": n, "base_url": self.upstream.url(n), "model": "vendor/" + n,
                           "api_key_file": str(key), "extra_body": (extra_body or {}).get(n, {})})
        (self.dir / "routes.json").write_text(json.dumps(routes))
        port = _env.free_port()
        self.log = self.dir / "gw.log"
        env = {k: v for k, v in os.environ.items() if not k.startswith("LLM_RACE_")}
        env.update(PYTHONUNBUFFERED="1", PYTHONPATH=_env.PACKAGE_DIR,
                   LLM_RACE_ROUTES_FILE=str(self.dir / "routes.json"),
                   LLM_RACE_MASTER_KEY_FILE=str(self.dir / "master.key"),
                   LLM_RACE_STATE_DIR=str(self.dir / "state"), LLM_RACE_PORT=str(port),
                   LLM_RACE_FIRST_TOKEN_SECS=str(window), LLM_RACE_TIMEOUT_SECS="10", LLM_RACE_SHARE=str(share),
                   LLM_RACE_RECHECK_SECS=str(recheck))
        out = self.log.open("w")
        p = subprocess.Popen([sys.executable, "-m", "llm_race_gateway", "serve"], stdout=out,
                             stderr=subprocess.STDOUT, env=env, cwd=str(self.dir))
        p.stdout_file = out
        self.procs.append(p)
        deadline = time.monotonic() + 90
        while True:
            try:
                if _env.get("http://127.0.0.1:%d/health/liveliness" % port) == 200:
                    break
            except (_env.URLError, OSError):
                pass
            if p.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("gateway did not start:\n" + self.log.read_text()[-3000:])
            time.sleep(0.3)
        self.base = "http://127.0.0.1:%d/v1" % port
        self.key = (self.dir / "master.key").read_text().strip()

    def logtext(self):
        return self.log.read_text()

    def wait_log(self, pattern, secs=5):
        deadline = time.monotonic() + secs
        while True:
            found = re.findall(pattern, self.logtext())
            if found:
                return found
            if time.monotonic() > deadline:
                self.fail("no %r in the log:\n%s" % (pattern, self.logtext()[-2000:]))
            time.sleep(0.1)

    def stream(self, model="race", probe=False, **body):
        """Returns (headers, chunks, seconds to first text)."""
        body = dict(body, model=model, messages=[{"role": "user", "content": "hello"}])
        if probe:
            body["metadata"] = {"llm_race_probe": True}
        return _env.post_stream(self.base, self.key, body)

    def ask(self, **kw):
        """One streamed turn; returns (route that wrote the reply, seconds to first content).

        The route is read from the reply text: through the public model LiteLLM reports the race
        provider as the deployment, and it rewrites every chunk's `model` to the requested name."""
        _, chunks, first = self.stream(**kw)
        m = ROUTE_IN_TEXT.search(_text(chunks))
        return (m.group(1) if m else _text(chunks)), first

    # -- routing -----------------------------------------------------------------------------------

    def test_the_first_route_is_the_default_and_its_key_and_body_arrive(self):
        self.start(["a", "b"], extra_body={"a": {"reasoning_effort": "none"}})
        self.ask(temperature=0.4, max_completion_tokens=321, tools=TOOLS)  # the fake answers a tool call
        self.assertEqual(list(self.upstream.bodies), ["a"])
        body = self.upstream.bodies["a"][-1]
        self.assertEqual((body["model"], body["temperature"], body["max_completion_tokens"], body["reasoning_effort"]),
                         ("vendor/a", 0.4, 321, "none"))
        self.assertEqual(body["tools"][0]["function"]["name"], "get_weather")
        self.assertNotIn("metadata", body)
        self.assertNotIn("llm_race_probe", json.dumps(body))

    def test_a_stalled_default_fails_over_inside_the_window_and_is_demoted_at_once(self):
        self.start(["a", "b"])
        self.assertEqual(self.ask()[0], "a")  # warm: a fresh gateway's first provider call carries setup
        self.upstream.delay["a"] = 5.0
        served, secs = self.ask()
        self.assertEqual(served, "b")
        self.assertLess(secs, WINDOW + 0.5)
        # The very next turn goes straight to b: the miss was recorded as a penalty, not averaged in.
        n_a = len(self.upstream.bodies["a"])
        served, secs = self.ask()
        self.assertEqual(served, "b")
        self.assertLess(secs, WINDOW)
        self.assertEqual(len(self.upstream.bodies["a"]), n_a)
        self.assertIn("default is now 'b'", self.logtext())

    def test_a_faster_route_takes_over_from_its_moving_average(self):
        # An unmeasured route ranks as if it took the window; a wide one keeps a's cold first call
        # (~0.5 s of one-off setup on top of its delay) from handing b the lead before b is measured.
        self.start(["a", "b"], window=2.0)
        self.upstream.delay["a"] = 0.3
        for _ in range(2):
            self.assertEqual(self.ask()[0], "a")
        # b is only measured when probed; once measured faster it becomes the default.
        self.assertEqual(self.ask(model="race-routes", probe=True)[0], "b")
        self.assertEqual(self.ask()[0], "b")
        self.assertIn("default is now 'b'", self.logtext())

    def test_a_probe_gets_a_route_other_than_the_best(self):
        self.start(["a", "b", "c"])
        for _ in range(6):
            self.ask(model="race-routes", probe=True)
        probes = [m.groups() for m in DECISION.finditer(self.logtext()) if m.group(2) == "probe"]
        self.assertEqual(len(probes), 6)
        # The ranking printed beside each decision starts with the best route at that instant.
        self.assertTrue(all(pick != best for pick, _, best in probes), probes)
        self.assertGreaterEqual(len({pick for pick, _, _ in probes}), 2, probes)

    def test_a_route_refusing_the_request_is_skipped_not_fatal(self):
        self.start(["a", "b"])
        self.upstream.status["a"] = 500
        self.assertEqual(self.ask()[0], "b")
        self.assertIn("route a failed", self.logtext())

    def test_when_every_route_stalls_the_request_still_ends_inside_its_windows(self):
        self.start(["a", "b"])
        self.assertEqual(self.ask()[0], "a")  # warm
        self.upstream.delay.update(a=5.0, b=5.0)
        t = time.monotonic()
        with self.assertRaises(Exception):
            _, chunks, _ = self.stream()
            if not _text(chunks):
                raise RuntimeError("no reply")
        self.assertLess(time.monotonic() - t, 2 * WINDOW + 1.0)

    def test_a_loser_closed_after_its_first_chunk_is_still_timed(self):
        self.start(["a", "b"])
        self.upstream.delay["b"] = 0.2
        headers, _, _ = self.stream(model="race-routes", probe=True, _stop_after_first_content=True)
        self.assertEqual(headers.get("x-litellm-model-id"), "b")
        samples = self.wait_log(r"llm_race: sample b (\d+\.\d+)s")
        self.assertEqual(len(samples), 1)
        self.assertGreaterEqual(float(samples[0]), 0.2)

    # -- the race ----------------------------------------------------------------------------------

    def test_a_race_speaks_the_faster_route_and_still_times_the_slower(self):
        self.start(["a", "b"], share=1.0)
        self.upstream.delay["a"] = 0.4
        self.assertEqual(self.ask()[0], "b")
        self.assertEqual((len(self.upstream.bodies["a"]), len(self.upstream.bodies["b"])), (1, 1))
        samples = self.wait_log(r"llm_race: sample a (\d+\.\d+)s")
        self.assertGreaterEqual(float(samples[0]), 0.4)
        self.assertIn("raced; probe answered first", self.logtext())
        # The sample reached the RANKING, not just the log: the next decision prints a measured a.
        self.ask()
        last = [ln for ln in self.logtext().splitlines() if "llm_race: race-routes -> " in ln][-1]
        self.assertRegex(last, r"\ba=0\.[4-9]\d*s/1\b|\ba=[1-9]\.\d+s/1\b")

    def test_a_demoted_route_is_raced_again_once_unmeasured_for_the_recheck_time(self):
        # A wide window, as above, keeps a's cold first call from losing to b's unmeasured prior.
        self.start(["a", "b"], window=2.0, recheck=3.0)
        self.upstream.delay["b"] = 0.3
        self.assertEqual(self.ask()[0], "a")
        self.upstream.delay["a"] = 5.0
        self.assertEqual(self.ask()[0], "b")  # a stalls: demoted at once
        self.wait_log(r"default is now 'b'")
        self.upstream.delay["a"] = 0.0  # a has recovered, but nothing measures it yet
        n_a = len(self.upstream.bodies["a"])
        self.assertEqual(self.ask()[0], "b")
        self.assertEqual(len(self.upstream.bodies["a"]), n_a)  # inside the recheck time: no race
        time.sleep(3.2)
        self.ask()
        self.assertEqual(len(self.upstream.bodies["a"]), n_a + 1)  # the next real turn raced a
        self.wait_log(r"llm_race: recheck a,")
        self.wait_log(r"llm_race: race-routes -> a \(recheck;")
        # Measured fast again, a wins back first place at once: the sample replaces the penalty.
        self.wait_log(r"default is now 'a'")
        self.assertEqual(self.ask()[0], "a")

    def test_tool_calls_and_usage_pass_through_the_race(self):
        self.start(["a", "b"], share=1.0)
        _, chunks, _ = self.stream(tools=TOOLS, stream_options={"include_usage": True})
        name, args, usage, finish = None, "", None, None
        for ch in chunks:
            usage = ch.get("usage") or usage
            for c in ch.get("choices") or ():
                finish = c.get("finish_reason") or finish
                for tc in (c.get("delta") or {}).get("tool_calls") or ():
                    fn = tc.get("function") or {}
                    name = name or fn.get("name")
                    args += fn.get("arguments") or ""
        self.assertEqual(name, "get_weather")
        self.assertEqual(json.loads(args), {"city": "Paris"})
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(usage["completion_tokens"], 7)

    def test_request_parameters_reach_the_route_unchanged_through_the_race(self):
        self.start(["a", "b"], extra_body={"a": {"reasoning_effort": "none"}})
        self.assertEqual(self.ask(temperature=0.4, max_completion_tokens=321)[0], "a")
        body = self.upstream.bodies["a"][-1]
        self.assertEqual((body["model"], body["temperature"], body["max_completion_tokens"], body["reasoning_effort"]),
                         ("vendor/a", 0.4, 321, "none"))
        self.assertNotIn("max_tokens", body)
        self.assertNotIn("llm_race_probe", json.dumps(body))

    def test_a_cut_reply_keeps_its_length_finish_and_the_providers_usage(self):
        for on_content in (False, True):
            with self.subTest(finish_on_last_text=on_content):
                self.start(["a", "b"])
                self.upstream.finish["a"] = "length"
                if on_content:
                    self.upstream.finish_on_content.add("a")
                _, chunks, _ = self.stream(stream_options={"include_usage": True})
                finishes = [c["finish_reason"] for ch in chunks for c in ch.get("choices") or () if c.get("finish_reason")]
                usage = [ch["usage"] for ch in chunks if ch.get("usage")][-1]
                self.assertEqual(_text(chunks), "Answer from a.")
                self.assertEqual(finishes, ["length"])
                self.assertEqual((usage["completion_tokens"], usage["completion_tokens_details"]["reasoning_tokens"]),
                                 (4, 3))
                self.upstream.finish.clear()
                self.upstream.finish_on_content.clear()

    def test_a_probe_that_misses_the_window_does_not_cost_the_caller(self):
        self.start(["a", "b"], share=1.0)
        self.assertEqual(self.ask()[0], "a")  # warm; b was probed too
        self.upstream.delay["b"] = 5.0
        served, secs = self.ask()
        self.assertEqual(served, "a")
        self.assertLess(secs, WINDOW)
        self.wait_log(r"route b failed")
        time.sleep(0.5)
        # The failed probe was not retried onto a: a was asked once per turn, never twice.
        self.assertEqual(len(self.upstream.bodies["a"]), 2)
        self.assertEqual(len(self.wait_log(r"llm_race: sample a [\d.]+s")), 2)

    def test_an_empty_stream_does_not_win_a_race(self):
        self.start(["a", "b"], share=1.0)
        self.upstream.empty.add("b")
        self.upstream.delay["a"] = 0.2
        self.assertEqual(self.ask()[0], "a")


@unittest.skipUnless(_env.HAVE_LITELLM, _env.SKIP_REASON)
class ChooseTest(unittest.TestCase):
    """EmaRouting.choose against a stand-in router: the metadata LiteLLM hands the strategy."""

    class _Router:
        def __init__(self, names):
            self.model_list = [{"model_name": "race-routes", "model_info": {"id": n}} for n in names] + [
                {"model_name": "race", "model_info": {"id": "llm-race"}}]

    def routing(self, names):
        from llm_race_gateway.ranker import RouteRanker
        from llm_race_gateway.routing import EmaRouting

        r = EmaRouting(RouteRanker(names, 0.3, 2.5), 2.5, 2.0, rng=random.Random(1))
        r._router = self._Router(names)
        return r

    def picked(self, routing, key="metadata", **md):
        return routing.choose("race-routes", {key: md})["model_info"]["id"]

    def test_retry_breadcrumbs_are_read_from_either_metadata_key(self):
        routing = self.routing(["a", "b", "c"])
        for key in ("metadata", "litellm_metadata"):
            self.assertEqual(self.picked(routing, key, previous_models=[{"deployment_id": "a"}]), "b")

    def test_a_probe_flag_nested_under_requester_metadata_counts(self):
        routing = self.routing(["a", "b"])
        self.assertEqual(self.picked(routing, requester_metadata={"llm_race_probe": True}), "b")

    def test_a_probe_naming_a_route_gets_that_route(self):
        routing = self.routing(["a", "b", "c"])
        self.assertEqual(self.picked(routing, llm_race_probe="c"), "c")
        self.assertEqual(self.picked(routing, requester_metadata={"llm_race_probe": "c"}), "c")

    def test_the_public_model_is_not_ranked(self):
        self.assertEqual(self.routing(["a", "b"]).choose("race", {})["model_info"]["id"], "llm-race")

    def test_a_failure_is_charged_from_this_attempts_start(self):
        import datetime as dt

        routing = self.routing(["a", "b"])
        t0 = dt.datetime(2026, 1, 1)
        routing._fail({"litellm_params": {"model_info": {"id": "a"}}, "api_call_start_time": t0 + dt.timedelta(seconds=3)},
                      t0, t0 + dt.timedelta(seconds=3.5))
        self.assertEqual(routing.ranker.snapshot()["a"]["ema_s"], 5.0)
        routing._observe({"litellm_params": {"model_info": {"id": "b"}}, "api_call_start_time": t0 + dt.timedelta(seconds=3),
                          "completion_start_time": t0 + dt.timedelta(seconds=3.2)}, t0, t0 + dt.timedelta(seconds=4))
        self.assertAlmostEqual(routing.ranker.snapshot()["b"]["ema_s"], 0.2)


if __name__ == "__main__":
    unittest.main()
