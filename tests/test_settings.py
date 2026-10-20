"""Configuration and the generated LiteLLM config. Standard library only."""

import json
import os
import tempfile
import unittest
from pathlib import Path

from tests import _env  # noqa: F401
from llm_race_gateway import cli
from llm_race_gateway.proxy_config import NO_KEY, RACE_PROVIDER, build_config, route_key_env
from llm_race_gateway.settings import ConfigError, load_routes, load_settings, read_route_key

# A key-shaped value built at run time; never a pasted literal.
FAKE_KEY = "test-" + "k" * 6 + "-" + "0123456789abcdef"


class _Dir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def env(self, **kw):
        base = {"LLM_RACE_ROUTES_FILE": str(self.dir / "routes.json"),
                "LLM_RACE_MASTER_KEY_FILE": str(self.dir / "master.key"),
                "LLM_RACE_STATE_DIR": str(self.dir / "state")}
        base.update({"LLM_RACE_" + k: str(v) for k, v in kw.items()})
        return base

    def routes(self, routes):
        (self.dir / "routes.json").write_text(json.dumps(routes))


class SettingsTest(_Dir):
    def test_defaults(self):
        s = load_settings(self.env())
        self.assertEqual((s.host, s.port, s.model, s.routes_group), ("127.0.0.1", 4000, "race", "race-routes"))
        self.assertEqual((s.first_token_secs, s.timeout_secs, s.ema_alpha, s.share), (2.5, 15.0, 0.3, 0.1))

    def test_a_required_key_missing_is_named(self):
        env = self.env()
        del env["LLM_RACE_STATE_DIR"]
        with self.assertRaisesRegex(ConfigError, "LLM_RACE_STATE_DIR is required"):
            load_settings(env)

    def test_the_config_file_is_read_and_the_environment_wins(self):
        cfg = self.dir / "gw.conf"
        cfg.write_text("# comment\n\nLLM_RACE_PORT=4100\nLLM_RACE_MODEL='dialogue'\nLLM_RACE_SHARE=0.5\n")
        s = load_settings(dict(self.env(SHARE=0.2), LLM_RACE_CONFIG_FILE=str(cfg)))
        self.assertEqual((s.port, s.model, s.share), (4100, "dialogue", 0.2))

    def test_an_unknown_key_in_the_config_file_is_refused(self):
        cfg = self.dir / "gw.conf"
        cfg.write_text("LLM_RACE_PROT=4100\n")
        with self.assertRaisesRegex(ConfigError, "unknown key LLM_RACE_PROT"):
            load_settings(dict(self.env(), LLM_RACE_CONFIG_FILE=str(cfg)))

    def test_bad_values_are_refused(self):
        for kw in ({"PORT": "x"}, {"SHARE": 1.5}, {"EMA_ALPHA": 0}, {"FIRST_TOKEN_SECS": 20},
                   {"MODEL": "a b"}, {"FAILURE_PENALTY_WINDOWS": 0.5}):
            with self.subTest(kw=kw), self.assertRaises(ConfigError):
                load_settings(self.env(**kw))


class RoutesTest(_Dir):
    def test_a_route_needs_name_url_and_model_and_nothing_unknown(self):
        for bad in ([], [{"name": "a", "base_url": "u"}], [{"name": "a", "base_url": "u", "model": "m", "x": 1}],
                    [{"name": "a", "base_url": "u", "model": "m"}, {"name": "a", "base_url": "u", "model": "m"}],
                    [{"name": "a", "base_url": "u", "model": "m", "api_key_env": "A", "api_key_file": "f"}]):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                self.routes(bad)
                load_routes(self.dir / "routes.json")

    def test_keys_come_from_an_env_var_or_a_file_or_nowhere(self):
        (self.dir / "b.key").write_text(FAKE_KEY + "\n")
        self.routes([{"name": "a", "base_url": "u", "model": "m", "api_key_env": "UPSTREAM_A"},
                     {"name": "b", "base_url": "u", "model": "m", "api_key_file": str(self.dir / "b.key")},
                     {"name": "c", "base_url": "u", "model": "m"}])
        a, b, c = load_routes(self.dir / "routes.json")
        self.assertEqual(read_route_key(a, {"UPSTREAM_A": FAKE_KEY}), FAKE_KEY)
        self.assertEqual(read_route_key(b, {}), FAKE_KEY)
        self.assertIsNone(read_route_key(c, {}))
        with self.assertRaisesRegex(ConfigError, "UPSTREAM_A is empty or unset"):
            read_route_key(a, {})


class ProxyConfigTest(_Dir):
    def test_the_written_config_names_key_variables_and_never_holds_a_key(self):
        (self.dir / "b.key").write_text(FAKE_KEY)
        self.routes([{"name": "a", "base_url": "http://up/a/v1", "model": "vendor/a", "extra_body": {"x": 1},
                      "price_prompt": 0.15, "price_completion": 0.6},
                     {"name": "b", "base_url": "http://up/b/v1", "model": "vendor/b",
                      "api_key_file": str(self.dir / "b.key")}])
        s = load_settings(self.env(MODEL="dialogue"))
        environ = {}
        cli.prepare(s, environ)
        text = cli.config_path(s).read_text()
        self.assertNotIn(FAKE_KEY, text)
        self.assertEqual(environ, {route_key_env(1): FAKE_KEY})
        self.assertEqual(os.stat(s.state_dir).st_mode & 0o777, 0o700)
        cfg = json.loads(text)
        a, b, public = cfg["model_list"]
        self.assertEqual((a["model_name"], a["litellm_params"]["model"], a["litellm_params"]["api_key"]),
                         ("dialogue-routes", "openai/vendor/a", NO_KEY))
        self.assertEqual(b["litellm_params"]["api_key"], "os.environ/" + route_key_env(1))
        self.assertEqual(a["litellm_params"]["extra_body"], {"x": 1})
        self.assertAlmostEqual(a["model_info"]["input_cost_per_token"], 0.15e-6)
        self.assertNotIn("input_cost_per_token", b["model_info"])
        self.assertEqual((public["model_name"], public["model_info"]["id"]), ("dialogue", RACE_PROVIDER))
        self.assertEqual(a["litellm_params"]["stream_timeout"], s.first_token_secs)
        self.assertEqual(a["litellm_params"]["max_retries"], 0)
        self.assertEqual(cfg["router_settings"]["num_retries"], 1)

    def test_build_config_is_pure(self):
        self.routes([{"name": "a", "base_url": "u", "model": "m"}])
        s = load_settings(self.env())
        routes = load_routes(s.routes_file)
        self.assertEqual(build_config(s, routes, [False]), build_config(s, routes, [False]))

    def test_the_master_key_file_is_created_private_and_reused(self):
        path = self.dir / "sub" / "master.key"
        first = cli.ensure_master_key(path)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertEqual(cli.ensure_master_key(path), first)


class CliTest(_Dir):
    def test_check_prints_a_summary_without_the_key(self):
        import contextlib
        import io

        self.routes([{"name": "a", "base_url": "u", "model": "m", "api_key_env": "UPSTREAM_A"}])
        old = dict(os.environ)
        os.environ.update(self.env(), UPSTREAM_A=FAKE_KEY)
        out = io.StringIO()
        try:
            with contextlib.redirect_stdout(out):
                rc = cli.main(["check"])
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(rc, 0)
        self.assertIn("1 routes ['a']", out.getvalue())
        self.assertIn("unpriced routes", out.getvalue())
        self.assertNotIn(FAKE_KEY, out.getvalue())

    def test_a_config_error_exits_2(self):
        import contextlib
        import io

        old = dict(os.environ)
        for k in [k for k in os.environ if k.startswith("LLM_RACE_")]:
            del os.environ[k]
        err = io.StringIO()
        try:
            with contextlib.redirect_stderr(err):
                rc = cli.main(["check"])
        finally:
            os.environ.clear()
            os.environ.update(old)
        self.assertEqual(rc, 2)
        self.assertIn("is required", err.getvalue())


if __name__ == "__main__":
    unittest.main()
