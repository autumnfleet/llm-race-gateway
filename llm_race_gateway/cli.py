"""`python3 -m llm_race_gateway serve|check`. Everything else comes from `LLM_RACE_*` (see README)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import secrets
import sys
from pathlib import Path
from typing import List, Optional

from . import __version__
from .proxy_config import build_config, route_key_env
from .settings import ConfigError, Route, Settings, load_routes, load_settings, read_route_key

log = logging.getLogger("llm_race_gateway")


def ensure_master_key(path: Path) -> str:
    """The key clients present. Created (mode 600, random) if the file does not exist."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write("sk-" + secrets.token_hex(24) + "\n")
        log.info("llm_race: created a master key file (mode 600)")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise ConfigError("%s: empty master key file" % path)
    return value


def prepare(settings: Settings, environ=None) -> List[Route]:
    """Load the routes, put each key into this process's environment, write the LiteLLM config.

    Returns the routes. The keys are read in-process only; the written config names variables.
    """
    environ = os.environ if environ is None else environ
    routes = load_routes(settings.routes_file)
    keyed = []
    for i, r in enumerate(routes):
        key = read_route_key(r, environ)
        if key is not None:
            environ[route_key_env(i)] = key
        keyed.append(key is not None)
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.state_dir.chmod(0o700)
    config_path(settings).write_text(json.dumps(build_config(settings, routes, keyed), indent=2))
    return routes


def config_path(settings: Settings) -> Path:
    return settings.state_dir / "litellm-config.json"


def install_runtime(settings: Settings, routes: List[Route]) -> None:
    from . import runtime
    from .race import RaceLLM
    from .ranker import RouteRanker
    from .routing import EmaRouting

    ranker = RouteRanker([r.name for r in routes], settings.ema_alpha, settings.first_token_secs)
    runtime.routing = EmaRouting(ranker, settings.first_token_secs, settings.failure_penalty_windows,
                                 priced={r.name for r in routes if r.priced})
    runtime.race = RaceLLM(settings.routes_group, settings.share, ranker, recheck_secs=settings.recheck_secs)


def cmd_check(settings: Settings) -> int:
    routes = load_routes(settings.routes_file)
    for r in routes:
        read_route_key(r)  # raises if a configured key cannot be read; the value is not printed
    print("llm-race-gateway %s: %d routes %s; public model %r, routes model %r; first token %.2fs, "
          "timeout %.1fs, race share %.2f, recheck %.0fs; listen %s:%d" % (
              __version__, len(routes), [r.name for r in routes], settings.model, settings.routes_group,
              settings.first_token_secs, settings.timeout_secs, settings.share, settings.recheck_secs,
              settings.host, settings.port))
    unpriced = [r.name for r in routes if not r.priced]
    if unpriced:
        print("unpriced routes (cost logged as 'unpriced'): %s" % unpriced)
    return 0


def cmd_serve(settings: Settings) -> int:
    # Start offline: LiteLLM otherwise fetches its model price map from the network at import.
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("LITELLM_TELEMETRY", "False")
    os.environ["LITELLM_MASTER_KEY"] = ensure_master_key(settings.master_key_file)
    routes = prepare(settings)
    install_runtime(settings, routes)
    log.info("llm_race: %s %d routes %s, first token %.2fs, race share %.2f, recheck %.0fs, %s:%d", __version__,
             len(routes), [r.name for r in routes], settings.first_token_secs, settings.share, settings.recheck_secs,
             settings.host, settings.port)
    from litellm.proxy.proxy_cli import run_server

    sys.argv = ["litellm", "--config", str(config_path(settings)), "--host", settings.host,
                "--port", str(settings.port)]
    run_server()  # click command: exits the process
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="llm_race_gateway", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the proxy")
    sub.add_parser("check", help="validate the configuration and print a summary (no secrets)")
    args = parser.parse_args(argv)
    try:
        settings = load_settings()
        logging.basicConfig(level=settings.log_level.upper(),
                            format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        if args.command == "check":
            return cmd_check(settings)
        return cmd_serve(settings)
    except (ConfigError, OSError) as e:
        print("llm_race_gateway: %s" % e, file=sys.stderr)
        return 2
