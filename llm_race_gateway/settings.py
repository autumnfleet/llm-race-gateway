"""Configuration, standard library only: `LLM_RACE_*` environment variables, an optional KEY=VALUE
file, and a JSON routes file.

Precedence: an environment variable wins over the same key in the config file
(`LLM_RACE_CONFIG_FILE`), which wins over the default. The README's Config section lists every key.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

PREFIX = "LLM_RACE_"
CONFIG_FILE_VAR = PREFIX + "CONFIG_FILE"

#: key -> (type, default); None as default = required.
_KEYS: Dict[str, tuple] = {
    "ROUTES_FILE": (str, None),
    "MASTER_KEY_FILE": (str, None),
    "STATE_DIR": (str, None),
    "HOST": (str, "127.0.0.1"),
    "PORT": (int, 4000),
    "MODEL": (str, "race"),
    "FIRST_TOKEN_SECS": (float, 2.5),
    "TIMEOUT_SECS": (float, 15.0),
    "EMA_ALPHA": (float, 0.3),
    "SHARE": (float, 0.1),
    "FAILURE_PENALTY_WINDOWS": (float, 2.0),
    "LOG_LEVEL": (str, "INFO"),
}

_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Route:
    """One OpenAI-compatible upstream: where it is, which model, and how to find its key."""

    name: str
    base_url: str
    model: str
    #: Name of an environment variable that holds the key, or a file that holds it (one line).
    #: Neither: the upstream needs no key (a local server); a placeholder is sent.
    api_key_env: Optional[str] = None
    api_key_file: Optional[str] = None
    #: Merged into every request body sent to this route (e.g. a provider's reasoning switch).
    extra_body: Dict[str, Any] = field(default_factory=dict)
    #: USD per million tokens. Absent: the route is logged as "unpriced" rather than as free.
    price_prompt: Optional[float] = None
    price_completion: Optional[float] = None

    @property
    def priced(self) -> bool:
        return self.price_prompt is not None and self.price_completion is not None


@dataclass(frozen=True)
class Settings:
    routes_file: Path
    master_key_file: Path
    state_dir: Path
    host: str
    port: int
    model: str
    first_token_secs: float
    timeout_secs: float
    ema_alpha: float
    share: float
    failure_penalty_windows: float
    log_level: str

    @property
    def routes_group(self) -> str:
        """The model name of the routes themselves; clients may ask for it directly (no race)."""
        return self.model + "-routes"


def read_kv_file(path: Path) -> Dict[str, str]:
    """KEY=VALUE per line; blank lines and `#` comments skipped; surrounding quotes stripped."""
    out: Dict[str, str] = {}
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ConfigError("%s:%d: expected KEY=VALUE" % (path, n))
        key, value = (s.strip() for s in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        out[key] = value
    return out


def load_settings(environ: Optional[Mapping[str, str]] = None) -> Settings:
    environ = os.environ if environ is None else environ
    merged: Dict[str, str] = {}
    cfg = environ.get(CONFIG_FILE_VAR)
    if cfg:
        for k, v in read_kv_file(Path(cfg).expanduser()).items():
            if not k.startswith(PREFIX) or k[len(PREFIX):] not in _KEYS:
                raise ConfigError("%s: unknown key %s" % (cfg, k))
            merged[k] = v
    for k, v in environ.items():
        if k.startswith(PREFIX) and k != CONFIG_FILE_VAR:
            merged[k] = v
    values: Dict[str, Any] = {}
    for key, (typ, default) in _KEYS.items():
        raw = merged.get(PREFIX + key)
        if raw is None or raw == "":
            if default is None:
                raise ConfigError("%s%s is required" % (PREFIX, key))
            values[key.lower()] = default
            continue
        try:
            values[key.lower()] = typ(raw)
        except ValueError:
            raise ConfigError("%s%s: expected %s, got %r" % (PREFIX, key, typ.__name__, raw)) from None
    for key in ("routes_file", "master_key_file", "state_dir"):
        values[key] = Path(values[key]).expanduser()
    s = Settings(**values)
    if not _NAME.match(s.model):
        raise ConfigError("%sMODEL: %r is not a plain name" % (PREFIX, s.model))
    if not 0.0 <= s.share <= 1.0:
        raise ConfigError("%sSHARE must be in [0, 1], got %r" % (PREFIX, s.share))
    if not 0.0 < s.ema_alpha <= 1.0:
        raise ConfigError("%sEMA_ALPHA must be in (0, 1], got %r" % (PREFIX, s.ema_alpha))
    if s.first_token_secs <= 0 or s.timeout_secs < s.first_token_secs:
        raise ConfigError("need 0 < %sFIRST_TOKEN_SECS <= %sTIMEOUT_SECS" % (PREFIX, PREFIX))
    if s.failure_penalty_windows < 1.0:
        raise ConfigError("%sFAILURE_PENALTY_WINDOWS must be >= 1" % PREFIX)
    return s


_ROUTE_FIELDS = set(Route.__dataclass_fields__)


def load_routes(path: Path) -> List[Route]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ConfigError("%s: not JSON (%s)" % (path, e)) from None
    if not isinstance(raw, list) or not raw:
        raise ConfigError("%s: expected a non-empty JSON list of routes" % path)
    routes = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise ConfigError("%s: route %d is not an object" % (path, i))
        unknown = set(r) - _ROUTE_FIELDS
        missing = {"name", "base_url", "model"} - set(r)
        if unknown or missing:
            raise ConfigError("%s: route %d: unknown %s, missing %s" % (path, i, sorted(unknown), sorted(missing)))
        if r.get("api_key_env") and r.get("api_key_file"):
            raise ConfigError("%s: route %r: give api_key_env or api_key_file, not both" % (path, r["name"]))
        if not isinstance(r["name"], str) or not _NAME.match(r["name"]):
            raise ConfigError("%s: route %d: name %r is not a plain name" % (path, i, r["name"]))
        routes.append(Route(**r))
    names = [r.name for r in routes]
    if len(set(names)) != len(names):
        raise ConfigError("%s: route names must be unique, got %s" % (path, names))
    return routes


def read_route_key(route: Route, environ: Optional[Mapping[str, str]] = None) -> Optional[str]:
    """The route's key, read in-process. Never logged, never written to the generated config."""
    environ = os.environ if environ is None else environ
    if route.api_key_env:
        value = environ.get(route.api_key_env, "").strip()
        if not value:
            raise ConfigError("route %r: environment variable %s is empty or unset" % (route.name, route.api_key_env))
        return value
    if route.api_key_file:
        value = Path(route.api_key_file).expanduser().read_text(encoding="utf-8").strip()
        if not value:
            raise ConfigError("route %r: key file is empty" % route.name)
        return value
    return None
