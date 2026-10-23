# llm-race-gateway

An OpenAI-compatible gateway for latency-critical chat, for example a voice agent where the user waits in silence for the first word. You give it several routes that can answer the same request: different providers or models, each with its own base URL and key. It serves them as one model and keeps answering through whichever route is fastest right now.

- **Ranking by first token.** Every answered request is a sample of its route's exponential moving average (EMA) of seconds to the first streamed chunk. The route with the lowest average answers the next request.
- **Failover inside a window.** A route that sends no chunk within the first-token window is abandoned and the request moves to the next route at once, with no backoff. The miss is recorded as a penalty of at least two windows, so a stalled default is demoted on the spot instead of being averaged away.
- **Racing a share of requests.** On a configurable share of requests a second route, never the current best, is asked at the same time. The caller hears whichever answers first. The loser is still read to its first chunk and timed, so slower routes keep being measured and a recovered route can win back first place.
- **Rechecking a demoted route.** A route ranked behind one configured after it, and unmeasured for ten minutes (configurable), is raced on the next real streamed request. No request is ever sent just to measure, so an idle gateway costs nothing. The first answer after a failure replaces the penalty instead of being averaged into it, so one fast answer is enough to win first place back.
- **Cost per request** in the log, for routes with a configured price. Unpriced routes are logged as `unpriced`, not as free.

It is a [LiteLLM](https://github.com/BerriAI/litellm) proxy with a custom routing strategy, a logging callback and a custom provider for the race. It is not a fork. LiteLLM still owns the HTTP server, auth, key handling and the OpenAI wire format, including streaming, tool calls and usage.

Version: see `VERSION`. Status: experimental. Maintained by Autumn Fleet.

```
llm_race_gateway/ranker.py        the policy: RouteRanker (EMA, penalties) and pick() (best / retry / probe)   stdlib
llm_race_gateway/settings.py      LLM_RACE_* variables, the KEY=VALUE file, the routes file                   stdlib
llm_race_gateway/proxy_config.py  settings + routes -> the LiteLLM proxy config                                stdlib
llm_race_gateway/routing.py       EmaRouting: LiteLLM routing strategy + success/failure callback              litellm
llm_race_gateway/race.py          RaceLLM: the custom provider that serves the public model                    litellm
llm_race_gateway/cli.py           `python3 -m llm_race_gateway serve|check`
```

### Why not LiteLLM's own routing

These were measured on LiteLLM 1.106.0 before writing any code here:

| LiteLLM feature | What happened |
|---|---|
| `routing_strategy: latency-based-routing` | Ranks by whole-response latency, or by time to first token only when streaming. It has no first-token deadline of its own and no penalty floor, so one stall is smoothed into the average. |
| Fallbacks with default retries | About 5.5 s to fail over, because of exponential backoff on the dead route first. With `num_retries` per route and `retry_after: 0`, as configured here: 0.33 to 0.43 s. |
| Retry after a timeout | A timeout is retryable, and LiteLLM may hand the retry to the route that just stalled. The strategy here skips routes this request already tried. |
| `fastest_response` (built-in race) | Cancels the loser before it is timed, and in the test credited the win to the wrong deployment. |
| Custom-provider streaming | `optional_params` renames `max_completion_tokens` to `max_tokens`, and the provider's usage was replaced by LiteLLM's own estimate (29 reported against a real 7). The race forwards the client's original body and re-emits the provider's usage. |

## Install

Python 3.10 to 3.14. Pin a release tag:

```sh
python3 -m venv .venv
.venv/bin/pip install "git+https://github.com/autumnfleet/llm-race-gateway@v0.1.1"
.venv/bin/llm-race-gateway check        # or: python -m llm_race_gateway check
```

This installs **`litellm[proxy]==1.106.0`**, pinned exactly because the integration relies on LiteLLM behaviour that is not a public API (see Compatibility contract). `ranker`, `settings` and `proxy_config` use only the standard library (3.9+), so they can be used without LiteLLM.

Example systemd user unit (paths are placeholders):

```ini
[Service]
EnvironmentFile=%h/.config/llm-race/gateway.env
ExecStart=/opt/llm-race-gateway/.venv/bin/llm-race-gateway serve
Restart=on-failure
```

## Config

All settings come from environment variables with the prefix `LLM_RACE_`. They can also come from a KEY=VALUE file whose path is in `LLM_RACE_CONFIG_FILE`, with one `KEY=VALUE` per line and `#` for comments. An environment variable wins over the same key in the file. An unknown `LLM_RACE_*` key in the file is an error, so a typo cannot pass silently.

| Variable | Default | Meaning |
|---|---|---|
| `LLM_RACE_ROUTES_FILE` | required | JSON file listing the routes (below). |
| `LLM_RACE_MASTER_KEY_FILE` | required | The key clients present as their API key. Created with a random value and mode 600 if it does not exist. |
| `LLM_RACE_STATE_DIR` | required | Where the generated LiteLLM config is written (`litellm-config.json`, directory mode 700). It never holds a key. |
| `LLM_RACE_HOST` | `127.0.0.1` | Listen address. |
| `LLM_RACE_PORT` | `4000` | Listen port. |
| `LLM_RACE_MODEL` | `race` | The public model name clients ask for. `<MODEL>-routes` is the routes group, which clients may also ask for directly: ranked and failed over, but never raced. |
| `LLM_RACE_FIRST_TOKEN_SECS` | `2.5` | The first-token window. LiteLLM applies it as a read timeout, so it also bounds the gap between chunks. |
| `LLM_RACE_TIMEOUT_SECS` | `15` | Total budget for one request. |
| `LLM_RACE_EMA_ALPHA` | `0.3` | Weight of the newest sample in the moving average. |
| `LLM_RACE_SHARE` | `0.1` | Share of requests to the public model that also race a second route. `0` disables racing. |
| `LLM_RACE_RECHECK_SECS` | `600` | A demoted route (ranked behind a route configured after it) unmeasured for this long is raced on the next streamed request to the public model (a non-streamed request is never raced). `0` disables rechecks. |
| `LLM_RACE_FAILURE_PENALTY_WINDOWS` | `2` | A failed or stalled attempt sets the route's average to at least this many windows. |
| `LLM_RACE_LOG_LEVEL` | `INFO` | Python logging level. |
| `LLM_RACE_CONFIG_FILE` | unset | Optional KEY=VALUE file (above). |

The launcher also sets `LITELLM_LOCAL_MODEL_COST_MAP=True`, so LiteLLM starts without fetching its price map from the network, and `LITELLM_TELEMETRY=False`. Both are defaults that your environment can override. It sets `LITELLM_MASTER_KEY` from the master key file, and `LLM_RACE_ROUTE_KEY_<i>` for each route's key, inside its own process only.

**Routes file.** A JSON list. The order matters: until anything has been measured, the first route is the default.

```json
[
  {"name": "primary", "base_url": "https://api.example.org/v1", "model": "<provider-model-id>",
   "api_key_env": "PRIMARY_API_KEY", "extra_body": {"reasoning_effort": "none"}},
  {"name": "backup", "base_url": "https://other.example.net/openai/v1", "model": "<provider-model-id>",
   "api_key_file": "~/.config/llm-race/backup.key", "price_prompt": 0.15, "price_completion": 0.6}
]
```

| Field | Meaning |
|---|---|
| `name` | Route id: letters, digits, `_ . -`. Used in logs and the `x-litellm-model-id` response header. |
| `base_url`, `model` | Any OpenAI-compatible chat-completions endpoint and the provider's own model id. |
| `api_key_env` or `api_key_file` | Where the key is: the name of an environment variable, or a file holding it. Give at most one. Give neither for an upstream that takes no key. Keys are read in-process and never written to the config, argv or the log. |
| `extra_body` | Merged into every request body sent to this route, for example a provider-specific reasoning switch. |
| `price_prompt`, `price_completion` | USD per million tokens. Without both, the route is logged as `unpriced`. |

`python3 -m llm_race_gateway check` validates everything and checks that every key can be read. It prints a summary without any key.

## Compatibility contract

Stable within a major version:

- The commands `llm-race-gateway serve|check` and `python3 -m llm_race_gateway serve|check`, and exit code 2 on a configuration error.
- Every `LLM_RACE_*` variable above, its meaning and its default, the KEY=VALUE file format, and the routes-file fields.
- The HTTP surface. It is LiteLLM's OpenAI-compatible proxy: `POST /v1/chat/completions` (streaming and not), `Authorization: Bearer <master key>`, and `GET /health/liveliness`. The public model is `LLM_RACE_MODEL`, the routes group `<MODEL>-routes`, and `x-litellm-model-id` names the route on direct routes-group calls.
- What passes through unchanged to the route: the client's request body, minus `model`, `messages`, `stream`, `metadata` and `user`, plus the route's `extra_body`. That includes `max_completion_tokens` under that name, `tools`, `temperature` and `stream_options`.
- What comes back: text and tool-call deltas as the route sent them, the route's `finish_reason` (including `length`), and the route's own `usage` (including `completion_tokens_details.reasoning_tokens`).
- The probe flag `{"metadata": {"llm_race_probe": true}}` on a routes-group request, meaning "any route but the best". Since 0.1.1 the value may also be a route name, meaning "that route", unless it is the best or not served.
- The policy: the order before any measurement, the penalty floor, the first sample after a failure replacing the penalty (since 0.1.1), rechecks only for demoted routes and only on real requests, no repeat of a tried route while an untried one remains, and a probe never getting the best route.

Not part of the contract: log line wording; the generated LiteLLM config; the Python module API; and the LiteLLM version. It is pinned, and a minor version of this package may move the pin. The integration uses LiteLLM behaviour that is not a public API: `Router.set_custom_routing_strategy`, which is re-installed per request because `routing_strategy_init` resets it; `previous_models` retry breadcrumbs; `api_call_start_time` and `completion_start_time` in callback kwargs; `litellm_params.proxy_server_request`; and the custom-provider stream wrapper. The integration tests pin each of these, so run them before moving the pin.

## How to adapt

- **Policy.** `ranker.py` is plain Python with no LiteLLM in it. Change the ranking (for example a decaying penalty) there and keep `pick()`'s contract.
- **Race share per client.** Today the share is global. A client that never wants a race can ask for `<MODEL>-routes` directly.
- **Persistence.** The ranking lives in memory and starts from the configured order after a restart. `RouteRanker.snapshot()` is the state to save if you need it to survive restarts.
- **Spend totals.** The proxy keeps no spend history without a database. The per-request cost is in the log line `llm_race: sample <route> <secs>s cost=$...`. Add LiteLLM's database or your own callback if you need totals.
- **Keep local:** the routes file, keys, prices, and the choice of models. None of them belong in this repository.

Known limit: a recheck happens only while streamed requests arrive, so after a quiet spell a demoted route is re-measured on the first request, not before. A route that is not demoted is measured only by the random share of races, or when the routes ahead of it fail.

## Tests

```sh
python3 -m unittest
```

Without LiteLLM the policy, settings and config tests run and the rest skip cleanly. With `litellm[proxy]==1.106.0` installed in the interpreter that runs them, all of them run (about two minutes): they launch the real proxy against a fake upstream on `127.0.0.1`, with no network and no real key, and pin each LiteLLM fact listed above.

## License

MIT, see [LICENSE](LICENSE).
