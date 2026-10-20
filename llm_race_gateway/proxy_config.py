"""Settings + routes -> the LiteLLM proxy config, standard library only.

Keys never enter the config: the launcher reads each route's key into this process's environment
as `LLM_RACE_ROUTE_KEY_<i>` and the config names that variable (`os.environ/...`), LiteLLM's own
indirection. Nothing secret reaches argv, the config file or a log line.

The config is written as JSON, which LiteLLM's YAML loader reads unchanged.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .settings import Route, Settings

#: The custom provider that serves the public model (`llm_race_gateway.race`).
RACE_PROVIDER = "llm-race"
#: Request metadata key that marks the second request of a race: "any route but the best".
PROBE_KEY = "llm_race_probe"
#: Placeholder sent to an upstream configured without a key.
NO_KEY = "no-key"


def route_key_env(i: int) -> str:
    return "LLM_RACE_ROUTE_KEY_%d" % i


def build_config(settings: Settings, routes: List[Route], keyed: List[bool]) -> Dict[str, Any]:
    """`keyed[i]`: route i has a key in `route_key_env(i)`."""
    model_list = []
    for i, r in enumerate(routes):
        params: Dict[str, Any] = {
            # "openai/" = LiteLLM's OpenAI-compatible provider; the rest is the provider's own id.
            "model": "openai/" + r.model,
            "api_base": r.base_url,
            "api_key": ("os.environ/" + route_key_env(i)) if keyed[i] else NO_KEY,
            # A read timeout, so it bounds the gaps between chunks too, not only the first one.
            "stream_timeout": settings.first_token_secs,
            "timeout": settings.timeout_secs,
            # The OpenAI client's own retries would re-ask a stalled route inside one attempt.
            "max_retries": 0,
        }
        if r.extra_body:
            params["extra_body"] = dict(r.extra_body)
        info: Dict[str, Any] = {"id": r.name}
        if r.price_prompt is not None:
            info["input_cost_per_token"] = r.price_prompt / 1e6
        if r.price_completion is not None:
            info["output_cost_per_token"] = r.price_completion / 1e6
        model_list.append({"model_name": settings.routes_group, "litellm_params": params, "model_info": info})
    # The public model. Its own deadline is the whole budget; the inner calls hold the first-token one.
    model_list.append({"model_name": settings.model,
                       "litellm_params": {"model": "%s/%s" % (RACE_PROVIDER, settings.model),
                                          "timeout": settings.timeout_secs},
                       "model_info": {"id": RACE_PROVIDER}})
    return {
        "model_list": model_list,
        "router_settings": {
            # One retry per other route: the routing strategy hands each retry the next untried route.
            "num_retries": len(routes) - 1,
            # LiteLLM's default backs off exponentially before a retry (measured ~5 s to fail over).
            "retry_after": 0,
            # The moving average is the health signal; a cooldown would hide a route from it.
            "disable_cooldowns": True,
        },
        "litellm_settings": {
            "callbacks": "llm_race_gateway.runtime.routing",
            "custom_provider_map": [{"provider": RACE_PROVIDER, "custom_handler": "llm_race_gateway.runtime.race"}],
            "drop_params": False,
            "request_timeout": settings.timeout_secs,
        },
        "general_settings": {},
    }
