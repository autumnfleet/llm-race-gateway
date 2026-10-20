"""The LiteLLM side of the policy: pick a route by moving average, learn from every first chunk.

One object is both the routing strategy (which deployment answers) and the logging callback that
feeds it (how long each deployment took to its first chunk, or that it failed). Facts measured on
LiteLLM 1.106.0, each of which shaped a line below:

- The success callback carries `completion_start_time` on a stream, and still fires when the
  client closes the stream right after its first chunk.
- A client that aborts BEFORE the first chunk produces no callback at all, so the first-token
  deadline has to be LiteLLM's own (`stream_timeout`), which does produce a failure callback.
- `Router.set_custom_routing_strategy` is undone by `routing_strategy_init`, and the proxy offers
  no config key for it, so the strategy is (re)installed from the pre-call hook of every request.
- A timeout is retryable, so LiteLLM does not skip the deployment that just timed out on the retry;
  this strategy does, from the `previous_models` breadcrumbs LiteLLM leaves in the request metadata.
- `start_time` is the whole request's start, so after a failover it would charge the answering
  route the dead route's wait; `api_call_start_time` is this provider call's own.
"""

from __future__ import annotations

import logging
import random
from typing import Any, Mapping, Optional, Set

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.router import CustomRoutingStrategyBase

from .proxy_config import PROBE_KEY
from .ranker import RouteRanker, pick

log = logging.getLogger("llm_race_gateway")


def deployment_id(obj: Optional[Mapping[str, Any]]) -> Optional[str]:
    """The deployment id (the route name) from a model_list entry or a callback's kwargs."""
    if not isinstance(obj, Mapping):
        return None
    params = obj.get("litellm_params") or {}
    for holder in (obj, params, params.get("metadata") or {}, obj.get("metadata") or {}):
        info = holder.get("model_info") if isinstance(holder, Mapping) else None
        if isinstance(info, Mapping) and isinstance(info.get("id"), str):
            return info["id"]
    return None


def _metadata(request_kwargs: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not isinstance(request_kwargs, Mapping):
        return {}
    for key in ("litellm_metadata", "metadata"):
        md = request_kwargs.get(key)
        if isinstance(md, Mapping):
            return md
    return {}


def _tried(md: Mapping[str, Any]) -> Set[str]:
    out = set()
    for crumb in md.get("previous_models") or ():
        if isinstance(crumb, Mapping) and isinstance(crumb.get("deployment_id"), str):
            out.add(crumb["deployment_id"])
    return out


def _is_probe(md: Mapping[str, Any]) -> bool:
    if md.get(PROBE_KEY):
        return True
    # A client's own `metadata` object arrives nested under requester metadata on some paths.
    inner = md.get("requester_metadata")
    return bool(isinstance(inner, Mapping) and inner.get(PROBE_KEY))


class EmaRouting(CustomLogger, CustomRoutingStrategyBase):
    def __init__(self, ranker: RouteRanker, first_token_secs: float, failure_penalty_windows: float,
                 priced: Optional[Set[str]] = None, rng: Optional[random.Random] = None):
        super().__init__()
        self.ranker = ranker
        self.first_token_secs = first_token_secs
        self.failure_penalty_windows = failure_penalty_windows
        #: Routes with a configured price; LiteLLM reports $0 for the rest, which would read as free.
        self.priced = set(priced or ())
        self._rng = rng or random.Random()
        self._router = None

    # -- install ----------------------------------------------------------------------------------

    def install(self, router) -> None:
        if router is None:
            return
        current = router.__dict__.get("async_get_available_deployment")
        if getattr(current, "__self__", None) is not self:
            router.set_custom_routing_strategy(self)
            self._router = router
            log.info("llm_race: routing installed over %s", self.ranker.names)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        from litellm.proxy import proxy_server

        self.install(proxy_server.llm_router)
        return data

    # -- choose -----------------------------------------------------------------------------------

    def choose(self, model: str, request_kwargs: Optional[Mapping[str, Any]]) -> dict:
        group = [d for d in self._router.model_list if d.get("model_name") == model]
        if not group:
            raise ValueError("llm_race: no deployment serves model %r" % (model,))
        by_id = {deployment_id(d): d for d in group}
        if not set(by_id) & set(self.ranker.names):
            return group[0]  # not the route group (the public model, served by the race provider)
        md = _metadata(request_kwargs)
        tried = _tried(md)
        name, why = pick(self.ranker.ranked(), [i for i in by_id if i is not None], tried, _is_probe(md), self._rng)
        log.info("llm_race: %s -> %s (%s; tried=%s; %s)", model, name, why, sorted(tried), self.ranker.describe())
        return by_id[name]

    async def async_get_available_deployment(self, model, messages=None, input=None,
                                             specific_deployment=False, request_kwargs=None):
        return self.choose(model, request_kwargs)

    def get_available_deployment(self, model, messages=None, input=None,
                                 specific_deployment=False, request_kwargs=None):
        return self.choose(model, request_kwargs)

    # -- learn ------------------------------------------------------------------------------------

    @staticmethod
    def _attempt_start(kwargs, start_time):
        return kwargs.get("api_call_start_time") or start_time

    def _observe(self, kwargs, start_time, end_time) -> None:
        name = deployment_id(kwargs)
        if name not in self.ranker.names:
            return  # the race provider's own outer call, or something unidentified
        first = kwargs.get("completion_start_time") or end_time
        secs = max(0.0, (first - self._attempt_start(kwargs, start_time)).total_seconds())
        self.ranker.observe(name, secs)
        # The proxy keeps no spend without a database; the per-request cost lives in this line.
        cost = kwargs.get("response_cost")
        log.info("llm_race: sample %s %.3fs cost=%s; %s", name, secs,
                 "$%.6f" % cost if cost is not None and name in self.priced else "unpriced",
                 self.ranker.describe())

    def _fail(self, kwargs, start_time, end_time) -> None:
        name = deployment_id(kwargs)
        if name not in self.ranker.names:
            return
        exc = kwargs.get("exception")
        log.warning("llm_race: route %s failed after %.2fs (%s)", name,
                    (end_time - self._attempt_start(kwargs, start_time)).total_seconds(), type(exc).__name__)
        self.ranker.fail(name, self.failure_penalty_windows * self.first_token_secs)

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._observe(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._fail(kwargs, start_time, end_time)
