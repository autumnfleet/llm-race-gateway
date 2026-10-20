"""The race, as a LiteLLM custom provider.

The public model is served by this provider. It asks LiteLLM's own router for the route group once,
or -- on a `share` of requests -- twice, the second marked as a probe so the routing strategy hands
it a route other than the best. The first stream to yield text or a tool call is passed through;
the loser is still read to its own first chunk, timed here, and then closed.

LiteLLM's built-in race (`fastest_response`) was measured and is not used: it cancels the loser
untimed, and its callback credited the wrong deployment with the win.

Chunks are yielded as `ModelResponseStream` objects, which LiteLLM's stream wrapper passes through
for a custom provider untouched -- tool-call deltas and all -- instead of the reduced
`GenericStreamingChunk` form.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, AsyncIterator, Dict, Optional, Set, Tuple

from litellm.llms.custom_llm import CustomLLM

from .proxy_config import PROBE_KEY
from .ranker import RouteRanker
from .routing import deployment_id

log = logging.getLogger("llm_race_gateway")

#: Keys of the client's request that are the outer call's business, not the route's.
_NOT_FORWARDED = {"model", "messages", "stream", "metadata", "user"}


def client_body(litellm_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The request exactly as the client sent it, when it came through the proxy.

    Forwarded instead of LiteLLM's `optional_params`, which for a custom provider is a TRANSLATION:
    measured on 1.106.0, `max_completion_tokens` arrived as `max_tokens` and LiteLLM's own
    `max_retries` was added. A client's token cap must reach the route under the name it used.
    """
    req = (litellm_params or {}).get("proxy_server_request")
    body = req.get("body") if isinstance(req, dict) else None
    return body if isinstance(body, dict) else None


def stream_route(stream) -> Optional[str]:
    """Which deployment a router stream came from."""
    hidden = getattr(stream, "_hidden_params", None) or {}
    if isinstance(hidden.get("model_id"), str):
        return hidden["model_id"]
    logging_obj = getattr(stream, "logging_obj", None)
    return deployment_id(getattr(logging_obj, "model_call_details", None))


def _router():
    from litellm.proxy import proxy_server

    return proxy_server.llm_router


async def _close(stream) -> None:
    for name in ("aclose", "close"):
        fn = getattr(stream, name, None)
        if fn is not None:
            try:
                r = fn()
                if asyncio.iscoroutine(r):
                    await r
            except Exception as e:  # noqa: BLE001 -- closing a loser must never fail the winner
                log.debug("llm_race: closing a stream raised %r", e)
            return


async def _usage_on_finish(lead: list, it) -> AsyncIterator:
    """Pass content and tool-call chunks through; end with ONE generic chunk: finish + real usage.

    LiteLLM's stream wrapper, for a custom provider, drops a `ModelResponseStream` chunk that has a
    finish_reason and no content, and ends the stream at the first content-less chunk after it. The
    provider's usage rides on exactly such chunks, so it was lost and the client got LiteLLM's own
    token ESTIMATE instead (measured: 29 reported for a provider's 7). Clients that record usage,
    reasoning tokens included, need the provider's own counts.
    The wrapper's generic-chunk path does carry a finish with usage, so the ending is sent in that
    form; it judges each chunk's form separately, so the two can share one stream.
    """
    finish, usage = None, None
    pending = list(lead)
    chunk = pending.pop(0) if pending else await _next(it)
    while chunk is not None:
        if _has_content(chunk, finish_counts=False):
            yield chunk
        for c in chunk.choices or ():
            finish = c.finish_reason or finish
        usage = getattr(chunk, "usage", None) or usage
        chunk = pending.pop(0) if pending else await _next(it)
    yield {"text": "", "tool_use": None, "is_finished": True, "finish_reason": finish or "stop", "index": 0,
           "usage": usage.model_dump(exclude_none=True) if usage is not None else None}


def _has_content(chunk, finish_counts: bool = True) -> bool:
    """Text or a tool-call delta (or, if `finish_counts`, a finish) -- not a bare usage record.

    The usage chunk measured through LiteLLM's router carries ONE choice with an empty delta, not
    an empty `choices` list as the OpenAI wire format has it, so emptiness is judged per delta."""
    return any((finish_counts and c.finish_reason) or (c.delta and (c.delta.content or c.delta.tool_calls))
               for c in (chunk.choices or ()))


async def _next(it):
    try:
        return await it.__anext__()
    except StopAsyncIteration:
        return None


class RaceLLM(CustomLLM):
    def __init__(self, inner_group: str, share: float, ranker: Optional[RouteRanker] = None,
                 rng: Optional[random.Random] = None):
        super().__init__()
        self.inner_group = inner_group
        self.ranker = ranker
        self.share = share
        self._rng = rng or random.Random()
        self._losers: Set[asyncio.Task] = set()

    def _inner_kwargs(self, messages, params: Dict[str, Any], probe: bool) -> Dict[str, Any]:
        kw = {k: v for k, v in (params or {}).items() if k not in _NOT_FORWARDED}
        kw.update(model=self.inner_group, messages=messages, stream=True)
        if probe:
            kw["metadata"] = {PROBE_KEY: True}
        return kw

    async def _contend(self, kwargs) -> Tuple[Any, Any, Any, float]:
        t0 = time.monotonic()
        stream = await _router().acompletion(**kwargs)
        it = stream.__aiter__()
        lead: list = []
        try:
            # A contender has answered only when it says something: LiteLLM turns an upstream that
            # sent nothing into a content-less chunk, which won a race (measured) and
            # left the caller with an empty reply.
            while True:
                chunk = await _next(it)
                if chunk is None:
                    raise ConnectionError("the route's stream ended without any text or tool call")
                lead.append(chunk)
                if _has_content(chunk, finish_counts=False):
                    break
        except BaseException:
            await _close(stream)
            raise
        return stream, it, lead, time.monotonic() - t0

    async def _settle(self, task: asyncio.Task) -> None:
        """Let a loser reach its first chunk, time it, then close it.

        Timed HERE because LiteLLM logs a success only for a stream that is exhausted or closed by
        the proxy's own client handling: measured, a router stream closed in-process
        after its first chunk produced no callback at all, so the loser was never ranked. A loser
        that fails instead is ranked by the failure callback as usual."""
        try:
            stream, _it, _lead, secs = await task
        except BaseException:
            return
        name = stream_route(stream)
        await _close(stream)
        if self.ranker is not None and name in self.ranker.names:
            self.ranker.observe(name, secs)
            log.info("llm_race: sample %s %.3fs (race loser); %s", name, secs, self.ranker.describe())

    async def astreaming(self, model, messages, api_base=None, custom_prompt_dict=None, model_response=None,
                         print_verbose=None, encoding=None, api_key=None, logging_obj=None,
                         optional_params=None, acompletion=None, litellm_params=None, logger_fn=None,
                         headers=None, timeout=None, client=None) -> AsyncIterator:
        body = client_body(litellm_params)
        if body is not None:
            optional_params, messages = body, body.get("messages", messages)
        else:
            optional_params = {k: v for k, v in (optional_params or {}).items() if k != "max_retries"}
        race = self._rng.random() < self.share
        t0 = time.monotonic()
        tasks = {asyncio.create_task(self._contend(self._inner_kwargs(messages, optional_params, False))): "lead"}
        if race:
            tasks[asyncio.create_task(self._contend(self._inner_kwargs(messages, optional_params, True)))] = "probe"
        winner = None
        errors = []
        try:
            pending = set(tasks)
            while pending and winner is None:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in sorted(done, key=lambda t: tasks[t] != "lead"):
                    if task.exception() is not None:
                        errors.append(task.exception())
                    elif winner is None:
                        winner = task
                    else:
                        self._keep(self._settle(task))
            for task in pending:
                self._keep(self._settle(task))
        except BaseException:
            for task in tasks:
                if task is not winner:
                    task.cancel()
            if winner is not None:
                await _close(winner.result()[0])
            raise
        if winner is None:
            raise errors[0]
        stream, it, lead, _secs = winner.result()
        if race:
            log.info("llm_race: raced; %s answered first in %.2fs", tasks[winner], time.monotonic() - t0)
        try:
            async for chunk in _usage_on_finish(lead, it):
                yield chunk
        finally:
            await _close(stream)

    def _keep(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._losers.add(task)
        task.add_done_callback(self._losers.discard)

    async def acompletion(self, model, messages, api_base=None, custom_prompt_dict=None, model_response=None,
                          print_verbose=None, encoding=None, api_key=None, logging_obj=None,
                          optional_params=None, acompletion=None, litellm_params=None, logger_fn=None,
                          headers=None, timeout=None, client=None):
        body = client_body(litellm_params)
        if body is not None:
            optional_params, messages = body, body.get("messages", messages)
        kw = self._inner_kwargs(messages, dict(optional_params or {}), False)
        kw["stream"] = False
        return await _router().acompletion(**kw)
