"""The routing policy, standard library only: rank routes by a moving average of time to first chunk.

Rules:

- Every answered request is a sample of its route's exponential moving average (EMA) of the
  seconds to the first streamed chunk. The route with the lowest average answers next.
- A route that misses the first-token window, or refuses, is not averaged in: its average becomes
  AT LEAST a penalty (default two windows), so one stall demotes it at once.
- A route with no measurement yet ranks as if it took exactly one window: behind every route that
  answered inside the window, ahead of every route that failed.
- Before anything is measured all routes tie and the configured order decides, so the first route
  listed is the default until evidence says otherwise.
"""

from __future__ import annotations

import logging
import random
import threading
from typing import Dict, Iterable, List, Optional, Sequence

log = logging.getLogger("llm_race_gateway")

#: A failure is recorded as this many first-token windows: it is known only to be longer than one
#: window, and must rank below any route that answered inside it.
DEFAULT_FAILURE_PENALTY_WINDOWS = 2.0


class RouteRanker:
    def __init__(self, names: Sequence[str], alpha: float, prior_secs: float):
        if not names:
            raise ValueError("a ranker needs at least one route")
        if len(set(names)) != len(names):
            raise ValueError("route names must be unique, got %r" % (list(names),))
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1], got %r" % (alpha,))
        self._names = list(names)
        self._alpha = alpha
        self._prior = prior_secs
        self._ema: Dict[str, Optional[float]] = dict.fromkeys(self._names)
        self._samples: Dict[str, int] = dict.fromkeys(self._names, 0)
        # LiteLLM calls the logging hooks from more than one thread (sync and async paths).
        self._lock = threading.Lock()

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def ranked(self) -> List[str]:
        order = {n: i for i, n in enumerate(self._names)}

        def key(n: str):
            ema = self._ema[n]
            return (self._prior if ema is None else ema, order[n])

        return sorted(self._names, key=key)

    def observe(self, name: str, secs: float) -> None:
        """A first chunk arrived after `secs`: an ordinary sample of the moving average."""
        self._update(name, secs, floor=0.0)

    def fail(self, name: str, penalty_secs: float) -> None:
        """No first chunk in time, or a refusal: the average becomes AT LEAST the penalty.

        Not smoothed: with alpha 0.3 a smoothed penalty can leave a failing default ahead of a
        healthy backup (0.1 s average + one 5 s penalty = 1.57 s, still ahead of a 2 s backup).
        """
        self._update(name, penalty_secs, floor=penalty_secs)

    def _update(self, name: str, secs: float, *, floor: float) -> None:
        with self._lock:
            if name not in self._ema:
                return
            before = self.ranked()[0]
            prev = self._ema[name]
            ema = secs if prev is None else self._alpha * secs + (1 - self._alpha) * prev
            self._ema[name] = max(ema, floor)
            self._samples[name] += 1
            after = self.ranked()[0]
        if after != before:
            log.warning("llm_race: default is now %r (was %r); %s", after, before, self.describe())

    def describe(self) -> str:
        parts = []
        for n in self.ranked():
            ema = self._ema[n]
            parts.append("%s=%s/%d" % (n, "-" if ema is None else "%.2fs" % ema, self._samples[n]))
        return ", ".join(parts)

    def snapshot(self) -> Dict[str, dict]:
        return {n: {"ema_s": self._ema[n], "samples": self._samples[n]} for n in self._names}


def pick(ranked: Sequence[str], available: Iterable[str], tried: Iterable[str], probe: bool,
         rng: random.Random) -> "tuple[str, str]":
    """Choose one route for one provider call. Returns (route, why), why in best / retry / probe.

    `ranked` is the ranker's order, `available` the routes the router can serve (a route the
    ranker does not know goes last, in the given order), `tried` the routes this request already
    used (LiteLLM's retry breadcrumbs). A retry never repeats a tried route while an untried one is
    left: a timeout is retryable, so LiteLLM would otherwise re-ask the route that just stalled. A
    probe asks for any route but the best, chosen at random, so a slower route keeps being measured.
    """
    available = list(available)
    tried = set(tried)
    order = [n for n in ranked if n in available] + [n for n in available if n not in ranked]
    if not order:
        raise ValueError("no route available")
    candidates = [n for n in order if n not in tried] or order
    if probe and not tried and len(candidates) > 1:
        return rng.choice(candidates[1:]), "probe"
    return candidates[0], ("retry" if tried else "best")
