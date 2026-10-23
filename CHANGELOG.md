# Changelog

## 0.1.1 (2026-10-23)

- Rechecks: a demoted route (ranked behind a route configured after it) that has gone unmeasured
  for `LLM_RACE_RECHECK_SECS` (default 600) is raced on the next real streamed request. No
  background requests. Before, with a share of 0.1 and two routes, recovery after one stall took about 80
  requests.
- The first sample after a failure replaces the penalty instead of being smoothed into it, so one
  fast answer wins first place back. Only an attempt that started after the failure counts.
- A probe is no longer retried: a failed probe used to be retried onto the lead's route, asking
  it twice and charging it the probe's wait.
- The probe flag may name a route (`"llm_race_probe": "<route>"`).
- Consumers: nothing to migrate. Set `LLM_RACE_RECHECK_SECS=0` to keep 0.1.0's behaviour on rechecks.

## 0.1.0 (2026-10-20)

- First release. A LiteLLM proxy (`litellm[proxy]==1.106.0`) that ranks OpenAI-compatible routes by
  a moving average of time to first chunk, fails over inside a first-token window, and races a
  share of requests against a second route, timing the loser too.
- Configuration through `LLM_RACE_*` environment variables and an optional KEY=VALUE file.
- Consumers: nothing to migrate.
