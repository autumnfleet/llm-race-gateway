# Changelog

## 0.1.0 (2026-10-20)

- First release. A LiteLLM proxy (`litellm[proxy]==1.106.0`) that ranks OpenAI-compatible routes by
  a moving average of time to first chunk, fails over inside a first-token window, and races a
  share of requests against a second route, timing the loser too.
- Configuration through `LLM_RACE_*` environment variables and an optional KEY=VALUE file.
- Consumers: nothing to migrate.
