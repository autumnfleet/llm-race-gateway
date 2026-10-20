"""Holds the routing and race instances the launcher builds; the proxy config names them.

LiteLLM resolves `llm_race_gateway.runtime.routing` with a normal import when no file of that name
sits next to the generated config, so it gets this module object back, with the instances the
launcher set.
"""

routing = None
race = None
