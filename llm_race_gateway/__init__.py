"""llm-race-gateway: a LiteLLM proxy that ranks OpenAI-compatible routes by time to first chunk,
fails over inside a first-token window, and races a share of requests against a second route."""

from pathlib import Path


def _version() -> str:
    # A source checkout or vendored copy carries VERSION beside the package; an installed wheel
    # carries the version in its metadata instead.
    path = Path(__file__).resolve().parent.parent / "VERSION"
    if path.exists():
        return path.read_text().strip()
    try:
        from importlib.metadata import version

        return version("llm-race-gateway")
    except Exception:  # noqa: BLE001
        return "0+unknown"


__version__ = _version()
