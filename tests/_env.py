"""Shared test helpers: the package on sys.path, a fake OpenAI-compatible upstream, a stdlib client.

Nothing here touches the network beyond 127.0.0.1, and no real key is used.
"""

import importlib.util
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PACKAGE_DIR not in sys.path:
    sys.path.insert(0, PACKAGE_DIR)

HAVE_LITELLM = importlib.util.find_spec("litellm") is not None
SKIP_REASON = "needs litellm[proxy] (see README, Tests)"
_proxy_ok = None


def require_proxy():
    """Skip unless LiteLLM's proxy imports in a child interpreter.

    A plain `litellm` without the `[proxy]` extra imports fine and fails only when the proxy starts,
    so finding the package is not enough. Checked once, in a child, so a broken install cannot
    leave half-imported modules in the test process.
    """
    global _proxy_ok
    if _proxy_ok is None:
        import subprocess

        _proxy_ok = HAVE_LITELLM and subprocess.run(
            [sys.executable, "-c", "import litellm.proxy.proxy_server"], stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, env=dict(os.environ, LITELLM_LOCAL_MODEL_COST_MAP="True"),
            timeout=120).returncode == 0
    if not _proxy_ok:
        raise unittest.SkipTest(SKIP_REASON)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Upstream:
    """`POST /<route>/v1/chat/completions` streams a short reply after `delay[route]` seconds.

    Headers go out only with the first chunk, as some real providers do, so a delay is a delay to
    the first byte. Every request body is recorded per route.
    """

    def __init__(self):
        self.delay = {}
        self.bodies = {}
        #: route -> finish_reason of its text replies; routes whose finish rides on the last text chunk.
        self.finish = {}
        self.finish_on_content = set()
        #: route -> HTTP status to refuse with; routes whose stream ends with no chunk at all.
        self.status = {}
        self.empty = set()
        self._lock = threading.Lock()
        upstream = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                upstream._handle(self)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self):
        self._server.shutdown()
        self._server.server_close()

    def url(self, route):
        return "http://127.0.0.1:%d/%s/v1" % (self.port, route)

    def _handle(self, h):
        parts = h.path.strip("/").split("/")
        route = parts[0]
        body = json.loads(h.rfile.read(int(h.headers.get("Content-Length", 0))) or b"{}")
        with self._lock:
            self.bodies.setdefault(route, []).append(body)
        time.sleep(self.delay.get(route, 0.0))
        if route in self.status:
            payload = json.dumps({"error": {"message": "refused by " + route, "type": "server_error"}}).encode()
            h.send_response(self.status[route])
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(payload)))
            h.end_headers()
            h.wfile.write(payload)
            return
        h.send_response(200)
        h.send_header("Content-Type", "text/event-stream")
        h.send_header("Connection", "close")
        h.end_headers()
        try:
            for event in self._events(route, body):
                h.wfile.write(("data: %s\n\n" % event).encode())
                h.wfile.flush()
                time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError):
            pass
        h.close_connection = True

    def _events(self, route, body):
        base = {"id": "c-" + route, "object": "chat.completion.chunk", "created": int(time.time()),
                "model": "model-" + route}

        def chunk(delta, finish=None, **extra):
            return json.dumps(dict(base, choices=[{"index": 0, "delta": delta, "finish_reason": finish}], **extra))

        if route in self.empty:
            yield "[DONE]"
            return
        if body.get("tools"):
            name = body["tools"][0]["function"]["name"]
            yield chunk({"role": "assistant", "tool_calls": [{"index": 0, "id": "call-" + route, "type": "function",
                                                              "function": {"name": name, "arguments": ""}}]})
            yield chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"city": "Paris"}'}}]})
            yield chunk({}, "tool_calls", usage={"prompt_tokens": 10, "completion_tokens": 7, "total_tokens": 17})
            yield "[DONE]"
            return
        finish = self.finish.get(route, "stop")
        words = ("Answer ", "from %s." % route)
        for i, word in enumerate(words):
            last = i == len(words) - 1 and route in self.finish_on_content
            yield chunk({"role": "assistant", "content": word}, finish if last else None)
        usage = {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                 "completion_tokens_details": {"reasoning_tokens": 3}}
        if route in self.finish_on_content:
            yield json.dumps(dict(base, choices=[], usage=usage))
        else:
            yield chunk({}, finish, usage=usage)
        yield "[DONE]"


def post_stream(base_url, key, body, timeout=30):
    """POST a streaming chat completion; returns (headers, parsed chunks, seconds to first text).

    `body["_stop_after_first_content"] = True` closes the stream right after the first text chunk.
    """
    stop_early = body.pop("_stop_after_first_content", False)
    req = urllib.request.Request(base_url + "/chat/completions", data=json.dumps(dict(body, stream=True)).encode(),
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
    chunks, first = [], None
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        headers = dict(resp.headers)
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            chunks.append(json.loads(data))
            if any((c.get("delta") or {}).get("content") for c in chunks[-1].get("choices") or ()):
                first = time.monotonic() - t0 if first is None else first
                if stop_early:
                    break
    return headers, chunks, first


def get(url, timeout=1):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status


HTTPError = urllib.error.HTTPError
URLError = urllib.error.URLError
