import json
import os
import stat
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import devin_bridge as b


def answer(text):
    """Echo the last line; a last line starting with 'JSON:' is returned verbatim (tool-call replies)."""
    last = text.splitlines()[-1]
    return last[5:] if last.startswith("JSON:") else "echo: " + last


class MockDevin(BaseHTTPRequestHandler):
    """Fake Devin v3: a session answers 'echo: <last user text>' and then waits for the user."""
    sessions = {}
    calls = []

    def log_message(self, *a):
        pass

    def reply(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockDevin.calls.append((self.path, body, self.headers["Authorization"]))
        if self.path.endswith("/sessions"):
            sid = f"devin-{len(self.sessions)}"
            text = body["prompt"]
            self.sessions[sid] = [("user", text), ("devin", answer(text))]
            return self.reply(200, {"session_id": sid, "url": "http://x/" + sid, "status": "running"})
        sid = self.path.split("/")[-2]
        if sid not in self.sessions:
            return self.reply(404, {"detail": "no session"})
        self.sessions[sid] += [("user", body["message"]), ("devin", answer(body["message"]))]
        self.reply(200, {})

    def do_GET(self):
        parts = self.path.split("?")[0].split("/")
        if parts[-1] == "messages":
            items = [{"event_id": f"e{i}", "source": s, "message": m, "created_at": i}
                     for i, (s, m) in enumerate(self.sessions[parts[-2]])]
            return self.reply(200, {"items": items, "has_next_page": False})
        self.reply(200, {"session_id": parts[-1], "url": "http://x", "status": "running",
                         "status_detail": "waiting_for_user"})


def post(url, body, key=None):
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(url, json.dumps(body).encode(), h)
    return urllib.request.urlopen(req, timeout=10)


class BridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mock = ThreadingHTTPServer(("127.0.0.1", 0), MockDevin)
        cls.bridge = b.make_server("127.0.0.1", 0)
        for s in (cls.mock, cls.bridge):
            threading.Thread(target=s.serve_forever, daemon=True).start()
        os.environ.update(
            DEVIN_API_KEY="cog_test", DEVIN_ORG_ID="org-1", DEVIN_POLL="0.01",
            DEVIN_API_BASE=f"http://127.0.0.1:{cls.mock.server_port}/v3",
            BRIDGE_API_KEY="secret")
        cls.url = f"http://127.0.0.1:{cls.bridge.server_port}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.mock.shutdown()
        cls.bridge.shutdown()

    def setUp(self):
        MockDevin.calls.clear()
        b.SESSIONS.clear()
        b.WARM.clear()

    def chat(self, messages, **kw):
        r = post(f"{self.url}/chat/completions", {"model": "devin-fast", "messages": messages, **kw}, "secret")
        return r.read().decode()

    def test_auth_required(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            post(f"{self.url}/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        self.assertEqual(c.exception.code, 401)

    def test_models(self):
        req = urllib.request.Request(f"{self.url}/models", headers={"Authorization": "Bearer secret"})
        ids = [m["id"] for m in json.load(urllib.request.urlopen(req))["data"]]
        self.assertIn("devin", ids)
        self.assertIn("devin-fast", ids)

    def test_new_session_and_mode(self):
        out = json.loads(self.chat([{"role": "user", "content": "hello"}]))
        self.assertEqual(out["choices"][0]["message"]["content"], "echo: hello")
        path, body, auth = MockDevin.calls[0]
        self.assertEqual(path, "/v3/organizations/org-1/sessions")
        self.assertEqual(body["devin_mode"], "fast")
        self.assertEqual(auth, "Bearer cog_test")

    def test_followup_reuses_session(self):
        m = [{"role": "user", "content": "one"}]
        self.chat(m)
        m += [{"role": "assistant", "content": "echo: one"}, {"role": "user", "content": "two"}]
        out = json.loads(self.chat(m))
        self.assertEqual(out["choices"][0]["message"]["content"], "echo: two")
        posts = [c[0] for c in MockDevin.calls]
        self.assertEqual(sum(p.endswith("/sessions") for p in posts), 1)
        self.assertTrue(any(p.endswith("/messages") for p in posts))

    def test_stream(self):
        raw = self.chat([{"role": "user", "content": "hi"}], stream=True)
        events = [l[6:] for l in raw.splitlines() if l.startswith("data: ")]
        self.assertEqual(events[-1], "[DONE]")
        text = "".join(json.loads(e)["choices"][0]["delta"].get("content", "") for e in events[:-1])
        self.assertIn("echo: hi", text)

    def test_warm_session_used_for_default_model(self):
        b.spawn_warm()
        warm = b.WARM[0]
        out = json.loads(post(f"{self.url}/chat/completions",
                              {"model": "devin", "messages": [{"role": "user", "content": "hello"}]},
                              "secret").read())
        text = out["choices"][0]["message"]["content"]
        self.assertIn("hello", text)
        self.assertNotIn("READY", text)  # warm-up reply must not leak as the answer
        self.assertTrue(any(c[0].endswith(f"{warm}/messages") for c in MockDevin.calls))

    def test_openai_shape_and_usage(self):
        out = json.loads(self.chat([{"role": "system", "content": "be brief"}, {"role": "user", "content": "hello"}]))
        msg = out["choices"][0]
        self.assertIsNone(msg["message"]["refusal"])
        self.assertIsNone(msg["logprobs"])
        self.assertGreater(out["usage"]["prompt_tokens"], 0)
        self.assertEqual(out["usage"]["total_tokens"], out["usage"]["prompt_tokens"] + out["usage"]["completion_tokens"])

    def test_openai_error_shape(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            post(f"{self.url}/chat/completions", {"messages": [{"role": "user", "content": "x"}]})
        err = json.loads(c.exception.read())["error"]
        self.assertEqual((err["code"], err["param"]), ("invalid_api_key", None))

    def test_stream_include_usage(self):
        raw = self.chat([{"role": "user", "content": "hi"}], stream=True, stream_options={"include_usage": True})
        events = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        self.assertEqual(events[-1]["choices"], [])
        self.assertGreater(events[-1]["usage"]["total_tokens"], 0)

    TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
    CALL = 'JSON:{"tool_calls":[{"name":"get_weather","arguments":{"city":"Jakarta"}}]}'

    def test_openai_tool_call(self):
        out = json.loads(self.chat([{"role": "user", "content": self.CALL}], tools=self.TOOLS))
        ch = out["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        fn = ch["message"]["tool_calls"][0]["function"]
        self.assertEqual((fn["name"], json.loads(fn["arguments"])), ("get_weather", {"city": "Jakarta"}))
        self.assertIsNone(ch["message"]["content"])

    def test_openai_tool_call_stream(self):
        raw = self.chat([{"role": "user", "content": self.CALL}], tools=self.TOOLS, stream=True)
        events = [json.loads(l[6:]) for l in raw.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
        calls = [e["choices"][0]["delta"]["tool_calls"] for e in events if e["choices"] and "tool_calls" in e["choices"][0]["delta"]]
        self.assertEqual(calls[0][0]["function"]["name"], "get_weather")
        self.assertEqual(events[-1]["choices"][0]["finish_reason"], "tool_calls")

    def test_tool_result_continues_same_session(self):
        m = [{"role": "user", "content": self.CALL}]
        self.chat(m, tools=self.TOOLS)
        m += [{"role": "assistant", "content": None, "tool_calls": [
                  {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}]},
              {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]
        out = json.loads(self.chat(m, tools=self.TOOLS))
        self.assertIn("sunny", out["choices"][0]["message"]["content"])
        self.assertEqual(sum(c[0].endswith("/sessions") for c in MockDevin.calls), 1)

    def anthropic(self, body, key="secret"):
        req = urllib.request.Request(f"http://127.0.0.1:{self.bridge.server_port}/v1/messages",
                                     json.dumps({"model": "devin", "max_tokens": 100, **body}).encode(),
                                     {"Content-Type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
        return urllib.request.urlopen(req, timeout=10).read().decode()

    def test_anthropic_message(self):
        out = json.loads(self.anthropic({"system": "s", "messages": [{"role": "user", "content": [{"type": "text", "text": "hello"}]}]}))
        self.assertEqual((out["type"], out["role"], out["stop_reason"]), ("message", "assistant", "end_turn"))
        self.assertEqual(out["content"][0]["type"], "text")
        self.assertIn("hello", out["content"][0]["text"])
        self.assertGreater(out["usage"]["input_tokens"], 0)

    def test_anthropic_tool_use(self):
        tools = [{"name": "get_weather", "description": "d", "input_schema": {"type": "object"}}]
        out = json.loads(self.anthropic({"tools": tools, "messages": [{"role": "user", "content": self.CALL}]}))
        self.assertEqual(out["stop_reason"], "tool_use")
        self.assertEqual((out["content"][0]["name"], out["content"][0]["input"]), ("get_weather", {"city": "Jakarta"}))

    def test_anthropic_stream_events(self):
        raw = self.anthropic({"stream": True, "messages": [{"role": "user", "content": "hi"}]})
        names = [l[7:] for l in raw.splitlines() if l.startswith("event: ")]
        self.assertEqual(names[0], "message_start")
        self.assertEqual(names[-2:], ["message_delta", "message_stop"])
        self.assertIn("content_block_delta", names)

    def test_anthropic_auth_error_shape(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            self.anthropic({"messages": [{"role": "user", "content": "x"}]}, key="wrong")
        body = json.loads(c.exception.read())
        self.assertEqual((c.exception.code, body["type"], body["error"]["type"]), (401, "error", "authentication_error"))

    @staticmethod
    def fake_cli():
        """A stand-in for `devin`: echoes the last prompt line, lists models, fails on model 'bad'."""
        d = tempfile.mkdtemp()
        path = os.path.join(d, "devin")
        open(path, "w").write(
            "#!/usr/bin/env python3\nimport sys\n"
            "a = sys.argv\nm = a[a.index('--model') + 1]\n"
            "prompt = open(a[a.index('--prompt-file') + 1]).read() if '--prompt-file' in a else a[-1]\n"
            "if m == 'zzz':\n    sys.stderr.write(\"Error: Unknown model: 'zzz'\\nAvailable: alpha, beta\\n\"); sys.exit(1)\n"
            "if m == 'bad':\n    sys.stderr.write('Error: Refusing to run in an untrusted workspace\\n'); sys.exit(1)\n"
            "last = prompt.splitlines()[-1]\n"
            "print(last[5:] if last.startswith('JSON:') else 'cli-' + m + ': ' + last)\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
        return path, d

    def test_cli_backend(self):
        path, d = self.fake_cli()
        os.environ.update(DEVIN_CLI_BIN=path, DEVIN_CLI_WORKDIR=os.path.join(d, "ws"))
        b.CLI_MODELS.update(at=0, ids=[])
        try:
            out = json.loads(post(f"{self.url}/chat/completions", {
                "model": "devin-cli/alpha", "messages": [{"role": "user", "content": "hello"}]}, "secret").read())
            self.assertEqual(out["choices"][0]["message"]["content"], "cli-alpha: hello")
            self.assertFalse(any(c[0].endswith("/sessions") for c in MockDevin.calls))  # no cloud session
            tool = json.loads(post(f"{self.url}/chat/completions", {
                "model": "devin-cli/beta", "tools": self.TOOLS,
                "messages": [{"role": "user", "content": self.CALL}]}, "secret").read())
            self.assertEqual(tool["choices"][0]["finish_reason"], "tool_calls")
            req = urllib.request.Request(f"{self.url}/models", headers={"Authorization": "Bearer secret"})
            ids = [m["id"] for m in json.load(urllib.request.urlopen(req))["data"]]
            self.assertIn("devin-cli/alpha", ids)
            deny = json.load(open(os.path.join(d, "ws", ".devin", "config.json")))["permissions"]["deny"]
            self.assertIn("Read(**)", deny)
            self.assertIn("exec", deny)
            big = "x" * 300_000 + "\nlast"  # larger than one argv entry may be on Linux
            out = json.loads(post(f"{self.url}/chat/completions", {
                "model": "devin-cli/alpha", "messages": [{"role": "user", "content": big}]}, "secret").read())
            self.assertEqual(out["choices"][0]["message"]["content"], "cli-alpha: last")
            with self.assertRaises(urllib.error.HTTPError) as c:
                post(f"{self.url}/chat/completions", {
                    "model": "devin-cli/bad", "messages": [{"role": "user", "content": "x"}]}, "secret")
            self.assertEqual(c.exception.code, 502)
            self.assertIn("untrusted workspace", json.loads(c.exception.read())["error"]["message"])
        finally:
            os.environ.pop("DEVIN_CLI_BIN"), os.environ.pop("DEVIN_CLI_WORKDIR")

    def test_body_too_large(self):
        old = b.MAX_BODY
        b.MAX_BODY = 100
        try:
            with self.assertRaises(urllib.error.HTTPError) as c:
                post(f"{self.url}/chat/completions", {"messages": [{"role": "user", "content": "x" * 500}]}, "secret")
            self.assertEqual(c.exception.code, 413)
        finally:
            b.MAX_BODY = old

    def test_bad_body(self):
        with self.assertRaises(urllib.error.HTTPError) as c:
            post(f"{self.url}/chat/completions", {"messages": []}, "secret")
        self.assertEqual(c.exception.code, 400)

    def test_parse_model(self):
        for m, want in [("devin", None), ("devin-ultra", "ultra"), ("dv/devin-lite", "lite"),
                        ("devin/fast", "fast"), ("gpt-4", None)]:
            self.assertEqual(b.parse_model(m), want, m)


if __name__ == "__main__":
    unittest.main()
