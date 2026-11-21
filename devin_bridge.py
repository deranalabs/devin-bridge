#!/usr/bin/env python3
"""OpenAI- and Anthropic-compatible bridge for the Devin v3 session API (stdlib only).

POST /v1/chat/completions   OpenAI Chat Completions (stream, tools, usage)
POST /v1/messages           Anthropic Messages (stream, tools, usage)
GET  /v1/models             OpenAI list, or Anthropic list when x-api-key/anthropic-version is sent

Each conversation maps to a Devin session; Devin's reply becomes the assistant message.
Tool calling is emulated: tools are described in the prompt and Devin answers with JSON.

Env:
  BRIDGE_API_KEY     required: clients must send it (Authorization: Bearer or x-api-key)
  DEVIN_API_KEY      service-user key (cog_...)           cloud models only
  DEVIN_ORG_ID       org-...                              cloud models only
  DEVIN_API_BASE     default https://api.devin.ai/v3
  BRIDGE_HOST/PORT   default 127.0.0.1 / 8320
  DEVIN_MAX_ACU      optional per-session ACU cap
  DEVIN_TIMEOUT      max seconds to wait for a reply (default 900)
  DEVIN_POLL         poll interval seconds (default 3)
  DEVIN_WARM         1 (default) keeps one pre-booted spare session so the first
                     reply is fast; 0 disables (a cold session takes ~5 min)
  DEVIN_PROMPT_PREFIX  override the instruction put before the transcript

CLI backend (fast, many models): model "devin-cli/<name>" runs `devin -p --model <name>`
in an empty workspace directory.
  DEVIN_CLI_BIN      default "devin"
  DEVIN_CLI_WORKDIR  default ~/.devin-bridge/workspace
  DEVIN_CLI_RESPECT_TRUST  "false" (default) skips the trust check, safe only because the
                     workspace is an empty dir and the CLI runs in read-only 'auto' mode
  DEVIN_CLI_TIMEOUT  seconds per call (default 300)
  DEVIN_CLI_CONCURRENCY  parallel CLI calls (default 4)
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODES = ["normal", "fast", "lite", "ultra", "fusion"]
DEFAULT_PREFIX = (
    "You are answering through an API chat bridge. Reply to the conversation below "
    "with a direct text answer. Do not open pull requests or change repositories "
    "unless the user explicitly asks."
)
CLI_PREFIX = (
    "You are answering through an API chat bridge. Reply to the conversation below with a "
    "direct text answer. Do not run commands, read or modify files."
)
TOOL_PROMPT = (
    "You can call these tools. To call one or more tools, reply with ONLY a JSON object "
    '{"tool_calls":[{"name":"<tool name>","arguments":{...}}]} and no other text. '
    "Otherwise reply with plain text. Tools: "
)
# status values after which Devin will not produce more output on its own
DONE_STATUS = {"exit", "error", "suspended"}
DONE_DETAIL = {"waiting_for_user", "finished", "waiting_for_approval"}

# conversation key -> {"sid": str, "turns": int}; shortcut: in-memory, lost on restart
SESSIONS = {}
WARM = []  # spare booted sessions
LOCK = threading.Lock()
CLI_SEM = threading.BoundedSemaphore(int(os.environ.get("DEVIN_CLI_CONCURRENCY", "4")))
CLI_MODELS = {"at": 0, "ids": []}


def cfg(name, default=None):
    return os.environ.get(name, default)


class UpstreamError(Exception):
    def __init__(self, status, detail):
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


class BadRequest(Exception):
    pass


def devin(method, path, body=None):
    url = f"{cfg('DEVIN_API_BASE', 'https://api.devin.ai/v3')}/organizations/{cfg('DEVIN_ORG_ID')}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {cfg('DEVIN_API_KEY')}",
        "Content-Type": "application/json",
        "User-Agent": "devin-bridge/1.0",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1 + attempt * 2)
                continue
            try:
                detail = json.loads(e.read()).get("detail") or e.reason
            except Exception:
                detail = e.reason
            raise UpstreamError(e.code, detail)
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(1)
                continue
            raise UpstreamError(502, str(e.reason))


def sid_of(session):
    sid = session["session_id"]
    return sid if sid.startswith("devin-") else f"devin-{sid}"


def all_messages(sid):
    items, after = [], None
    while True:
        q = "?first=200" + (f"&after={after}" if after else "")
        page = devin("GET", f"/sessions/{sid}/messages{q}")
        items += page.get("items", [])
        if not page.get("has_next_page") or not page.get("end_cursor"):
            return items
        after = page["end_cursor"]


# ---------- request normalisation: both APIs become {"messages", "tools", "system"} ----------

def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def norm_openai(req):
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("'messages' must be a non-empty array")
    out, system = [], []
    for m in msgs:
        role, t = m.get("role"), text_of(m.get("content"))
        if role in ("system", "developer"):
            system.append(t)
        elif role == "assistant":
            calls = [{"name": c["function"]["name"], "arguments": c["function"].get("arguments", "")}
                     for c in m.get("tool_calls") or []]
            out.append({"role": "assistant", "text": t + (("\n" if t else "") + "tool_calls: " + json.dumps(calls) if calls else "")})
        elif role == "tool":
            out.append({"role": "user", "text": f"[tool result {m.get('tool_call_id', '')}]\n{t}"})
        else:
            out.append({"role": "user", "text": t})
    tools = [{"name": t["function"]["name"], "description": t["function"].get("description", ""),
              "parameters": t["function"].get("parameters", {})}
             for t in req.get("tools") or [] if t.get("type") == "function"]
    if req.get("tool_choice") == "none":
        tools = []
    rf = (req.get("response_format") or {}).get("type")
    if rf in ("json_object", "json_schema"):
        system.append("Reply with only valid JSON and no other text.")
    return finish_norm(out, system, tools, req.get("tool_choice") == "required")


def norm_anthropic(req):
    msgs = req.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages: Field required")
    system = [text_of(req.get("system"))] if req.get("system") else []
    out = []
    for m in msgs:
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": m["role"], "text": c})
            continue
        parts = []
        for b in c or []:
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
            elif b.get("type") == "tool_use":
                parts.append("tool_calls: " + json.dumps([{"name": b["name"], "arguments": b.get("input", {})}]))
            elif b.get("type") == "tool_result":
                parts.append(f"[tool result {b.get('tool_use_id', '')}]\n{text_of(b.get('content'))}")
        out.append({"role": m["role"], "text": "\n".join(parts)})
    tools = [{"name": t["name"], "description": t.get("description", ""), "parameters": t.get("input_schema", {})}
             for t in req.get("tools") or []]
    tc = (req.get("tool_choice") or {}).get("type")
    if tc == "none":
        tools = []
    return finish_norm(out, system, tools, tc in ("any", "tool"))


def finish_norm(msgs, system, tools, required):
    if not any(m["text"] for m in msgs if m["role"] == "user"):
        raise BadRequest("no user message with text")
    return {"messages": msgs, "system": system, "tools": tools, "required": required}


def transcript(norm):
    return "\n\n".join([f"[system]\n{s}" for s in norm["system"] if s] +
                       [f"[{m['role']}]\n{m['text']}" for m in norm["messages"] if m["text"]])


def build_prompt(norm, prefix=DEFAULT_PREFIX):
    parts = [cfg("DEVIN_PROMPT_PREFIX", prefix)]
    if norm["tools"]:
        parts.append(TOOL_PROMPT + json.dumps(norm["tools"]) +
                     (" You must call a tool." if norm["required"] else ""))
    parts.append(transcript(norm))
    return "\n\n".join(parts)


def conv_key(norm, model):
    first = next((m["text"] for m in norm["messages"] if m["role"] == "user"), "")
    return hashlib.sha256(json.dumps([model, norm["system"], first]).encode()).hexdigest()


def parse_model(model):
    """'dv/devin-fast', 'devin-fast', 'devin/fast' -> 'fast'; plain 'devin' -> None (org default)."""
    m = (model or "devin").rsplit("/", 1)[-1].removeprefix("devin-")
    return m if m in MODES else None


def parse_tool_calls(text, tools):
    """Return [{'name','arguments': dict}] if Devin answered with a tool-call JSON, else None."""
    if not tools:
        return None
    names = {t["name"] for t in tools}
    s = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.S)
    for cand in (fenced.group(1) if fenced else s, s[s.find("{"):s.rfind("}") + 1]):
        try:
            obj = json.loads(cand)
            break
        except ValueError:
            obj = None
    calls = []
    for c in (obj or {}).get("tool_calls") or [] if isinstance(obj, dict) else []:
        args = c.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args or "{}")
            except ValueError:
                continue
        if c.get("name") in names and isinstance(args, dict):
            calls.append({"name": c["name"], "arguments": args})
    return calls or None


# ---------- Devin session driving ----------

def create_session(prompt, mode=None):
    body = {"prompt": prompt, "title": "chat bridge"}
    if mode:
        body["devin_mode"] = mode
    if cfg("DEVIN_MAX_ACU"):
        body["max_acu_limit"] = int(cfg("DEVIN_MAX_ACU"))
    return sid_of(devin("POST", "/sessions", body))


def spawn_warm():
    """Boot a spare session ahead of time: a cold Devin session takes ~5 min to answer."""
    try:
        sid = create_session("Reply with the single word READY, then wait for the next message.")
        with LOCK:
            WARM.append(sid)
    except UpstreamError:
        pass


def take_warm():
    with LOCK:
        sid = WARM.pop(0) if WARM else None
    if sid:
        threading.Thread(target=spawn_warm, daemon=True).start()
    return sid


def cli_bin():
    return cfg("DEVIN_CLI_BIN", "devin")


def cli_workdir():
    return os.path.expanduser(cfg("DEVIN_CLI_WORKDIR", "~/.devin-bridge/workspace"))


def cli_models():
    """Model names the logged-in Devin CLI accepts (parsed from its 'Available:' error), cached 1h."""
    if time.time() - CLI_MODELS["at"] > 3600:
        try:
            workdir = cli_workdir()
            os.makedirs(workdir, exist_ok=True)
            p = subprocess.run([cli_bin(), "--respect-workspace-trust", cfg("DEVIN_CLI_RESPECT_TRUST", "false"),
                                "-p", "--model", "zzz", "--", "x"],
                               capture_output=True, text=True, timeout=30, cwd=workdir)
            m = re.search(r"Available:\s*(.+)", p.stdout + p.stderr)
            CLI_MODELS.update(at=time.time(), ids=[x.strip() for x in m.group(1).split(",")] if m else [])
        except (OSError, subprocess.SubprocessError):
            CLI_MODELS.update(at=time.time(), ids=[])
    return CLI_MODELS["ids"]


def run_cli(norm, model):
    """Stateless CLI call: the full transcript goes in each prompt, so there is no session to lose."""
    name = model.split("/", 1)[1]
    workdir = cli_workdir()
    os.makedirs(workdir, exist_ok=True)
    cmd = [cli_bin(), "--respect-workspace-trust", cfg("DEVIN_CLI_RESPECT_TRUST", "false"),
           "-p", "--model", name, "--permission-mode", "auto", "--", build_prompt(norm, CLI_PREFIX)]
    box = {}

    def work():
        try:
            with CLI_SEM:
                box["p"] = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                                          timeout=float(cfg("DEVIN_CLI_TIMEOUT", "300")))
        except subprocess.TimeoutExpired:
            box["err"] = "devin cli timed out"
        except OSError as e:
            box["err"] = f"cannot run devin cli: {e}"

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while t.is_alive():
        yield None
        t.join(2)
    p = box.get("p")
    if p is None or p.returncode != 0:
        detail = box.get("err") or ((p.stderr or p.stdout).strip().splitlines() or ["devin cli failed"])[0]
        raise UpstreamError(502, detail)
    yield p.stdout.strip()


def run(norm, model):
    """Generator: yields None (keepalive) or reply text from Devin for this conversation."""
    if (model or "").startswith("devin-cli/"):
        yield from run_cli(norm, model)
        return
    turns = sum(1 for m in norm["messages"] if m["role"] == "assistant")
    key = conv_key(norm, model)
    with LOCK:
        known = SESSIONS.get(key)
    # everything after the last assistant message is new to an existing session
    tail = []
    for m in reversed(norm["messages"]):
        if m["role"] == "assistant":
            break
        tail.append(m["text"])
    new_text = "\n\n".join(reversed(tail))
    sid = seen = None
    if known and turns == known["turns"] + 1 and new_text:
        try:
            sid = known["sid"]
            seen = {m["event_id"] for m in all_messages(sid)}
            devin("POST", f"/sessions/{sid}/messages", {"message": new_text})
        except UpstreamError:
            sid = seen = None  # session gone/archived: fall back to a fresh one with full history
    prompt = build_prompt(norm)
    mode = parse_model(model)
    if sid is None and mode is None and cfg("DEVIN_WARM", "1") != "0":
        sid = take_warm()
        if sid:
            # wait for its READY reply so it cannot be mistaken for the answer
            deadline = time.time() + float(cfg("DEVIN_TIMEOUT", "900"))
            while True:
                msgs = all_messages(sid)
                if any(m["source"] == "devin" for m in msgs) or time.time() > deadline:
                    break
                yield None
                time.sleep(float(cfg("DEVIN_POLL", "3")))
            seen = {m["event_id"] for m in msgs}
            devin("POST", f"/sessions/{sid}/messages", {"message": prompt})
    if sid is None:
        sid, seen = create_session(prompt, mode), set()
    with LOCK:
        SESSIONS[key] = {"sid": sid, "turns": turns}
    yield from replies(sid, seen)


def replies(sid, seen):
    """Yield new Devin message texts as they appear; return when the session settles."""
    deadline = time.time() + float(cfg("DEVIN_TIMEOUT", "900"))
    got = False
    while time.time() < deadline:
        yield None  # keepalive tick so streaming clients do not time out
        time.sleep(float(cfg("DEVIN_POLL", "3")))
        s = devin("GET", f"/sessions/{sid}")
        for m in all_messages(sid):
            if m["event_id"] not in seen and m["source"] == "devin":
                seen.add(m["event_id"])
                got = True
                yield m["message"]
        status, detail = s.get("status"), s.get("status_detail")
        # a stale waiting_for_user right after a follow-up must not end the wait
        if status in DONE_STATUS or (status == "running" and detail in DONE_DETAIL and got):
            if detail == "waiting_for_approval":
                yield f"\n\n(Devin is waiting for approval: {s.get('url')})"
            elif not got:
                yield f"(Devin ended with status {status}/{detail} without replying: {s.get('url')})"
            return
    yield f"\n\n(bridge timeout; Devin is still working: {sid})"


def events(norm, model):
    """Yield None (ping), ('text', str) or ('calls', [..]). With tools the reply is buffered."""
    if not norm["tools"]:
        for t in run(norm, model):
            yield None if t is None else ("text", t)
        return
    buf = []
    for t in run(norm, model):
        if t is None:
            yield None
        else:
            buf.append(t)
    text = "\n\n".join(buf)
    calls = parse_tool_calls(text, norm["tools"])
    yield ("calls", calls) if calls else ("text", text)


def est(s):
    """Rough token estimate (chars/4): Devin does not report tokens."""
    return max(1, len(s) // 4)


# ---------- HTTP ----------

def oa_completion(model, cid, text, calls, prompt_tokens):
    msg = {"role": "assistant", "content": text if not calls else None, "refusal": None}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                              "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls]
    out = est(text) if not calls else est(json.dumps(calls))
    return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "logprobs": None,
                         "finish_reason": "tool_calls" if calls else "stop"}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": out,
                      "total_tokens": prompt_tokens + out}}


def oa_chunk(model, cid, delta, finish=None, usage=None, choices=True):
    c = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
         "choices": [{"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}] if choices else []}
    if usage:
        c["usage"] = usage
    return c


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        # access log to stderr (journald): method, path, status only - never headers or bodies
        sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def anthropic(self):
        return self.path.startswith("/v1/messages") or "anthropic-version" in self.headers or "x-api-key" in self.headers

    def send_json(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def error(self, status, message, etype="api_error", anthropic=None):
        if self.anthropic() if anthropic is None else anthropic:
            etype = {401: "authentication_error", 404: "not_found_error", 429: "rate_limit_error"}.get(
                status, "invalid_request_error" if status < 500 else "api_error")
            return self.send_json(status, {"type": "error", "error": {"type": etype, "message": message}})
        self.send_json(status, {"error": {"message": message, "type": etype, "param": None,
                                          "code": {401: "invalid_api_key", 404: "not_found"}.get(status)}})

    def authed(self):
        key = cfg("BRIDGE_API_KEY")
        sent = self.headers.get("x-api-key") or (self.headers.get("Authorization") or "").removeprefix("Bearer ")
        if key and sent != key:
            self.error(401, "invalid api key", "invalid_request_error")
            return False
        return True

    def sse(self, obj, event=None):
        head = f"event: {event}\n" if event else ""
        self.wfile.write(f"{head}data: {json.dumps(obj)}\n\n".encode())
        self.wfile.flush()

    def sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path == "/health":
            return self.send_json(200, {"ok": True})
        if not self.authed():
            return
        if path == "/v1/models":
            ids = ["devin"] + [f"devin-{m}" for m in MODES] + [f"devin-cli/{m}" for m in cli_models()]
            if self.anthropic():
                data = [{"type": "model", "id": i, "display_name": i, "created_at": "2026-01-01T00:00:00Z"} for i in ids]
                return self.send_json(200, {"data": data, "has_more": False, "first_id": ids[0], "last_id": ids[-1]})
            return self.send_json(200, {"object": "list", "data": [
                {"id": i, "object": "model", "created": 0, "owned_by": "devin"} for i in ids]})
        self.error(404, "not found", "invalid_request_error")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path not in ("/v1/chat/completions", "/v1/messages"):
            return self.error(404, "not found", "invalid_request_error")
        is_anthropic = path == "/v1/messages"
        if not self.authed():
            return
        try:
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            norm = (norm_anthropic if is_anthropic else norm_openai)(req)
        except BadRequest as e:
            return self.error(400, str(e), "invalid_request_error", is_anthropic)
        except Exception:
            return self.error(400, "invalid request body", "invalid_request_error", is_anthropic)
        model = req.get("model") or "devin"
        ptok = est(transcript(norm))
        streaming = False
        try:
            if req.get("stream"):
                self.sse_start()
                streaming = True
                (self.stream_anthropic if is_anthropic else self.stream_openai)(model, norm, ptok, req)
                return
            text, calls = "", None
            for ev in events(norm, model):
                if ev and ev[0] == "text":
                    text += ("\n\n" if text else "") + ev[1]
                elif ev:
                    calls = ev[1]
            if is_anthropic:
                return self.send_json(200, an_message(model, text, calls, ptok))
            self.send_json(200, oa_completion(model, f"chatcmpl-{uuid.uuid4().hex[:24]}", text, calls, ptok))
        except UpstreamError as e:
            if streaming:  # status code already sent: surface the error as text
                msg = f"\n\n(devin api error: {e.detail})"
                if is_anthropic:
                    self.sse({"type": "error", "error": {"type": "api_error", "message": msg.strip()}}, "error")
                else:
                    self.sse(oa_chunk(model, "chatcmpl-err", {"content": msg}, "stop"))
                    self.wfile.write(b"data: [DONE]\n\n")
            else:
                self.error(502 if e.status >= 500 else e.status, f"devin api: {e.detail}", anthropic=is_anthropic)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def stream_openai(self, model, norm, ptok, req):
        cid, out = f"chatcmpl-{uuid.uuid4().hex[:24]}", ""
        self.sse(oa_chunk(model, cid, {"role": "assistant", "content": "", "refusal": None}))
        finish = "stop"
        for ev in events(norm, model):
            if ev is None:
                self.wfile.write(b": ping\n\n")
                self.wfile.flush()
            elif ev[0] == "text":
                out += ev[1]
                self.sse(oa_chunk(model, cid, {"content": ev[1] + "\n\n"}))
            else:
                finish = "tool_calls"
                out = json.dumps(ev[1])
                self.sse(oa_chunk(model, cid, {"tool_calls": [
                    {"index": i, "id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}}
                    for i, c in enumerate(ev[1])]}))
        self.sse(oa_chunk(model, cid, {}, finish))
        if (req.get("stream_options") or {}).get("include_usage"):
            self.sse(oa_chunk(model, cid, {}, usage={"prompt_tokens": ptok, "completion_tokens": est(out),
                                                     "total_tokens": ptok + est(out)}, choices=False))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def stream_anthropic(self, model, norm, ptok, req):
        mid, idx, open_text, out = f"msg_{uuid.uuid4().hex[:24]}", 0, False, ""
        self.sse({"type": "message_start", "message": {
            "id": mid, "type": "message", "role": "assistant", "content": [], "model": model,
            "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": ptok, "output_tokens": 1}}},
            "message_start")
        stop = "end_turn"
        for ev in events(norm, model):
            if ev is None:
                self.sse({"type": "ping"}, "ping")
            elif ev[0] == "text":
                if not open_text:
                    self.sse({"type": "content_block_start", "index": idx,
                              "content_block": {"type": "text", "text": ""}}, "content_block_start")
                    open_text = True
                out += ev[1]
                self.sse({"type": "content_block_delta", "index": idx,
                          "delta": {"type": "text_delta", "text": ev[1] + "\n\n"}}, "content_block_delta")
            else:
                stop = "tool_use"
                out = json.dumps(ev[1])
                for c in ev[1]:
                    self.sse({"type": "content_block_start", "index": idx, "content_block": {
                        "type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": c["name"], "input": {}}},
                        "content_block_start")
                    self.sse({"type": "content_block_delta", "index": idx, "delta": {
                        "type": "input_json_delta", "partial_json": json.dumps(c["arguments"])}},
                        "content_block_delta")
                    self.sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
                    idx += 1
        if open_text:
            self.sse({"type": "content_block_stop", "index": idx}, "content_block_stop")
        self.sse({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                  "usage": {"output_tokens": est(out)}}, "message_delta")
        self.sse({"type": "message_stop"}, "message_stop")


def an_message(model, text, calls, ptok):
    content = ([{"type": "tool_use", "id": f"toolu_{uuid.uuid4().hex[:24]}", "name": c["name"], "input": c["arguments"]}
                for c in calls] if calls else [{"type": "text", "text": text}])
    return {"id": f"msg_{uuid.uuid4().hex[:24]}", "type": "message", "role": "assistant", "model": model,
            "content": content, "stop_reason": "tool_use" if calls else "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": ptok, "output_tokens": est(json.dumps(calls) if calls else text)}}


def make_server(host, port):
    return ThreadingHTTPServer((host, port), Handler)


if __name__ == "__main__":
    if not cfg("BRIDGE_API_KEY"):
        raise SystemExit("missing env BRIDGE_API_KEY (refusing to run an open bridge)")
    cloud = bool(cfg("DEVIN_API_KEY") and cfg("DEVIN_ORG_ID"))
    if not cloud:
        os.environ["DEVIN_WARM"] = "0"  # warm sessions need the cloud API
        print("cloud API not configured: only devin-cli/<model> will work", flush=True)
    host, port = cfg("BRIDGE_HOST", "127.0.0.1"), int(cfg("BRIDGE_PORT", "8320"))
    if cfg("DEVIN_WARM", "1") != "0":
        threading.Thread(target=spawn_warm, daemon=True).start()
    print(f"devin-bridge on http://{host}:{port}/v1", flush=True)
    make_server(host, port).serve_forever()
