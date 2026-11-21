# devin-bridge

OpenAI- and Anthropic-compatible API for your own [Devin](https://devin.ai) account. Single Python file, standard library only.

```
OpenAI / Anthropic client  ──►  devin-bridge  ──►  Devin CLI   (fast, many models)
                                              └─►  Devin API   (cloud sessions)
```

> Unofficial; not affiliated with Cognition. Run it with your own account and token, and do not expose it to other people: requests consume your Devin quota, and sharing an account may violate Devin's terms.

## Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions: streaming, tools, `stream_options.include_usage` |
| `POST /v1/messages` | Anthropic Messages: streaming, `tool_use` / `tool_result` |
| `GET /v1/models` | Model list (Anthropic format when `anthropic-version` or `x-api-key` is sent) |
| `GET /health` | Liveness check |

## Models

| Model | Backend | Latency | Notes |
|---|---|---|---|
| `devin-cli/<name>` | `devin -p --model <name>` | 2–7 s | Any model your account supports (list is read from the CLI). Stateless: the full conversation is sent on each request. **Recommended.** |
| `devin`, `devin-fast`, `devin-lite`, `devin-ultra`, `devin-fusion` | Devin API v3 sessions | ~10 s warm, ~5 min cold | For long agent tasks. Consumes ACUs. |

## Quick start

Requirements: Python 3.9+. For `devin-cli/*`, an installed and authenticated [Devin CLI](https://docs.devin.ai/cli/index.md) (`devin auth login`).

```bash
git clone https://github.com/deranalabs/devin-bridge && cd devin-bridge
cp .env.example .env        # set BRIDGE_API_KEY at minimum
set -a; . ./.env; set +a
python3 devin_bridge.py
```

```bash
curl -s localhost:8320/v1/chat/completions \
  -H "Authorization: Bearer $BRIDGE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"devin-cli/swe-2-flash","messages":[{"role":"user","content":"hello"}]}'
```

To run as a service, adapt `devin-bridge.service` (systemd user unit).

The CLI backend needs the workspace directory trusted once:

```bash
mkdir -p ~/.devin-bridge/workspace && cd ~/.devin-bridge/workspace && devin   # accept the trust prompt, then exit
```

## Configuration

| Variable | Default | Description |
|---|---|---|
| `BRIDGE_API_KEY` | – | **Required.** Clients send it as `Authorization: Bearer` or `x-api-key`. |
| `BRIDGE_HOST`, `BRIDGE_PORT` | `127.0.0.1`, `8320` | Do not bind publicly without a firewall or private network. |
| `DEVIN_CLI_BIN` | `devin` | Path to the CLI binary. |
| `DEVIN_CLI_WORKDIR` | `~/.devin-bridge/workspace` | Empty directory the CLI runs in. |
| `DEVIN_CLI_RESPECT_TRUST` | `true` | Require the CLI workspace to be trusted: run `devin` once in `DEVIN_CLI_WORKDIR` and accept. Set `false` to skip the check; reasonable only because the directory is empty and the CLI runs read-only (`--permission-mode auto`). |
| `DEVIN_CLI_CONCURRENCY` | `4` | Parallel CLI calls; extra requests queue. |
| `DEVIN_CLI_TIMEOUT` | `300` | Seconds per CLI call. |
| `DEVIN_API_KEY`, `DEVIN_ORG_ID` | – | Service-user token (`cog_…`) and org ID. Cloud models only. |
| `DEVIN_WARM` | `1` | Keep one pre-booted cloud session so the first reply is fast. |
| `DEVIN_MAX_ACU` | – | Per-session ACU cap for cloud sessions. |
| `DEVIN_TIMEOUT`, `DEVIN_POLL` | `900`, `3` | Cloud reply timeout and poll interval, in seconds. |

## 9router and other gateways

Add a custom OpenAI-compatible provider pointing at `http://127.0.0.1:8320/v1` with `BRIDGE_API_KEY` as its key. Send `"stream": false` explicitly if the gateway treats a missing `stream` as streaming. Keep Devin models out of automatic fallback chains: they are slower and consume quota.

## Limitations

- Devin is an agent, not a raw model; replies may read like an agent's.
- Tool calling is emulated: tools are described in the prompt and Devin answers with JSON, which the bridge converts to `tool_calls` / `tool_use`. Models occasionally answer in plain text instead. With tools, replies are buffered until complete (keepalive pings are still sent).
- `usage` is an estimate (characters ÷ 4).
- `stop`, `n`, `max_tokens`, `temperature` are ignored. `/v1/responses` is not implemented.
- Cloud session state is in memory and is lost on restart. The CLI backend keeps no state.
- Long conversations are resent on every request, so cost and latency grow with context.

## License

MIT. See [LICENSE](LICENSE).

## Tests

```bash
python3 -m unittest
```

Uses fake Devin API and CLI servers, so it consumes no quota.
