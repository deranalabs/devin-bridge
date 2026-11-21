# devin-bridge

Jembatan OpenAI + Anthropic API untuk akun **Devin** milikmu sendiri. Satu file Python, tanpa dependensi (stdlib saja).

```
Klien (Hermes, Cline, 9router, curl)  ──OpenAI / Anthropic──►  devin-bridge  ──►  Devin CLI   (cepat, banyak model)
                                                                              └─►  Devin API   (sesi cloud, lambat)
```

> Tidak resmi dan tidak berafiliasi dengan Cognition/Devin. Pakai **akun dan token milikmu sendiri**. Jangan membagikan endpoint bridge-mu ke orang lain: itu memakai kuota ACU akunmu, dan berbagi akun bisa melanggar syarat layanan Devin. Untuk teman: biarkan masing-masing menjalankan bridge sendiri dengan akun sendiri.

## Endpoint

| Endpoint | Format |
|---|---|
| `POST /v1/chat/completions` | OpenAI Chat Completions (stream, tools, `stream_options.include_usage`) |
| `POST /v1/messages` | Anthropic Messages (stream, `tool_use`/`tool_result`) |
| `GET /v1/models` | Daftar model (format Anthropic kalau header `anthropic-version`/`x-api-key` dikirim) |
| `GET /health` | Cek hidup |

## Dua jalur backend

| Model | Cara kerja | Kecepatan | Catatan |
|---|---|---|---|
| `devin-cli/<nama>` | Menjalankan `devin -p --model <nama>` | 2–7 detik | Pilih model apa pun yang dikenali akunmu (`claude-*`, `gpt-*`, `gemini-*`, `kimi-*`, `swe-*`, ...). Tanpa sesi: seluruh percakapan dikirim tiap request. **Disarankan.** |
| `devin`, `devin-fast`, `devin-lite`, `devin-ultra`, `devin-fusion` | Membuat sesi cloud lewat Devin API v3 | ~10 detik kalau sesi cadangan hangat, ~5 menit kalau dingin | Untuk tugas agent panjang. Memakai ACU. |

Daftar `devin-cli/*` dibaca otomatis dari CLI-mu, jadi selalu sesuai akun.

## Pasang

Syarat: Python 3.9+. Untuk jalur CLI: [Devin CLI](https://docs.devin.ai/cli/index.md) sudah login (`devin auth login`; di server tanpa browser pakai `--force-manual-token-flow`).

```bash
git clone <repo-ini> && cd devin-bridge
cp .env.example .env     # isi nilainya
set -a; . ./.env; set +a
python3 devin_bridge.py
```

Cek:

```bash
curl -s localhost:8320/v1/chat/completions \
  -H "Authorization: Bearer $BRIDGE_API_KEY" -H "Content-Type: application/json" \
  -d '{"model":"devin-cli/swe-2-flash","messages":[{"role":"user","content":"say hi"}]}'
```

Sebagai service (Linux, systemd user): salin `devin-bridge.service`, ubah path, lalu `systemctl --user enable --now devin-bridge`.

## Konfigurasi (env)

| Variabel | Default | Fungsi |
|---|---|---|
| `BRIDGE_API_KEY` | – | **Wajib diisi.** Key yang harus dikirim klien (`Authorization: Bearer` atau `x-api-key`). |
| `BRIDGE_HOST` / `BRIDGE_PORT` | `127.0.0.1` / `8320` | Jangan bind ke `0.0.0.0` tanpa firewall atau tailnet. |
| `DEVIN_API_KEY` | – | Token service user `cog_...` (hanya untuk jalur cloud). |
| `DEVIN_ORG_ID` | – | `org-...`, ada di Settings → Devin API (hanya jalur cloud). |
| `DEVIN_CLI_BIN` | `devin` | Path binary CLI. |
| `DEVIN_CLI_WORKDIR` | `~/.devin-bridge/workspace` | Folder kosong tempat CLI dijalankan. |
| `DEVIN_CLI_RESPECT_TRUST` | `false` | Melewati pengecekan workspace trust. Aman hanya karena foldernya kosong dan CLI berjalan dalam mode baca-saja (`--permission-mode auto`). Set `true` kalau mau trust manual (jalankan `devin` interaktif di folder itu sekali). |
| `DEVIN_CLI_CONCURRENCY` | `4` | Panggilan CLI paralel. |
| `DEVIN_CLI_TIMEOUT` | `300` | Detik per panggilan CLI. |
| `DEVIN_WARM` | `1` | Jaga satu sesi cloud cadangan agar balasan pertama cepat. `0` untuk mematikan. |
| `DEVIN_MAX_ACU` | – | Batas ACU per sesi cloud. |
| `DEVIN_TIMEOUT` / `DEVIN_POLL` | `900` / `3` | Batas tunggu dan interval polling sesi cloud. |

## Pakai dengan 9router / Hermes

Di dashboard 9router: tambah provider **Custom OpenAI-compatible** (prefix `dv`, base URL `http://127.0.0.1:8320/v1`), lalu tambah connection berisi `BRIDGE_API_KEY`. Model jadi `dv/devin-cli/claude-sonnet-5.5` dst.

Catatan 9router: kalau `stream` tidak dikirim, 9router membalas sebagai event-stream. Kirim `"stream": false` eksplisit untuk JSON biasa.

Jangan masukkan `dv/...` ke combo atau fallback chain: lambat dan memakai kuota.

## Batasan yang jujur

- Devin itu **agent**, bukan model mentah. Balasan bisa terdengar seperti agent.
- **Tool calling hasil tiruan**: tool dijelaskan di prompt, Devin diminta membalas JSON, bridge mengubahnya jadi `tool_calls`/`tool_use`. Kadang model membalas teks atau mencoba alat internalnya sendiri. Dengan tools, balasan ditahan sampai lengkap (ping keepalive tetap dikirim).
- `usage` hanya estimasi (karakter ÷ 4).
- `stop`, `n`, `max_tokens`, `temperature` diabaikan. Belum ada `/v1/responses`.
- Peta sesi cloud ada di memori: hilang saat restart. Jalur CLI tidak punya sesi.
- Prompt CLI dikirim lewat file sementara (`--prompt-file`), jadi panjang percakapan tidak dibatasi ukuran argumen OS; batasnya konteks model.
- Respons langsung memakai kuota akunmu.

## Tes

```bash
python3 -m unittest
```

19 test, memakai Devin dan CLI tiruan (tidak memakai kuota). Cek terhadap Devin asli dilakukan manual.
