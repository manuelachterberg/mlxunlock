# MLX Auto-Router

```text
▄▀▀▄▀▀▄ █     █    █      ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀▄      ▄▀▀▀▀▄ ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀ ▄▀▀▀▀▄
▀  ▀  ▀ ▀     ▀    ▀      ▀    ▀ ▀    ▀    ▀    ▀    ▀      ▀    ▀ ▀    ▀ ▀    ▀    ▀    ▀     ▀    ▀
█  █  █ █     ▄▀▀▀▀▄      █▀▀▀▀█ █    █    █    █    █      █▀▀▀▀▄ █    █ █    █    █    ▄▀▀▀  █▀▀▀▀▄
█  ▀  █ █     █    █      █    █ █    █    █    █    █      █    █ █    █ █    █    █    █     █    █
▀     ▀  ▀▀▀▀ ▀    ▀      ▀    ▀  ▀▀▀▀     ▀     ▀▀▀▀       ▀    ▀  ▀▀▀▀   ▀▀▀▀     ▀     ▀▀▀▀ ▀    ▀
```

A local LLM router dashboard for macOS / Apple Silicon. It runs two `mlx_lm.server` instances side-by-side — a large primary model for quality and a smaller fallback model for speed — and automatically routes OpenAI-compatible API requests between them.

Built for a locally discovered MLX primary model and fallback model, configurable for any MLX-compatible model.

## Demo Video

Watch the demo on YouTube:

- https://www.youtube.com/watch?v=zgPQV-Zi7ok

---

## What It Does

- **Starts both servers automatically**
  - Primary model on port `8080`
  - Fallback model on port `8081`
- **Proxy / OpenAI-compatible router** on port `8082`
- **Auto-routing** based on prompt length, RAM pressure, and primary health
- **Manual override** with a single key press
- **Live dashboard** with system stats, routing info, token throughput, and log stream

---

## Architecture

```text
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Primary   │     │   Fallback  │     │     Proxy       │
│ 27B / 8080  │     │  9B / 8081  │     │   8082 /v1      │
└──────┬──────┘     └──────┬──────┘     └────────┬────────┘
       │                   │                    │
       └───────────────────┴────────────────────┘
                                ▲
                                │
                         OpenWebUI / API client
```

---

## Requirements

- macOS with Apple Silicon
- Python 3.10+
- `mlx-lm` installed
- `psutil` and `rich` installed
- Enough unified memory to run both discovered models simultaneously
  - 27B 4-bit: ~24–35 GB
  - 9B 4-bit: ~6–10 GB
  - The startup scan calculates conservative suggestions from available RAM and model specs

---

## Installation

```bash
./start.sh
```

`start.sh` creates `.venv` when needed, installs [requirements.txt](requirements.txt), and starts the dashboard. At startup the dashboard scans `MODEL_ROOT` (default `./models`) for local MLX model directories. If none are found, an interactive terminal wizard offers to download the default `mlx-community/Qwen3.5-27B-4bit` and `mlx-community/Qwen3.5-9B-4bit` models or accepts custom Hugging Face model IDs for the primary and fallback models. Set `PRIMARY_MODEL_PATH` and `FALLBACK_MODEL_PATH` to override automatic selection.

---

## Configuration

Press `K` in the running dashboard to open the configuration TUI. Changes apply after restarting the affected servers.
The selected values are persisted in `router_config.json` and loaded on the next start. Model specifications are rescanned when the configured model paths change.

```python
MODEL_ROOT = "./models"
PRIMARY_PORT = 8080
PROXY_PORT = 8082
HOST = "0.0.0.0"

FALLBACK_TYPE = "mlx_lm"
FALLBACK_HOST = "localhost"
FALLBACK_PORT = 8081
FALLBACK_MEMORY_LIMIT = "derived by startup scan"
AUTO_START_FALLBACK = True

REASONING_EFFORT_27B = "low"
TOKEN_LIMIT_27B = "derived by startup scan"
```

| Setting | Description |
| --- | --- |
| `MODEL_ROOT` | Directory scanned for local MLX models |
| `PRIMARY_MODEL_PATH` | Optional override for the automatically selected primary model |
| `FALLBACK_MODEL_PATH` | Optional override for the automatically selected fallback model |
| `FALLBACK_MEMORY_LIMIT` | Metal memory limit for the fallback server in bytes |
| `REASONING_EFFORT_27B` | Controls primary thinking depth (`xhigh`, `medium`, or `low`) |
| `TOKEN_LIMIT_27B` | Token threshold above which requests go to fallback |
| `SWAP_LIMIT_27B` | SWAP threshold in GB above which requests go to fallback |

### Controlling thinking mode on the 27B model

Qwen3.5-27B defaults to `xhigh` reasoning, which produces very long internal thinking traces. The router automatically injects the chosen `reasoning_effort` into every `/v1/chat/completions` request sent to the primary model via `chat_template_kwargs`. Set `REASONING_EFFORT_27B = "low"` for brief reasoning, `"medium"` for moderate reasoning, or `"xhigh"` to match the model default. Set it to `None` to leave the model default untouched.

---

## Usage

```bash
./start.sh
```

The dashboard will:

1. Start the 27B primary server on port `8080`
2. Start the 9B fallback server on port `8081`
3. Start the proxy router on port `8082`

Point your OpenAI-compatible client (e.g. OpenWebUI) at:

```text
http://<your-mac-ip>:8082/v1
```

Use any non-empty string as the API key, for example `dummy`.

---

## Dashboard Hotkeys

| Key | Action |
| --- | --- |
| `Q` | Quit |
| `R` | Restart primary (27B) server |
| `F` | Restart fallback server |
| `S` | Toggle forced fallback / auto routing |
| `C` | Clear logs and history |
| `P` | Save stats to file |

---

## Routing Logic

| Condition | Router Decision | Dashboard Display |
| --- | --- | --- |
| Short prompt, Primary healthy | → Primary | ▶ PRIMARY |
| Prompt > `TOKEN_LIMIT_27B` tokens | → Fallback | ▶ FALLBACK (5000t > 3000t limit) |
| SWAP > `SWAP_LIMIT_27B` GB | → Fallback | ▶ FALLBACK (SWAP 6.2GB > 5GB) |
| 27B crashed / unhealthy | → Fallback | ▶ FALLBACK (27B crashed) |
| You press `S` | → Fallback | ▶ FALLBACK (FORCED) |
| No fallback available | → 27B only | 27B (no fallback) |

If the fallback server is offline, all requests are routed to the primary model.

---

## Files

| File | Purpose |
| --- | --- |
| `mlx_dashboard.py` | Main dashboard, proxy router, and process manager |
| `mlx_server_safe_wrapper.py` | Starts the primary 27B `mlx_lm.server` on port 8080 |
| `mlx_server_fallback_wrapper.py` | Starts the fallback `mlx_lm.server` on port 8081 |
| `requirements.txt` | Python dependencies |
| `DASHBOARD.md` | German setup guide |
| `endpoint.md` | Legacy endpoint notes |

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'mlx_lm'`

Install or upgrade `mlx-lm`:

```bash
source .venv/bin/activate
python -m pip install --upgrade mlx-lm
```

### Fallback server fails to start

- Check that `FALLBACK_MODEL_PATH` is valid and accessible
- Verify you have enough free disk space and RAM
- Try starting it manually to see the error:

```bash
python -m mlx_lm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8081
```

### Model downloads are slow

`mlx_lm.server` caches models in `~/.cache/huggingface/`. You can pre-download with:

```bash
python -m mlx_lm.server --model mlx-community/Qwen3.5-9B-4bit --host 0.0.0.0 --port 8081
```

Then stop it and start the dashboard.

### OpenWebUI cannot connect

- Make sure the dashboard is running
- Use the Mac's LAN IP, not `127.0.0.1`, if OpenWebUI runs on another machine
- Use any non-empty API key

---

## Credits

- Default primary model: [`mlx-community/Qwen3.5-27B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-27B-4bit)
- Default fallback model: [`mlx-community/Qwen3.5-9B-4bit`](https://huggingface.co/mlx-community/Qwen3.5-9B-4bit)
- Routing and dashboard code: custom, built on top of `mlx-lm`, `psutil`, and `rich`
