#!/usr/bin/env python3
"""
MLX Server Dashboard v11 - Auto-Router Edition
Routes between 27B (quality) and fallback (speed) based on prompt length.
Proxy runs on port 8082. OpenAI-compatible clients should connect to http://host:8082/v1
"""

import subprocess
import sys
import os
import re
import time
import threading
import logging
import tty
import termios
import select
import json
import http.client
import socketserver
from pathlib import Path
from datetime import datetime
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align

try:
    from terminaltexteffects.effects.effect_matrix import Matrix
    from terminaltexteffects.effects.effect_synthgrid import SynthGrid
except ImportError:
    Matrix = None
    SynthGrid = None

console = Console()
LOGGER = logging.getLogger("mlx_router")

AUTO_FALLBACK_HEADROOM = 0.85

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_PRIMARY_MODEL_REPO = "mlx-community/Qwen3.5-27B-4bit"
DEFAULT_FALLBACK_MODEL_REPO = "mlx-community/Qwen3.5-9B-4bit"


def find_local_model_paths(root: Path) -> list[tuple[int, str]]:
    candidates = []
    for config_path in root.glob("**/config.json"):
        model_dir = config_path.parent
        weight_size = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
        if weight_size:
            candidates.append((weight_size, str(model_dir)))
    return sorted(candidates, reverse=True)


def download_model(repo_id: str, root: Path) -> str:
    from huggingface_hub import snapshot_download

    target_dir = root / repo_id.replace("/", "--")
    console.print("[bold cyan]Lade " + repo_id + " nach " + str(target_dir) + "...[/bold cyan]")
    return snapshot_download(repo_id=repo_id, local_dir=target_dir)


def run_model_setup_wizard(root: Path) -> None:
    if not sys.stdin.isatty():
        raise RuntimeError(
            "Keine lokalen MLX-Modelle gefunden. Starte das Dashboard in einem Terminal, "
            "um den Modell-Wizard auszuführen, oder setze PRIMARY_MODEL_PATH und FALLBACK_MODEL_PATH."
        )

    console.print("[bold cyan]◈ MLX-MODELL-SETUP ◈[/bold cyan]")
    console.print("Es wurden keine lokalen MLX-Modelle gefunden.")
    console.print("[1] Qwen3.5-27B + Qwen3.5-9B herunterladen (Standard)")
    console.print("[2] Eigene Hugging-Face-Modell-IDs eingeben")
    choice = console.input("Auswahl [1]: ").strip() or "1"

    if choice == "2":
        primary_repo = console.input("Primary-Modell-ID: ").strip()
        fallback_repo = console.input("Fallback-Modell-ID: ").strip()
        if not primary_repo or not fallback_repo:
            raise RuntimeError("Für Primary und Fallback ist jeweils eine Hugging-Face-Modell-ID erforderlich.")
    elif choice == "1":
        primary_repo = DEFAULT_PRIMARY_MODEL_REPO
        fallback_repo = DEFAULT_FALLBACK_MODEL_REPO
    else:
        raise RuntimeError("Ungültige Auswahl im Modell-Setup.")

    root.mkdir(parents=True, exist_ok=True)
    try:
        primary_path = download_model(primary_repo, root)
        fallback_path = download_model(fallback_repo, root)
    except Exception as error:
        raise RuntimeError("Modell-Download fehlgeschlagen: " + str(error)) from error

    os.environ["PRIMARY_MODEL_PATH"] = primary_path
    os.environ["FALLBACK_MODEL_PATH"] = fallback_path
    console.print("[bold green]Modelle wurden heruntergeladen. Starte den Router...[/bold green]")


def discover_model_paths():
    root = Path(os.environ.get("MODEL_ROOT", "./models"))
    candidates = find_local_model_paths(root)
    if not candidates:
        run_model_setup_wizard(root)
        candidates = find_local_model_paths(root)
    if not candidates:
        raise RuntimeError("Nach dem Modell-Setup wurden keine MLX-Modelle unter " + str(root) + " gefunden.")
    primary = os.environ.get("PRIMARY_MODEL_PATH", candidates[0][1])
    fallback = os.environ.get("FALLBACK_MODEL_PATH")
    if not fallback:
        fallback = next((path for _, path in reversed(candidates) if path != primary), primary)
    return primary, fallback


def scan_model_spec(model_path):
    path = Path(model_path)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    weight_bytes = sum(item.stat().st_size for item in path.glob("*.safetensors"))
    return {
        "path": str(path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "weight_bytes": weight_bytes,
        "weight_gb": round(weight_bytes / 1024 ** 3, 2),
        "quantization": config.get("quantization", config.get("quantization_config", {})),
        "native_context_tokens": text_config.get("max_position_embeddings"),
        "layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "attention_layers": sum(1 for item in text_config.get("layer_types", []) if item == "full_attention"),
        "kv_heads": text_config.get("num_key_value_heads", text_config.get("num_attention_heads")),
        "head_dim": text_config.get("head_dim"),
        "dtype": text_config.get("dtype", "float16"),
    }


def suggest_model_limits(primary_spec, fallback_spec):
    total_ram_gb = psutil.virtual_memory().total / 1024 ** 3
    system_reserve_gb = max(4.0, total_ram_gb * 0.15)
    loaded_model_gb = primary_spec["weight_gb"]

    def context_for(spec):
        native = int(spec.get("native_context_tokens") or 8192)
        available_gb = max(1.0, total_ram_gb - spec["weight_gb"] - system_reserve_gb)
        attention_layers = max(1, int(spec.get("attention_layers") or spec.get("layers") or 1))
        kv_heads = max(1, int(spec.get("kv_heads") or 1))
        head_dim = max(1, int(spec.get("head_dim") or 128))
        dtype_bytes = 4 if "32" in str(spec.get("dtype", "")) else 2
        kv_bytes_per_token = 2 * attention_layers * kv_heads * head_dim * dtype_bytes
        cache_budget_bytes = available_gb * 1024 ** 3 * 0.20
        memory_limited = int(cache_budget_bytes / kv_bytes_per_token)
        suggested = 1
        while suggested * 2 <= memory_limited:
            suggested *= 2
        return min(native, max(2048, suggested)), available_gb, kv_bytes_per_token

    context, available_gb, kv_bytes_per_token = context_for(primary_spec)
    reserve = max(256, min(1024, context // 8))
    max_generation = max(512, min(8192, context // 4))
    primary_memory_limit = min(total_ram_gb * 0.80, primary_spec["weight_gb"] + available_gb * 0.35)
    fallback_available_gb = max(1.0, total_ram_gb - fallback_spec["weight_gb"] - system_reserve_gb)
    fallback_limit = min(total_ram_gb * 0.80, fallback_spec["weight_gb"] + fallback_available_gb * 0.35)
    return {
        "max_context_tokens": context,
        "context_safety_margin": reserve,
        "max_generation_tokens": max_generation,
        "primary_prompt_limit": max(1024, int((context - reserve - max_generation) * AUTO_FALLBACK_HEADROOM)),
        "primary_memory_limit_gb": round(primary_memory_limit, 2),
        "fallback_memory_limit_gb": round(fallback_limit, 2),
        "primary_prompt_cache_bytes": int(available_gb * 1024 ** 3 * 0.20),
        "total_ram_gb": round(total_ram_gb, 2),
        "system_reserve_gb": round(system_reserve_gb, 2),
        "loaded_model_gb": round(loaded_model_gb, 2),
        "available_for_cache_gb": round(available_gb, 2),
        "primary_kv_bytes_per_token": kv_bytes_per_token,
    }


MODEL_PATH, FALLBACK_MODEL_PATH = discover_model_paths()
MODEL_SPECS = {
    "primary": scan_model_spec(MODEL_PATH),
    "fallback": scan_model_spec(FALLBACK_MODEL_PATH),
}
MODEL_LIMIT_SUGGESTIONS = suggest_model_limits(MODEL_SPECS["primary"], MODEL_SPECS["fallback"])
Path("logs").mkdir(exist_ok=True)
Path("logs/model_specs.json").write_text(json.dumps({
    "scanned_at": datetime.now().isoformat(),
    "models": MODEL_SPECS,
    "suggestions": MODEL_LIMIT_SUGGESTIONS,
}, indent=2))
PRIMARY_PORT = 8080
PROXY_PORT = 8082
HOST = "0.0.0.0"
PREFILL_STEP_SIZE = 4096
PROMPT_CONCURRENCY = 1
DECODE_CONCURRENCY = 1
PROMPT_CACHE_SIZE = 1
PROMPT_CACHE_BYTES = MODEL_LIMIT_SUGGESTIONS["primary_prompt_cache_bytes"]
SERVER_MAX_TOKENS = MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"]
PRIMARY_MEMORY_LIMIT = int(MODEL_LIMIT_SUGGESTIONS["primary_memory_limit_gb"] * 1024 ** 3)

# Fallback configuration - CHANGE THIS TO YOUR SETUP
FALLBACK_TYPE = "mlx_lm"    # "ollama" or "mlx_lm"
FALLBACK_HOST = "localhost"
FALLBACK_PORT = 8081        # Ollama default=11434, second mlx_lm=8081
FALLBACK_MEMORY_LIMIT = int(MODEL_LIMIT_SUGGESTIONS["fallback_memory_limit_gb"] * 1024 ** 3)
AUTO_START_FALLBACK = False
LAZY_FALLBACK = True

MAX_CONTEXT_TOKENS = MODEL_LIMIT_SUGGESTIONS["max_context_tokens"]
CONTEXT_SAFETY_MARGIN = MODEL_LIMIT_SUGGESTIONS["context_safety_margin"]
TOP_PANEL_HEIGHT = 18
LOG_PANEL_HEIGHT = 22
EFFECT_PANEL_HEIGHT = 13
EFFECT_ROWS = 9
TOKEN_LIMIT_27B = MODEL_LIMIT_SUGGESTIONS["primary_prompt_limit"]
SWAP_LIMIT_27B = 4.0
ROUTE_ON_SWAP = True
AUTO_RESTART_27B = True
RESTART_AFTER_REQUESTS_27B = 20
RESTART_ON_SWAP_GB_27B = 8.0
RESTART_COOLDOWN_SECONDS = 60

# Thinking control for primary model. The template has no separate thinking
# budget, so the total generation limit provides a practical upper bound.
THINKING_ENABLED_27B = True
REASONING_EFFORT_27B = "low"
MAX_GENERATION_TOKENS_27B = MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"]
MIN_GENERATION_TOKENS_27B = max(256, min(2048, MAX_GENERATION_TOKENS_27B // 2))
DYNAMIC_FALLBACK = False
CONFIG_FILE = Path("router_config.json")


def load_saved_config():
    if not CONFIG_FILE.exists():
        return
    try:
        saved = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Could not read %s", CONFIG_FILE)
        return
    for name in (
        "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT",
        "MAX_CONTEXT_TOKENS", "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B",
        "PREFILL_STEP_SIZE", "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY",

        "MAX_GENERATION_TOKENS_27B", "THINKING_ENABLED_27B",
        "REASONING_EFFORT_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B",
    ):
        if name in saved:
            globals()[name] = saved[name]


load_saved_config()

if Path(MODEL_PATH).exists() and Path(FALLBACK_MODEL_PATH).exists():
    MODEL_SPECS = {
        "primary": scan_model_spec(MODEL_PATH),
        "fallback": scan_model_spec(FALLBACK_MODEL_PATH),
    }
    MODEL_LIMIT_SUGGESTIONS = suggest_model_limits(MODEL_SPECS["primary"], MODEL_SPECS["fallback"])


def save_config():
    values = {}
    for name in (
        "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT",
        "MAX_CONTEXT_TOKENS", "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B",
        "PREFILL_STEP_SIZE", "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY",
        "MAX_GENERATION_TOKENS_27B", "THINKING_ENABLED_27B",
        "REASONING_EFFORT_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B",
    ):
        values[name] = globals()[name]
    CONFIG_FILE.write_text(json.dumps(values, indent=2) + "\n")


def dynamic_primary_prompt_limit():
    spec = MODEL_SPECS["primary"]
    native = int(spec.get("native_context_tokens") or MAX_CONTEXT_TOKENS)
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    attention_layers = max(1, int(spec.get("attention_layers") or spec.get("layers") or 1))
    kv_heads = max(1, int(spec.get("kv_heads") or 1))
    head_dim = max(1, int(spec.get("head_dim") or 128))
    dtype_bytes = 4 if "32" in str(spec.get("dtype", "")) else 2
    kv_bytes_per_token = 2 * attention_layers * kv_heads * head_dim * dtype_bytes
    cache_budget_bytes = max(0.5, available_gb * 0.20) * 1024 ** 3
    memory_limited = max(2048, int(cache_budget_bytes / kv_bytes_per_token))
    context_limit = min(native, MAX_CONTEXT_TOKENS, memory_limited)
    return max(1024, int((context_limit - CONTEXT_SAFETY_MARGIN - MAX_GENERATION_TOKENS_27B) * AUTO_FALLBACK_HEADROOM))


# ═══════════════════════════════════════════════════════════════
#  KEYBOARD
# ═══════════════════════════════════════════════════════════════

class NonBlockingInput:
    def __init__(self):
        self.commands = deque()
        self.running = True
        self.old_settings = None

    def start(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self.old_settings = None
        threading.Thread(target=self._read_keys, daemon=True).start()

    def _read_keys(self):
        while self.running:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        self.commands.append(char.lower())
            except Exception:
                time.sleep(0.1)

    def get_command(self):
        if self.commands:
            return self.commands.popleft()
        return None

    def stop(self):
        self.running = False
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════
#  FX
# ═══════════════════════════════════════════════════════════════

IDLE_FRAMES = ["◐", "◓", "◑", "◒"]
MAX_SPEED = 1500.0

def dither_bar(percent: float, width: int = 40) -> Text:
    filled = width * (percent / 100)
    full_blocks = int(filled)
    remainder = filled - full_blocks
    bar = Text()
    for i in range(full_blocks):
        hue = 120 + int(60 * (i / width))
        bar.append("█", style=f"bold rgb(0,{hue},200)")
    if full_blocks < width:
        if remainder > 0.75:
            char = "▓"
        elif remainder > 0.5:
            char = "▒"
        elif remainder > 0.25:
            char = "░"
        else:
            char = "▒"
        hue = 120 + int(60 * full_blocks / width)
        bar.append(char, style=f"rgb(0,{hue},180)")
    for i in range(full_blocks + 1, width):
        bar.append("░", style="dim rgb(30,30,50)")
    return bar

def speedometer(speed: float, max_speed: float = MAX_SPEED, width: int = 30) -> Text:
    ratio = min(speed / max_speed, 1.0)
    filled = int(width * ratio)
    result = Text()
    for i in range(width):
        if i < filled:
            if ratio < 0.33:
                color = "rgb(50,255,150)"
            elif ratio < 0.66:
                color = "rgb(255,220,50)"
            else:
                color = "rgb(255,80,50)"
            result.append("━", style=f"bold {color}")
        else:
            result.append("─", style="dim rgb(30,30,50)")
    result.append(f"  {speed:.0f} t/s", style="bold cyan")
    return result

def ram_blocks(used_gb: float, total_gb: float, width: int = 32) -> Text:
    ratio = min(used_gb / total_gb, 1.0)
    filled = int(width * ratio)
    result = Text()
    for i in range(width):
        if i < filled:
            if ratio > 0.92:
                color = "rgb(255,30,30)"
            elif ratio > 0.75:
                color = "rgb(255,180,30)"
            else:
                color = "rgb(50,255,150)"
            result.append("▪", style=color)
        else:
            result.append("▫", style="dim rgb(30,30,50)")
    result.append(f"  {used_gb:.1f}/{total_gb:.0f}GB", style="bold white")
    return result

def sparkline(data: deque, width: int = 40, max_val: float = MAX_SPEED) -> Text:
    if not data or len(data) < 2:
        return Text("░" * width, style="dim rgb(30,30,50)")
    chars = "▁▂▃▄▅▆▇█"
    values = list(data)[-width:]
    min_v = min(values)
    ceiling = max(max(values), max_val * 0.3)
    range_v = max(ceiling - min_v, 0.001)
    result = Text()
    for v in values:
        idx = min(int(((v - min_v) / range_v) * (len(chars) - 1)), len(chars) - 1)
        result.append(chars[idx], style="rgb(100,255,150)")
    while len(result) < width:
        result.append("░", style="dim")
    return result


def matrix_strip(width: int = 48) -> Text:
    chars = "01{}[]<>/\\*+-"
    frame = int(time.time() * 8)
    result = Text()
    for index in range(width):
        value = (index * 37 + frame * (index % 5 + 3)) % 101
        if value < 18:
            char = chars[(index + frame) % len(chars)]
            result.append(char, style="bold bright_green")
        elif value < 42:
            char = chars[(index * 3 + frame) % len(chars)]
            result.append(char, style="green")
        else:
            result.append(" ")
    return result


def synthwave_grid(width: int = 48) -> Text:
    frame = int(time.time() * 6) % 12
    result = Text()
    for index in range(width):
        if (index + frame) % 7 == 0:
            result.append("/", style="bold magenta")
        elif (index + frame) % 5 == 0:
            result.append("_", style="bright_blue")
        else:
            result.append(".", style="rgb(90,30,120)")
    return result


ASCII_FONT = {
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "D": ["████ ", "█   █", "█   █", "█   █", "████ "],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "F": ["█████", "█    ", "████ ", "█    ", "█    "],
    "G": [" ████", "█    ", "█ ███", "█   █", " ███ "],
    "I": ["█████", "  █  ", "  █  ", "  █  ", "█████"],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "P": ["████ ", "█   █", "████ ", "█    ", "█    "],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    " " : ["     ", "     ", "     ", "     ", "     "],
}


def ascii_banner(text: str, color: str) -> list[Text]:
    rows = []
    for row in range(5):
        line = Text()
        for char in text:
            line.append(ASCII_FONT.get(char, ASCII_FONT[" "])[row], style=color)
            line.append(" ")
        rows.append(line)
    return rows

# ═══════════════════════════════════════════════════════════════
#  PROXY
# ═══════════════════════════════════════════════════════════════

class ProxyState:
    def __init__(self):
        self.active_backend = "27B"
        self.routing_mode = "AUTO"
        self.routing_reason = "initial"
        self.last_request_tokens = 0
        self.last_request_time = ""
        self.total_routed_27b = 0
        self.total_routed_fallback = 0
        self.lock = threading.Lock()
        self.primary_healthy = True
        self.fallback_available = False
        self.request_lock = threading.Lock()

PROXY_STATE = ProxyState()

def estimate_tokens_from_body(body: bytes) -> int:
    try:
        data = json.loads(body)
        messages = data.get("messages", [])
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_chars += len(part["text"])
        return total_chars // 4
    except Exception:
        return 0


def inject_chat_template_kwargs(body: bytes, kwargs: dict) -> bytes:
    """Merge chat_template_kwargs into a request body."""
    try:
        data = json.loads(body)
    except Exception:
        return body

    existing = data.get("chat_template_kwargs", {}) or {}
    merged = {**existing, **kwargs}
    data["chat_template_kwargs"] = merged
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def limit_generation_to_context(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    prompt_tokens = estimate_tokens_from_body(body)
    available_tokens = max(
        1,
        MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_MARGIN,
    )
    for field in ("max_tokens", "max_completion_tokens"):
        requested = data.get(field)
        if isinstance(requested, int) and requested > available_tokens:
            data[field] = available_tokens
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def ensure_generation_budget_for_primary(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    fields = ("max_tokens", "max_completion_tokens")
    found_limit = False
    for field in fields:
        requested = data.get(field)
        if isinstance(requested, int):
            data[field] = max(requested, MIN_GENERATION_TOKENS_27B)
            found_limit = True
    if not found_limit:
        data["max_tokens"] = MIN_GENERATION_TOKENS_27B
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def apply_primary_sampling_policy(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    data.setdefault("repetition_penalty", 1.12)
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def limit_generation_for_primary(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    for field in ("max_tokens", "max_completion_tokens"):
        requested = data.get(field)
        if isinstance(requested, int) and requested > MAX_GENERATION_TOKENS_27B:
            data[field] = MAX_GENERATION_TOKENS_27B
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def normalize_backend_model(body: bytes, target_port: int) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    if target_port == PRIMARY_PORT:
        data["model"] = MODEL_PATH
    elif target_port == FALLBACK_PORT:
        data["model"] = FALLBACK_MODEL_PATH
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def normalize_model_list(body: bytes, target_port: int) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    role = "primary" if target_port == PRIMARY_PORT else "fallback"
    models = data.get("data", [])
    template = dict(models[0]) if models else {"object": "model"}
    template["id"] = role
    data["data"] = [template]
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def model_list_response():
    return json.dumps({
        "object": "list",
        "data": [{
            "id": "primary",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }).encode("utf-8")


def health_response():
    return json.dumps({"status": "ok", "service": "mlx-router"}).encode("utf-8")

def check_backend_available(port: int, path: str) -> bool:
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        conn.close()
        return response.status == 200
    except Exception:
        return False

def check_fallback_available() -> bool:
    try:
        conn = http.client.HTTPConnection(FALLBACK_HOST, FALLBACK_PORT, timeout=2)
        path = "/v1/models" if FALLBACK_TYPE in ("ollama", "mlx_lm") else "/"
        conn.request("GET", path)
        response = conn.getresponse()
        conn.close()
        return response.status == 200
    except Exception:
        return False

class ProxyHandler(BaseHTTPRequestHandler):
    dashboard = None

    def _record_decode_result(self, response_body: bytes, started_at: float):
        if not self.dashboard or not response_body:
            return
        try:
            payload = json.loads(response_body)
            generated_tokens = int(payload.get("usage", {}).get("completion_tokens", 0) or 0)
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        if generated_tokens <= 0:
            return
        elapsed = max(time.perf_counter() - started_at, 0.001)
        with self.dashboard.lock:
            self.dashboard.inference_phase = "DECODE"
            self.dashboard.gen_tokens = generated_tokens
            self.dashboard.decode_speed = generated_tokens / elapsed
            self.dashboard.last_infer_time = time.time()
        LOGGER.info(
            "proxy decode completed tokens=%d elapsed=%.3fs rate=%.1f tok/s",
            generated_tokens,
            elapsed,
            generated_tokens / elapsed,
        )

    def log_message(self, format, *args):
        pass

    def _route_request(self, body: bytes) -> tuple[int, str]:
        estimated_tokens = estimate_tokens_from_body(body)
        dynamic_limit = dynamic_primary_prompt_limit() if DYNAMIC_FALLBACK else TOKEN_LIMIT_27B

        with PROXY_STATE.lock:
            PROXY_STATE.last_request_tokens = estimated_tokens
            PROXY_STATE.last_request_time = datetime.now().strftime("%H:%M:%S")

            if self.command == "GET" and self.path.startswith("/v1/models"):
                preferred_port = FALLBACK_PORT if PROXY_STATE.routing_mode == "FALLBACK" else PRIMARY_PORT
                if check_backend_available(preferred_port, "/v1/models"):
                    PROXY_STATE.active_backend = "FALLBACK" if preferred_port == FALLBACK_PORT else "27B"
                    PROXY_STATE.routing_reason = "model list"
                    return preferred_port, PROXY_STATE.routing_reason
                alternate_port = FALLBACK_PORT if preferred_port == PRIMARY_PORT else PRIMARY_PORT
                if check_backend_available(alternate_port, "/v1/models"):
                    PROXY_STATE.active_backend = "FALLBACK" if alternate_port == FALLBACK_PORT else "27B"
                    PROXY_STATE.routing_reason = "preferred backend unavailable"
                    return alternate_port, PROXY_STATE.routing_reason
                PROXY_STATE.routing_reason = "no backend available"
                raise ConnectionError("No model backend available")

            fallback_ok = check_fallback_available()
            fallback_configured = fallback_ok or (LAZY_FALLBACK and self.dashboard is not None)
            PROXY_STATE.fallback_available = fallback_ok

            if PROXY_STATE.routing_mode == "PRIMARY":
                if check_backend_available(PRIMARY_PORT, "/v1/models"):
                    PROXY_STATE.active_backend = "27B"
                    PROXY_STATE.routing_reason = "27B direct"
                    PROXY_STATE.total_routed_27b += 1
                    return PRIMARY_PORT, PROXY_STATE.routing_reason
            elif PROXY_STATE.routing_mode == "FALLBACK":
                if fallback_ok:
                    PROXY_STATE.active_backend = "FALLBACK"
                    PROXY_STATE.routing_reason = "9B direct"
                    PROXY_STATE.total_routed_fallback += 1
                    return FALLBACK_PORT, PROXY_STATE.routing_reason

            if not fallback_configured:
                PROXY_STATE.active_backend = "27B"
                PROXY_STATE.routing_reason = str(estimated_tokens) + "t (no fallback)"
                PROXY_STATE.total_routed_27b += 1
                return PRIMARY_PORT, PROXY_STATE.routing_reason

            swap = psutil.swap_memory().used / (1024**3)
            mem = psutil.virtual_memory()
            ram_pct = mem.percent

            if estimated_tokens > dynamic_limit:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = str(estimated_tokens) + "t > " + str(dynamic_limit) + "t limit"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if ROUTE_ON_SWAP and swap > SWAP_LIMIT_27B:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "SWAP " + str(round(swap, 1)) + "GB > " + str(SWAP_LIMIT_27B) + "GB"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if ram_pct > 90 and estimated_tokens > 1000:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "RAM " + str(int(ram_pct)) + "% + " + str(estimated_tokens) + "t"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if not PROXY_STATE.primary_healthy:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "27B crashed/unhealthy"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            PROXY_STATE.active_backend = "27B"
            PROXY_STATE.routing_reason = str(estimated_tokens) + "t < " + str(TOKEN_LIMIT_27B) + "t, healthy"
            PROXY_STATE.total_routed_27b += 1
            return PRIMARY_PORT, PROXY_STATE.routing_reason

    def _prepare_body(self, target_port: int, body: bytes) -> bytes:
        if self.path != "/v1/chat/completions":
            return body
        body = normalize_backend_model(body, target_port)
        if target_port == PRIMARY_PORT:
            body = ensure_generation_budget_for_primary(body)
            body = limit_generation_to_context(body)
            body = limit_generation_for_primary(body)
            body = apply_primary_sampling_policy(body)
            body = inject_chat_template_kwargs(
                body,
                {
                    "enable_thinking": THINKING_ENABLED_27B,
                    "reasoning_effort": REASONING_EFFORT_27B,
                },
            )
        elif target_port == FALLBACK_PORT:
            body = inject_chat_template_kwargs(body, {"enable_thinking": False})
        return body

    def _forward(self, method: str, body: bytes = None):
        if method in ("GET", "HEAD") and self.path in ("/", "/health", "/v1/health"):
            health_body = health_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(health_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            if method == "GET":
                self.wfile.write(health_body)
                self.wfile.flush()
            return
        if method in ("GET", "HEAD") and self.path in ("/models", "/v1/models"):
            model_body = model_list_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(model_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            if method == "GET":
                self.wfile.write(model_body)
                self.wfile.flush()
            return
        request_lock = None
        if method == "POST" and self.path == "/v1/chat/completions" and self.dashboard:
            request_lock = PROXY_STATE.request_lock
            request_lock.acquire()
        try:
            target_port, reason = self._route_request(body or b"")
        except ConnectionError as error:
            LOGGER.error("no backend available method=%s path=%s error=%s", method, self.path, error)
            self.send_error(503, "No model backend available")
            if request_lock:
                request_lock.release()
            return
        if self.command == "POST" and self.path == "/v1/chat/completions" and self.dashboard:
            if not self.dashboard.ensure_backend(target_port):
                LOGGER.error("selected backend failed to start target=%s path=%s", target_port, self.path)
                self.send_error(503, "Selected model backend failed to start")
                if request_lock:
                    request_lock.release()
                return
            if not check_backend_available(target_port, "/v1/models"):
                LOGGER.error("selected backend unavailable target=%s path=%s", target_port, self.path)
                self.send_error(503, "Selected model backend is unavailable")
                if request_lock:
                    request_lock.release()
                return
            with self.dashboard.lock:
                self.dashboard.inference_phase = "PREFILL"
                self.dashboard.is_processing = True
                self.dashboard.prompt_current = 0
                self.dashboard.prompt_total = estimate_tokens_from_body(body or b"")
                self.dashboard.gen_tokens = 0
                self.dashboard.last_activity = datetime.now().strftime("%H:%M:%S")
                self.dashboard.last_infer_time = time.time()
        target_host = "localhost"
        prepared_body = self._prepare_body(target_port, body or b"")
        LOGGER.info(
            "route method=%s path=%s target=%s reason=%s prompt_estimate=%d "
            "body_bytes=%d prepared_bytes=%d",
            method,
            self.path,
            target_port,
            reason,
            estimate_tokens_from_body(body or b""),
            len(body or b""),
            len(prepared_body),
        )

        try:
            started_at = time.perf_counter()
            decode_started_at = None
            streamed_tokens = 0
            stream_buffer = b""
            response_body = bytearray()
            conn = http.client.HTTPConnection(target_host, target_port, timeout=300)
            headers = {k: v for k, v in self.headers.items()}
            headers['Content-Length'] = str(len(prepared_body))
            path = self.path

            conn.request(method, path, body=prepared_body, headers=headers)
            response = conn.getresponse()

            if method == "GET" and path.startswith("/v1/models"):
                model_body = normalize_model_list(response.read(), target_port)
                self.send_response(response.status)
                for header, value in response.getheaders():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                self.send_header("Content-Length", str(len(model_body)))
                self.end_headers()
                self.wfile.write(model_body)
                self.wfile.flush()
                conn.close()
                return

            self.send_response(response.status)
            for header, value in response.getheaders():
                if header.lower() not in ('transfer-encoding',):
                    self.send_header(header, value)
            self.end_headers()

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                if method == "POST" and path == "/v1/chat/completions":
                    response_body.extend(chunk)
                if method == "POST" and path == "/v1/chat/completions":
                    stream_buffer += chunk
                    while b"\n" in stream_buffer:
                        raw_line, stream_buffer = stream_buffer.split(b"\n", 1)
                        if not raw_line.startswith(b"data: "):
                            continue
                        payload = raw_line[6:].strip()
                        if payload in (b"", b"[DONE]"):
                            continue
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices", [])
                        delta = choices[0].get("delta", {}) if choices else {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            if decode_started_at is None:
                                decode_started_at = time.perf_counter()
                            streamed_tokens += 1
                            elapsed = time.perf_counter() - decode_started_at
                            rate = streamed_tokens / elapsed if elapsed > 0 else 0.0
                            with self.dashboard.lock:
                                self.dashboard.inference_phase = "DECODE"
                                self.dashboard.gen_tokens = streamed_tokens
                                self.dashboard.decode_speed = rate
                                self.dashboard.is_processing = True
                                self.dashboard.last_infer_time = time.time()
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except BrokenPipeError:
                    break

            conn.close()
            if method == "POST" and path == "/v1/chat/completions" and not streamed_tokens:
                self._record_decode_result(bytes(response_body), started_at)
            if method == "POST" and path == "/v1/chat/completions" and streamed_tokens:
                elapsed = time.perf_counter() - decode_started_at
                LOGGER.info(
                    "proxy decode completed tokens=%d elapsed=%.3fs rate=%.1f tok/s",
                    streamed_tokens,
                    elapsed,
                    streamed_tokens / elapsed if elapsed > 0 else 0,
                )
            if method == "POST" and path == "/v1/chat/completions" and self.dashboard:
                with self.dashboard.lock:
                    self.dashboard.is_processing = False
                    self.dashboard.inference_phase = "IDLE"
            LOGGER.info(
                "response method=%s path=%s target=%s status=%d elapsed=%.3fs",
                method,
                self.path,
                target_port,
                response.status,
                time.perf_counter() - started_at,
            )

        except Exception:
            LOGGER.exception("backend failure target=%s path=%s", target_port, self.path)
            if target_port == PRIMARY_PORT and LAZY_FALLBACK and self.dashboard:
                with PROXY_STATE.lock:
                    PROXY_STATE.primary_healthy = False
                    PROXY_STATE.fallback_available = True
                if self.dashboard.ensure_backend(FALLBACK_PORT):
                    self._forward_fallback(method, self._prepare_body(FALLBACK_PORT, body or b""))
                    return
            if target_port == PRIMARY_PORT and PROXY_STATE.fallback_available:
                with PROXY_STATE.lock:
                    PROXY_STATE.primary_healthy = False
                self._forward_fallback(method, body)
            else:
                self.send_error(502, "Bad Gateway")
        finally:
            if request_lock:
                request_lock.release()

    def _forward_fallback(self, method: str, body: bytes):
        try:
            conn = http.client.HTTPConnection("localhost", FALLBACK_PORT, timeout=300)
            headers = {k: v for k, v in self.headers.items()}
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()

            self.send_response(response.status)
            for header, value in response.getheaders():
                if header.lower() not in ('transfer-encoding',):
                    self.send_header(header, value)
            self.end_headers()

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except BrokenPipeError:
                    break
            conn.close()
        except Exception:
            self.send_error(502, "Both backends failed")

    def do_GET(self):
        self._forward("GET")

    def do_HEAD(self):
        self._forward("HEAD")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        self._forward("POST", body)

# ═══════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════

class ManagedServer:
    def __init__(self, command, environment, log_buffer, log_handler, fallback=False):
        self.command = command
        self.environment = environment
        self.log_buffer = log_buffer
        self.log_handler = log_handler
        self.fallback = fallback
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=self.environment,
        )
        threading.Thread(
            target=self.log_handler,
            args=(self.process, self.log_buffer, "ONLINE", "ERROR", self.fallback),
            daemon=True,
        ).start()
        return self.process

    def stop(self):
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class MLXDashboard:
    def __init__(self):
        self.server_process = None
        self.fallback_process = None
        self.proxy_server = None
        self.status = "BOOT"
        self.fallback_status = "BOOT" if AUTO_START_FALLBACK else "MANUAL"
        self.prompt_current = 0
        self.prompt_total = 0
        self.prompt_speed = 0.0
        self.prefill_speed = 0.0
        self.decode_speed = 0.0
        self.memory_gb = 0.0
        self.cache_sequences = 0
        self.cache_size_gb = 0.0
        self.last_activity = "--:--:--"
        self.is_processing = False
        self.inference_phase = "IDLE"
        self.effect_iterator = None
        self.effect_iterator_phase = None
        self.process_start_time = None
        self.request_count = 0
        self.uptime_start = time.time()
        self.speed_history = deque(maxlen=80)
        self.token_history = deque(maxlen=80)
        self.last_history_sample = 0.0
        self.log_buffer = deque(maxlen=30)
        self.fallback_log_buffer = deque(maxlen=30)
        self.primary_server = None
        self.fallback_server = None
        self.running = True
        self.lock = threading.Lock()
        self.keyboard = NonBlockingInput()
        self.notification = ""
        self.notif_time = 0
        self.idle_frame_idx = 0
        self.last_infer_time = 0
        self.gen_tokens = 0
        self.eta_seconds = 0

        self.total_ram = psutil.virtual_memory().total / (1024**3)
        self.used_ram = 0.0
        self.swap_used = 0.0
        self.top_processes = []
        self.ram_warning = False
        self.force_fallback = False
        self.config_mode = False
        self.last_primary_restart = 0.0
        self.restart_lock = threading.Lock()
        self.backend_switch_lock = threading.Lock()
        self.last_restart_request_count = 0
        self.swap_restart_armed = True
        with PROXY_STATE.lock:
            PROXY_STATE.active_backend = "27B"
            PROXY_STATE.routing_reason = "initial"
            PROXY_STATE.last_request_tokens = 0
            PROXY_STATE.last_request_time = ""
            PROXY_STATE.total_routed_27b = 0
            PROXY_STATE.total_routed_fallback = 0
            PROXY_STATE.primary_healthy = True
            PROXY_STATE.fallback_available = False

    def start_proxy(self):
        try:
            ProxyHandler.dashboard = self
            self.proxy_server = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
            threading.Thread(target=self.proxy_server.serve_forever, daemon=True).start()
            self._notify("PROXY STARTED ON PORT " + str(PROXY_PORT))
        except Exception as e:
            self._notify("PROXY FAILED: " + str(e))

    def start_server(self):
        cmd = [sys.executable, "mlx_server_safe_wrapper.py"]
        env = os.environ.copy()
        env.update({
            "PRIMARY_MODEL_PATH": MODEL_PATH,
            "PRIMARY_PORT": str(PRIMARY_PORT),
            "PREFILL_STEP_SIZE": str(PREFILL_STEP_SIZE),
            "PROMPT_CONCURRENCY": str(PROMPT_CONCURRENCY),
            "DECODE_CONCURRENCY": str(DECODE_CONCURRENCY),
            "PROMPT_CACHE_SIZE": str(PROMPT_CACHE_SIZE),
            "PROMPT_CACHE_BYTES": str(PROMPT_CACHE_BYTES),
            "PRIMARY_MEMORY_LIMIT": str(PRIMARY_MEMORY_LIMIT),
            "SERVER_MAX_TOKENS": str(SERVER_MAX_TOKENS),
        })
        self.primary_server = ManagedServer(cmd, env, self.log_buffer, self._parse_logs)
        self.server_process = self.primary_server.start()
        if not hasattr(self, "monitor_thread") or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()

    def start_fallback_server(self):
        if not AUTO_START_FALLBACK and not LAZY_FALLBACK:
            return
        cmd = [sys.executable, "mlx_server_fallback_wrapper.py"]
        env = os.environ.copy()
        env["FALLBACK_MODEL_PATH"] = FALLBACK_MODEL_PATH
        env["FALLBACK_PORT"] = str(FALLBACK_PORT)
        env["FALLBACK_MEMORY_LIMIT"] = str(FALLBACK_MEMORY_LIMIT)
        env["PREFILL_STEP_SIZE"] = str(PREFILL_STEP_SIZE)
        env["PROMPT_CONCURRENCY"] = str(PROMPT_CONCURRENCY)
        env["DECODE_CONCURRENCY"] = str(DECODE_CONCURRENCY)
        env["PROMPT_CACHE_SIZE"] = str(PROMPT_CACHE_SIZE)
        env["PROMPT_CACHE_BYTES"] = str(PROMPT_CACHE_BYTES)
        env["SERVER_MAX_TOKENS"] = str(SERVER_MAX_TOKENS)
        self.fallback_server = ManagedServer(
            cmd, env, self.fallback_log_buffer, self._parse_logs, fallback=True
        )
        self.fallback_process = self.fallback_server.start()

    def ensure_backend(self, target_port):
        with self.backend_switch_lock:
            desired_process = self.server_process if target_port == PRIMARY_PORT else self.fallback_process
            if desired_process and desired_process.poll() is None:
                return check_backend_available(target_port, "/v1/models")

            if target_port == PRIMARY_PORT:
                if self.fallback_server:
                    self.fallback_server.stop()
                self.start_server()
            else:
                if self.primary_server:
                    self.primary_server.stop()
                self.start_fallback_server()

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if check_backend_available(target_port, "/v1/models"):
                    return True
                time.sleep(0.5)
            return False

    def restart_server(self):
        if not self.restart_lock.acquire(blocking=False):
            return
        try:
            self.last_primary_restart = time.time()
            self._restart_server_locked()
        finally:
            self.restart_lock.release()

    def _restart_server_locked(self):
        self._notify("RESTARTING PRIMARY SERVER...")
        if self.primary_server:
            self.primary_server.stop()
        self.prompt_current = 0
        self.prompt_total = 0
        self.prompt_speed = 0.0
        self.prefill_speed = 0.0
        self.decode_speed = 0.0
        self.inference_phase = "IDLE"
        self.gen_tokens = 0
        self.eta_seconds = 0
        self.speed_history.clear()
        self.token_history.clear()
        with PROXY_STATE.lock:
            PROXY_STATE.primary_healthy = True
        self.start_server()
        self.last_restart_request_count = self.request_count
        self._notify("PRIMARY SERVER RESTARTED")

    def _maybe_auto_restart(self):
        if LAZY_FALLBACK or not AUTO_RESTART_27B or self.is_processing or self.force_fallback:
            return
        now = time.time()
        if now - self.last_primary_restart < RESTART_COOLDOWN_SECONDS:
            return
        swap_triggered = ROUTE_ON_SWAP and self.swap_used >= RESTART_ON_SWAP_GB_27B
        if not swap_triggered:
            self.swap_restart_armed = True
        request_triggered = self.request_count - self.last_restart_request_count >= RESTART_AFTER_REQUESTS_27B
        swap_triggered = swap_triggered and self.swap_restart_armed
        if swap_triggered or request_triggered:
            reason = "SWAP threshold" if swap_triggered else "request limit"
            self._notify("AUTO-RESTART: " + reason)
            self.swap_restart_armed = False
            self.restart_server()

    def restart_fallback_server(self):
        self._notify("RESTARTING FALLBACK SERVER...")
        if self.fallback_server:
            self.fallback_server.stop()
        self.fallback_status = "BOOT"
        self.start_fallback_server()
        self._notify("FALLBACK SERVER RESTARTED")

    def toggle_fallback(self):
        self.set_routing_mode("AUTO" if self.force_fallback else "FALLBACK")

    def set_routing_mode(self, mode):
        global DYNAMIC_FALLBACK
        self.force_fallback = mode == "FALLBACK"
        DYNAMIC_FALLBACK = mode == "DYNAMIC"
        with PROXY_STATE.lock:
            PROXY_STATE.routing_mode = mode
            PROXY_STATE.primary_healthy = True
        self._notify("ROUTING MODE: " + mode)

    def stop_process(self, process, label):
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self._notify(label + " SERVER STOPPED")

    def toggle_primary_server(self):
        if self.server_process and self.server_process.poll() is None:
            self.stop_process(self.server_process, "27B")
            return
        self.start_server()
        self._notify("27B SERVER STARTED")

    def toggle_fallback_server(self):
        if self.fallback_process and self.fallback_process.poll() is None:
            self.stop_process(self.fallback_process, "FALLBACK")
            self.fallback_status = "MANUAL"
            return
        self.start_fallback_server()
        self._notify("FALLBACK SERVER STARTED")

    def clear_logs(self):
        with self.lock:
            self.log_buffer.clear()
            self.fallback_log_buffer.clear()
            self.speed_history.clear()
            self.token_history.clear()
        self._notify("LOGS CLEARED")

    def print_stats(self):
        Path("logs").mkdir(exist_ok=True)
        filename = "logs/mlx_stats_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
        with open(filename, "w") as f:
            f.write("MLX Router Stats - " + str(datetime.now()) + "\n")
            f.write("Status: " + self.status + "\n")
            f.write("Requests: " + str(self.request_count) + "\n")
            f.write("Routed to 27B: " + str(PROXY_STATE.total_routed_27b) + "\n")
            f.write("Routed to Fallback: " + str(PROXY_STATE.total_routed_fallback) + "\n")
            f.write("RAM: " + str(round(self.used_ram, 1)) + "/" + str(int(self.total_ram)) + " GB\n")
        self._notify("STATS SAVED: " + filename)

    def configure(self, live):
        global MODEL_PATH, FALLBACK_MODEL_PATH, MAX_CONTEXT_TOKENS
        global CONTEXT_SAFETY_MARGIN, TOKEN_LIMIT_27B, PREFILL_STEP_SIZE
        global PROMPT_CONCURRENCY, DECODE_CONCURRENCY, THINKING_ENABLED_27B
        global REASONING_EFFORT_27B, MAX_GENERATION_TOKENS_27B
        global ROUTE_ON_SWAP, AUTO_RESTART_27B
        global FALLBACK_MEMORY_LIMIT
        live.stop()
        self.keyboard.stop()
        settings = [
            ("Primary Modellpfad", MODEL_PATH, str),
            ("Fallback Modellpfad", FALLBACK_MODEL_PATH, str),
            ("Fallback Metal-Limit GB", FALLBACK_MEMORY_LIMIT / 1024 ** 3, float),
            ("Gesamtkontext", MAX_CONTEXT_TOKENS, int),
            ("Kontextreserve", CONTEXT_SAFETY_MARGIN, int),
            ("AUTO: Primary bis Prompt", TOKEN_LIMIT_27B, int),
            ("Prefill Schrittgröße", PREFILL_STEP_SIZE, int),
            ("Prefill Parallelität", PROMPT_CONCURRENCY, int),
            ("Decode Parallelität", DECODE_CONCURRENCY, int),
            ("Primary max. Ausgabe", MAX_GENERATION_TOKENS_27B, int),
            ("Primary Thinking", "AN" if THINKING_ENABLED_27B else "AUS", str),
            ("Primary Reasoning", REASONING_EFFORT_27B, str),
            ("SWAP-Routing", "AN" if ROUTE_ON_SWAP else "AUS", str),
            ("Auto-Restart Primary", "AN" if AUTO_RESTART_27B else "AUS", str),
        ]
        values = [item[1] for item in settings]

        def render_config(selected):
            console.clear()
            table = Table(title="MLX KONFIGURATION", border_style="cyan", expand=True)
            table.add_column("", width=3)
            table.add_column("Parameter", style="bold cyan")
            table.add_column("Wert", style="white")
            for index, (label, _, _) in enumerate(settings):
                marker = "▶" if index == selected else " "
                style = "bold yellow" if index == selected else "white"
                table.add_row(marker, label, str(values[index]), style=style)
            console.print(table)
            console.print(
                "[dim]Scan-Vorschlag: Kontext "
                + str(MODEL_LIMIT_SUGGESTIONS["max_context_tokens"])
                + " | Primary-Prompt "
                + str(MODEL_LIMIT_SUGGESTIONS["primary_prompt_limit"])
                + " | Ausgabe "
                + str(MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"])
                + " | Fallback-Metal "
                + str(MODEL_LIMIT_SUGGESTIONS["fallback_memory_limit_gb"])
                + " GB[/dim]"
            )
            console.print("\n[dim][↑/↓ oder J/K] Auswahl  [E/Enter] Bearbeiten  [S] Speichern  [Q/Esc] Abbrechen[/dim]")

        def read_key():
            stdin_fd = sys.stdin.fileno()
            if not select.select([stdin_fd], [], [], 0.1)[0]:
                return None
            key = os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
            if key == "\x1b":
                sequence = ""
                while select.select([stdin_fd], [], [], 0.1)[0]:
                    sequence += os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
                    if sequence[-1:] in ("A", "B", "C", "D", "~"):
                        break
                if sequence.endswith("A"):
                    return "up"
                if sequence.endswith("B"):
                    return "down"
                return "escape"
            if key in ("\r", "\n"):
                return "enter"
            return key.lower()

        try:
            selected = 0
            saved = False
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            render_config(selected)
            while not saved:
                command = read_key()
                if command is None:
                    continue
                if command in ("q", "esc", "escape", "quit"):
                    break
                if command in ("j", "down", "v"):
                    selected = (selected + 1) % len(settings)
                    render_config(selected)
                    continue
                if command in ("k", "up", "^"):
                    selected = (selected - 1) % len(settings)
                    render_config(selected)
                    continue
                if command in ("e", "enter"):
                    label, _, convert = settings[selected]
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    value = console.input("[bold yellow]" + label + "[/bold yellow] [dim][" + str(values[selected]) + "][/dim]: ").strip()
                    if value:
                        values[selected] = convert(value)
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setcbreak(sys.stdin.fileno())
                    render_config(selected)
                    continue
                if command in ("s", "save"):
                    saved = True

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            if saved:
                config_values = [
                    "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT", "MAX_CONTEXT_TOKENS",
                    "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B", "PREFILL_STEP_SIZE",
                    "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY", "MAX_GENERATION_TOKENS_27B",
                    "THINKING_ENABLED_27B", "REASONING_EFFORT_27B", "ROUTE_ON_SWAP",
                    "AUTO_RESTART_27B",
                ]
                for name, value in zip(config_values, values):
                    if name == "FALLBACK_MEMORY_LIMIT":
                        value = int(float(value) * 1024 ** 3)
                    elif name in ("THINKING_ENABLED_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B"):
                        value = str(value).lower() in ("an", "j", "ja", "y", "yes", "1", "true")
                    globals()[name] = value
                save_config()
                console.clear()
                console.print("[bold green]Konfiguration gespeichert.[/bold green]")
                restart = console.input("Server jetzt neu starten? [J/n]: ").strip().lower()
                if restart not in ("n", "nein", "no", "0"):
                    acquired = self.restart_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            self._restart_server_locked()
                        finally:
                            self.restart_lock.release()
                    if AUTO_START_FALLBACK:
                        self.restart_fallback_server()
                console.input("\nEnter zum Fortfahren...")
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("\n[bold red]Ungültige Eingabe oder Konfiguration abgebrochen.[/bold red]")
            console.input("Enter zum Fortfahren...")
        finally:
            self.keyboard = NonBlockingInput()
            self.keyboard.start()
            live.start()

    def _notify(self, msg: str):
        self.notification = msg
        self.notif_time = time.time()

    def _calculate_eta(self) -> str:
        if not self.is_processing or self.prompt_total <= 0 or self.prompt_speed <= 0:
            return "--"
        remaining = self.prompt_total - self.prompt_current
        if remaining <= 0:
            return "0s"
        eta = remaining / self.prompt_speed
        self.eta_seconds = eta
        if eta < 60:
            return "~" + str(int(eta)) + "s"
        elif eta < 3600:
            mins = int(eta / 60)
            secs = int(eta % 60)
            return "~" + str(mins) + "m " + str(secs) + "s"
        else:
            hrs = int(eta / 3600)
            mins = int((eta % 3600) / 60)
            return "~" + str(hrs) + "h " + str(mins) + "m"

    def _sample_histograms(self) -> None:
        now = time.monotonic()
        if not self.is_processing or now - self.last_history_sample < 0.5:
            return
        self.last_history_sample = now
        if self.inference_phase == "DECODE":
            self.speed_history.append(self.decode_speed)
            self.token_history.append(self.gen_tokens)
        else:
            self.speed_history.append(self.prefill_speed)
            self.token_history.append(self.prompt_current)

    def _parse_logs(self, process, buffer, online_status, error_status, is_fallback=False):
        if not process:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            with self.lock:
                buffer.append(line)
                if any(marker in line for marker in (
                    "Request started:",
                    "Generation queued:",
                    "Prefill completed:",
                    "Decode started:",
                    "Decode progress:",
                    "Request completed:",
                    "Request failed:",
                )):
                    LOGGER.info("server=%s %s", "fallback" if is_fallback else "primary", line)
                if is_fallback:
                    if "Starting httpd" in line or "Application startup complete" in line or "Uvicorn running" in line:
                        self.fallback_status = online_status
                    if "Error" in line or "RuntimeError" in line or "OutOfMemory" in line:
                        self.fallback_status = error_status

                if "Starting httpd" in line or "Application startup complete" in line or "Uvicorn running" in line:
                    self.status = "ONLINE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                if "POST /v1/chat/completions" in line or "GET /v1/models" in line:
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                if "Request started: endpoint=/chat/completions" in line:
                    self.request_count += 1
                    self.is_processing = True
                    self.inference_phase = "PREFILL"
                    self.prompt_current = 0
                    self.prompt_total = 0
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()

                queued_match = re.search(
                    r"Generation queued:.*prompt_tokens=(\d+).*max_tokens=(\d+)",
                    line,
                )
                if queued_match:
                    self.prompt_current = 0
                    self.prompt_total = int(queued_match.group(1))
                    self.is_processing = True
                    self.process_start_time = time.time()
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                if "Prefill started:" in line:
                    prefill_started_match = re.search(r"prompt_tokens=(\d+)", line)
                    if prefill_started_match:
                        self.prompt_current = 0
                        self.prompt_total = int(prefill_started_match.group(1))
                    self.is_processing = True
                    self.inference_phase = "PREFILL"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                prefill_match = re.search(
                    r"Prefill progress:.*tokens=(\d+)/(\d+)", line,
                )
                if prefill_match:
                    self.prompt_current = int(prefill_match.group(1))
                    self.prompt_total = int(prefill_match.group(2))
                    self.is_processing = True
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        if elapsed > 0:
                            self.prefill_speed = self.prompt_current / elapsed
                            self.prompt_speed = self.prefill_speed
                            self.speed_history.append(self.prefill_speed)
                            self.token_history.append(self.prompt_current)

                prefill_complete_match = re.search(
                    r"Prefill completed:.*prompt_tokens=(\d+).*rate=([\d.]+)",
                    line,
                )
                if prefill_complete_match:
                    self.prompt_current = int(prefill_complete_match.group(1))
                    self.prompt_total = self.prompt_current
                    self.prefill_speed = float(prefill_complete_match.group(2))
                    self.prompt_speed = self.prefill_speed
                    self.speed_history.append(self.prefill_speed)
                    self.token_history.append(self.prompt_current)
                    self.process_start_time = None
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()
                    self.inference_phase = "DECODE"

                decode_match = re.search(
                    r"Decode progress:.*generated_tokens=(\d+).*rate=([\d.]+)",
                    line,
                )
                if decode_match:
                    self.gen_tokens = int(decode_match.group(1))
                    rate_text = decode_match.group(2)
                    if rate_text:
                        self.decode_speed = float(rate_text)
                        self.speed_history.append(self.decode_speed)
                    self.is_processing = True
                    self.inference_phase = "DECODE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()

                if "Request completed: endpoint=/chat/completions" in line:
                    self.is_processing = False
                    self.inference_phase = "IDLE"
                    self.last_infer_time = time.time()
                    self.process_start_time = None

                progress_match = re.search(r'Prompt processing progress:\s+(\d+)/(\d+)', line)
                if progress_match:
                    self.prompt_current = int(progress_match.group(1))
                    self.prompt_total = int(progress_match.group(2))
                    self.is_processing = True
                    self.inference_phase = "PREFILL" if self.prompt_current < self.prompt_total else "DECODE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    now = time.time()
                    if not self.process_start_time:
                        self.process_start_time = self.last_infer_time or now
                    self.last_infer_time = now
                    elapsed = now - self.process_start_time
                    if elapsed > 0 and self.prompt_current > 0:
                        self.prefill_speed = self.prompt_current / elapsed
                        self.prompt_speed = self.prefill_speed
                        self.speed_history.append(self.prefill_speed)
                        self.token_history.append(self.prompt_current)

                gen_match = re.search(r'Generation:\s+(\d+)\s+tokens?', line)
                if gen_match:
                    self.gen_tokens = int(gen_match.group(1))
                    self.is_processing = False
                    self.inference_phase = "IDLE"
                    self.last_infer_time = time.time()
                    self.eta_seconds = 0
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        self.prefill_speed = self.prompt_total / elapsed if elapsed > 0 else 0
                        self.prompt_speed = self.prefill_speed
                        self.process_start_time = None
                        self.speed_history.append(self.prefill_speed)

                if self.prompt_current > 0 and self.prompt_current >= self.prompt_total:
                    self.is_processing = True
                    self.inference_phase = "DECODE"
                    self.last_infer_time = time.time()
                    self.eta_seconds = 0
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        self.prefill_speed = self.prompt_total / elapsed if elapsed > 0 else 0
                        self.prompt_speed = self.prefill_speed
                        self.process_start_time = None
                        self.speed_history.append(self.prefill_speed)

                cache_match = re.search(r'Prompt Cache:\s+(\d+)\s+sequences?,\s+([\d.]+)\s*(GB|MB)', line)
                if cache_match:
                    self.cache_sequences = int(cache_match.group(1))
                    val = float(cache_match.group(2))
                    unit = cache_match.group(3)
                    if unit == "MB":
                        val = val / 1024
                    self.cache_size_gb = val
                    self.memory_gb = val

                mem_match = re.search(r'Peak memory:\s+([\d.]+)\s*(GB|MB)', line, re.IGNORECASE)
                if mem_match:
                    val = float(mem_match.group(1))
                    unit = mem_match.group(2)
                    if unit == "MB":
                        val = val / 1024
                    self.memory_gb = val

                if "Error" in line or "RuntimeError" in line or "OutOfMemory" in line:
                    self.status = "ERROR"
                    with PROXY_STATE.lock:
                        PROXY_STATE.primary_healthy = False

    def _monitor_system(self):
        while self.running:
            try:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                self.used_ram = mem.used / (1024**3)
                self.swap_used = swap.used / (1024**3)
                self.ram_warning = mem.percent > 88
                self._maybe_auto_restart()

                procs = []
                for p in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
                    try:
                        info = p.info
                        if info['memory_percent'] and info['memory_percent'] > 0.05:
                            procs.append({
                                'pid': info['pid'],
                                'name': info['name'][:22],
                                'rss': info['memory_info'].rss / (1024**3),
                                'percent': info['memory_percent']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                procs.sort(key=lambda x: x['rss'], reverse=True)
                self.top_processes = procs[:8]
            except Exception:
                pass
            time.sleep(2)

    def _check_hotkeys(self, live):
        cmd = self.keyboard.get_command()
        if cmd == 'q':
            self._notify("SHUTTING DOWN...")
            self.running = False
        elif cmd == 'r':
            self.restart_server()
        elif cmd == 'f':
            self.restart_fallback_server()
        elif cmd == 'c':
            self.clear_logs()
        elif cmd == 'p':
            self.print_stats()
        elif cmd == 'k':
            self.configure(live)
        elif cmd == 's':
            self.toggle_fallback()
        elif cmd == 'a':
            self.set_routing_mode("AUTO")
        elif cmd == 'd':
            self.set_routing_mode("AUTO" if DYNAMIC_FALLBACK else "DYNAMIC")
        elif cmd == '1':
            self.set_routing_mode("PRIMARY")
        elif cmd == '2':
            self.set_routing_mode("FALLBACK")
        elif cmd == 'x':
            self.toggle_fallback_server()
        elif cmd == 'z':
            self.toggle_primary_server()

    def _idle_indicator(self) -> Text:
        self.idle_frame_idx = (self.idle_frame_idx + 1) % len(IDLE_FRAMES)
        frame = IDLE_FRAMES[self.idle_frame_idx]

        if self.is_processing:
            return Text("⚡ PROCESSING", style="bold yellow")

        idle_secs = int(time.time() - self.last_infer_time) if self.last_infer_time > 0 else 0
        if idle_secs < 5:
            return Text("✓ READY", style="bold green")

        return Text(frame + " IDLE (" + str(idle_secs) + "s)", style="dim cyan")

    def _hotkey_bar(self) -> Text:
        bar = Text()
        hotkeys = [
            ("Q", "Quit", "bold magenta"),
            ("R", "Restart Primary", "bold cyan"),
            ("", "Restart [F]allback", "bold cyan"),
            ("A", "Auto-Routing", "bold green"),
            ("D", "Dynamic Fallback", "bold cyan"),
            ("1", "Immer Primary", "bold green"),
            ("2", "Immer Fallback", "bold yellow"),
            ("X", "Fallback on/off", "bold red"),
            ("Z", "Primary on/off", "bold red"),
            ("C", "Clear", "bold yellow"),
            ("P", "Print", "bold green"),
            ("K", "Konfig", "bold magenta"),
        ]
        for key, label, style in hotkeys:
            if key:
                bar.append("[", style="dim")
                bar.append(key, style=style)
                bar.append("] " + label + "  ", style="dim")
            else:
                bar.append("Restart ", style="dim")
                bar.append("[F]", style=style)
                bar.append("allback  ", style="dim")
        bar.no_wrap = True
        bar.overflow = "ellipsis"
        return bar

    def _notification_bar(self) -> Text:
        if time.time() - self.notif_time < 3 and self.notification:
            return Text("▶ " + self.notification, style="bold yellow")
        return Text("")

    def _header(self) -> Panel:
        wide_logo = (
            "▄▀▀▄▀▀▄ █     █    █      ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀▄      ▄▀▀▀▀▄ ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀ ▄▀▀▀▀▄\n"
            "▀  ▀  ▀ ▀     ▀    ▀      ▀    ▀ ▀    ▀    ▀    ▀    ▀      ▀    ▀ ▀    ▀ ▀    ▀    ▀    ▀     ▀    ▀\n"
            "█  █  █ █     ▄▀▀▀▀▄      █▀▀▀▀█ █    █    █    █    █      █▀▀▀▀▄ █    █ █    █    █    ▄▀▀▀  █▀▀▀▀▄\n"
            "█  ▀  █ █     █    █      █    █ █    █    █    █    █      █    █ █    █ █    █    █    █     █    █\n"
            "▀     ▀  ▀▀▀▀ ▀    ▀      ▀    ▀  ▀▀▀▀     ▀     ▀▀▀▀       ▀    ▀  ▀▀▀▀   ▀▀▀▀     ▀     ▀▀▀▀ ▀    ▀"
        )
        logo = Text()
        if console.width >= 118:
            for index, row in enumerate(wide_logo.splitlines()):
                logo.append(row, style="bold cyan" if index < 2 else "bold magenta")
                if index < 4:
                    logo.append("\n")
        else:
            logo.append("MLX AUTO-ROUTER", style="bold cyan")
        logo.justify = "center"
        uptime = int(time.time() - self.uptime_start)
        uptime_str = str(uptime // 3600).zfill(2) + ":" + str((uptime % 3600) // 60).zfill(2) + ":" + str(uptime % 60).zfill(2)
        subtitle = Text("  Proxy: http://" + HOST + ":" + str(PROXY_PORT) + "/v1  |  Uptime: " + uptime_str, style="dim cyan")
        content = Text.assemble(logo, "\n", subtitle)
        return Panel(content, border_style="cyan", padding=(0, 1))

    def _status_panel(self) -> Panel:
        with self.lock:
            status_color = {"BOOT": "yellow", "ONLINE": "green", "ERROR": "red"}.get(self.status, "white")
            icon = "▶" if self.status == "ONLINE" else "◉"

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold blue", width=16)
            table.add_column()

            table.add_row("Status", Text(icon + " " + self.status, style="bold " + status_color))
            if AUTO_START_FALLBACK:
                fb_status_color = {"BOOT": "yellow", "ONLINE": "green", "ERROR": "red"}.get(self.fallback_status, "white")
                fb_icon = "▶" if self.fallback_status == "ONLINE" else "◉"
                table.add_row("Fallback", Text(fb_icon + " " + self.fallback_status, style="bold " + fb_status_color))
            table.add_row("Activity", self._idle_indicator())
            table.add_row("Last", Text(self.last_activity, style="bold yellow"))
            table.add_row("Requests", Text(str(self.request_count), style="bold magenta"))
            table.add_row("Cache", Text(str(self.cache_sequences) + " seq / " + str(round(self.cache_size_gb, 2)) + "GB", style="bold cyan"))

            return Panel(table, title="[bold blue]● SYSTEM[/bold blue]", border_style="blue", box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _prompt_panel(self) -> Panel:
        with self.lock:
            percent = (self.prompt_current / self.prompt_total * 100) if self.prompt_total > 0 else 0
            eta = self._calculate_eta()

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold yellow", width=14)
            table.add_column()

            table.add_row("Tokens", Text(str(self.prompt_current) + " / " + str(self.prompt_total), style="bold white"))
            table.add_row("Gen Tokens", Text(str(self.gen_tokens), style="bold white"))
            table.add_row("Prefill", speedometer(self.prefill_speed))
            table.add_row("Decode", speedometer(self.decode_speed))
            table.add_row("Progress", dither_bar(percent))
            table.add_row("Percent", Text(str(round(percent, 1)) + "%", style="bold green" if percent >= 100 else "bold yellow"))
            table.add_row("ETA", Text(eta, style="bold magenta"))

            if self.inference_phase == "PREFILL":
                title = "[bold yellow]⚡ PREFILL[/bold yellow]"
            elif self.inference_phase == "DECODE":
                title = "[bold cyan]⚡ DECODE / STREAMING[/bold cyan]"
            else:
                title = "[bold dim]⚡ ENGINE IDLE[/bold dim]"
            border = "yellow" if self.is_processing else "dim"

            return Panel(table, title=title, border_style=border, box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _viz_panel(self) -> Panel:
        with self.lock:
            self._sample_histograms()
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(ratio=1)
            effect_width = max(48, console.width - 8)
            effect_frame = Text(" " * effect_width)
            effect_class = None
            prompt_complete = self.prompt_total > 0 and self.prompt_current >= self.prompt_total
            decode_active = self.gen_tokens > 0 or prompt_complete or self.inference_phase == "DECODE"
            if self.is_processing and decode_active:
                effect_class = Matrix
            elif self.is_processing and not decode_active and self.inference_phase == "PREFILL":
                effect_class = SynthGrid
            if effect_class is not None:
                if self.effect_iterator is None or self.effect_iterator_phase != self.inference_phase:
                    phase_text = "DECODING" if decode_active else "PREFILLING"
                    banner = [row.plain for row in ascii_banner(phase_text, "white")]
                    banner = [row.center(effect_width) for row in banner]
                    effect = effect_class("\n".join(banner))
                    effect.terminal_config.canvas_width = effect_width
                    effect.terminal_config.canvas_height = EFFECT_ROWS
                    effect.terminal_config.anchor_text = "w"
                    effect.terminal_config.ignore_terminal_dimensions = True
                    self.effect_iterator = iter(effect)
                    self.effect_iterator_phase = self.inference_phase
                try:
                    effect_frame = Text.from_ansi(next(self.effect_iterator))
                except StopIteration:
                    self.effect_iterator = None
                    effect_frame = Text(" " * effect_width)
            else:
                self.effect_iterator = None
                self.effect_iterator_phase = None
            table.add_row(effect_frame)
            table.add_row(sparkline(self.speed_history, width=effect_width, max_val=MAX_SPEED))
            table.add_row(sparkline(self.token_history, width=effect_width, max_val=40000))

            return Panel(table, border_style="magenta", box=box.ROUNDED, height=EFFECT_PANEL_HEIGHT)

    def _router_panel(self) -> Panel:
        with PROXY_STATE.lock:
            active = PROXY_STATE.active_backend
            reason = PROXY_STATE.routing_reason
            tokens = PROXY_STATE.last_request_tokens
            fallback_ok = PROXY_STATE.fallback_available
            routed_27b = PROXY_STATE.total_routed_27b
            routed_fb = PROXY_STATE.total_routed_fallback
            routing_mode = PROXY_STATE.routing_mode

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold green", width=16)
            table.add_column()

            if active == "27B":
                model_text = Text("▶ PRIMARY", style="bold green")
            else:
                model_text = Text("▶ FALLBACK", style="bold yellow")

            if self.force_fallback:
                model_text = Text("▶ FALLBACK (FORCED)", style="bold red")

            table.add_row("Active Model", model_text)
            table.add_row("Mode", Text(routing_mode, style="bold cyan"))
            if routing_mode == "DYNAMIC":
                table.add_row("Dynamic Limit", Text(str(dynamic_primary_prompt_limit()) + " tokens", style="bold cyan"))
            table.add_row("Reason", Text(reason, style="bold white"))
            table.add_row("Last Prompt", Text(str(tokens) + " tokens", style="bold cyan"))
            table.add_row("Fallback", Text("✓ Online" if fallback_ok else "✗ Offline",
                          style="bold green" if fallback_ok else "bold red"))
            if FALLBACK_TYPE == "mlx_lm":
                fb_model = os.path.basename(FALLBACK_MODEL_PATH.rstrip("/").rstrip("\\"))
                table.add_row("Fallback Model", Text(fb_model, style="bold cyan"))
            table.add_row("Stats", Text("27B:" + str(routed_27b) + " | FB:" + str(routed_fb), style="dim"))

            title = "[bold green]◉ ROUTER[/bold green]"
            border = "green" if active == "27B" else "yellow"

            return Panel(table, title=title, border_style=border, box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _ram_panel(self) -> Panel:
        with self.lock:
            table = Table(show_header=False, box=None, padding=(0, 0))
            table.add_column(style="bold red", width=12)
            table.add_column()

            ram_percent = (self.used_ram / self.total_ram) * 100 if self.total_ram else 0
            ram_color = "red" if ram_percent > 88 else "yellow" if ram_percent > 75 else "green"
            table.add_row("RAM", ram_blocks(self.used_ram, self.total_ram))
            table.add_row("Usage", Text(str(round(ram_percent, 1)) + "%", style="bold " + ram_color))

            swap_critical = ROUTE_ON_SWAP and self.swap_used > SWAP_LIMIT_27B
            if swap_critical:
                table.add_row("SWAP", Text("🔥 " + str(round(self.swap_used, 1)) + " GB SWAP!", style="bold red blink"))
            else:
                table.add_row("SWAP", Text(str(round(self.swap_used, 1)) + " GB", style="dim"))

            ram_critical = ram_percent > 88
            if ram_critical:
                table.add_row("", Text("⚠️  RAM CRITICAL", style="bold red blink"))

            table.add_row("", Text("─" * 40, style="dim"))
            table.add_row("Processes", Text("PID      Name                 RSS", style="bold underline"))

            for proc in self.top_processes[:10]:
                name = proc['name'][:20]
                rss = proc['rss']
                pid = proc['pid']
                is_mlx = "mlx" in name.lower() or "python" in name.lower() or pid == (self.server_process.pid if self.server_process else -1)
                style = "bold yellow" if is_mlx else "white"
                table.add_row("", Text(str(pid).ljust(8) + " " + name.ljust(20) + " " + str(round(rss, 1)) + "G", style=style))

            return Panel(table, title="[bold red]◉ MEMORY[/bold red]",
                        border_style="red", box=box.ROUNDED,
                        height=TOP_PANEL_HEIGHT)

    def _logs_panel(self) -> Panel:
        with self.lock:
            lines = list(self.log_buffer)
            colored_lines = []
            for line in lines[-18:]:
                if "Prompt processing progress" in line:
                    colored_lines.append(Text(line, style="bold green", no_wrap=True, overflow="ellipsis"))
                elif "Prompt Cache" in line:
                    colored_lines.append(Text(line, style="blue", no_wrap=True, overflow="ellipsis"))
                elif "Generation" in line or "Prefill" in line or "Decode" in line:
                    colored_lines.append(Text(line, style="bold cyan", no_wrap=True, overflow="ellipsis"))
                elif "Error" in line or "RuntimeError" in line:
                    colored_lines.append(Text(line, style="bold red", no_wrap=True, overflow="ellipsis"))
                elif "POST" in line or "GET" in line:
                    colored_lines.append(Text(line, style="dim", no_wrap=True, overflow="ellipsis"))
                else:
                    colored_lines.append(Text(line, style="white", no_wrap=True, overflow="ellipsis"))

            content = Text("\n").join(colored_lines) if colored_lines else Text("Waiting for logs...", style="dim")

            return Panel(content, title="[bold green]◉ LOG STREAM[/bold green]",
                        border_style="green", box=box.ROUNDED, height=LOG_PANEL_HEIGHT)

    def generate_display(self):
        top = Table(show_header=False, box=None, expand=True, padding=(0, 0))
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_row(self._status_panel(), self._prompt_panel(), self._router_panel(), self._ram_panel())

        full = Table(show_header=False, box=None, expand=True, padding=(0, 0))
        full.add_column()
        full.add_row(self._header())

        notif = self._notification_bar()
        if notif.plain:
            full.add_row(Align.center(notif))

        full.add_row(top)
        full.add_row(self._viz_panel())
        full.add_row(self._logs_panel())
        full.add_row(Align.center(self._hotkey_bar()))

        return full

    def run(self):
        console.clear()
        console.print("[bold cyan]◈ Initializing MLX Auto-Router...[/bold cyan]")
        console.print("[dim]Primary: 27B on port " + str(PRIMARY_PORT) + "[/dim]")
        console.print("[dim]Proxy: http://" + HOST + ":" + str(PROXY_PORT) + "/v1[/dim]")
        if AUTO_START_FALLBACK:
            console.print("[dim]Fallback: " + FALLBACK_TYPE + " '" + FALLBACK_MODEL_PATH + "' on port " + str(FALLBACK_PORT) + "[/dim]")
        else:
            console.print("[dim]Fallback: " + FALLBACK_TYPE + " on " + FALLBACK_HOST + ":" + str(FALLBACK_PORT) + " (manual)[/dim]")
        console.print()
        console.print("[bold yellow]IMPORTANT:[/bold yellow] Set your API client to http://<host>:" + str(PROXY_PORT) + "/v1")
        console.print()

        self.keyboard.start()
        self.start_server()
        if AUTO_START_FALLBACK:
            time.sleep(1)
            self.start_fallback_server()
            time.sleep(2)
        else:
            time.sleep(2)
        self.start_proxy()

        try:
            with Live(self.generate_display(), refresh_per_second=10, screen=True) as live:
                while self.running:
                    self._check_hotkeys(live)
                    live.update(self.generate_display())
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        self.keyboard.stop()
        if self.proxy_server:
            self.proxy_server.shutdown()
        console.print("\n[bold red]◈ Shutting down...[/bold red]")
        if self.primary_server:
            self.primary_server.stop()
        if self.fallback_server:
            console.print("[bold red]◈ Stopping fallback server...[/bold red]")
            self.fallback_server.stop()
        console.print("[bold green]◈ Done.[/bold green]")


def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler("logs/router.log"),
        ],
        force=True,
    )
    dashboard = MLXDashboard()
    dashboard.run()


if __name__ == "__main__":
    raise SystemExit(main())
#!/usr/bin/env python3
"""
MLX Server Dashboard v11 - Auto-Router Edition
Routes between 27B (quality) and fallback (speed) based on prompt length.
Proxy runs on port 8082. OpenAI-compatible clients should connect to http://host:8082/v1
"""

import subprocess
import sys
import os
import re
import time
import threading
import logging
import tty
import termios
import select
import json
import http.client
import socketserver
from pathlib import Path
from datetime import datetime
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psutil

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import box
from rich.align import Align

try:
    from terminaltexteffects.effects.effect_matrix import Matrix
    from terminaltexteffects.effects.effect_synthgrid import SynthGrid
except ImportError:
    Matrix = None
    SynthGrid = None

console = Console()
LOGGER = logging.getLogger("mlx_router")
AUTO_FALLBACK_HEADROOM = 0.85

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

def discover_model_paths():
    root = Path(os.environ.get("MODEL_ROOT", "./models"))
    candidates = []
    for config_path in root.glob("**/config.json"):
        model_dir = config_path.parent
        weight_size = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
        if weight_size:
            candidates.append((weight_size, str(model_dir)))
    candidates.sort(reverse=True)
    if not candidates:
        raise RuntimeError("Keine MLX-Modelle unter " + str(root) + " gefunden.")
    primary = os.environ.get("PRIMARY_MODEL_PATH", candidates[0][1])
    fallback = os.environ.get("FALLBACK_MODEL_PATH")
    if not fallback:
        fallback = next((path for _, path in reversed(candidates) if path != primary), primary)
    return primary, fallback


def scan_model_spec(model_path):
    path = Path(model_path)
    config_path = path / "config.json"
    config = json.loads(config_path.read_text())
    text_config = config.get("text_config", config)
    weight_bytes = sum(item.stat().st_size for item in path.glob("*.safetensors"))
    return {
        "path": str(path),
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "weight_bytes": weight_bytes,
        "weight_gb": round(weight_bytes / 1024 ** 3, 2),
        "quantization": config.get("quantization", config.get("quantization_config", {})),
        "native_context_tokens": text_config.get("max_position_embeddings"),
        "layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "attention_layers": sum(1 for item in text_config.get("layer_types", []) if item == "full_attention"),
        "kv_heads": text_config.get("num_key_value_heads", text_config.get("num_attention_heads")),
        "head_dim": text_config.get("head_dim"),
        "dtype": text_config.get("dtype", "float16"),
    }


def suggest_model_limits(primary_spec, fallback_spec):
    total_ram_gb = psutil.virtual_memory().total / 1024 ** 3
    system_reserve_gb = max(4.0, total_ram_gb * 0.15)
    loaded_model_gb = primary_spec["weight_gb"]

    def context_for(spec):
        native = int(spec.get("native_context_tokens") or 8192)
        available_gb = max(1.0, total_ram_gb - loaded_model_gb - system_reserve_gb)
        attention_layers = max(1, int(spec.get("attention_layers") or spec.get("layers") or 1))
        kv_heads = max(1, int(spec.get("kv_heads") or 1))
        head_dim = max(1, int(spec.get("head_dim") or 128))
        dtype_bytes = 4 if "32" in str(spec.get("dtype", "")) else 2
        kv_bytes_per_token = 2 * attention_layers * kv_heads * head_dim * dtype_bytes
        cache_budget_bytes = available_gb * 1024 ** 3 * 0.20
        memory_limited = int(cache_budget_bytes / kv_bytes_per_token)
        suggested = 1
        while suggested * 2 <= memory_limited:
            suggested *= 2
        return min(native, max(2048, suggested)), available_gb, kv_bytes_per_token

    context, available_gb, kv_bytes_per_token = context_for(primary_spec)
    reserve = max(256, min(1024, context // 8))
    max_generation = max(512, min(8192, context // 4))
    primary_memory_limit = min(total_ram_gb * 0.80, primary_spec["weight_gb"] + available_gb * 0.35)
    fallback_available_gb = max(1.0, total_ram_gb - fallback_spec["weight_gb"] - system_reserve_gb)
    fallback_limit = min(total_ram_gb * 0.80, fallback_spec["weight_gb"] + fallback_available_gb * 0.35)
    return {
        "max_context_tokens": context,
        "context_safety_margin": reserve,
        "max_generation_tokens": max_generation,
        "primary_prompt_limit": max(1024, int((context - reserve - max_generation) * AUTO_FALLBACK_HEADROOM)),
        "primary_memory_limit_gb": round(primary_memory_limit, 2),
        "fallback_memory_limit_gb": round(fallback_limit, 2),
        "primary_prompt_cache_bytes": int(available_gb * 1024 ** 3 * 0.20),
        "total_ram_gb": round(total_ram_gb, 2),
        "system_reserve_gb": round(system_reserve_gb, 2),
        "loaded_model_gb": round(loaded_model_gb, 2),
        "available_for_cache_gb": round(available_gb, 2),
        "primary_kv_bytes_per_token": kv_bytes_per_token,
    }


MODEL_PATH, FALLBACK_MODEL_PATH = discover_model_paths()
MODEL_SPECS = {
    "primary": scan_model_spec(MODEL_PATH),
    "fallback": scan_model_spec(FALLBACK_MODEL_PATH),
}
MODEL_LIMIT_SUGGESTIONS = suggest_model_limits(MODEL_SPECS["primary"], MODEL_SPECS["fallback"])
Path("logs").mkdir(exist_ok=True)
Path("logs/model_specs.json").write_text(json.dumps({
    "scanned_at": datetime.now().isoformat(),
    "models": MODEL_SPECS,
    "suggestions": MODEL_LIMIT_SUGGESTIONS,
}, indent=2))
PRIMARY_PORT = 8080
PROXY_PORT = 8082
HOST = "0.0.0.0"
PREFILL_STEP_SIZE = 4096
PROMPT_CONCURRENCY = 1
DECODE_CONCURRENCY = 1
PROMPT_CACHE_SIZE = 1
PROMPT_CACHE_BYTES = MODEL_LIMIT_SUGGESTIONS["primary_prompt_cache_bytes"]
SERVER_MAX_TOKENS = MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"]
PRIMARY_MEMORY_LIMIT = int(MODEL_LIMIT_SUGGESTIONS["primary_memory_limit_gb"] * 1024 ** 3)

# Fallback configuration - CHANGE THIS TO YOUR SETUP
FALLBACK_TYPE = "mlx_lm"    # "ollama" or "mlx_lm"
FALLBACK_HOST = "localhost"
FALLBACK_PORT = 8081        # Ollama default=11434, second mlx_lm=8081
FALLBACK_MEMORY_LIMIT = int(MODEL_LIMIT_SUGGESTIONS["fallback_memory_limit_gb"] * 1024 ** 3)
AUTO_START_FALLBACK = False
LAZY_FALLBACK = True

MAX_CONTEXT_TOKENS = MODEL_LIMIT_SUGGESTIONS["max_context_tokens"]
CONTEXT_SAFETY_MARGIN = MODEL_LIMIT_SUGGESTIONS["context_safety_margin"]
TOP_PANEL_HEIGHT = 18
LOG_PANEL_HEIGHT = 22
EFFECT_PANEL_HEIGHT = 13
EFFECT_ROWS = 9
TOKEN_LIMIT_27B = MODEL_LIMIT_SUGGESTIONS["primary_prompt_limit"]
SWAP_LIMIT_27B = 4.0
ROUTE_ON_SWAP = True
AUTO_RESTART_27B = True
RESTART_AFTER_REQUESTS_27B = 20
RESTART_ON_SWAP_GB_27B = 8.0
RESTART_COOLDOWN_SECONDS = 60

# Thinking control for primary model. The template has no separate thinking
# budget, so the total generation limit provides a practical upper bound.
THINKING_ENABLED_27B = True
REASONING_EFFORT_27B = "low"
MAX_GENERATION_TOKENS_27B = MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"]
MIN_GENERATION_TOKENS_27B = max(256, min(2048, MAX_GENERATION_TOKENS_27B // 2))
DYNAMIC_FALLBACK = False
CONFIG_FILE = Path("router_config.json")


def load_saved_config():
    if not CONFIG_FILE.exists():
        return
    try:
        saved = json.loads(CONFIG_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        LOGGER.warning("Could not read %s", CONFIG_FILE)
        return
    for name in (
        "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT",
        "MAX_CONTEXT_TOKENS", "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B",
        "PREFILL_STEP_SIZE", "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY",
        "MAX_GENERATION_TOKENS_27B", "THINKING_ENABLED_27B",
        "REASONING_EFFORT_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B",
    ):
        if name in saved:
            globals()[name] = saved[name]


load_saved_config()

if Path(MODEL_PATH).exists() and Path(FALLBACK_MODEL_PATH).exists():
    MODEL_SPECS = {
        "primary": scan_model_spec(MODEL_PATH),
        "fallback": scan_model_spec(FALLBACK_MODEL_PATH),
    }
    MODEL_LIMIT_SUGGESTIONS = suggest_model_limits(MODEL_SPECS["primary"], MODEL_SPECS["fallback"])


def save_config():
    values = {}
    for name in (
        "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT",
        "MAX_CONTEXT_TOKENS", "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B",
        "PREFILL_STEP_SIZE", "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY",
        "MAX_GENERATION_TOKENS_27B", "THINKING_ENABLED_27B",
        "REASONING_EFFORT_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B",
    ):
        values[name] = globals()[name]
    CONFIG_FILE.write_text(json.dumps(values, indent=2) + "\n")


def dynamic_primary_prompt_limit():
    spec = MODEL_SPECS["primary"]
    native = int(spec.get("native_context_tokens") or MAX_CONTEXT_TOKENS)
    available_gb = psutil.virtual_memory().available / 1024 ** 3
    attention_layers = max(1, int(spec.get("attention_layers") or spec.get("layers") or 1))
    kv_heads = max(1, int(spec.get("kv_heads") or 1))
    head_dim = max(1, int(spec.get("head_dim") or 128))
    dtype_bytes = 4 if "32" in str(spec.get("dtype", "")) else 2
    kv_bytes_per_token = 2 * attention_layers * kv_heads * head_dim * dtype_bytes
    cache_budget_bytes = max(0.5, available_gb * 0.20) * 1024 ** 3
    memory_limited = max(2048, int(cache_budget_bytes / kv_bytes_per_token))
    context_limit = min(native, MAX_CONTEXT_TOKENS, memory_limited)
    return max(1024, int((context_limit - CONTEXT_SAFETY_MARGIN - MAX_GENERATION_TOKENS_27B) * AUTO_FALLBACK_HEADROOM))


# ═══════════════════════════════════════════════════════════════
#  KEYBOARD
# ═══════════════════════════════════════════════════════════════

class NonBlockingInput:
    def __init__(self):
        self.commands = deque()
        self.running = True
        self.old_settings = None

    def start(self):
        try:
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:
            self.old_settings = None
        threading.Thread(target=self._read_keys, daemon=True).start()

    def _read_keys(self):
        while self.running:
            try:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    char = sys.stdin.read(1)
                    if char:
                        self.commands.append(char.lower())
            except Exception:
                time.sleep(0.1)

    def get_command(self):
        if self.commands:
            return self.commands.popleft()
        return None

    def stop(self):
        self.running = False
        if self.old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

# ═══════════════════════════════════════════════════════════════
#  FX
# ═══════════════════════════════════════════════════════════════

IDLE_FRAMES = ["◐", "◓", "◑", "◒"]
MAX_SPEED = 1500.0

def dither_bar(percent: float, width: int = 40) -> Text:
    filled = width * (percent / 100)
    full_blocks = int(filled)
    remainder = filled - full_blocks
    bar = Text()
    for i in range(full_blocks):
        hue = 120 + int(60 * (i / width))
        bar.append("█", style=f"bold rgb(0,{hue},200)")
    if full_blocks < width:
        if remainder > 0.75:
            char = "▓"
        elif remainder > 0.5:
            char = "▒"
        elif remainder > 0.25:
            char = "░"
        else:
            char = "▒"
        hue = 120 + int(60 * full_blocks / width)
        bar.append(char, style=f"rgb(0,{hue},180)")
    for i in range(full_blocks + 1, width):
        bar.append("░", style="dim rgb(30,30,50)")
    return bar

def speedometer(speed: float, max_speed: float = MAX_SPEED, width: int = 30) -> Text:
    ratio = min(speed / max_speed, 1.0)
    filled = int(width * ratio)
    result = Text()
    for i in range(width):
        if i < filled:
            if ratio < 0.33:
                color = "rgb(50,255,150)"
            elif ratio < 0.66:
                color = "rgb(255,220,50)"
            else:
                color = "rgb(255,80,50)"
            result.append("━", style=f"bold {color}")
        else:
            result.append("─", style="dim rgb(30,30,50)")
    result.append(f"  {speed:.0f} t/s", style="bold cyan")
    return result

def ram_blocks(used_gb: float, total_gb: float, width: int = 32) -> Text:
    ratio = min(used_gb / total_gb, 1.0)
    filled = int(width * ratio)
    result = Text()
    for i in range(width):
        if i < filled:
            if ratio > 0.92:
                color = "rgb(255,30,30)"
            elif ratio > 0.75:
                color = "rgb(255,180,30)"
            else:
                color = "rgb(50,255,150)"
            result.append("▪", style=color)
        else:
            result.append("▫", style="dim rgb(30,30,50)")
    result.append(f"  {used_gb:.1f}/{total_gb:.0f}GB", style="bold white")
    return result

def sparkline(data: deque, width: int = 40, max_val: float = MAX_SPEED) -> Text:
    if not data or len(data) < 2:
        return Text("░" * width, style="dim rgb(30,30,50)")
    chars = "▁▂▃▄▅▆▇█"
    values = list(data)[-width:]
    min_v = min(values)
    ceiling = max(max(values), max_val * 0.3)
    range_v = max(ceiling - min_v, 0.001)
    result = Text()
    for v in values:
        idx = min(int(((v - min_v) / range_v) * (len(chars) - 1)), len(chars) - 1)
        result.append(chars[idx], style="rgb(100,255,150)")
    while len(result) < width:
        result.append("░", style="dim")
    return result


def matrix_strip(width: int = 48) -> Text:
    chars = "01{}[]<>/\\*+-"
    frame = int(time.time() * 8)
    result = Text()
    for index in range(width):
        value = (index * 37 + frame * (index % 5 + 3)) % 101
        if value < 18:
            char = chars[(index + frame) % len(chars)]
            result.append(char, style="bold bright_green")
        elif value < 42:
            char = chars[(index * 3 + frame) % len(chars)]
            result.append(char, style="green")
        else:
            result.append(" ")
    return result


def synthwave_grid(width: int = 48) -> Text:
    frame = int(time.time() * 6) % 12
    result = Text()
    for index in range(width):
        if (index + frame) % 7 == 0:
            result.append("/", style="bold magenta")
        elif (index + frame) % 5 == 0:
            result.append("_", style="bright_blue")
        else:
            result.append(".", style="rgb(90,30,120)")
    return result


ASCII_FONT = {
    "A": [" ███ ", "█   █", "█████", "█   █", "█   █"],
    "C": [" ████", "█    ", "█    ", "█    ", " ████"],
    "D": ["████ ", "█   █", "█   █", "█   █", "████ "],
    "E": ["█████", "█    ", "████ ", "█    ", "█████"],
    "F": ["█████", "█    ", "████ ", "█    ", "█    "],
    "G": [" ████", "█    ", "█ ███", "█   █", " ███ "],
    "I": ["█████", "  █  ", "  █  ", "  █  ", "█████"],
    "L": ["█    ", "█    ", "█    ", "█    ", "█████"],
    "N": ["█   █", "██  █", "█ █ █", "█  ██", "█   █"],
    "O": [" ███ ", "█   █", "█   █", "█   █", " ███ "],
    "P": ["████ ", "█   █", "████ ", "█    ", "█    "],
    "R": ["████ ", "█   █", "████ ", "█  █ ", "█   █"],
    "T": ["█████", "  █  ", "  █  ", "  █  ", "  █  "],
    " " : ["     ", "     ", "     ", "     ", "     "],
}


def ascii_banner(text: str, color: str) -> list[Text]:
    rows = []
    for row in range(5):
        line = Text()
        for char in text:
            line.append(ASCII_FONT.get(char, ASCII_FONT[" "])[row], style=color)
            line.append(" ")
        rows.append(line)
    return rows

# ═══════════════════════════════════════════════════════════════
#  PROXY
# ═══════════════════════════════════════════════════════════════

class ProxyState:
    def __init__(self):
        self.active_backend = "27B"
        self.routing_mode = "AUTO"
        self.routing_reason = "initial"
        self.last_request_tokens = 0
        self.last_request_time = ""
        self.total_routed_27b = 0
        self.total_routed_fallback = 0
        self.lock = threading.Lock()
        self.primary_healthy = True
        self.fallback_available = False

PROXY_STATE = ProxyState()

def estimate_tokens_from_body(body: bytes) -> int:
    try:
        data = json.loads(body)
        messages = data.get("messages", [])
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and "text" in part:
                        total_chars += len(part["text"])
        return total_chars // 4
    except Exception:
        return 0


def inject_chat_template_kwargs(body: bytes, kwargs: dict) -> bytes:
    """Merge chat_template_kwargs into a request body."""
    try:
        data = json.loads(body)
    except Exception:
        return body

    existing = data.get("chat_template_kwargs", {}) or {}
    merged = {**existing, **kwargs}
    data["chat_template_kwargs"] = merged
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def limit_generation_to_context(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    prompt_tokens = estimate_tokens_from_body(body)
    available_tokens = max(
        1,
        MAX_CONTEXT_TOKENS - prompt_tokens - CONTEXT_SAFETY_MARGIN,
    )
    for field in ("max_tokens", "max_completion_tokens"):
        requested = data.get(field)
        if isinstance(requested, int) and requested > available_tokens:
            data[field] = available_tokens
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def ensure_generation_budget_for_primary(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    fields = ("max_tokens", "max_completion_tokens")
    found_limit = False
    for field in fields:
        requested = data.get(field)
        if isinstance(requested, int):
            data[field] = max(requested, MIN_GENERATION_TOKENS_27B)
            found_limit = True
    if not found_limit:
        data["max_tokens"] = MIN_GENERATION_TOKENS_27B
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def apply_primary_sampling_policy(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body

    data.setdefault("repetition_penalty", 1.12)
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def limit_generation_for_primary(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    for field in ("max_tokens", "max_completion_tokens"):
        requested = data.get(field)
        if isinstance(requested, int) and requested > MAX_GENERATION_TOKENS_27B:
            data[field] = MAX_GENERATION_TOKENS_27B
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def normalize_backend_model(body: bytes, target_port: int) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    if target_port == PRIMARY_PORT:
        data["model"] = MODEL_PATH
    elif target_port == FALLBACK_PORT:
        data["model"] = FALLBACK_MODEL_PATH
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def normalize_model_list(body: bytes, target_port: int) -> bytes:
    try:
        data = json.loads(body)
    except Exception:
        return body
    role = "primary" if target_port == PRIMARY_PORT else "fallback"
    models = data.get("data", [])
    template = dict(models[0]) if models else {"object": "model"}
    template["id"] = role
    data["data"] = [template]
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def model_list_response():
    return json.dumps({
        "object": "list",
        "data": [{
            "id": "primary",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local",
        }],
    }).encode("utf-8")


def health_response():
    return json.dumps({"status": "ok", "service": "mlx-router"}).encode("utf-8")

def check_backend_available(port: int, path: str) -> bool:
    try:
        conn = http.client.HTTPConnection("localhost", port, timeout=2)
        conn.request("GET", path)
        response = conn.getresponse()
        conn.close()
        return response.status == 200
    except Exception:
        return False

def check_fallback_available() -> bool:
    try:
        conn = http.client.HTTPConnection(FALLBACK_HOST, FALLBACK_PORT, timeout=2)
        path = "/v1/models" if FALLBACK_TYPE in ("ollama", "mlx_lm") else "/"
        conn.request("GET", path)
        response = conn.getresponse()
        conn.close()
        return response.status == 200
    except Exception:
        return False

class ProxyHandler(BaseHTTPRequestHandler):
    dashboard = None

    def _record_decode_result(self, response_body: bytes, started_at: float):
        if not self.dashboard or not response_body:
            return
        try:
            payload = json.loads(response_body)
            generated_tokens = int(payload.get("usage", {}).get("completion_tokens", 0) or 0)
        except (ValueError, TypeError, json.JSONDecodeError):
            return
        if generated_tokens <= 0:
            return
        elapsed = max(time.perf_counter() - started_at, 0.001)
        with self.dashboard.lock:
            self.dashboard.inference_phase = "DECODE"
            self.dashboard.gen_tokens = generated_tokens
            self.dashboard.decode_speed = generated_tokens / elapsed
            self.dashboard.last_infer_time = time.time()
        LOGGER.info(
            "proxy decode completed tokens=%d elapsed=%.3fs rate=%.1f tok/s",
            generated_tokens,
            elapsed,
            generated_tokens / elapsed,
        )

    def log_message(self, format, *args):
        pass

    def _route_request(self, body: bytes) -> tuple[int, str]:
        estimated_tokens = estimate_tokens_from_body(body)
        dynamic_limit = dynamic_primary_prompt_limit() if DYNAMIC_FALLBACK else TOKEN_LIMIT_27B

        with PROXY_STATE.lock:
            PROXY_STATE.last_request_tokens = estimated_tokens
            PROXY_STATE.last_request_time = datetime.now().strftime("%H:%M:%S")

            if self.command == "GET" and self.path.startswith("/v1/models"):
                preferred_port = FALLBACK_PORT if PROXY_STATE.routing_mode == "FALLBACK" else PRIMARY_PORT
                if check_backend_available(preferred_port, "/v1/models"):
                    PROXY_STATE.active_backend = "FALLBACK" if preferred_port == FALLBACK_PORT else "27B"
                    PROXY_STATE.routing_reason = "model list"
                    return preferred_port, PROXY_STATE.routing_reason
                alternate_port = FALLBACK_PORT if preferred_port == PRIMARY_PORT else PRIMARY_PORT
                if check_backend_available(alternate_port, "/v1/models"):
                    PROXY_STATE.active_backend = "FALLBACK" if alternate_port == FALLBACK_PORT else "27B"
                    PROXY_STATE.routing_reason = "preferred backend unavailable"
                    return alternate_port, PROXY_STATE.routing_reason
                PROXY_STATE.routing_reason = "no backend available"
                raise ConnectionError("No model backend available")

            fallback_ok = check_fallback_available()
            fallback_configured = fallback_ok or (LAZY_FALLBACK and self.dashboard is not None)
            PROXY_STATE.fallback_available = fallback_ok

            if PROXY_STATE.routing_mode == "PRIMARY":
                if check_backend_available(PRIMARY_PORT, "/v1/models"):
                    PROXY_STATE.active_backend = "27B"
                    PROXY_STATE.routing_reason = "27B direct"
                    PROXY_STATE.total_routed_27b += 1
                    return PRIMARY_PORT, PROXY_STATE.routing_reason
            elif PROXY_STATE.routing_mode == "FALLBACK":
                if fallback_ok:
                    PROXY_STATE.active_backend = "FALLBACK"
                    PROXY_STATE.routing_reason = "9B direct"
                    PROXY_STATE.total_routed_fallback += 1
                    return FALLBACK_PORT, PROXY_STATE.routing_reason

            if not fallback_ok:
                PROXY_STATE.active_backend = "27B"
                PROXY_STATE.routing_reason = str(estimated_tokens) + "t (no fallback)"
                PROXY_STATE.total_routed_27b += 1
                return PRIMARY_PORT, PROXY_STATE.routing_reason

            swap = psutil.swap_memory().used / (1024**3)
            mem = psutil.virtual_memory()
            ram_pct = mem.percent

            if estimated_tokens > dynamic_limit:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = str(estimated_tokens) + "t > " + str(dynamic_limit) + "t limit"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if ROUTE_ON_SWAP and swap > SWAP_LIMIT_27B:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "SWAP " + str(round(swap, 1)) + "GB > " + str(SWAP_LIMIT_27B) + "GB"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if ram_pct > 90 and estimated_tokens > 1000:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "RAM " + str(int(ram_pct)) + "% + " + str(estimated_tokens) + "t"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            if not PROXY_STATE.primary_healthy:
                PROXY_STATE.active_backend = "FALLBACK"
                PROXY_STATE.routing_reason = "27B crashed/unhealthy"
                PROXY_STATE.total_routed_fallback += 1
                return FALLBACK_PORT, PROXY_STATE.routing_reason

            PROXY_STATE.active_backend = "27B"
            PROXY_STATE.routing_reason = str(estimated_tokens) + "t < " + str(TOKEN_LIMIT_27B) + "t, healthy"
            PROXY_STATE.total_routed_27b += 1
            return PRIMARY_PORT, PROXY_STATE.routing_reason

    def _prepare_body(self, target_port: int, body: bytes) -> bytes:
        if self.path != "/v1/chat/completions":
            return body
        body = normalize_backend_model(body, target_port)
        if target_port == PRIMARY_PORT:
            body = ensure_generation_budget_for_primary(body)
            body = limit_generation_to_context(body)
            body = limit_generation_for_primary(body)
            body = apply_primary_sampling_policy(body)
            body = inject_chat_template_kwargs(
                body,
                {
                    "enable_thinking": THINKING_ENABLED_27B,
                    "reasoning_effort": REASONING_EFFORT_27B,
                },
            )
        elif target_port == FALLBACK_PORT:
            body = inject_chat_template_kwargs(body, {"enable_thinking": False})
        return body

    def _forward(self, method: str, body: bytes = None):
        if method in ("GET", "HEAD") and self.path in ("/", "/health", "/v1/health"):
            health_body = health_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(health_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            if method == "GET":
                self.wfile.write(health_body)
                self.wfile.flush()
            return
        if method in ("GET", "HEAD") and self.path in ("/models", "/v1/models"):
            model_body = model_list_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(model_body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "*")
            self.end_headers()
            if method == "GET":
                self.wfile.write(model_body)
                self.wfile.flush()
            return
        try:
            target_port, reason = self._route_request(body or b"")
        except ConnectionError as error:
            LOGGER.error("no backend available method=%s path=%s error=%s", method, self.path, error)
            self.send_error(503, "No model backend available")
            if request_lock:
                request_lock.release()
            return
        if self.command == "POST" and self.path == "/v1/chat/completions" and self.dashboard:
            if not check_backend_available(target_port, "/v1/models"):
                LOGGER.error("selected backend unavailable target=%s path=%s", target_port, self.path)
                self.send_error(503, "Selected model backend is unavailable")
                if request_lock:
                    request_lock.release()
                return
            with self.dashboard.lock:
                self.dashboard.inference_phase = "PREFILL"
                self.dashboard.is_processing = True
                self.dashboard.prompt_current = 0
                self.dashboard.prompt_total = estimate_tokens_from_body(body or b"")
                self.dashboard.gen_tokens = 0
                self.dashboard.last_activity = datetime.now().strftime("%H:%M:%S")
                self.dashboard.last_infer_time = time.time()
        target_host = "localhost"
        prepared_body = self._prepare_body(target_port, body or b"")
        LOGGER.info(
            "route method=%s path=%s target=%s reason=%s prompt_estimate=%d "
            "body_bytes=%d prepared_bytes=%d",
            method,
            self.path,
            target_port,
            reason,
            estimate_tokens_from_body(body or b""),
            len(body or b""),
            len(prepared_body),
        )

        try:
            started_at = time.perf_counter()
            decode_started_at = None
            streamed_tokens = 0
            stream_buffer = b""
            response_body = bytearray()
            conn = http.client.HTTPConnection(target_host, target_port, timeout=300)
            headers = {k: v for k, v in self.headers.items()}
            headers['Content-Length'] = str(len(prepared_body))
            path = self.path

            conn.request(method, path, body=prepared_body, headers=headers)
            response = conn.getresponse()

            if method == "GET" and path.startswith("/v1/models"):
                model_body = normalize_model_list(response.read(), target_port)
                self.send_response(response.status)
                for header, value in response.getheaders():
                    if header.lower() not in ("transfer-encoding", "content-length"):
                        self.send_header(header, value)
                self.send_header("Content-Length", str(len(model_body)))
                self.end_headers()
                self.wfile.write(model_body)
                self.wfile.flush()
                conn.close()
                return

            self.send_response(response.status)
            for header, value in response.getheaders():
                if header.lower() not in ('transfer-encoding',):
                    self.send_header(header, value)
            self.end_headers()

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                if method == "POST" and path == "/v1/chat/completions":
                    response_body.extend(chunk)
                if method == "POST" and path == "/v1/chat/completions":
                    stream_buffer += chunk
                    while b"\n" in stream_buffer:
                        raw_line, stream_buffer = stream_buffer.split(b"\n", 1)
                        if not raw_line.startswith(b"data: "):
                            continue
                        payload = raw_line[6:].strip()
                        if payload in (b"", b"[DONE]"):
                            continue
                        try:
                            event = json.loads(payload)
                        except json.JSONDecodeError:
                            continue
                        choices = event.get("choices", [])
                        delta = choices[0].get("delta", {}) if choices else {}
                        if delta.get("content") or delta.get("reasoning_content"):
                            if decode_started_at is None:
                                decode_started_at = time.perf_counter()
                            streamed_tokens += 1
                            elapsed = time.perf_counter() - decode_started_at
                            rate = streamed_tokens / elapsed if elapsed > 0 else 0.0
                            with self.dashboard.lock:
                                self.dashboard.inference_phase = "DECODE"
                                self.dashboard.gen_tokens = streamed_tokens
                                self.dashboard.decode_speed = rate
                                self.dashboard.is_processing = True
                                self.dashboard.last_infer_time = time.time()
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except BrokenPipeError:
                    break

            conn.close()
            if method == "POST" and path == "/v1/chat/completions" and not streamed_tokens:
                self._record_decode_result(bytes(response_body), started_at)
            if method == "POST" and path == "/v1/chat/completions" and streamed_tokens:
                elapsed = time.perf_counter() - decode_started_at
                LOGGER.info(
                    "proxy decode completed tokens=%d elapsed=%.3fs rate=%.1f tok/s",
                    streamed_tokens,
                    elapsed,
                    streamed_tokens / elapsed if elapsed > 0 else 0,
                )
            if method == "POST" and path == "/v1/chat/completions" and self.dashboard:
                with self.dashboard.lock:
                    self.dashboard.is_processing = False
                    self.dashboard.inference_phase = "IDLE"
            LOGGER.info(
                "response method=%s path=%s target=%s status=%d elapsed=%.3fs",
                method,
                self.path,
                target_port,
                response.status,
                time.perf_counter() - started_at,
            )

        except Exception:
            LOGGER.exception("backend failure target=%s path=%s", target_port, self.path)
            if target_port == PRIMARY_PORT and LAZY_FALLBACK and self.dashboard:
                with PROXY_STATE.lock:
                    PROXY_STATE.primary_healthy = False
                    PROXY_STATE.fallback_available = True
                if self.dashboard.ensure_backend(FALLBACK_PORT):
                    self._forward_fallback(method, self._prepare_body(FALLBACK_PORT, body or b""))
                    return
            if target_port == PRIMARY_PORT and PROXY_STATE.fallback_available:
                with PROXY_STATE.lock:
                    PROXY_STATE.primary_healthy = False
                self._forward_fallback(method, body)
            else:
                self.send_error(502, "Bad Gateway")
        finally:
            if request_lock:
                request_lock.release()

    def _forward_fallback(self, method: str, body: bytes):
        try:
            conn = http.client.HTTPConnection("localhost", FALLBACK_PORT, timeout=300)
            headers = {k: v for k, v in self.headers.items()}
            conn.request(method, self.path, body=body, headers=headers)
            response = conn.getresponse()

            self.send_response(response.status)
            for header, value in response.getheaders():
                if header.lower() not in ('transfer-encoding',):
                    self.send_header(header, value)
            self.end_headers()

            while True:
                chunk = response.read(8192)
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except BrokenPipeError:
                    break
            conn.close()
        except Exception:
            self.send_error(502, "Both backends failed")

    def do_GET(self):
        self._forward("GET")

    def do_HEAD(self):
        self._forward("HEAD")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, HEAD, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        self._forward("POST", body)

# ═══════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════

class ManagedServer:
    def __init__(self, command, environment, log_buffer, log_handler, fallback=False):
        self.command = command
        self.environment = environment
        self.log_buffer = log_buffer
        self.log_handler = log_handler
        self.fallback = fallback
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=self.environment,
        )
        threading.Thread(
            target=self.log_handler,
            args=(self.process, self.log_buffer, "ONLINE", "ERROR", self.fallback),
            daemon=True,
        ).start()
        return self.process

    def stop(self):
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()


class MLXDashboard:
    def __init__(self):
        self.server_process = None
        self.fallback_process = None
        self.proxy_server = None
        self.status = "BOOT"
        self.fallback_status = "BOOT" if AUTO_START_FALLBACK else "MANUAL"
        self.prompt_current = 0
        self.prompt_total = 0
        self.prompt_speed = 0.0
        self.prefill_speed = 0.0
        self.decode_speed = 0.0
        self.memory_gb = 0.0
        self.cache_sequences = 0
        self.cache_size_gb = 0.0
        self.last_activity = "--:--:--"
        self.is_processing = False
        self.inference_phase = "IDLE"
        self.effect_iterator = None
        self.effect_iterator_phase = None
        self.process_start_time = None
        self.request_count = 0
        self.uptime_start = time.time()
        self.speed_history = deque(maxlen=80)
        self.token_history = deque(maxlen=80)
        self.last_history_sample = 0.0
        self.log_buffer = deque(maxlen=30)
        self.fallback_log_buffer = deque(maxlen=30)
        self.primary_server = None
        self.fallback_server = None
        self.running = True
        self.lock = threading.Lock()
        self.keyboard = NonBlockingInput()
        self.notification = ""
        self.notif_time = 0
        self.idle_frame_idx = 0
        self.last_infer_time = 0
        self.gen_tokens = 0
        self.eta_seconds = 0

        self.total_ram = psutil.virtual_memory().total / (1024**3)
        self.used_ram = 0.0
        self.swap_used = 0.0
        self.top_processes = []
        self.ram_warning = False
        self.force_fallback = False
        self.config_mode = False
        self.last_primary_restart = 0.0
        self.restart_lock = threading.Lock()
        self.backend_switch_lock = threading.Lock()
        self.last_restart_request_count = 0
        self.swap_restart_armed = True
        with PROXY_STATE.lock:
            PROXY_STATE.active_backend = "27B"
            PROXY_STATE.routing_reason = "initial"
            PROXY_STATE.last_request_tokens = 0
            PROXY_STATE.last_request_time = ""
            PROXY_STATE.total_routed_27b = 0
            PROXY_STATE.total_routed_fallback = 0
            PROXY_STATE.primary_healthy = True
            PROXY_STATE.fallback_available = False

    def start_proxy(self):
        try:
            ProxyHandler.dashboard = self
            self.proxy_server = ThreadingHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
            threading.Thread(target=self.proxy_server.serve_forever, daemon=True).start()
            self._notify("PROXY STARTED ON PORT " + str(PROXY_PORT))
        except Exception as e:
            self._notify("PROXY FAILED: " + str(e))

    def start_server(self):
        cmd = [sys.executable, "mlx_server_safe_wrapper.py"]
        env = os.environ.copy()
        env.update({
            "PRIMARY_MODEL_PATH": MODEL_PATH,
            "PRIMARY_PORT": str(PRIMARY_PORT),
            "PREFILL_STEP_SIZE": str(PREFILL_STEP_SIZE),
            "PROMPT_CONCURRENCY": str(PROMPT_CONCURRENCY),
            "DECODE_CONCURRENCY": str(DECODE_CONCURRENCY),
            "PROMPT_CACHE_SIZE": str(PROMPT_CACHE_SIZE),
            "PROMPT_CACHE_BYTES": str(PROMPT_CACHE_BYTES),
            "PRIMARY_MEMORY_LIMIT": str(PRIMARY_MEMORY_LIMIT),
            "SERVER_MAX_TOKENS": str(SERVER_MAX_TOKENS),
        })
        self.primary_server = ManagedServer(cmd, env, self.log_buffer, self._parse_logs)
        self.server_process = self.primary_server.start()
        if not hasattr(self, "monitor_thread") or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_system, daemon=True)
            self.monitor_thread.start()

    def start_fallback_server(self):
        if not AUTO_START_FALLBACK and not LAZY_FALLBACK:
            return
        cmd = [sys.executable, "mlx_server_fallback_wrapper.py"]
        env = os.environ.copy()
        env["FALLBACK_MODEL_PATH"] = FALLBACK_MODEL_PATH
        env["FALLBACK_PORT"] = str(FALLBACK_PORT)
        env["FALLBACK_MEMORY_LIMIT"] = str(FALLBACK_MEMORY_LIMIT)
        env["PREFILL_STEP_SIZE"] = str(PREFILL_STEP_SIZE)
        env["PROMPT_CONCURRENCY"] = str(PROMPT_CONCURRENCY)
        env["DECODE_CONCURRENCY"] = str(DECODE_CONCURRENCY)
        env["PROMPT_CACHE_SIZE"] = str(PROMPT_CACHE_SIZE)
        env["PROMPT_CACHE_BYTES"] = str(PROMPT_CACHE_BYTES)
        env["SERVER_MAX_TOKENS"] = str(SERVER_MAX_TOKENS)
        self.fallback_server = ManagedServer(
            cmd, env, self.fallback_log_buffer, self._parse_logs, fallback=True
        )
        self.fallback_process = self.fallback_server.start()

    def ensure_backend(self, target_port):
        with self.backend_switch_lock:
            desired_process = self.server_process if target_port == PRIMARY_PORT else self.fallback_process
            if desired_process and desired_process.poll() is None:
                return check_backend_available(target_port, "/v1/models")

            if target_port == PRIMARY_PORT:
                if self.fallback_server:
                    self.fallback_server.stop()
                self.start_server()
            else:
                if self.primary_server:
                    self.primary_server.stop()
                self.start_fallback_server()

            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                if check_backend_available(target_port, "/v1/models"):
                    return True
                time.sleep(0.5)
            return False

    def restart_server(self):
        if not self.restart_lock.acquire(blocking=False):
            return
        try:
            self.last_primary_restart = time.time()
            self._restart_server_locked()
        finally:
            self.restart_lock.release()

    def _restart_server_locked(self):
        self._notify("RESTARTING PRIMARY SERVER...")
        if self.primary_server:
            self.primary_server.stop()
        self.prompt_current = 0
        self.prompt_total = 0
        self.prompt_speed = 0.0
        self.prefill_speed = 0.0
        self.decode_speed = 0.0
        self.inference_phase = "IDLE"
        self.gen_tokens = 0
        self.eta_seconds = 0
        self.speed_history.clear()
        self.token_history.clear()
        with PROXY_STATE.lock:
            PROXY_STATE.primary_healthy = True
        self.start_server()
        self.last_restart_request_count = self.request_count
        self._notify("PRIMARY SERVER RESTARTED")

    def _maybe_auto_restart(self):
        if LAZY_FALLBACK or not AUTO_RESTART_27B or self.is_processing or self.force_fallback:
            return
        now = time.time()
        if now - self.last_primary_restart < RESTART_COOLDOWN_SECONDS:
            return
        swap_triggered = ROUTE_ON_SWAP and self.swap_used >= RESTART_ON_SWAP_GB_27B
        if not swap_triggered:
            self.swap_restart_armed = True
        request_triggered = self.request_count - self.last_restart_request_count >= RESTART_AFTER_REQUESTS_27B
        swap_triggered = swap_triggered and self.swap_restart_armed
        if swap_triggered or request_triggered:
            reason = "SWAP threshold" if swap_triggered else "request limit"
            self._notify("AUTO-RESTART: " + reason)
            self.swap_restart_armed = False
            self.restart_server()

    def restart_fallback_server(self):
        self._notify("RESTARTING FALLBACK SERVER...")
        if self.fallback_server:
            self.fallback_server.stop()
        self.fallback_status = "BOOT"
        self.start_fallback_server()
        self._notify("FALLBACK SERVER RESTARTED")

    def toggle_fallback(self):
        self.set_routing_mode("AUTO" if self.force_fallback else "FALLBACK")

    def set_routing_mode(self, mode):
        global DYNAMIC_FALLBACK
        self.force_fallback = mode == "FALLBACK"
        DYNAMIC_FALLBACK = mode == "DYNAMIC"
        with PROXY_STATE.lock:
            PROXY_STATE.routing_mode = mode
            PROXY_STATE.primary_healthy = True
        self._notify("ROUTING MODE: " + mode)

    def stop_process(self, process, label):
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        self._notify(label + " SERVER STOPPED")

    def toggle_primary_server(self):
        if self.server_process and self.server_process.poll() is None:
            self.stop_process(self.server_process, "27B")
            return
        self.start_server()
        self._notify("27B SERVER STARTED")

    def toggle_fallback_server(self):
        if self.fallback_process and self.fallback_process.poll() is None:
            self.stop_process(self.fallback_process, "FALLBACK")
            self.fallback_status = "MANUAL"
            return
        self.start_fallback_server()
        self._notify("FALLBACK SERVER STARTED")

    def clear_logs(self):
        with self.lock:
            self.log_buffer.clear()
            self.fallback_log_buffer.clear()
            self.speed_history.clear()
            self.token_history.clear()
        self._notify("LOGS CLEARED")

    def print_stats(self):
        Path("logs").mkdir(exist_ok=True)
        filename = "logs/mlx_stats_" + datetime.now().strftime('%Y%m%d_%H%M%S') + ".txt"
        with open(filename, "w") as f:
            f.write("MLX Router Stats - " + str(datetime.now()) + "\n")
            f.write("Status: " + self.status + "\n")
            f.write("Requests: " + str(self.request_count) + "\n")
            f.write("Routed to 27B: " + str(PROXY_STATE.total_routed_27b) + "\n")
            f.write("Routed to Fallback: " + str(PROXY_STATE.total_routed_fallback) + "\n")
            f.write("RAM: " + str(round(self.used_ram, 1)) + "/" + str(int(self.total_ram)) + " GB\n")
        self._notify("STATS SAVED: " + filename)

    def configure(self, live):
        global MODEL_PATH, FALLBACK_MODEL_PATH, MAX_CONTEXT_TOKENS
        global CONTEXT_SAFETY_MARGIN, TOKEN_LIMIT_27B, PREFILL_STEP_SIZE
        global PROMPT_CONCURRENCY, DECODE_CONCURRENCY, THINKING_ENABLED_27B
        global REASONING_EFFORT_27B, MAX_GENERATION_TOKENS_27B
        global ROUTE_ON_SWAP, AUTO_RESTART_27B
        global FALLBACK_MEMORY_LIMIT
        live.stop()
        self.keyboard.stop()
        settings = [
            ("Primary Modellpfad", MODEL_PATH, str),
            ("Fallback Modellpfad", FALLBACK_MODEL_PATH, str),
            ("Fallback Metal-Limit GB", FALLBACK_MEMORY_LIMIT / 1024 ** 3, float),
            ("Gesamtkontext", MAX_CONTEXT_TOKENS, int),
            ("Kontextreserve", CONTEXT_SAFETY_MARGIN, int),
            ("AUTO: Primary bis Prompt", TOKEN_LIMIT_27B, int),
            ("Prefill Schrittgröße", PREFILL_STEP_SIZE, int),
            ("Prefill Parallelität", PROMPT_CONCURRENCY, int),
            ("Decode Parallelität", DECODE_CONCURRENCY, int),
            ("Primary max. Ausgabe", MAX_GENERATION_TOKENS_27B, int),
            ("Primary Thinking", "AN" if THINKING_ENABLED_27B else "AUS", str),
            ("Primary Reasoning", REASONING_EFFORT_27B, str),
            ("SWAP-Routing", "AN" if ROUTE_ON_SWAP else "AUS", str),
            ("Auto-Restart Primary", "AN" if AUTO_RESTART_27B else "AUS", str),
        ]
        values = [item[1] for item in settings]

        def render_config(selected):
            console.clear()
            table = Table(title="MLX KONFIGURATION", border_style="cyan", expand=True)
            table.add_column("", width=3)
            table.add_column("Parameter", style="bold cyan")
            table.add_column("Wert", style="white")
            for index, (label, _, _) in enumerate(settings):
                marker = "▶" if index == selected else " "
                style = "bold yellow" if index == selected else "white"
                table.add_row(marker, label, str(values[index]), style=style)
            console.print(table)
            console.print(
                "[dim]Scan-Vorschlag: Kontext "
                + str(MODEL_LIMIT_SUGGESTIONS["max_context_tokens"])
                + " | Primary-Prompt "
                + str(MODEL_LIMIT_SUGGESTIONS["primary_prompt_limit"])
                + " | Ausgabe "
                + str(MODEL_LIMIT_SUGGESTIONS["max_generation_tokens"])
                + " | Fallback-Metal "
                + str(MODEL_LIMIT_SUGGESTIONS["fallback_memory_limit_gb"])
                + " GB[/dim]"
            )
            console.print("\n[dim][↑/↓ oder J/K] Auswahl  [E/Enter] Bearbeiten  [S] Speichern  [Q/Esc] Abbrechen[/dim]")

        def read_key():
            stdin_fd = sys.stdin.fileno()
            if not select.select([stdin_fd], [], [], 0.1)[0]:
                return None
            key = os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
            if key == "\x1b":
                sequence = ""
                while select.select([stdin_fd], [], [], 0.1)[0]:
                    sequence += os.read(stdin_fd, 1).decode("utf-8", errors="ignore")
                    if sequence[-1:] in ("A", "B", "C", "D", "~"):
                        break
                if sequence.endswith("A"):
                    return "up"
                if sequence.endswith("B"):
                    return "down"
                return "escape"
            if key in ("\r", "\n"):
                return "enter"
            return key.lower()

        try:
            selected = 0
            saved = False
            old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            render_config(selected)
            while not saved:
                command = read_key()
                if command is None:
                    continue
                if command in ("q", "esc", "escape", "quit"):
                    break
                if command in ("j", "down", "v"):
                    selected = (selected + 1) % len(settings)
                    render_config(selected)
                    continue
                if command in ("k", "up", "^"):
                    selected = (selected - 1) % len(settings)
                    render_config(selected)
                    continue
                if command in ("e", "enter"):
                    label, _, convert = settings[selected]
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                    value = console.input("[bold yellow]" + label + "[/bold yellow] [dim][" + str(values[selected]) + "][/dim]: ").strip()
                    if value:
                        values[selected] = convert(value)
                    old_settings = termios.tcgetattr(sys.stdin)
                    tty.setcbreak(sys.stdin.fileno())
                    render_config(selected)
                    continue
                if command in ("s", "save"):
                    saved = True

            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

            if saved:
                config_values = [
                    "MODEL_PATH", "FALLBACK_MODEL_PATH", "FALLBACK_MEMORY_LIMIT", "MAX_CONTEXT_TOKENS",
                    "CONTEXT_SAFETY_MARGIN", "TOKEN_LIMIT_27B", "PREFILL_STEP_SIZE",
                    "PROMPT_CONCURRENCY", "DECODE_CONCURRENCY", "MAX_GENERATION_TOKENS_27B",
                    "THINKING_ENABLED_27B", "REASONING_EFFORT_27B", "ROUTE_ON_SWAP",
                    "AUTO_RESTART_27B",
                ]
                for name, value in zip(config_values, values):
                    if name == "FALLBACK_MEMORY_LIMIT":
                        value = int(float(value) * 1024 ** 3)
                    elif name in ("THINKING_ENABLED_27B", "ROUTE_ON_SWAP", "AUTO_RESTART_27B"):
                        value = str(value).lower() in ("an", "j", "ja", "y", "yes", "1", "true")
                    globals()[name] = value
                save_config()
                console.clear()
                console.print("[bold green]Konfiguration gespeichert.[/bold green]")
                restart = console.input("Server jetzt neu starten? [J/n]: ").strip().lower()
                if restart not in ("n", "nein", "no", "0"):
                    acquired = self.restart_lock.acquire(blocking=False)
                    if acquired:
                        try:
                            self._restart_server_locked()
                        finally:
                            self.restart_lock.release()
                    if AUTO_START_FALLBACK:
                        self.restart_fallback_server()
                console.input("\nEnter zum Fortfahren...")
        except (ValueError, EOFError, KeyboardInterrupt):
            console.print("\n[bold red]Ungültige Eingabe oder Konfiguration abgebrochen.[/bold red]")
            console.input("Enter zum Fortfahren...")
        finally:
            self.keyboard = NonBlockingInput()
            self.keyboard.start()
            live.start()

    def _notify(self, msg: str):
        self.notification = msg
        self.notif_time = time.time()

    def _calculate_eta(self) -> str:
        if not self.is_processing or self.prompt_total <= 0 or self.prompt_speed <= 0:
            return "--"
        remaining = self.prompt_total - self.prompt_current
        if remaining <= 0:
            return "0s"
        eta = remaining / self.prompt_speed
        self.eta_seconds = eta
        if eta < 60:
            return "~" + str(int(eta)) + "s"
        elif eta < 3600:
            mins = int(eta / 60)
            secs = int(eta % 60)
            return "~" + str(mins) + "m " + str(secs) + "s"
        else:
            hrs = int(eta / 3600)
            mins = int((eta % 3600) / 60)
            return "~" + str(hrs) + "h " + str(mins) + "m"

    def _sample_histograms(self) -> None:
        now = time.monotonic()
        if not self.is_processing or now - self.last_history_sample < 0.5:
            return
        self.last_history_sample = now
        if self.inference_phase == "DECODE":
            self.speed_history.append(self.decode_speed)
            self.token_history.append(self.gen_tokens)
        else:
            self.speed_history.append(self.prefill_speed)
            self.token_history.append(self.prompt_current)

    def _parse_logs(self, process, buffer, online_status, error_status, is_fallback=False):
        if not process:
            return
        for line in process.stdout:
            line = line.strip()
            if not line:
                continue
            with self.lock:
                buffer.append(line)
                if any(marker in line for marker in (
                    "Request started:",
                    "Generation queued:",
                    "Prefill completed:",
                    "Decode started:",
                    "Decode progress:",
                    "Request completed:",
                    "Request failed:",
                )):
                    LOGGER.info("server=%s %s", "fallback" if is_fallback else "primary", line)
                if is_fallback:
                    if "Starting httpd" in line or "Application startup complete" in line or "Uvicorn running" in line:
                        self.fallback_status = online_status
                    if "Error" in line or "RuntimeError" in line or "OutOfMemory" in line:
                        self.fallback_status = error_status

                if "Starting httpd" in line or "Application startup complete" in line or "Uvicorn running" in line:
                    self.status = "ONLINE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                if "POST /v1/chat/completions" in line or "GET /v1/models" in line:
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                if "Request started: endpoint=/chat/completions" in line:
                    self.request_count += 1
                    self.is_processing = True
                    self.inference_phase = "PREFILL"
                    self.prompt_current = 0
                    self.prompt_total = 0
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()

                queued_match = re.search(
                    r"Generation queued:.*prompt_tokens=(\d+).*max_tokens=(\d+)",
                    line,
                )
                if queued_match:
                    self.prompt_current = 0
                    self.prompt_total = int(queued_match.group(1))
                    self.is_processing = True
                    self.process_start_time = time.time()
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                if "Prefill started:" in line:
                    prefill_started_match = re.search(r"prompt_tokens=(\d+)", line)
                    if prefill_started_match:
                        self.prompt_current = 0
                        self.prompt_total = int(prefill_started_match.group(1))
                    self.is_processing = True
                    self.inference_phase = "PREFILL"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")

                prefill_match = re.search(
                    r"Prefill progress:.*tokens=(\d+)/(\d+)", line,
                )
                if prefill_match:
                    self.prompt_current = int(prefill_match.group(1))
                    self.prompt_total = int(prefill_match.group(2))
                    self.is_processing = True
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        if elapsed > 0:
                            self.prefill_speed = self.prompt_current / elapsed
                            self.prompt_speed = self.prefill_speed
                            self.speed_history.append(self.prefill_speed)
                            self.token_history.append(self.prompt_current)

                prefill_complete_match = re.search(
                    r"Prefill completed:.*prompt_tokens=(\d+).*rate=([\d.]+)",
                    line,
                )
                if prefill_complete_match:
                    self.prompt_current = int(prefill_complete_match.group(1))
                    self.prompt_total = self.prompt_current
                    self.prefill_speed = float(prefill_complete_match.group(2))
                    self.prompt_speed = self.prefill_speed
                    self.speed_history.append(self.prefill_speed)
                    self.token_history.append(self.prompt_current)
                    self.process_start_time = None
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()
                    self.inference_phase = "DECODE"

                decode_match = re.search(
                    r"Decode progress:.*generated_tokens=(\d+).*rate=([\d.]+)",
                    line,
                )
                if decode_match:
                    self.gen_tokens = int(decode_match.group(1))
                    rate_text = decode_match.group(2)
                    if rate_text:
                        self.decode_speed = float(rate_text)
                        self.speed_history.append(self.decode_speed)
                    self.is_processing = True
                    self.inference_phase = "DECODE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    self.last_infer_time = time.time()

                if "Request completed: endpoint=/chat/completions" in line:
                    self.is_processing = False
                    self.inference_phase = "IDLE"
                    self.last_infer_time = time.time()
                    self.process_start_time = None

                progress_match = re.search(r'Prompt processing progress:\s+(\d+)/(\d+)', line)
                if progress_match:
                    self.prompt_current = int(progress_match.group(1))
                    self.prompt_total = int(progress_match.group(2))
                    self.is_processing = True
                    self.inference_phase = "PREFILL" if self.prompt_current < self.prompt_total else "DECODE"
                    self.last_activity = datetime.now().strftime("%H:%M:%S")
                    now = time.time()
                    if not self.process_start_time:
                        self.process_start_time = self.last_infer_time or now
                    self.last_infer_time = now
                    elapsed = now - self.process_start_time
                    if elapsed > 0 and self.prompt_current > 0:
                        self.prefill_speed = self.prompt_current / elapsed
                        self.prompt_speed = self.prefill_speed
                        self.speed_history.append(self.prefill_speed)
                        self.token_history.append(self.prompt_current)

                gen_match = re.search(r'Generation:\s+(\d+)\s+tokens?', line)
                if gen_match:
                    self.gen_tokens = int(gen_match.group(1))
                    self.is_processing = False
                    self.inference_phase = "IDLE"
                    self.last_infer_time = time.time()
                    self.eta_seconds = 0
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        self.prefill_speed = self.prompt_total / elapsed if elapsed > 0 else 0
                        self.prompt_speed = self.prefill_speed
                        self.process_start_time = None
                        self.speed_history.append(self.prefill_speed)

                if self.prompt_current > 0 and self.prompt_current >= self.prompt_total:
                    self.is_processing = True
                    self.inference_phase = "DECODE"
                    self.last_infer_time = time.time()
                    self.eta_seconds = 0
                    if self.process_start_time:
                        elapsed = time.time() - self.process_start_time
                        self.prefill_speed = self.prompt_total / elapsed if elapsed > 0 else 0
                        self.prompt_speed = self.prefill_speed
                        self.process_start_time = None
                        self.speed_history.append(self.prefill_speed)

                cache_match = re.search(r'Prompt Cache:\s+(\d+)\s+sequences?,\s+([\d.]+)\s*(GB|MB)', line)
                if cache_match:
                    self.cache_sequences = int(cache_match.group(1))
                    val = float(cache_match.group(2))
                    unit = cache_match.group(3)
                    if unit == "MB":
                        val = val / 1024
                    self.cache_size_gb = val
                    self.memory_gb = val

                mem_match = re.search(r'Peak memory:\s+([\d.]+)\s*(GB|MB)', line, re.IGNORECASE)
                if mem_match:
                    val = float(mem_match.group(1))
                    unit = mem_match.group(2)
                    if unit == "MB":
                        val = val / 1024
                    self.memory_gb = val

                if "Error" in line or "RuntimeError" in line or "OutOfMemory" in line:
                    self.status = "ERROR"
                    with PROXY_STATE.lock:
                        PROXY_STATE.primary_healthy = False

    def _monitor_system(self):
        while self.running:
            try:
                mem = psutil.virtual_memory()
                swap = psutil.swap_memory()
                self.used_ram = mem.used / (1024**3)
                self.swap_used = swap.used / (1024**3)
                self.ram_warning = mem.percent > 88
                self._maybe_auto_restart()

                procs = []
                for p in psutil.process_iter(['pid', 'name', 'memory_info', 'memory_percent']):
                    try:
                        info = p.info
                        if info['memory_percent'] and info['memory_percent'] > 0.05:
                            procs.append({
                                'pid': info['pid'],
                                'name': info['name'][:22],
                                'rss': info['memory_info'].rss / (1024**3),
                                'percent': info['memory_percent']
                            })
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                procs.sort(key=lambda x: x['rss'], reverse=True)
                self.top_processes = procs[:8]
            except Exception:
                pass
            time.sleep(2)

    def _check_hotkeys(self, live):
        cmd = self.keyboard.get_command()
        if cmd == 'q':
            self._notify("SHUTTING DOWN...")
            self.running = False
        elif cmd == 'r':
            self.restart_server()
        elif cmd == 'f':
            self.restart_fallback_server()
        elif cmd == 'c':
            self.clear_logs()
        elif cmd == 'p':
            self.print_stats()
        elif cmd == 'k':
            self.configure(live)
        elif cmd == 's':
            self.toggle_fallback()
        elif cmd == 'a':
            self.set_routing_mode("AUTO")
        elif cmd == 'd':
            self.set_routing_mode("AUTO" if DYNAMIC_FALLBACK else "DYNAMIC")
        elif cmd == '1':
            self.set_routing_mode("PRIMARY")
        elif cmd == '2':
            self.set_routing_mode("FALLBACK")
        elif cmd == 'x':
            self.toggle_fallback_server()
        elif cmd == 'z':
            self.toggle_primary_server()

    def _idle_indicator(self) -> Text:
        self.idle_frame_idx = (self.idle_frame_idx + 1) % len(IDLE_FRAMES)
        frame = IDLE_FRAMES[self.idle_frame_idx]

        if self.is_processing:
            return Text("⚡ PROCESSING", style="bold yellow")

        idle_secs = int(time.time() - self.last_infer_time) if self.last_infer_time > 0 else 0
        if idle_secs < 5:
            return Text("✓ READY", style="bold green")

        return Text(frame + " IDLE (" + str(idle_secs) + "s)", style="dim cyan")

    def _hotkey_bar(self) -> Text:
        bar = Text()
        hotkeys = [
            ("Q", "Quit", "bold magenta"),
            ("R", "Restart Primary", "bold cyan"),
            ("", "Restart [F]allback", "bold cyan"),
            ("A", "Auto-Routing", "bold green"),
            ("D", "Dynamic Fallback", "bold cyan"),
            ("1", "Immer Primary", "bold green"),
            ("2", "Immer Fallback", "bold yellow"),
            ("X", "Fallback on/off", "bold red"),
            ("Z", "Primary on/off", "bold red"),
            ("C", "Clear", "bold yellow"),
            ("P", "Print", "bold green"),
            ("K", "Konfig", "bold magenta"),
        ]
        for key, label, style in hotkeys:
            if key:
                bar.append("[", style="dim")
                bar.append(key, style=style)
                bar.append("] " + label + "  ", style="dim")
            else:
                bar.append("Restart ", style="dim")
                bar.append("[F]", style=style)
                bar.append("allback  ", style="dim")
        bar.no_wrap = True
        bar.overflow = "ellipsis"
        return bar

    def _notification_bar(self) -> Text:
        if time.time() - self.notif_time < 3 and self.notification:
            return Text("▶ " + self.notification, style="bold yellow")
        return Text("")

    def _header(self) -> Panel:
        wide_logo = (
            "▄▀▀▄▀▀▄ █     █    █      ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀▄      ▄▀▀▀▀▄ ▄▀▀▀▀▄ █    █ ▀▀▀█▀▀▀ ▄▀▀▀▀ ▄▀▀▀▀▄\n"
            "▀  ▀  ▀ ▀     ▀    ▀      ▀    ▀ ▀    ▀    ▀    ▀    ▀      ▀    ▀ ▀    ▀ ▀    ▀    ▀    ▀     ▀    ▀\n"
            "█  █  █ █     ▄▀▀▀▀▄      █▀▀▀▀█ █    █    █    █    █      █▀▀▀▀▄ █    █ █    █    █    ▄▀▀▀  █▀▀▀▀▄\n"
            "█  ▀  █ █     █    █      █    █ █    █    █    █    █      █    █ █    █ █    █    █    █     █    █\n"
            "▀     ▀  ▀▀▀▀ ▀    ▀      ▀    ▀  ▀▀▀▀     ▀     ▀▀▀▀       ▀    ▀  ▀▀▀▀   ▀▀▀▀     ▀     ▀▀▀▀ ▀    ▀"
        )
        logo = Text()
        if console.width >= 118:
            for index, row in enumerate(wide_logo.splitlines()):
                logo.append(row, style="bold cyan" if index < 2 else "bold magenta")
                if index < 4:
                    logo.append("\n")
        else:
            logo.append("MLX AUTO-ROUTER", style="bold cyan")
        logo.justify = "center"
        uptime = int(time.time() - self.uptime_start)
        uptime_str = str(uptime // 3600).zfill(2) + ":" + str((uptime % 3600) // 60).zfill(2) + ":" + str(uptime % 60).zfill(2)
        subtitle = Text("  Proxy: http://" + HOST + ":" + str(PROXY_PORT) + "/v1  |  Uptime: " + uptime_str, style="dim cyan")
        content = Text.assemble(logo, "\n", subtitle)
        return Panel(content, border_style="cyan", padding=(0, 1))

    def _status_panel(self) -> Panel:
        with self.lock:
            status_color = {"BOOT": "yellow", "ONLINE": "green", "ERROR": "red"}.get(self.status, "white")
            icon = "▶" if self.status == "ONLINE" else "◉"

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold blue", width=16)
            table.add_column()

            table.add_row("Status", Text(icon + " " + self.status, style="bold " + status_color))
            if AUTO_START_FALLBACK:
                fb_status_color = {"BOOT": "yellow", "ONLINE": "green", "ERROR": "red"}.get(self.fallback_status, "white")
                fb_icon = "▶" if self.fallback_status == "ONLINE" else "◉"
                table.add_row("Fallback", Text(fb_icon + " " + self.fallback_status, style="bold " + fb_status_color))
            table.add_row("Activity", self._idle_indicator())
            table.add_row("Last", Text(self.last_activity, style="bold yellow"))
            table.add_row("Requests", Text(str(self.request_count), style="bold magenta"))
            table.add_row("Cache", Text(str(self.cache_sequences) + " seq / " + str(round(self.cache_size_gb, 2)) + "GB", style="bold cyan"))

            return Panel(table, title="[bold blue]● SYSTEM[/bold blue]", border_style="blue", box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _prompt_panel(self) -> Panel:
        with self.lock:
            percent = (self.prompt_current / self.prompt_total * 100) if self.prompt_total > 0 else 0
            eta = self._calculate_eta()

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold yellow", width=14)
            table.add_column()

            table.add_row("Tokens", Text(str(self.prompt_current) + " / " + str(self.prompt_total), style="bold white"))
            table.add_row("Gen Tokens", Text(str(self.gen_tokens), style="bold white"))
            table.add_row("Prefill", speedometer(self.prefill_speed))
            table.add_row("Decode", speedometer(self.decode_speed))
            table.add_row("Progress", dither_bar(percent))
            table.add_row("Percent", Text(str(round(percent, 1)) + "%", style="bold green" if percent >= 100 else "bold yellow"))
            table.add_row("ETA", Text(eta, style="bold magenta"))

            if self.inference_phase == "PREFILL":
                title = "[bold yellow]⚡ PREFILL[/bold yellow]"
            elif self.inference_phase == "DECODE":
                title = "[bold cyan]⚡ DECODE / STREAMING[/bold cyan]"
            else:
                title = "[bold dim]⚡ ENGINE IDLE[/bold dim]"
            border = "yellow" if self.is_processing else "dim"

            return Panel(table, title=title, border_style=border, box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _viz_panel(self) -> Panel:
        with self.lock:
            self._sample_histograms()
            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(ratio=1)
            effect_width = max(48, console.width - 8)
            effect_frame = Text(" " * effect_width)
            effect_class = None
            prompt_complete = self.prompt_total > 0 and self.prompt_current >= self.prompt_total
            decode_active = self.gen_tokens > 0 or prompt_complete or self.inference_phase == "DECODE"
            if self.is_processing and decode_active:
                effect_class = Matrix
            elif self.is_processing and not decode_active and self.inference_phase == "PREFILL":
                effect_class = SynthGrid
            if effect_class is not None:
                if self.effect_iterator is None or self.effect_iterator_phase != self.inference_phase:
                    phase_text = "DECODING" if decode_active else "PREFILLING"
                    banner = [row.plain for row in ascii_banner(phase_text, "white")]
                    banner = [row.center(effect_width) for row in banner]
                    effect = effect_class("\n".join(banner))
                    effect.terminal_config.canvas_width = effect_width
                    effect.terminal_config.canvas_height = EFFECT_ROWS
                    effect.terminal_config.anchor_text = "w"
                    effect.terminal_config.ignore_terminal_dimensions = True
                    self.effect_iterator = iter(effect)
                    self.effect_iterator_phase = self.inference_phase
                try:
                    effect_frame = Text.from_ansi(next(self.effect_iterator))
                except StopIteration:
                    self.effect_iterator = None
                    effect_frame = Text(" " * effect_width)
            else:
                self.effect_iterator = None
                self.effect_iterator_phase = None
            table.add_row(effect_frame)
            table.add_row(sparkline(self.speed_history, width=effect_width, max_val=MAX_SPEED))
            table.add_row(sparkline(self.token_history, width=effect_width, max_val=40000))

            return Panel(table, border_style="magenta", box=box.ROUNDED, height=EFFECT_PANEL_HEIGHT)

    def _router_panel(self) -> Panel:
        with PROXY_STATE.lock:
            active = PROXY_STATE.active_backend
            reason = PROXY_STATE.routing_reason
            tokens = PROXY_STATE.last_request_tokens
            fallback_ok = PROXY_STATE.fallback_available
            routed_27b = PROXY_STATE.total_routed_27b
            routed_fb = PROXY_STATE.total_routed_fallback
            routing_mode = PROXY_STATE.routing_mode

            table = Table(show_header=False, box=None, padding=(0, 1))
            table.add_column(style="bold green", width=16)
            table.add_column()

            if active == "27B":
                model_text = Text("▶ PRIMARY", style="bold green")
            else:
                model_text = Text("▶ FALLBACK", style="bold yellow")

            if self.force_fallback:
                model_text = Text("▶ FALLBACK (FORCED)", style="bold red")

            table.add_row("Active Model", model_text)
            table.add_row("Mode", Text(routing_mode, style="bold cyan"))
            if routing_mode == "DYNAMIC":
                table.add_row("Dynamic Limit", Text(str(dynamic_primary_prompt_limit()) + " tokens", style="bold cyan"))
            table.add_row("Reason", Text(reason, style="bold white"))
            table.add_row("Last Prompt", Text(str(tokens) + " tokens", style="bold cyan"))
            table.add_row("Fallback", Text("✓ Online" if fallback_ok else "✗ Offline",
                          style="bold green" if fallback_ok else "bold red"))
            if FALLBACK_TYPE == "mlx_lm":
                fb_model = os.path.basename(FALLBACK_MODEL_PATH.rstrip("/").rstrip("\\"))
                table.add_row("Fallback Model", Text(fb_model, style="bold cyan"))
            table.add_row("Stats", Text("27B:" + str(routed_27b) + " | FB:" + str(routed_fb), style="dim"))

            title = "[bold green]◉ ROUTER[/bold green]"
            border = "green" if active == "27B" else "yellow"

            return Panel(table, title=title, border_style=border, box=box.ROUNDED, height=TOP_PANEL_HEIGHT)

    def _ram_panel(self) -> Panel:
        with self.lock:
            table = Table(show_header=False, box=None, padding=(0, 0))
            table.add_column(style="bold red", width=12)
            table.add_column()

            ram_percent = (self.used_ram / self.total_ram) * 100 if self.total_ram else 0
            ram_color = "red" if ram_percent > 88 else "yellow" if ram_percent > 75 else "green"
            table.add_row("RAM", ram_blocks(self.used_ram, self.total_ram))
            table.add_row("Usage", Text(str(round(ram_percent, 1)) + "%", style="bold " + ram_color))

            swap_critical = ROUTE_ON_SWAP and self.swap_used > SWAP_LIMIT_27B
            if swap_critical:
                table.add_row("SWAP", Text("🔥 " + str(round(self.swap_used, 1)) + " GB SWAP!", style="bold red blink"))
            else:
                table.add_row("SWAP", Text(str(round(self.swap_used, 1)) + " GB", style="dim"))

            ram_critical = ram_percent > 88
            if ram_critical:
                table.add_row("", Text("⚠️  RAM CRITICAL", style="bold red blink"))

            table.add_row("", Text("─" * 40, style="dim"))
            table.add_row("Processes", Text("PID      Name                 RSS", style="bold underline"))

            for proc in self.top_processes[:10]:
                name = proc['name'][:20]
                rss = proc['rss']
                pid = proc['pid']
                is_mlx = "mlx" in name.lower() or "python" in name.lower() or pid == (self.server_process.pid if self.server_process else -1)
                style = "bold yellow" if is_mlx else "white"
                table.add_row("", Text(str(pid).ljust(8) + " " + name.ljust(20) + " " + str(round(rss, 1)) + "G", style=style))

            return Panel(table, title="[bold red]◉ MEMORY[/bold red]",
                        border_style="red", box=box.ROUNDED,
                        height=TOP_PANEL_HEIGHT)

    def _logs_panel(self) -> Panel:
        with self.lock:
            lines = list(self.log_buffer)
            colored_lines = []
            for line in lines[-18:]:
                if "Prompt processing progress" in line:
                    colored_lines.append(Text(line, style="bold green", no_wrap=True, overflow="ellipsis"))
                elif "Prompt Cache" in line:
                    colored_lines.append(Text(line, style="blue", no_wrap=True, overflow="ellipsis"))
                elif "Generation" in line or "Prefill" in line or "Decode" in line:
                    colored_lines.append(Text(line, style="bold cyan", no_wrap=True, overflow="ellipsis"))
                elif "Error" in line or "RuntimeError" in line:
                    colored_lines.append(Text(line, style="bold red", no_wrap=True, overflow="ellipsis"))
                elif "POST" in line or "GET" in line:
                    colored_lines.append(Text(line, style="dim", no_wrap=True, overflow="ellipsis"))
                else:
                    colored_lines.append(Text(line, style="white", no_wrap=True, overflow="ellipsis"))

            content = Text("\n").join(colored_lines) if colored_lines else Text("Waiting for logs...", style="dim")

            return Panel(content, title="[bold green]◉ LOG STREAM[/bold green]",
                        border_style="green", box=box.ROUNDED, height=LOG_PANEL_HEIGHT)

    def generate_display(self):
        top = Table(show_header=False, box=None, expand=True, padding=(0, 0))
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_column(ratio=1)
        top.add_row(self._status_panel(), self._prompt_panel(), self._router_panel(), self._ram_panel())

        full = Table(show_header=False, box=None, expand=True, padding=(0, 0))
        full.add_column()
        full.add_row(self._header())

        notif = self._notification_bar()
        if notif.plain:
            full.add_row(Align.center(notif))

        full.add_row(top)
        full.add_row(self._viz_panel())
        full.add_row(self._logs_panel())
        full.add_row(Align.center(self._hotkey_bar()))

        return full

    def run(self):
        console.clear()
        console.print("[bold cyan]◈ Initializing MLX Auto-Router...[/bold cyan]")
        console.print("[dim]Primary: 27B on port " + str(PRIMARY_PORT) + "[/dim]")
        console.print("[dim]Proxy: http://" + HOST + ":" + str(PROXY_PORT) + "/v1[/dim]")
        if AUTO_START_FALLBACK:
            console.print("[dim]Fallback: " + FALLBACK_TYPE + " '" + FALLBACK_MODEL_PATH + "' on port " + str(FALLBACK_PORT) + "[/dim]")
        else:
            console.print("[dim]Fallback: " + FALLBACK_TYPE + " on " + FALLBACK_HOST + ":" + str(FALLBACK_PORT) + " (manual)[/dim]")
        console.print()
        console.print("[bold yellow]IMPORTANT:[/bold yellow] Set your API client to http://<host>:" + str(PROXY_PORT) + "/v1")
        console.print()

        self.keyboard.start()
        self.start_server()
        if AUTO_START_FALLBACK:
            time.sleep(1)
            self.start_fallback_server()
            time.sleep(2)
        else:
            time.sleep(2)
        self.start_proxy()

        try:
            with Live(self.generate_display(), refresh_per_second=10, screen=True) as live:
                while self.running:
                    self._check_hotkeys(live)
                    live.update(self.generate_display())
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def shutdown(self):
        self.running = False
        self.keyboard.stop()
        if self.proxy_server:
            self.proxy_server.shutdown()
        console.print("\n[bold red]◈ Shutting down...[/bold red]")
        if self.primary_server:
            self.primary_server.stop()
        if self.fallback_server:
            console.print("[bold red]◈ Stopping fallback server...[/bold red]")
            self.fallback_server.stop()
        console.print("[bold green]◈ Done.[/bold green]")


def main():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[
            logging.FileHandler("logs/router.log"),
        ],
        force=True,
    )
    dashboard = MLXDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()