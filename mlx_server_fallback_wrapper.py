import os
import mlx.core as mx

# Allow dashboard to override memory limit via env var; default 12 GB for fallback.
FALLBACK_MEMORY_LIMIT = int(os.environ.get("FALLBACK_MEMORY_LIMIT", str(12 * 1024**3)))
mx.set_memory_limit(FALLBACK_MEMORY_LIMIT)

import sys

FALLBACK_MODEL_PATH = os.environ.get("FALLBACK_MODEL_PATH")
if not FALLBACK_MODEL_PATH:
    raise RuntimeError("FALLBACK_MODEL_PATH ist nicht gesetzt.")
FALLBACK_PORT = os.environ.get("FALLBACK_PORT", "8081")
PREFILL_STEP_SIZE = os.environ.get("PREFILL_STEP_SIZE", "4096")
PROMPT_CONCURRENCY = os.environ.get("PROMPT_CONCURRENCY", "1")
DECODE_CONCURRENCY = os.environ.get("DECODE_CONCURRENCY", "1")
PROMPT_CACHE_SIZE = os.environ.get("PROMPT_CACHE_SIZE", "1")
SERVER_MAX_TOKENS = os.environ.get("SERVER_MAX_TOKENS", "4096")

sys.argv = [
    "mlx_lm.server",
    "--model", FALLBACK_MODEL_PATH,
    "--host", "0.0.0.0",
    "--port", FALLBACK_PORT,
    "--prefill-step-size", PREFILL_STEP_SIZE,
    "--prompt-concurrency", PROMPT_CONCURRENCY,
    "--decode-concurrency", DECODE_CONCURRENCY,
    "--prompt-cache-size", PROMPT_CACHE_SIZE,
    "--max-tokens", SERVER_MAX_TOKENS,
]

from mlx_lm.server import main
main()
