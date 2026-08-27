import os
import mlx.core as mx
mx.set_memory_limit(int(os.environ.get("PRIMARY_MEMORY_LIMIT", str(20 * 1024**3))))

import sys
model_path = os.environ.get("PRIMARY_MODEL_PATH")
if not model_path:
	raise RuntimeError("PRIMARY_MODEL_PATH ist nicht gesetzt.")
primary_port = os.environ.get("PRIMARY_PORT", "8080")
prefill_step_size = os.environ.get("PREFILL_STEP_SIZE", "4096")
prompt_concurrency = os.environ.get("PROMPT_CONCURRENCY", "1")
decode_concurrency = os.environ.get("DECODE_CONCURRENCY", "1")
prompt_cache_size = os.environ.get("PROMPT_CACHE_SIZE", "1")
prompt_cache_bytes = os.environ.get("PROMPT_CACHE_BYTES")
max_tokens = os.environ.get("SERVER_MAX_TOKENS", "4096")
sys.argv = [
	"mlx_lm.server",
	"--model", model_path,
	"--host", "0.0.0.0",
	"--port", primary_port,
	"--prefill-step-size", prefill_step_size,
	"--prompt-concurrency", prompt_concurrency,
	"--decode-concurrency", decode_concurrency,
	"--prompt-cache-size", prompt_cache_size,
	"--max-tokens", max_tokens,
]
if prompt_cache_bytes:
	sys.argv.extend(["--prompt-cache-bytes", prompt_cache_bytes])

from mlx_lm.server import main
main()