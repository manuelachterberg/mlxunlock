#!/usr/bin/env python3
"""mlx_ondemand.py — OpenAI-Endpunkt, der kein Modell permanent im RAM haelt.

Kompakter Gegenentwurf zum Dashboard fuer Hintergrund-Aufgaben (z. B. Hermes-Compaction):
  * Aussen: http://127.0.0.1:8082/v1  (Model-ID "primary", Key beliebig)
  * Beim ersten Request wird mlx_lm.server mit dem 9B-Fallback auf Port 8090
    gestartet (laedt aus NVMe, ~10-30 s).
  * Bleibt ein Request laenger als IDLE_TIMEOUT aus, wird der Server beendet
    und der RAM freigegeben.

Start:  .venv/bin/python mlx_ondemand.py
Notiz:  Nicht parallel zum Dashboard laufen lassen (Port 8082 doppelt belegt).
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = str(ROOT / "models" / "Qwen3.8-9B-Abliterated-MLX" / "4bit")
PYTHON = str(ROOT / ".venv" / "bin" / "python")
INNER_PORT = int(os.environ.get("ONDEMAND_INNER_PORT", "8090"))
OUTER_PORT = int(os.environ.get("ONDEMAND_PORT", "8082"))
IDLE_TIMEOUT = int(os.environ.get("ONDEMAND_IDLE", "120"))   # s ohne Request -> entladen
SERVER_MAX_TOKENS = os.environ.get("SERVER_MAX_TOKENS", "8192")

_lock = threading.Lock()
_proc: subprocess.Popen | None = None
_last_hit = 0.0
_active = 0            # laufende Requests — Modell entlaedt nie mid-request


def log(msg: str) -> None:
    print(time.strftime("[%H:%M:%S]"), msg, flush=True)


def _port_up(port: int, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            s.settimeout(1)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def _stop() -> None:
    global _proc
    if _proc and _proc.poll() is None:
        _proc.terminate()
        try:
            _proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _proc.kill()
    _proc = None


def _ensure_server() -> None:
    global _proc, _last_hit
    _last_hit = time.time()
    with _lock:
        if _proc and _proc.poll() is None and _port_up(INNER_PORT, timeout=2):
            return
        _stop()
        log(f"starte mlx_lm.server (9B) auf :{INNER_PORT} — lade Modell…")
        env = {**os.environ,
               "FALLBACK_MODEL_PATH": MODEL,
               "FALLBACK_PORT": str(INNER_PORT),
               "SERVER_MAX_TOKENS": SERVER_MAX_TOKENS}
        _proc = subprocess.Popen(
            [PYTHON, str(ROOT / "mlx_server_fallback_wrapper.py")],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if not _port_up(INNER_PORT, timeout=300):
            _stop()
            raise RuntimeError("innerer MLX-Server nicht erreichbar")
        log("Modell bereit.")


def _idle_watch() -> None:
    while True:
        time.sleep(5)
        if _active > 0:          # Request laeuft noch — nicht entladen
            continue
        with _lock:
            if _proc and _proc.poll() is None and time.time() - _last_hit > IDLE_TIMEOUT:
                log(f"Idle > {IDLE_TIMEOUT}s — Modell entladen, RAM frei.")
                _stop()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"    # noetig fuer Chunked-Transfer (SSE)

    def _send(self, code: int, payload: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_chunk(self, payload: bytes) -> None:
        self.wfile.write(("%x\r\n" % len(payload)).encode() + payload + b"\r\n")
        self.wfile.flush()

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            self._send(200, json.dumps({"object": "list", "data": [
                {"id": "primary", "object": "model", "created": 0,
                 "owned_by": "mlx-ondemand"}]}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send(404, b'{"error":"not found"}')
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        except (KeyError, json.JSONDecodeError):
            self._send(400, b'{"error":"bad json"}')
            return
        body["model"] = MODEL          # Router-ID -> echten Pfad ersetzen
        # stream=True MUSS als SSE (text/event-stream) durchgereicht werden —
        # Hermes liest Compaction-Antworten als Stream; normal JSON => leerer Inhalt.
        stream = bool(body.get("stream", False))
        # Qwen3 wuerde max_tokens in "reasoning" verbrennen und content leer lassen.
        ctk = body.get("chat_template_kwargs") or {}
        ctk.setdefault("enable_thinking", False)
        body["chat_template_kwargs"] = ctk
        global _active
        _active += 1
        headers_sent = False
        try:
            _ensure_server()
            req = urllib.request.Request(
                f"http://127.0.0.1:{INNER_PORT}/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=1800) as r:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    headers_sent = True
                    read = getattr(r, "read1", None) or (lambda n: r.read(1))
                    while True:
                        chunk = read(65536)
                        if not chunk:
                            break
                        self._send_chunk(chunk)
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                else:
                    self._send(r.status, r.read())
        except Exception as e:  # noqa: BLE001 — Endpoint soll antworten, nie haengen
            log(f"FEHLER: {e}")
            if not headers_sent:
                try:
                    self._send(502, json.dumps({"error": str(e)}).encode())
                except OSError:
                    pass
        finally:
            global _last_hit
            _last_hit = time.time()
            _active -= 1

    def log_message(self, format, *args):  # Request-Zeilen stumm, eigene log() reicht
        pass


if __name__ == "__main__":
    threading.Thread(target=_idle_watch, daemon=True).start()
    log(f"On-Demand-Router auf :{OUTER_PORT}  (Modell: {MODEL}, Idle: {IDLE_TIMEOUT}s)")
    try:
        ThreadingHTTPServer(("127.0.0.1", OUTER_PORT), Handler).serve_forever()
    finally:
        _stop()
