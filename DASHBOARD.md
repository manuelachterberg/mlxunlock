# MLX Dashboard – Auto-Router mit zwei MLX-Modellen

Dieses Dashboard routed zwischen zwei lokalen `mlx_lm.server`-Instanzen und hält standardmäßig nur eine davon im Speicher:

- **Primary:** automatisch erkanntes Modell auf Port `8080`
- **Fallback:** automatisch erkanntes zweites MLX-Modell auf Port `8081`, wird bei Bedarf geladen
- **Proxy:** Ein OpenAI-kompatibler Client spricht den Router auf Port `8082` an

## Dateien

| Datei | Zweck |
| --- | --- |
| `mlx_dashboard.py` | Dashboard + Proxy + Router |
| `mlx_server_safe_wrapper.py` | Startet den 27B-Server auf Port 8080 |
| `mlx_server_fallback_wrapper.py` | Startet den Fallback-Server auf Port 8081 |

## Konfiguration

Die Modellpfade werden beim Start automatisch erkannt. Mit `K` öffnest du die Konfigurations-TUI. Gespeicherte Werte liegen in `router_config.json` und werden beim nächsten Start geladen:

```python
MODEL_ROOT = "./models"
PRIMARY_PORT = 8080
PROXY_PORT = 8082

FALLBACK_TYPE = "mlx_lm"
FALLBACK_PORT = 8081
FALLBACK_MEMORY_LIMIT = "Vorschlag aus dem Startscan"
AUTO_START_FALLBACK = False
LAZY_FALLBACK = True

REASONING_EFFORT_27B = "low"
TOKEN_LIMIT_27B = "Vorschlag aus dem Startscan"
```

> Das Fallback-Modell wird beim ersten Start automatisch von Hugging Face geladen. Wenn du ein lokales Modell bevorzugst, setze hier den lokalen Pfad.

## Starten

```bash
source .venv/bin/activate
python mlx_dashboard.py
```

Das Dashboard startet automatisch:
1. 27B-Server auf Port 8080
2. Proxy auf Port 8082

Der Fallback-Server wird erst gestartet, wenn das Routing eine Anfrage dorthin schickt. Dafür wird der 27B-Server zuerst beendet. Beim Rückwechsel wird der Fallback beendet und der 27B-Server wieder geladen. Der erste Request nach einem Wechsel wartet daher auf den Modellstart.

## API-Client einstellen

Admin Panel → Settings → Connections → OpenAI API:

- **URL:** `http://<dein-macbook-ip>:8082/v1`
- **Key:** `dummy` (oder beliebig)

Stelle im Client die Anzahl paralleler Requests auf `1`. Der Router serialisiert Chat-Anfragen zusätzlich selbst, damit mehrere Anfragen nicht gleichzeitig GPU-Speicher und KV-Caches anfordern.

## Hotkeys im Dashboard

| Taste | Aktion |
| --- | --- |
| `Q` | Beenden |
| `R` | 27B-Server neu starten |
| `F` | Fallback-Server neu starten |
| `S` | Force Fallback / Auto-Routing umschalten |
| `C` | Logs löschen |
| `P` | Statistik in Datei speichern |
| `K` | Konfiguration öffnen |
| `A` | Auto-Routing |
| `1` | Immer Primary |
| `2` | Immer Fallback |
| `X` | Fallback starten/stoppen |
| `Z` | Primary starten/stoppen |

## Wie das Routing funktioniert

| Situation | Entscheidung | Anzeige |
| --- | --- | --- |
| Kurze Frage, Primary läuft | → Primary | ▶ PRIMARY |
| Langer Prompt > 3000 Tokens | → Fallback | ▶ FALLBACK (5000t > 3000t limit) |
| SWAP > 5 GB | → Fallback | ▶ FALLBACK (SWAP 6.2GB > 5GB) |
| 27B crashed/unhealthy | → Fallback | ▶ FALLBACK (27B crashed) |
| Du drückst `S` | → Fallback | ▶ FALLBACK (FORCED) |

Der Proxy startet den Fallback bei aktiviertem `LAZY_FALLBACK` automatisch. Ohne konfiguriertes Fallback-Modell routed er immer zu 27B.

## Thinking-Stärke steuern

Qwen3.8-27B verwendet standardmäßig `xhigh`-Reasoning und erzeugt sehr lange interne Denkblöcke. Der Router injiziert automatisch `reasoning_effort` in alle `/v1/chat/completions`-Requests an das 27B-Modell:

```python
REASONING_EFFORT_27B = "low"   # kurzes Thinking
# REASONING_EFFORT_27B = "medium"
# REASONING_EFFORT_27B = "xhigh"  # Modell-Default
# REASONING_EFFORT_27B = None     # nicht injizieren
```

Werte: `xhigh` | `medium` | `low` | `None`

## Manuelles Fallback (statt Auto-Start)

Wenn du den Fallback-Server selbst starten willst (z. B. weil das Modell woanders läuft):

```python
AUTO_START_FALLBACK = False
```

Dann startest du ihn manuell:

```bash
python -m mlx_lm.server --model ./models/Qwen3.8-7B-MLX/4-bit --host 0.0.0.0 --port 8081
```

Der Router erkennt ihn trotzdem, sobald er auf Port 8081 antwortet.
