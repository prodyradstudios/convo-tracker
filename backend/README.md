# Convo Tracker — backend

FastAPI server that enrolls voices, then labels who said what in a live
conversation (VAD → speaker embedding → cosine match → speech-to-text).
The API contract lives in [`../docs/SPEC.md`](../docs/SPEC.md).

## Modules (`app/`)

| Module | Purpose |
|---|---|
| `main.py` | `create_app(settings=None, *, embedder=None, transcriber=None)` factory + module-level `app`. Builds state in the lifespan handler, mounts `/api`, `/ws/session`, then serves `../frontend` at `/`. |
| `__main__.py` | `python -m app` — runs uvicorn on `CT_HOST`:`CT_PORT`. |
| `config.py` | `Settings` dataclass read from `CT_*` env vars. |
| `state.py` | `AppState` (settings, embedder, transcriber, registry), built once. |
| `api/speakers.py` | REST enrollment routes (`/api/health`, `/api/speakers…`). |
| `api/session.py` | WebSocket `/ws/session`: receive loop + one worker task per connection; ML work runs in `asyncio.to_thread`. |
| `audio/io.py` | WAV decode (any rate/channels → 16 kHz mono float32), PCM16 helpers. |
| `audio/vad.py` | Streaming energy VAD / utterance `Segmenter`, `trim_silence`. |
| `speakers/embedder.py` | `Embedder` protocol, `EcapaEmbedder` (SpeechBrain, lazy), `MfccEmbedder` (numpy/scipy). |
| `speakers/registry.py` | Thread-safe `SpeakerRegistry`, cosine identification with continuity smoothing, atomic JSON persistence. |
| `transcription/transcriber.py` | `WhisperTranscriber` (faster-whisper, lazy) and `NullTranscriber`. |
| `pipeline.py` | `SessionPipeline`: segment → `Utterance`. |

## Setup

```bash
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements-dev.txt          # core + tests (no ML)
# optional, real models (CPU wheels):
../.venv/bin/pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Without the ML extras the server still starts (auto mode) using the weak
MFCC embedder and placeholder transcripts (`[speech 2.3s]`), logging a warning.
Models (ECAPA, Whisper) are downloaded on first start; ECAPA is cached under
`$CT_DATA_DIR/models`, Whisper in the Hugging Face cache.

## Run

```bash
cd backend
../.venv/bin/python -m app                         # http://127.0.0.1:8000
# or, with auto-reload:
../.venv/bin/uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 (the browser only allows the microphone on
`localhost`/`127.0.0.1` or HTTPS).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `CT_DATA_DIR` | `./data` | Where `speakers.json` (and the ECAPA model) are stored. |
| `CT_EMBEDDER` | `auto` | `auto` \| `ecapa` \| `mfcc` (auto → ecapa if importable). |
| `CT_TRANSCRIBER` | `auto` | `auto` \| `whisper` \| `null`. |
| `CT_MATCH_THRESHOLD` | embedder default (ecapa 0.30, mfcc 0.80) | Minimum cosine similarity to label a speaker. |
| `CT_SILENCE_MS` | `700` | Silence that ends an utterance. |
| `CT_MAX_UTTERANCE_S` | `15` | Force-cut long utterances. |
| `CT_MIN_UTTERANCE_S` | `0.5` | Drop utterances with less speech than this. |
| `CT_WHISPER_MODEL` | `base.en` | faster-whisper model name. |
| `CT_DEVICE` | `cpu` | `cpu` or `cuda` (int8 on CPU, float16 otherwise). |
| `CT_HOST` / `CT_PORT` | `127.0.0.1` / `8000` | Used by `python -m app`. |

## Test

```bash
cd backend
../.venv/bin/python -m pytest -q
```

Tests need only `requirements-dev.txt`; they use fake/MFCC embedders and
fake/null transcribers, synthetic audio, and FastAPI's `TestClient`
(REST and WebSocket).
