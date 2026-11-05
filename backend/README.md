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
| `speakers/embedder.py` | `Embedder` protocol, `EcapaEmbedder` (SpeechBrain, lazy, batched), `MfccEmbedder` (numpy/scipy); loudness normalisation; enrollment augmentation (speed/pitch perturbation). |
| `speakers/registry.py` | Thread-safe `SpeakerRegistry`: prototype scoring (centroid / top-2 prototypes), continuity smoothing, in-memory adaptive prototypes, atomic JSON persistence. |
| `speakers/turns.py` | `analyze_turns` / `detect_turns`: chunk pre-check, then sliding-window speaker-change detection inside one segment. |
| `transcription/transcriber.py` | `WhisperTranscriber` (faster-whisper, lazy; text or word timestamps) and `NullTranscriber`. |
| `pipeline.py` | `SessionPipeline`: segment → turns → one `Utterance` per speaker turn. |

## Setup

```bash
python3 -m venv ../.venv
../.venv/bin/pip install -r requirements-dev.txt          # real transcription + tests
# optional, stronger ECAPA speaker recognition (CPU wheels):
../.venv/bin/pip install -r requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

Without the ECAPA extras the server uses the basic MFCC speaker matcher and
real Whisper transcripts. `auto` and `whisper` both load real speech-to-text;
missing packages or unavailable models stop startup with an actionable error.
Only explicit `CT_TRANSCRIBER=null` emits `[speech 2.3s]` placeholders, with a
warning in the terminal and UI. Remove that override or set it to `whisper`
to restore spoken words.
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
| `CT_TRANSCRIBER` | `auto` | `auto` / `whisper`: real speech-to-text, required to load successfully. `null`: explicit development placeholders only. |
| `CT_MATCH_THRESHOLD` | embedder default (ecapa 0.30, mfcc 0.80) | Minimum cosine similarity to label a speaker. |
| `CT_SILENCE_MS` | `700` | Silence that ends an utterance. |
| `CT_MAX_UTTERANCE_S` | `15` | Force-cut long utterances. |
| `CT_MIN_UTTERANCE_S` | `0.5` | Drop utterances with less speech than this. |
| `CT_WHISPER_MODEL` | `base.en` | faster-whisper model name or a downloaded model directory (offline). |
| `CT_DEVICE` | `cpu` | `cpu` or `cuda` (int8 on CPU, float16 otherwise). |
| `CT_TURN_MIN_SEGMENT_S` | `2.0` | Look for speaker changes only in segments with at least this much speech (>= 0). |
| `CT_TURN_WINDOW_S` | `1.5` | Window length for speaker-change detection (0.25–10 s). |
| `CT_TURN_HOP_S` | `0.5` | Window hop (> 0, <= window; grows to keep at most 40 windows per segment). |
| `CT_TURN_MIN_S` | `1.0` | Coarse smoothing merges shorter turns (>= 0); confirmed rapid changes are preserved. |
| `CT_TURN_CHANGE_WINDOW_S` | `0.4` | Short audio window for rapid speaker-change evidence (0.2–2 s). |
| `CT_TURN_CHANGE_HOP_S` | `0.1` | Probe interval (0.03 s to change-window length). |
| `CT_TURN_CHANGE_CONFIRM_S` | `0.2` | Sustained alternate-speaker evidence required before switching (hop to 2 s; at least two probes). |
| `CT_TURN_CHANGE_MARGIN` | `0.15` | Required lead over competing enrolled voices; also separates Unknown evidence from ambiguous matches (0–2). |
| `CT_TURN_CHANGE_THRESHOLD` | `0.35` | Minimum cosine distance from the active voice before switching (0–2). Lower values switch more readily but risk false interruptions. |
| `CT_ENROLL_AUGMENT` | `1` | `1`/`0`: also store speed- and pitch-perturbed embeddings of each enrollment sample. |
| `CT_ADAPT` | `1` | `1`/`0`: learn in-memory prototypes from very confident live speech (max 20 per person). |
| `CT_HOST` / `CT_PORT` | `127.0.0.1` / `8000` | Used by `python -m app`. |

## Test

Speaker changes are checked while audio arrives, without waiting for the silence
timeout. A confirmed change cuts the active audio, queues the old turn for
transcription, and sends a live `speaker` event for the incoming voice. The UI
shows that name before the final transcript is available. Short, confidently
different turns are protected from the conservative coarse turn smoothing.

The default detector uses 400 ms windows and 100 ms hops, then requires 200 ms
of confirming evidence. These are audio evidence settings, not a wall-clock
latency guarantee: model inference and transcription add processing time.
Cheap acoustic checks reduce model calls, with periodic full-model probes to
avoid missing changes that sound similar to the cheap checker. Short-window
probes never adapt enrolled voiceprints. Simultaneous voices are still assigned
to the dominant voice; this does not separate overlapping audio sources.

```bash
cd backend
../.venv/bin/python -m pytest -q
```

Tests need only `requirements-dev.txt` and do not download models; they use fake/MFCC embedders and
fake/null transcribers, synthetic audio, and FastAPI's `TestClient`
(REST and WebSocket).
