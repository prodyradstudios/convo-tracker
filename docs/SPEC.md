# Convo Tracker — Build Spec (v1, enrollment-first)

This file is the contract between the frontend and backend. Both sides must follow it exactly.

## Repo layout

```
backend/
  app/
    __init__.py
    main.py                 # FastAPI app factory, mounts /api, /ws, serves ../frontend at /
    config.py               # Settings dataclass read from env vars (CT_*)
    state.py                # AppState: holds registry, embedder, transcriber (built once at startup)
    api/__init__.py
    api/speakers.py         # REST routes for enrollment
    api/session.py          # WebSocket route /ws/session
    audio/__init__.py
    audio/io.py             # decode WAV bytes -> float32 mono 16 kHz numpy; pcm16 bytes -> float32
    audio/vad.py            # streaming energy VAD + utterance segmenter
    speakers/__init__.py
    speakers/embedder.py    # Embedder protocol, EcapaEmbedder (SpeechBrain), MfccEmbedder (numpy-only fallback)
    speakers/registry.py    # SpeakerRegistry: enroll, delete, list, identify (cosine similarity), JSON persistence
    transcription/__init__.py
    transcription/transcriber.py  # Transcriber protocol, WhisperTranscriber (faster-whisper), NullTranscriber
    pipeline.py             # SessionPipeline: VAD -> embed -> identify -> transcribe -> Utterance
  tests/                    # pytest; must run with NO ML packages installed (use mfcc + null/fake transcriber)
  requirements.txt          # core: fastapi, uvicorn[standard], numpy, scipy, soundfile, python-multipart
  requirements-ml.txt       # -r requirements.txt + faster-whisper, speechbrain, torch, torchaudio
  requirements-dev.txt      # -r requirements.txt + pytest, httpx
  pyproject.toml            # pytest config (testpaths, pythonpath)
frontend/
  index.html
  styles.css
  js/app.js                 # UI wiring
  js/api.js                 # fetch wrappers for REST
  js/audio.js               # mic capture via AudioWorklet, emits 16 kHz Int16 chunks
  js/pcm-worklet.js         # AudioWorkletProcessor: downsample to 16 kHz mono, post Int16Array chunks
  js/wav.js                 # encode Int16 PCM -> WAV Blob (16 kHz mono 16-bit)
  js/session.js             # WebSocket client for /ws/session
docs/SPEC.md
README.md
```

## Audio format (everywhere)

16 000 Hz, mono, signed 16-bit little-endian PCM. The browser does all resampling.
Backend works internally in float32 in [-1, 1].

## REST API (JSON unless noted). Prefix `/api`.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/api/health` | – | `{"status":"ok","embedder":"ecapa"|"mfcc","transcriber":"whisper"|"null"}` |
| GET | `/api/speakers` | – | `{"speakers":[Speaker...]}` |
| POST | `/api/speakers` | multipart/form-data: `name` (str, 1–40 chars after strip), `audio` (WAV file) | `201` + `Speaker` |
| POST | `/api/speakers/{id}/samples` | multipart: `audio` (WAV) — adds another sample to an existing speaker | `200` + `Speaker` |
| DELETE | `/api/speakers/{id}` | – | `204` |
| DELETE | `/api/speakers` | – | `204` (remove all) |

`Speaker` = `{"id": str (uuid4 hex), "name": str, "color": str ("#rrggbb"), "num_samples": int, "created_at": float (unix seconds)}`

Errors: `{"detail": str}` with status 400 (bad name, unreadable audio, too short: < 2.0 s of speech after trimming silence), 404 (unknown id), 409 (duplicate name, case-insensitive), 413 (upload > 10 MB or > 60 s).

Colors are assigned server-side from a fixed palette of 8 distinct colors, cycling.

## WebSocket `/ws/session`

Client → server:
- Text frame `{"type":"start"}` — begin a session (resets segmenter). Server replies `{"type":"status","state":"listening"}`.
- Binary frames — raw PCM16LE 16 kHz mono, any length (browser sends ~100 ms chunks). Ignored unless started. Odd byte counts: drop the trailing byte.
- Text frame `{"type":"stop"}` — flush any in-progress utterance, finish processing queued ones, then reply `{"type":"status","state":"stopped"}`.

Server → client:
- `{"type":"status","state":"listening"|"stopped"}`
- `{"type":"speech","active":true|false}` — VAD speech start/end (for a "someone is talking" indicator).
- `{"type":"utterance","id":str,"seq":int,"speaker_id":str|null,"speaker_name":str,"color":str,"text":str,"start":float,"end":float,"confidence":float}`
  - `seq` starts at 1 per session and increases in order. `start`/`end` are seconds since session start.
  - `speaker_id` null and `speaker_name` "Unknown" (color `#888888`) when no enrolled speaker scores >= threshold.
  - `confidence` = best cosine similarity, rounded to 3 decimals.
  - Utterances whose transcript is empty/whitespace are NOT sent.
- `{"type":"error","message":str}`

Utterances must be emitted in seq order. Heavy work (embedding, transcription) runs off the event loop
(`asyncio.to_thread` / a single worker per session) so audio reception never blocks.

## Segmentation (audio/vad.py)

Frame = 30 ms (480 samples). Per-frame RMS energy in dBFS. Adaptive noise floor (slow EMA of non-speech frames,
initialised from the first 300 ms). Speech if energy > max(noise_floor + 10 dB, -50 dBFS).
- Utterance starts after 3 consecutive speech frames; keep 300 ms pre-roll.
- Utterance ends after `CT_SILENCE_MS` (default 700) ms of non-speech; keep 200 ms tail.
- Force-cut at `CT_MAX_UTTERANCE_S` (default 15) s.
- Drop utterances shorter than `CT_MIN_UTTERANCE_S` (default 0.5) s.
Segmenter API: `feed(samples: np.ndarray) -> list[Segment]`, `flush() -> list[Segment]`, plus events for speech start/end.
`Segment(audio: np.ndarray, start: float, end: float)`.

## Speaker identification

- `Embedder.embed(audio: np.ndarray) -> np.ndarray` (1-D float32, L2-normalised). `Embedder.name`, `Embedder.default_threshold`.
- `EcapaEmbedder`: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb` (192-d). default_threshold 0.30. Lazy import.
- `MfccEmbedder`: numpy/scipy only: 40 log-mel → 20 MFCC + deltas, mean and std pooled, CMN; ~80-d. default_threshold 0.80 (it is weaker; used for tests/offline only).
- Selection: `CT_EMBEDDER=auto|ecapa|mfcc` (auto → ecapa if importable else mfcc, log a warning).
- Enrollment: trim silence with the VAD, require >= 2.0 s speech, embed in 3 s windows with 1.5 s hop, average, re-normalise. Each speaker keeps a list of sample embeddings; centroid = normalised mean.
- Identify: cosine similarity to each centroid; best >= threshold (`CT_MATCH_THRESHOLD`, default = embedder.default_threshold) → that speaker, else Unknown. Utterances < 1.0 s: still identify, but if best - second_best < 0.05 and there was a previous identified speaker within that pair, prefer the previous speaker (continuity smoothing).
- Persistence: `CT_DATA_DIR` (default `./data`), `speakers.json` storing id, name, color, created_at, embedder name, sample embeddings as lists. On load, discard speakers whose embedder name differs from the active one (log warning). Writes are atomic (tmp + os.replace). Registry is thread-safe (threading.Lock).

## Transcription

- `Transcriber.transcribe(audio: np.ndarray) -> str`.
- `WhisperTranscriber`: faster-whisper, model `CT_WHISPER_MODEL` (default `base.en`), device `CT_DEVICE` (default `cpu`), compute_type int8 on cpu, `beam_size=1`, `vad_filter=False`, `condition_on_previous_text=False`. Lazy import; model loaded once at startup.
- `NullTranscriber`: returns `"[speech {duration:.1f}s]"` (used in tests/offline when faster-whisper missing).
- Selection: `CT_TRANSCRIBER=auto|whisper|null`.

## Frontend behaviour

- Plain HTML/CSS/vanilla ES modules. No build step, no frameworks. Served by FastAPI at `/`.
- Panel 1 "People": name input + Record button. Record → button becomes "Stop (0:07)" with live timer and a level meter; auto-stops at 15 s; requires >= 3 s before Stop is enabled. On stop, encode WAV and POST `/api/speakers`. Show errors inline. Below: list of enrolled people (color dot, name, sample count, "+ sample" button, delete ✕).
- Panel 2 "Conversation": Start / Stop listening button (disabled with a hint while 0 people enrolled), speech indicator, Clear and Export (.txt download) buttons. Feed of lines: colored name chip, text, mm:ss timestamp, confidence shown as tooltip. Auto-scrolls to bottom unless user scrolled up.
- Mic: `getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true}})`. One shared AudioContext; worklet posts Int16 chunks of 1600 samples (100 ms at 16 kHz).
- WebSocket URL derived from `location` (ws/wss). Handle close/error: show banner, reset button state.
- Accessible: buttons have labels, feed is `aria-live="polite"`.
