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
    speakers/registry.py    # SpeakerRegistry: enroll, delete, list, identify (prototype scoring), JSON persistence
    speakers/turns.py       # analyze_turns: speaker-change detection inside one segment (chunk pre-check + sliding windows)
    transcription/__init__.py
    transcription/transcriber.py  # Transcriber protocol, WhisperTranscriber (faster-whisper), NullTranscriber
    pipeline.py             # SessionPipeline: VAD -> turns -> embed -> identify -> transcribe -> Utterances
  tests/                    # pytest; no model downloads required (use mfcc + null/fake transcriber)
  requirements.txt          # core: fastapi, uvicorn[standard], numpy, scipy, soundfile, python-multipart
  requirements-ml.txt       # -r requirements.txt + speechbrain, torch, torchaudio
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

Errors: `{"detail": str}` with status 400 (bad name, unreadable audio or sample rate outside 4–384 kHz, too short: < 2.0 s of speech after trimming silence — counted from VAD speech onset to last speech frame, excluding pre-roll/tail padding), 404 (unknown id), 409 (duplicate name, case-insensitive), 413 (upload > 10 MB or > 60 s; a request whose body exceeds 10 MB + 64 KB is refused before it is read).

Colors are assigned server-side from a fixed palette of 8 distinct colors, cycling.

Speech-to-text is included in the standard runtime dependencies. `CT_TRANSCRIBER=auto`
and `whisper` both load faster-whisper; missing packages, inaccessible model
downloads, and model-load errors fail startup with an actionable error. They do
not silently fall back to duration text. Only explicit `CT_TRANSCRIBER=null`
selects the development placeholder, with a terminal warning and UI banner.
`CT_WHISPER_MODEL` defaults to `base.en` and also accepts a downloaded model
directory for offline operation. Tests inject fake models and require no model
downloads.

## WebSocket `/ws/session`

Client → server:
- Text frame `{"type":"start"}` — begin a session (resets segmenter). Server replies `{"type":"status","state":"listening"}`.
  A `start` while already listening ends the current run without a `stopped`: its in-progress utterance is flushed (with `speech` `false` if needed) and its results are sent before the new `listening`.
- Binary frames — raw PCM16LE 16 kHz mono, any length (browser sends ~100 ms chunks). Ignored unless started. Odd byte counts: drop the trailing byte.
- Text frame `{"type":"stop"}` — flush any in-progress utterance, finish processing queued ones, then reply `{"type":"status","state":"stopped"}`.

Server → client:
- `{"type":"status","state":"listening"|"stopped"}`
- `{"type":"speech","active":true|false}` — VAD speech start/end (for a "someone is talking" indicator).
- `{"type":"speaker","speaker_id":str|null,"speaker_name":str,"color":str,"confidence":float,"start":float}` — live speaker identification or a confirmed change. `start` is the estimated turn boundary in session seconds. The client updates its speaking indicator immediately; this event contains no transcript text and does not consume an utterance sequence number. Final text follows in `utterance` messages after transcription.
- `{"type":"utterance","id":str,"seq":int,"speaker_id":str|null,"speaker_name":str,"color":str,"text":str,"start":float,"end":float,"confidence":float}`
  - `seq` starts at 1 per session and increases in order. `start`/`end` are seconds since session start.
  - `speaker_id` null and `speaker_name` "Unknown" (color `#888888`) when no enrolled speaker scores >= threshold.
  - `confidence` = best cosine similarity, rounded to 3 decimals.
  - Utterances whose transcript is empty/whitespace are NOT sent.
  - One VAD segment can yield several utterances when the speaker changes without a pause
    (see "Speaker turns"); they are sent consecutively, in time order, each with its own `id` and `seq`.
- `{"type":"error","message":str}`
  - If more than 40 utterances are waiting to be processed, the oldest waiting one is dropped and `{"type":"error","message":"Falling behind: dropped an utterance"}` is sent.

Utterances must be emitted in seq order. Heavy work (embedding, transcription) runs off the event loop
(`asyncio.to_thread` / a single worker per session) so audio reception never blocks.

## Segmentation (audio/vad.py)

Frame = 30 ms (480 samples). Per-frame RMS energy in dBFS. Adaptive noise floor (slow EMA of non-speech frames,
initialised from the first 300 ms). Speech if energy > max(noise_floor + 10 dB, -50 dBFS).
If every frame of a 5 s window counts as speech, the floor is raised to that window's quietest frame (the background got louder).
- Utterance starts after 3 consecutive speech frames; keep 300 ms pre-roll.
- Utterance ends after `CT_SILENCE_MS` (default 700) ms of non-speech; keep 200 ms tail.
- Force-cut at `CT_MAX_UTTERANCE_S` (default 15) s.
- Drop utterances shorter than `CT_MIN_UTTERANCE_S` (default 0.5) s.
Segmenter API: `feed(samples: np.ndarray) -> list[Segment]`, `flush() -> list[Segment]`, plus events for speech start/end.
`Segment(audio: np.ndarray, start: float, end: float, speech_duration: float | None = None)`; `speech_duration` is onset → last speech frame
(no pre-roll/tail) and is what the < 1.0 s continuity-smoothing rule uses.

## Speaker identification

- `Embedder.embed(audio: np.ndarray) -> np.ndarray` (1-D float32, L2-normalised). `Embedder.name`, `Embedder.default_threshold`.
  Optional `Embedder.embed_batch(list[np.ndarray]) -> np.ndarray` (one row per clip); `embed_many()` uses it when present, else loops `embed`.
- **Loudness normalisation** (shared `prepare_audio()`, used by every real embedder for enrollment AND live audio): the RMS of the
  voiced 30 ms frames (within 30 dB of the loudest frame) is scaled to -23 dBFS, the gain is limited so no sample exceeds 0.99
  (no clipping) and to +60 dB; near-silent input (frame RMS < 1e-5) is left unchanged.
- `EcapaEmbedder`: SpeechBrain `speechbrain/spkrec-ecapa-voxceleb` (192-d). default_threshold 0.30. Lazy import.
  `embed_batch` zero-pads clips to the longest and passes relative `wav_lens` to `encode_batch`.
- `MfccEmbedder`: numpy/scipy only: 40 log-mel → 20 MFCC + deltas, mean and std pooled, CMN; ~80-d. default_threshold 0.80 (it is weaker; used for tests/offline only).
- Selection: `CT_EMBEDDER=auto|ecapa|mfcc` (auto → ecapa if importable else mfcc, log a warning).
- Enrollment: trim silence with the VAD, require >= 2.0 s speech, embed in 3 s windows with 1.5 s hop, average, re-normalise. Each speaker keeps a list of sample embeddings; centroid = normalised mean of the real samples.
- **Enrollment augmentation** (`CT_ENROLL_AUGMENT=1|0`, default 1): each sample is also embedded (same 3 s windows) after
  speed perturbation with `scipy.signal.resample_poly` at factors 0.9 and 1.1 (pitch and tempo change by about -1.8 / +1.65 semitones),
  and, when torchaudio is importable, after a pure pitch shift of ±2 semitones (phase-vocoder time stretch via
  `torchaudio.functional.phase_vocoder`, then `resample_poly`; the same algorithm as `torchaudio.functional.pitch_shift`, whose own
  resampler takes many seconds for these ratios). A pitch-shift failure is logged and skipped. These embeddings are stored with
  their sample as `augmented`; `num_samples` counts real samples only.
- **Scoring** against each speaker: `score = max(cos(u, centroid), mean of the top-2 cos(u, p) over the speaker's prototypes)`,
  where prototypes = real sample embeddings + their augmented embeddings + adaptive prototypes. (With one sample and no extras this is
  plain cosine to the centroid.) The **static** score is the same without adaptive prototypes; adaptive prototypes only count for a
  speaker whose static score is already >= threshold (they can never turn an Unknown voice into a match).
- Identify: best score >= threshold (`CT_MATCH_THRESHOLD`, default = embedder.default_threshold) → that speaker, else Unknown.
  `confidence` = that score. Utterances < 1.0 s: still identify, but if best - second_best < 0.05 and there was a previous identified
  speaker within that pair, prefer the previous speaker (continuity smoothing).
- **Adaptive prototypes** (`CT_ADAPT=1|0`, default 1): a live utterance with >= 2.0 s of speech identified with
  **static** score >= threshold + 0.15 AND at least 0.10 above the second-best speaker's static score (with only one speaker enrolled:
  static score >= threshold + 0.25), AND cos(embedding, centroid) >= threshold adds its embedding as an adaptive prototype of that speaker.
  Only whole, unsplit segments whose turn analysis was clean (every chunk/window had the same label) adapt; split pieces never do.
  In memory only (never persisted), FIFO capped at 20 per speaker, thread-safe, dropped when the speaker is deleted (or all are removed).
- Persistence: `CT_DATA_DIR` (default `./data`), `speakers.json` storing id, name, color, created_at, embedder name, sample embeddings as
  lists and `augmented` (one list of embeddings per sample). Files without `augmented` (older versions) load fine; an entry whose
  `augmented` is malformed (wrong count, size, non-finite or zero vectors) is skipped like any other invalid entry. On load, discard
  speakers whose embedder name differs from the active one (log warning). Writes are atomic (tmp + os.replace). Registry is thread-safe (threading.Lock).

## Speaker turns (splitting a segment when the speaker changes without a pause)

### Online handoff

While audio arrives, a separate short-window detector checks active speech using
`CT_TURN_CHANGE_WINDOW_S` (default 0.4 s) and `CT_TURN_CHANGE_HOP_S` (0.1 s).
A switch needs a different confidently enrolled identity (or confidently Unknown),
a cosine distance of at least `CT_TURN_CHANGE_THRESHOLD` (0.35) from the active
voice, and sustained evidence for `CT_TURN_CHANGE_CONFIRM_S` (0.2 s, at least two
probes). `CT_TURN_CHANGE_MARGIN` (0.15) rejects ambiguous identities.
These controls apply independently of the coarse minimum segment/turn durations.

A cheap MFCC novelty check reduces full-model calls; periodic full-model probes
prevent the cheap check from indefinitely suppressing a genuine change. Candidate
changes bypass the cheap gate for confirmation. Short-window evidence never
adapts speaker voiceprints. Model work runs off the event loop.

Confirmed changes cut the active VAD audio at a refined acoustic boundary without
waiting for silence or sending a false speech-end event. The old turn is queued
for transcription and a `speaker` event announces the new live identity. The
suffix keeps its audio and a handoff marker so previous-speaker continuity cannot
pull it back into the old speaker. Confirmation and model compute add latency;
the configured audio windows are not a wall-clock latency guarantee.

### Completed-segment analysis

A bounded confidence-gated short-window pass protects clear brief interruptions
before the conservative coarse pass below. Strong changes require sustained,
contiguous evidence and a large embedding distance; they are not erased by the
coarse minimum-turn smoothing. A single enrolled speaker can transition to Unknown.

The coarse fallback runs for a segment with `speech_seconds >= CT_TURN_MIN_SEGMENT_S`
(default 2.0) when at least 2 speakers are enrolled.

1. Windows of `CT_TURN_WINDOW_S` (default 1.5 s) every `CT_TURN_HOP_S` (default 0.5 s), evenly spread over the segment audio;
   at most 40 windows (the hop grows if needed). Windows with fewer than 50 % voiced 30 ms frames are skipped. Voiced uses the VAD's
   absolute criterion: frame energy > `max(min_speech_db, min(noise floor, -40 dBFS) + margin_db)`, where the noise floor is the
   segmenter's (carried on the `Segment`) or, if unknown, the segment's 2nd-percentile frame energy. Nothing is relative to the
   loudest frame, so a speaker much quieter than the other still counts as voiced.
2. **Chunk pre-check**: first only the windows that tile the segment end to end (about one window apart, so about the segment's
   length of audio) are embedded, in one batch. If there are >= 2 and every one has the same best speaker by **static** score, at or
   above the threshold and >= 0.10 above the runner-up, the segment is one turn of that speaker and the normalised mean of the
   chunk embeddings is used as its whole-segment embedding (no further embedding). Otherwise the remaining windows are embedded
   (one more batch; chunk embeddings are reused) and scored with the registry's scoring. Each window gets its best speaker, or
   Unknown below the threshold.
3. Smoothing: a window whose two neighbours agree takes their label (majority over 3 windows; the first/last window takes the label
   of its two inner neighbours when they agree). Then, until stable: an Unknown run spanning <= `CT_TURN_WINDOW_S` between two
   *different* speakers (the boundary mixture) is split at its midpoint between them; otherwise the shortest run spanning
   <= `CT_TURN_MIN_S` (default 1.0 s) or holding a single window is merged into the neighbour whose speaker scores higher on that
   run's windows (an Unknown neighbour counts as scoring the threshold).
4. Turns `(start, end, speaker_id | None)` cover the segment; a boundary lies midway between the centres of the last window of one
   run and the first window of the next. Where two adjacent turns' median voiced levels differ by >= 10 dB, the boundary moves to
   the sharpest level step within half a window of it (the louder voice dominates mixed windows).

If there is one turn, the segment is handled as a single utterance (identity from the chunk-mean embedding, or a whole-segment
embedding when turn analysis did not run). Otherwise:
- `transcribe_words` returns the words with timestamps, `[]` when nothing was said, or `None` when the transcriber has no word
  timing (the null transcriber; a transcriber without the method counts as `None`; Whisper returns `None` if recognized text has
  no usable word metadata, including missing, invalid or reversed timestamps). This uses per-turn audio transcription rather than dropping recognized speech.
- With words: each word goes to the turn containing its midpoint (words sorted by midpoint; a non-finite time borrows the other
  end or the previous word's time); consecutive words of one turn form a piece whose `start`/`end` snap to its first/last word,
  clamped to the clip; overlapping pieces are cut at the turn boundary, and a piece with no length takes its turn's span. Pieces
  are in time order with `start < end`. Text is the words concatenated as given (Whisper words carry their own leading space; CJK
  words none). `[]`: no utterance for the segment.
- With `None`: the audio is cut at the turn boundaries and each piece is transcribed separately.
- Each piece of >= 1.0 s is re-identified from its turn's acoustic span, without previous-speaker continuity bias; shorter pieces keep their turn's label and score. Word-timestamp spans are clamped to their assigned turn so a mistimed word cannot pull another speaker's audio into re-identification.
  Empty-text pieces are dropped, then adjacent pieces with the same speaker (or both Unknown) are merged (confidence = duration-weighted
  mean). If every turn produced text and they all merge into one speaker (a false split), the segment is handled as unsplit.
- Each remaining piece is its own utterance with its own `seq`, in time order.

Overlapping speech (two people talking at the same time) is not separated: each window goes to whichever voice dominates it.

## Configuration added for turns and voice variation

| Variable | Default | Valid | Meaning |
|---|---|---|---|
| `CT_TURN_MIN_SEGMENT_S` | `2.0` | >= 0 | Only look for speaker changes in segments with at least this much speech. |
| `CT_TURN_WINDOW_S` | `1.5` | 0.25–10 | Window length for turn detection. |
| `CT_TURN_HOP_S` | `0.5` | > 0 and <= window | Window hop (grows to keep <= 40 windows). |
| `CT_TURN_MIN_S` | `1.0` | >= 0 | Shorter turns are merged into a neighbour. |
| `CT_ENROLL_AUGMENT` | `1` | 1/0 (true/false, yes/no, on/off) | Store speed/pitch-perturbed enrollment embeddings. |
| `CT_ADAPT` | `1` | 1/0 (true/false, yes/no, on/off) | Learn in-memory adaptive prototypes from confident live speech. |

Invalid values fail at startup with a `ValueError` naming the variable.

## Frontend behaviour

- Plain HTML/CSS/vanilla ES modules. No build step, no frameworks. Served by FastAPI at `/`.
- Panel 1 "People": name input + Record button. Record → button becomes "Stop (0:07)" with live timer and a level meter; auto-stops at 15 s; requires >= 3 s before Stop is enabled. On stop, encode WAV and POST `/api/speakers`. Show errors inline. Under the record button a tip suggests 2–3 samples per person
  (normal voice, louder/more animated, quieter) via "+ sample". Below: list of enrolled people (color dot, name, sample count, "+ sample" button, delete ✕).
- Panel 2 "Conversation": Start / Stop listening button (disabled with a hint while 0 people enrolled), speech indicator, Clear and Export (.txt download) buttons. Feed of lines: colored name chip, text, mm:ss timestamp, confidence shown as tooltip. Auto-scrolls to bottom unless user scrolled up.
- Mic: `getUserMedia({audio:{channelCount:1, echoCancellation:true, noiseSuppression:true, autoGainControl:true}})`. One shared AudioContext; worklet posts Int16 chunks of 1600 samples (100 ms at 16 kHz).
- WebSocket URL derived from `location` (ws/wss). Handle close/error: show banner, reset button state.
- Accessible: buttons have labels, feed is `aria-live="polite"`.
