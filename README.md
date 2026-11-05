# Convo Tracker

Tracks a conversation and labels who said what.

Version 1 enrolls voices first. Each person records a short voice sample and types their name. The app then listens to the room and writes a live transcript where every line is attributed to the closest enrolled voice. Lines that match no one well enough are labeled **Unknown**.

- **Backend** (`backend/`): Python, FastAPI. A streaming VAD splits the audio into utterances. SpeechBrain ECAPA voiceprints, compared by cosine similarity, identify the speaker, and faster-whisper transcribes the speech.
- **Frontend** (`frontend/`): plain HTML, CSS and JavaScript, with no build step. Mic audio is captured with an AudioWorklet, resampled to 16 kHz and streamed over a WebSocket.
- **Contract** (`docs/SPEC.md`): the REST and WebSocket API, audio format and algorithms.
- **Run guide** (`docs/run-system.md`): step-by-step local setup, configuration and troubleshooting.
- **Design guide** (`docs/BUILD_GUIDE.md`): architecture, components, algorithm and roadmap. Start here to iterate on the idea.

## Quick start

Requires **Python 3.11–3.13** (3.12 recommended). Check with `python3 --version`. On a Mac, `brew install python@3.12` and use `python3.12` below.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-ml.txt
cd backend && python -m app
```

Open <http://localhost:8000>. Browsers only allow the microphone on `localhost` or HTTPS.

1. In **People**, type a name, press **Record**, speak for about 8 seconds, and press **Stop**. Repeat this for everyone. Use **+ sample** to improve a person's match.
2. In **Conversation**, press **Start listening** and talk.
3. Press **Stop listening** when you're done. **Export** downloads the transcript as `.txt`.

On Linux or Windows, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the install line for smaller CPU-only PyTorch.

If the install fails with `pkg-config is required for building PyAV`, your Python is too new or too old for the prebuilt packages. Delete `.venv` and recreate it with Python 3.12.

The first start downloads the Whisper (`base.en`) and ECAPA models from Hugging Face. For a smaller install, `pip install -r backend/requirements.txt` includes real Whisper transcription with the basic MFCC speaker matcher; ECAPA is optional. `CT_TRANSCRIBER=auto` (the default) and `whisper` both require real speech-to-text. If dependencies or model downloads fail, startup reports the error instead of silently replacing words with `[speech …s]`.

If you see only speech durations, remove `CT_TRANSCRIBER=null` from your startup settings or start with `CT_TRANSCRIBER=whisper python -m app` from `backend/`. The explicit `null` mode is only for interface development and tests. Model files are cached after the first successful download; `CT_WHISPER_MODEL` can also point to a downloaded faster-whisper model directory for offline use.

See [`backend/README.md`](backend/README.md) for configuration (`CT_*` environment variables) and tests.

## Tests

```bash
pip install -r backend/requirements-dev.txt
cd backend && python -m pytest -q
```

## Roadmap

- Approach two: automatically discover new speakers. When a voice matches nobody, it becomes "Person N" and can be renamed later.
- Split an utterance when the speaker changes mid-stream, to handle quick back-and-forth.
- Adaptive voiceprints, so matching improves with high-confidence lines.
