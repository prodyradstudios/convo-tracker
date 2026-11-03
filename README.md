# Convo Tracker

Tracks a conversation and labels who said what.

Version 1 enrolls voices first. Each person records a short voice sample and types their name. The app then listens to the room and writes a live transcript where every line is attributed to the closest enrolled voice. Lines that match no one well enough are labeled **Unknown**.

- **Backend** (`backend/`): Python, FastAPI. A streaming VAD splits the audio into utterances. SpeechBrain ECAPA voiceprints, compared by cosine similarity, identify the speaker, and faster-whisper transcribes the speech.
- **Frontend** (`frontend/`): plain HTML, CSS and JavaScript, with no build step. Mic audio is captured with an AudioWorklet, resampled to 16 kHz and streamed over a WebSocket.
- **Contract** (`docs/SPEC.md`): the REST and WebSocket API, audio format and algorithms.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
cd backend && python -m app
```

Open <http://localhost:8000>. Browsers only allow the microphone on `localhost` or HTTPS.

1. In **People**, type a name, press **Record**, speak for about 8 seconds, and press **Stop**. Repeat this for everyone. Use **+ sample** to improve a person's match.
2. In **Conversation**, press **Start listening** and talk.
3. Press **Stop listening** when you're done. **Export** downloads the transcript as `.txt`.

The first start downloads the Whisper (`base.en`) and ECAPA models from Hugging Face. If those packages aren't installed, run `pip install -r backend/requirements.txt` instead. The app then falls back to a weak MFCC speaker matcher and placeholder transcripts, which is useful for development only.

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
