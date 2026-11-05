# Convo Tracker

Tracks a conversation and labels who said what.

Version 1 enrolls voices first. Each person records a short voice sample and types their name. The app then listens to the selected microphone or shared computer audio and writes a live transcript where every line is attributed to the closest enrolled voice. Lines that match no one well enough are labeled **Unknown**.

- **Backend** (`backend/`): Python, FastAPI. A streaming VAD splits the audio into utterances. SpeechBrain ECAPA voiceprints, compared by cosine similarity, identify the speaker, and faster-whisper transcribes the speech.
- **Frontend** (`frontend/`): plain HTML, CSS and JavaScript, with no build step. Microphone or browser-shared computer audio passes through an AudioWorklet, is resampled to 16 kHz mono, and streams over a WebSocket.
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

Open <http://localhost:8000>. Audio capture requires `localhost` or HTTPS. Select **Microphone** or **Computer audio** under **Audio input**; the same selection is used for enrollment and listening.

1. In **People**, type a name, press **Record**, capture about 8 seconds of that person speaking alone, and press **Stop**. Repeat this for everyone, using the input source where their voice is heard. Use **+ sample** to improve a person's match.
2. In **Conversation**, press **Start listening** and begin the conversation. In **Computer audio** mode, choose the audio source in the browser sharing dialog each time you start recording or listening.
3. Press **Stop listening** when you're done. **Export** downloads the transcript as `.txt`.

On Linux or Windows, add `--extra-index-url https://download.pytorch.org/whl/cpu` to the install line for smaller CPU-only PyTorch.

If the install fails with `pkg-config is required for building PyAV`, your Python is too new or too old for the prebuilt packages. Delete `.venv` and recreate it with Python 3.12.

The first start downloads the Whisper (`base.en`) and ECAPA models from Hugging Face. For a smaller install, `pip install -r backend/requirements.txt` includes real Whisper transcription with the basic MFCC speaker matcher; ECAPA is optional. `CT_TRANSCRIBER=auto` (the default) and `whisper` both require real speech-to-text. If dependencies or model downloads fail, startup reports the error instead of silently replacing words with `[speech …s]`.

If you see only speech durations, remove `CT_TRANSCRIBER=null` from your startup settings or start with `CT_TRANSCRIBER=whisper python -m app` from `backend/`. The explicit `null` mode is only for interface development and tests. Model files are cached after the first successful download; `CT_WHISPER_MODEL` can also point to a downloaded faster-whisper model directory for offline use.

See [`backend/README.md`](backend/README.md) for configuration (`CT_*` environment variables) and tests.

## Discord and computer audio

Select **Computer audio** to transcribe audio played by a browser tab or another shareable source. For Discord, prefer Discord in a browser tab: choose that tab in the browser sharing dialog and enable **Share audio**. For the native Discord app, select an entire screen and enable system audio **if your browser and operating system offer it**. Window/screen audio capture varies by browser, OS, and source; native app capture is not universally available. If it is unavailable, use Discord in a browser tab with audio sharing.

Enroll remote speakers one at a time using the same shared source, with everyone else quiet. The browser asks you to select the shared source again for each enrollment sample and each listening session. It may require a screen/tab video selection to grant audio capture; only audio is sent to Convo Tracker. **Computer audio** does not mix in your local microphone, so your own microphone voice is excluded unless it is already present in the shared audio. The app reports an error if capture is unsupported or the selected share contains no audio track; choose an audio-capable source and enable sharing audio. This feature uses browser APIs and needs no additional application dependencies.

## Tests

```bash
pip install -r backend/requirements-dev.txt
cd backend && python -m pytest -q
```

Frontend capture and socket tests (Node 24, from the repository root):

```bash
node --test --test-isolation=none frontend/tests/*.test.mjs
```

The browser integration tests use Playwright and Chromium when available and otherwise skip. They simulate the shared media source; actual tab/system audio sharing still depends on the browser and OS. The additional `frontend/tests/audio-worklet.browser.mjs` check exercises real browser resampling and final-chunk flushing with generated audio; use `PLAYWRIGHT_MODULE` and `CHROMIUM_PATH` to select local browser tools.

## Roadmap

- Approach two: automatically discover new speakers. When a voice matches nobody, it becomes "Person N" and can be renamed later.
- Split an utterance when the speaker changes mid-stream, to handle quick back-and-forth.
- Adaptive voiceprints, so matching improves with high-confidence lines.
