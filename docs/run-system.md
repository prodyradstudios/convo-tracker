# run-system

> How to run Convo Tracker on your own computer. Exported from the doc at <https://claude.ai/artifact/7L8MrnBaxuyaA1g4kwsWmb> on 2026-11-03; update this file when the setup changes.

Install once, then start the server and open http://localhost:8000.

## Requirements

You need Python 3.11 to 3.13 (3.12 recommended), git, a Chromium-based browser and a microphone. Newer or older Pythons may lack prebuilt packages, and the install then fails trying to compile them.

| Item | Requirement | Check |
| --- | --- | --- |
| Python | 3.11, 3.12 or 3.13 | `python3 --version` |
| OS | macOS (Apple Silicon or Intel), Linux, or Windows | none |
| Browser | Chrome or Edge (Safari and Firefox untested) | none |
| Microphone | Any; the browser asks for permission | none |
| Disk | About 3 GB for PyTorch and models | none |
| Network | Internet on first start, to download the models from Hugging Face | none |

On a Mac without Python 3.12, install it with Homebrew: `brew install python@3.12`.

## Install

Install once, from the repo root. Every command below assumes you are in the `convo-tracker` folder unless it says otherwise.

1. Get the code:

```bash
git clone https://github.com/prodyradstudios/convo-tracker.git
cd convo-tracker
```

2. Until PR #1 is merged, switch to the work branch: `git checkout claude/keen-cannon-sxbxab`. After the merge, stay on `main`.
3. Create and activate a virtual environment with Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate` instead. Your prompt should now start with `(.venv)`.

4. Install the packages. This takes a few minutes.

| OS | Command |
| --- | --- |
| macOS | `pip install -r backend/requirements-ml.txt` |
| Linux / Windows | `pip install -r backend/requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu` |

The extra index on Linux and Windows gets the smaller CPU-only PyTorch. For a smaller install with real transcription, install `backend/requirements.txt` instead; it uses Whisper for words and the basic MFCC speaker matcher. Speech-duration placeholders require explicitly setting `CT_TRANSCRIBER=null` and are intended only for interface development.

To update later: `git pull`, then rerun step 4 if `backend/requirements*.txt` changed.

## Windows (PowerShell)

On a Windows laptop, use these PowerShell commands for the whole setup; everything else in this guide is the same. Windows has no `python3.12` or `source` command: use the `py` launcher and the `Activate.ps1` script.

1. Install Python 3.12 and Git (skip what you have). Close and reopen PowerShell afterwards.

```powershell
winget install Python.Python.3.12
winget install Git.Git
```

2. Get the code and create the virtual environment:

```powershell
git clone https://github.com/prodyradstudios/convo-tracker.git
cd convo-tracker
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
```

3. Install the packages (CPU-only PyTorch):

```powershell
pip install -r backend\requirements-ml.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

4. Start the server, then open http://localhost:8000 in Chrome or Edge:

```powershell
cd backend
python -m app
```

5. Change a setting by setting it first, in the same window:

```powershell
$env:CT_MATCH_THRESHOLD = "0.25"
python -m app
```

The setting lasts until you close that PowerShell window. To run the tests, from the `backend` folder: `pip install -r requirements-dev.txt`, then `python -m pytest -q`.

**Windows-only errors**

| Symptom | Fix |
| --- | --- |
| `Activate.ps1 cannot be loaded because running scripts is disabled` | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, answer Y, then activate again |
| `py` or `python` is not recognized | Reopen PowerShell after installing Python; or reinstall from python.org with "Add python.exe to PATH" ticked |
| `WinError 126` or a missing `.dll` when starting | Install the Visual C++ runtime: `winget install Microsoft.VCRedist.2015+.x64`, then retry |
| Windows Firewall prompt on first start | Choose **Cancel** or private networks only; the app only needs `localhost` |

Using Command Prompt instead of PowerShell: activate with `.venv\Scripts\activate.bat`, and set a variable with `set CT_MATCH_THRESHOLD=0.25` on its own line before `python -m app`.

## Run

Start the server from the `backend` folder with the virtual environment active, then use the app at http://localhost:8000.

1. Start the server:

```bash
source .venv/bin/activate
cd backend
python -m app
```

2. Wait for `Uvicorn running on http://127.0.0.1:8000`. The first start downloads the Whisper `base.en` and ECAPA models (a few hundred MB), so it can take a few minutes. Later starts take seconds.
3. Open **http://localhost:8000** in Chrome and allow the microphone. Use `localhost` exactly: browsers block the mic on other plain-http addresses.
4. Check the footer. It should name `ecapa` (or `mfcc` for the basic speaker matcher) and `whisper`. If it says `null`, transcription was explicitly disabled; remove `CT_TRANSCRIBER=null` or set it to `whisper`, then restart. Model load failures now stop startup rather than silently substituting speech durations.
5. **Enroll** each person in the People panel: type a name, press **Record**, talk normally for about 8 seconds, press **Stop**. Add 2 to 3 samples per person with **+ sample**: one normal, one louder or more animated, one quieter.
6. **Listen**: press **Start listening** and talk. Each line appears with the speaker's name a second or two after they pause, and interruptions become separate lines.
7. Press **Stop listening** when done. **Export** downloads the transcript as a `.txt` file.

To stop the server, press **Ctrl+C** in the terminal. Enrolled voices are saved in `backend/data/speakers.json` and come back on the next start; **Remove all** in the People panel deletes them.

## Configuration

The defaults work for most rooms. To change a setting, put it before the start command, for example `CT_MATCH_THRESHOLD=0.25 python -m app`. Invalid values stop the server at startup with a message naming the variable.

| Variable | Default | When to change it |
| --- | --- | --- |
| `CT_MATCH_THRESHOLD` | `0.30` (ECAPA) | Lower (0.25) if too many lines are Unknown; raise (0.35) if people get mixed up |
| `CT_SILENCE_MS` | `700` | Pause that ends a line; lower for fast back-and-forth |
| `CT_TURN_MIN_S` | `1.0` | Shortest interruption that becomes its own line |
| `CT_WHISPER_MODEL` | `base.en` | `small.en` for better transcripts (slower); a name without `.en` for other languages |
| `CT_DEVICE` | `cpu` | `cuda` on a machine with an NVIDIA GPU |
| `CT_ENROLL_AUGMENT` | `1` | `0` to store only the original samples (faster enrollment) |
| `CT_ADAPT` | `1` | `0` to stop learning voices during the conversation |
| `CT_PORT` / `CT_HOST` | `8000` / `127.0.0.1` | Run on another port, or `0.0.0.0` to expose on your network |
| `CT_DATA_DIR` | `data` | Where enrolled voices are saved |

The full list, including `CT_MAX_UTTERANCE_S`, `CT_TURN_WINDOW_S` and `CT_TURN_HOP_S`, is in `backend/README.md`.

## Run the tests

The automated tests need no microphone and no AI models; they use synthetic voices and stand-in models. Expect 228 passed.

```bash
source .venv/bin/activate
pip install -r backend/requirements-dev.txt
cd backend
python -m pytest -q
```

## Troubleshooting

Most problems come from the wrong Python version, the wrong folder, or an inactive virtual environment.

| Symptom | Cause | Fix |
| --- | --- | --- |
| `pkg-config is required for building PyAV` | Python too new or too old for the prebuilt packages | Delete `.venv` and recreate it with `python3.12`, then reinstall |
| `No module named 'uvicorn'` | The install failed or the venv isn't active | Activate the venv (`(.venv)` in the prompt) and rerun the install |
| `Could not open requirements file: backend/requirements-ml.txt` | You are already inside `backend` | Drop the `backend/` prefix, or `cd ..` first |
| Footer says `mfcc` | Basic speaker matcher is active; real transcription still works | Install ECAPA extras if better speaker recognition is needed |
| Footer says `null`, or lines show `[speech …s]` | `CT_TRANSCRIBER=null` explicitly disables transcription | Remove that setting or use `CT_TRANSCRIBER=whisper`, then restart |
| Startup says Whisper cannot load | Missing runtime, blocked model download, or incompatible device | Install `backend/requirements.txt`, allow Hugging Face model downloads, use `CT_DEVICE=cpu`, or point `CT_WHISPER_MODEL` at a downloaded model directory |
| Terminal warning mentioning `huggingface.co` | No internet on first start, or a proxy blocks it | Connect and restart; later starts work offline |
| Mic blocked or no prompt | Page not opened on `localhost`, or permission denied earlier | Use http://localhost:8000; reset site permissions in the browser |
| `Address already in use` | Another server is on port 8000 | Stop it, or start with `CT_PORT=8001 python -m app` |
| Lines labeled Unknown | Voice far from the enrollment samples | Add samples with **+ sample**, or lower `CT_MATCH_THRESHOLD` |
| Two people merged into one line | They talked at the same time | Known limit: overlapping speech is not separated |
| Mic doesn't work from another device | Browsers only allow the mic on `localhost` or HTTPS | Use the computer running the server, or put it behind HTTPS |
