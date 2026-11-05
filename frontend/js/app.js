// UI wiring for Convo Tracker. All server/user text is rendered with textContent.

import * as api from './api.js';
import { AudioCapture, primeAudio } from './audio.js';
import { encodeWav } from './wav.js';
import { SessionClient } from './session.js';

const MIN_RECORD_S = 3;
const MAX_RECORD_S = 15;
// After Stop the server may still be transcribing (CPU Whisper can be slow), so
// give up only after this much silence from it, or after the hard cap.
const STOP_IDLE_TIMEOUT_MS = 15000;
const STOP_MAX_MS = 120000;
const SPEAKERS_RETRY_MAX_MS = 15000;
const SAMPLE_RATE = 16000;
const UNKNOWN_COLOR = '#888888';
const TRANSCRIPTION_DISABLED_MESSAGE =
  'Speech-to-text is disabled. Restart the server with CT_TRANSCRIBER=whisper to transcribe spoken words.';

const $ = (id) => document.getElementById(id);
const el = {
  banner: $('banner'),
  bannerText: $('banner-text'),
  bannerClose: $('banner-close'),
  audioSource: $('audio-source'),
  audioSourceHint: $('audio-source-hint'),
  removeAll: $('remove-all'),
  form: $('enroll-form'),
  name: $('name-input'),
  record: $('record-btn'),
  enrollMeter: $('enroll-meter'),
  enrollError: $('enroll-error'),
  enrollStatus: $('enroll-status'),
  enrollHint: $('enroll-hint'),
  people: $('people-list'),
  peopleEmpty: $('people-empty'),
  clear: $('clear-btn'),
  exportBtn: $('export-btn'),
  listen: $('listen-btn'),
  listenHint: $('listen-hint'),
  listenMeter: $('listen-meter'),
  speech: $('speech-indicator'),
  speechText: $('speech-text'),
  feed: $('feed'),
  feedEmpty: $('feed-empty'),
  jump: $('jump-btn'),
  health: $('health-text'),
};

const capture = new AudioCapture();
const selectedSource = () => el.audioSource.value === 'computer' ? 'computer' : 'microphone';

const state = {
  speakers: [],
  loadedSpeakers: false,
  speakersError: null, // message while the initial load keeps failing
  speakersVersion: 0, // bumped on every local change to `speakers`
  /** 'idle' | 'recording' | 'saving' | 'connecting' | 'listening' | 'finishing' */
  mode: 'idle',
  busy: 0, // in-flight people requests (delete etc.)
  recording: null, // { target: {kind:'new', name} | {kind:'sample', id, name}, chunks, samples, startedAt, timer }
  session: null,
  stopTimer: null,
  finishDeadline: 0, // performance.now() after which finishing gives up regardless
  lastServerError: null, // last {"type":"error"} message of the current session
  lines: [], // { kind:'utterance', ...u } | { kind:'separator' }
  seenIds: new Set(),
  sessionCount: 0,
  stickToBottom: true,
};

// ---------- helpers ----------

function safeColor(c) {
  return typeof c === 'string' && /^#[0-9a-fA-F]{6}$/.test(c) ? c : UNKNOWN_COLOR;
}

/** Black or white text, whichever contrasts better with the hex background. */
function textColorFor(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lin = (v) => {
    const s = v / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  const L = 0.2126 * lin((n >> 16) & 255) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
  return (L + 0.05) / 0.05 > 1.05 / (L + 0.05) ? '#111111' : '#ffffff';
}

function fmtTime(seconds) {
  const s = Math.max(0, Math.floor(Number(seconds) || 0));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}

function fmtTimer(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function levelToScale(rms) {
  if (!(rms > 0)) return 0;
  const db = 20 * Math.log10(rms);
  return Math.min(1, Math.max(0, (db + 60) / 60)); // -60 dBFS .. 0 dBFS
}

function setMeter(meterEl, rms) {
  meterEl.firstElementChild.style.transform = `scaleX(${levelToScale(rms).toFixed(3)})`;
}

function showBanner(msg) {
  el.bannerText.textContent = msg;
  el.banner.hidden = false;
}

function hideBanner() {
  el.banner.hidden = true;
  el.bannerText.textContent = '';
}

function setEnrollError(msg) {
  el.enrollError.textContent = msg || '';
  el.enrollError.hidden = !msg;
}

function setEnrollStatus(msg) {
  el.enrollStatus.textContent = msg || '';
}

const isListeningMode = () => ['connecting', 'listening', 'finishing'].includes(state.mode);
const isEnrollMode = () => state.mode === 'recording' || state.mode === 'saving';

// ---------- rendering ----------

function render() {
  const { mode } = state;
  const rec = state.recording;
  const hasPeople = state.speakers.length > 0;
  const computerAudio = selectedSource() === 'computer';
  el.audioSource.disabled = mode !== 'idle' || state.busy > 0;
  el.audioSourceHint.textContent = computerAudio
    ? 'Choose the call’s tab or a screen with audio sharing, and enable “Share audio”. Available sources depend on your browser and computer. Only audio is sent; your microphone is not included.'
    : 'Uses your microphone for voice samples and live transcription.';
  el.enrollHint.textContent = computerAudio
    ? 'Type a name, press Record, share the call’s audio and have only that person speak for 5–15 seconds.'
    : 'Type a name, press Record and speak naturally for 5–15 seconds.';

  // Record button
  if (mode === 'recording' && rec) {
    const elapsed = (performance.now() - rec.startedAt) / 1000;
    el.record.textContent = `Stop (${fmtTimer(elapsed)})`;
    el.record.classList.add('recording');
    el.record.disabled = !rec.live || elapsed < MIN_RECORD_S;
  } else {
    el.record.textContent = mode === 'saving' ? 'Saving…' : 'Record';
    el.record.classList.remove('recording');
    el.record.disabled = mode !== 'idle' || state.busy > 0;
  }
  el.name.disabled = mode !== 'idle';
  el.enrollMeter.hidden = mode !== 'recording';

  // People list buttons
  const peopleLocked = isEnrollMode() || state.busy > 0;
  for (const btn of el.people.querySelectorAll('button[data-action="sample"]')) {
    btn.disabled = mode !== 'idle' || state.busy > 0;
  }
  for (const btn of el.people.querySelectorAll('button[data-action="delete"]')) {
    btn.disabled = peopleLocked;
  }
  for (const li of el.people.children) {
    li.classList.toggle('recording', !!rec && rec.target.kind === 'sample' && rec.target.id === li.dataset.id);
  }
  el.removeAll.disabled = !hasPeople || peopleLocked;
  if (!state.loadedSpeakers) {
    el.peopleEmpty.textContent = state.speakersError ? 'Couldn’t load people — retrying…' : 'Loading people…';
    el.peopleEmpty.hidden = false;
  } else {
    el.peopleEmpty.textContent = 'Nobody enrolled yet.';
    el.peopleEmpty.hidden = hasPeople;
  }

  // Listen button
  el.listen.classList.toggle('active', mode === 'listening' || mode === 'finishing');
  switch (mode) {
    case 'connecting':
      el.listen.textContent = 'Connecting…';
      el.listen.disabled = true;
      break;
    case 'listening':
      el.listen.textContent = 'Stop listening';
      el.listen.disabled = false;
      break;
    case 'finishing':
      el.listen.textContent = 'Finishing…';
      el.listen.disabled = true;
      break;
    default:
      el.listen.textContent = 'Start listening';
      el.listen.disabled = !hasPeople || mode !== 'idle';
  }
  if (!hasPeople && !isListeningMode()) {
    if (state.loadedSpeakers) el.listenHint.textContent = 'Enroll at least one person to start listening.';
    else if (state.speakersError) el.listenHint.textContent = 'Couldn’t load people — retrying…';
    else el.listenHint.textContent = 'Loading people…';
    el.listenHint.hidden = false;
  } else if (isEnrollMode()) {
    el.listenHint.textContent = 'Finish the voice recording first.';
    el.listenHint.hidden = false;
  } else {
    el.listenHint.hidden = true;
  }
  el.speech.hidden = !(mode === 'listening' || mode === 'finishing');
  el.listenMeter.hidden = mode !== 'listening';

  // Feed toolbar
  const hasLines = state.lines.some((l) => l.kind === 'utterance');
  el.clear.disabled = state.lines.length === 0;
  el.exportBtn.disabled = !hasLines;
  el.feedEmpty.hidden = state.lines.length > 0;
}

function renderPeople() {
  const frag = document.createDocumentFragment();
  for (const sp of state.speakers) {
    const li = document.createElement('li');
    li.dataset.id = sp.id;

    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.backgroundColor = safeColor(sp.color);
    dot.setAttribute('aria-hidden', 'true');

    const main = document.createElement('div');
    main.className = 'person-main';
    const name = document.createElement('span');
    name.className = 'person-name';
    name.textContent = sp.name;
    name.title = sp.name;
    const meta = document.createElement('span');
    meta.className = 'person-meta';
    const n = Number(sp.num_samples) || 0;
    meta.textContent = `${n} sample${n === 1 ? '' : 's'}`;
    main.append(name, meta);

    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'btn btn-small';
    add.dataset.action = 'sample';
    add.textContent = '+ sample';
    add.setAttribute('aria-label', `Record another voice sample for ${sp.name}`);

    const del = document.createElement('button');
    del.type = 'button';
    del.className = 'icon-btn danger';
    del.dataset.action = 'delete';
    del.textContent = '✕';
    del.setAttribute('aria-label', `Remove ${sp.name}`);
    del.title = `Remove ${sp.name}`;

    li.append(dot, main, add, del);
    frag.append(li);
  }
  el.people.replaceChildren(frag);
  render();
}

function makeLineElement(line) {
  const li = document.createElement('li');
  if (line.kind === 'separator') {
    li.className = 'separator';
    li.textContent = line.label;
    return li;
  }
  const unknown = line.speaker_id == null;
  li.className = unknown ? 'line unknown' : 'line';
  const conf = Number(line.confidence);
  li.title = `${line.speaker_name} · confidence ${Number.isFinite(conf) ? conf.toFixed(3) : 'n/a'}`;

  const chip = document.createElement('span');
  chip.className = 'chip';
  const color = safeColor(line.color);
  chip.style.backgroundColor = color;
  chip.style.color = textColorFor(color);
  chip.textContent = line.speaker_name;

  const text = document.createElement('p');
  text.className = 'line-text';
  text.textContent = line.text;

  const time = document.createElement('time');
  time.className = 'line-time';
  time.textContent = fmtTime(line.start);
  time.setAttribute('aria-label', `at ${fmtTime(line.start)}`);

  li.append(chip, text, time);
  return li;
}

function appendLine(line) {
  state.lines.push(line);
  el.feed.append(makeLineElement(line));
  if (state.stickToBottom) {
    el.feed.scrollTop = el.feed.scrollHeight;
  } else if (line.kind === 'utterance') {
    el.jump.hidden = false;
  }
  render();
}

function isNearBottom() {
  const f = el.feed;
  return f.scrollHeight - f.scrollTop - f.clientHeight < 24;
}

// ---------- people ----------

const speakersLoad = { inFlight: null, retryTimer: null, retryDelay: 1000 };

/**
 * (Re)load the people list. Until the first load succeeds, failures are retried
 * with exponential backoff (1 s, 2 s, 4 s … capped at 15 s).
 */
function refreshSpeakers() {
  if (speakersLoad.inFlight) return speakersLoad.inFlight;
  clearTimeout(speakersLoad.retryTimer);
  speakersLoad.retryTimer = null;
  speakersLoad.inFlight = (async () => {
    try {
      let list;
      let version;
      do {
        // A local change (enroll/delete) while the request was in flight makes
        // the response stale; fetch again.
        version = state.speakersVersion;
        list = await api.listSpeakers();
      } while (version !== state.speakersVersion);
      const recovered = !!state.speakersError;
      state.speakers = list;
      state.loadedSpeakers = true;
      state.speakersError = null;
      speakersLoad.retryDelay = 1000;
      renderPeople();
      if (recovered && healthError) loadHealth();
    } catch (err) {
      if (state.loadedSpeakers) {
        console.warn('Could not refresh people:', err.message); // keep the list we have
      } else {
        state.speakersError = err.message;
        speakersLoad.retryTimer = setTimeout(refreshSpeakers, speakersLoad.retryDelay);
        speakersLoad.retryDelay = Math.min(speakersLoad.retryDelay * 2, SPEAKERS_RETRY_MAX_MS);
        render();
      }
    } finally {
      speakersLoad.inFlight = null;
    }
  })();
  return speakersLoad.inFlight;
}

function setSpeakers(list) {
  state.speakers = list;
  state.speakersVersion++;
}

function upsertSpeaker(sp) {
  const i = state.speakers.findIndex((s) => s.id === sp.id);
  if (i >= 0) state.speakers[i] = sp;
  else state.speakers.push(sp);
  state.speakersVersion++;
  renderPeople();
}

async function withBusy(fn) {
  state.busy++;
  render();
  try {
    return await fn();
  } finally {
    state.busy--;
    render();
  }
}

async function deleteSpeaker(id) {
  const sp = state.speakers.find((s) => s.id === id);
  if (!sp) return;
  if (!window.confirm(`Remove ${sp.name}? Their voice samples will be deleted.`)) return;
  setEnrollError('');
  await withBusy(async () => {
    try {
      await api.deleteSpeaker(id);
      setSpeakers(state.speakers.filter((s) => s.id !== id));
      setEnrollStatus(`Removed ${sp.name}.`);
    } catch (err) {
      if (err.status === 404) setSpeakers(state.speakers.filter((s) => s.id !== id));
      else setEnrollError(`Could not remove ${sp.name}: ${err.message}`);
    }
  });
  renderPeople();
}

async function deleteAll() {
  if (!state.speakers.length) return;
  if (!window.confirm(`Remove all ${state.speakers.length} enrolled people? This cannot be undone.`)) return;
  setEnrollError('');
  await withBusy(async () => {
    try {
      await api.deleteAllSpeakers();
      setSpeakers([]);
      setEnrollStatus('Removed everyone.');
    } catch (err) {
      setEnrollError(`Could not remove everyone: ${err.message}`);
    }
  });
  renderPeople();
}

// ---------- enrollment recording ----------

async function startRecording(target) {
  if (state.mode !== 'idle') return;
  primeAudio(); // must run inside the click, before any await (Safari/iOS)
  setEnrollError('');
  setEnrollStatus('');
  hideBanner();

  const rec = { target, source: selectedSource(), chunks: [], samples: 0, startedAt: performance.now(), timer: null, live: false };
  state.recording = rec;
  state.mode = 'recording';
  setMeter(el.enrollMeter, 0);
  render();
  capture.onended = () => {
    if (state.recording === rec) finishRecording({ deviceLost: true });
  };

  try {
    await capture.start(
      (chunk) => {
        rec.chunks.push(chunk);
        rec.samples += chunk.length;
      },
      (rms) => setMeter(el.enrollMeter, rms),
      { source: rec.source },
    );
  } catch (err) {
    if (state.recording === rec) {
      capture.onended = null;
      state.recording = null;
      state.mode = 'idle';
      setEnrollError(err.message);
      render();
    }
    return;
  }
  if (state.recording !== rec) return; // cancelled meanwhile

  rec.live = true;
  rec.startedAt = performance.now();
  const instruction = rec.source === 'computer' ? 'have only this person speak in the call.' : 'speak now.';
  setEnrollStatus(target.kind === 'sample' ? `Recording another sample for ${target.name}… ${instruction}` : `Recording… ${instruction}`);
  rec.timer = setInterval(() => {
    const elapsed = (performance.now() - rec.startedAt) / 1000;
    if (elapsed >= MAX_RECORD_S) finishRecording();
    else render();
  }, 200);
  render();
  el.record.focus();
}

async function finishRecording({ deviceLost = false } = {}) {
  const rec = state.recording;
  if (!rec || state.mode !== 'recording') return;
  clearInterval(rec.timer);
  state.mode = 'saving';
  capture.onended = null;
  render();
  await capture.stop();
  if (state.recording !== rec) return; // cancelled while leaving the page
  state.recording = null;
  setMeter(el.enrollMeter, 0);

  const seconds = rec.samples / SAMPLE_RATE;
  if (deviceLost && seconds < MIN_RECORD_S) {
    state.mode = 'idle';
    setEnrollStatus('');
    setEnrollError(rec.source === 'computer'
      ? 'Audio sharing stopped before enough speech was recorded. Share audio and try again.'
      : 'The microphone stopped unexpectedly. Please try again.');
    render();
    return;
  }

  setEnrollStatus(`Saving ${seconds.toFixed(1)} s sample…`);
  const wav = encodeWav(rec.chunks, SAMPLE_RATE);
  try {
    if (rec.target.kind === 'new') {
      const sp = await api.createSpeaker(rec.target.name, wav);
      upsertSpeaker(sp);
      el.name.value = '';
      setEnrollStatus(`Enrolled ${sp.name}.`);
    } else {
      const sp = await api.addSample(rec.target.id, wav);
      upsertSpeaker(sp);
      setEnrollStatus(`Added a sample for ${sp.name} (${sp.num_samples} total).`);
    }
  } catch (err) {
    setEnrollStatus('');
    if (err.status === 409 && rec.target.kind === 'new') {
      setEnrollError(`${rec.target.name} is already enrolled. Use “+ sample” next to their name to add more audio.`);
    } else {
      setEnrollError(err.message);
    }
    if (err.status === 404 || err.status === 409) refreshSpeakers();
  } finally {
    state.mode = 'idle';
    render();
    if (rec.target.kind === 'new' && !el.name.value) el.name.focus();
  }
}

function onRecordSubmit(e) {
  e.preventDefault();
  if (state.mode === 'recording') {
    finishRecording();
    return;
  }
  if (state.mode !== 'idle') return;
  const name = el.name.value.trim();
  if (!name) {
    setEnrollError('Please type a name first.');
    el.name.focus();
    return;
  }
  if (name.length > 40) {
    setEnrollError('Names can be at most 40 characters.');
    el.name.focus();
    return;
  }
  if (state.speakers.some((s) => s.name.toLowerCase() === name.toLowerCase())) {
    setEnrollError(`${name} is already enrolled. Use “+ sample” next to their name to add more audio.`);
    el.name.focus();
    return;
  }
  startRecording({ kind: 'new', name });
}

// ---------- listening ----------

function clearStopTimer() {
  if (state.stopTimer) {
    clearTimeout(state.stopTimer);
    state.stopTimer = null;
  }
}

async function startListening() {
  if (state.mode !== 'idle' || !state.speakers.length) return;
  primeAudio(); // must run inside the click, before any await (Safari/iOS)
  const source = selectedSource();
  hideBanner();
  state.mode = 'connecting';
  state.lastServerError = null;
  render();

  // While finishing, any message shows the server is still working: push the
  // give-up timer back.
  const heardFromServer = () => {
    if (state.mode === 'finishing' && state.stopTimer) armStopTimer(session);
  };

  const session = new SessionClient({
    onStatus: (st) => {
      if (session !== state.session) return;
      heardFromServer();
      if (st === 'stopped' && state.mode === 'finishing') endSession();
    },
    onSpeech: (active) => {
      if (session !== state.session) return;
      heardFromServer();
      if (state.mode === 'finishing') return; // keep showing "Processing…"
      el.speech.classList.toggle('active', active);
      el.speechText.textContent = active ? 'Someone is speaking…' : 'Listening…';
    },
    onSpeaker: (speaker) => {
      if (session !== state.session) return;
      heardFromServer();
      if (state.mode === 'finishing') return;
      el.speech.classList.add('active');
      el.speechText.textContent = `${String(speaker.speaker_name ?? 'Unknown')} is speaking…`;
    },
    onUtterance: (u) => {
      if (session !== state.session) return;
      heardFromServer();
      if (u.id != null && state.seenIds.has(u.id)) return;
      if (u.id != null) state.seenIds.add(u.id);
      appendLine({
        kind: 'utterance',
        id: u.id,
        seq: u.seq,
        speaker_id: u.speaker_id ?? null,
        speaker_name: String(u.speaker_name ?? 'Unknown'),
        color: u.speaker_id == null ? UNKNOWN_COLOR : u.color,
        text: String(u.text ?? ''),
        start: Number(u.start) || 0,
        end: Number(u.end) || 0,
        confidence: u.confidence,
      });
    },
    onError: (message) => {
      if (session !== state.session) return;
      heardFromServer();
      state.lastServerError = message;
      showBanner(`Server error: ${message}`);
    },
    onClose: ({ expected }) => {
      if (session !== state.session || expected) return;
      const wasFinishing = state.mode === 'finishing';
      const serverError = state.lastServerError;
      endSession({ socketGone: true });
      showBanner(
        (wasFinishing
          ? 'The connection closed before the last speech was processed.'
          : 'Connection to the server was lost. Listening stopped.') +
          (serverError ? ` Server error: ${serverError}` : ''),
      );
    },
  });
  state.session = session;
  let audioReady = false;
  capture.onended = () => {
    if (session !== state.session) return;
    const message = source === 'computer'
      ? 'Audio sharing ended. Listening stopped.'
      : 'The microphone stopped unexpectedly. Listening stopped.';
    if (state.mode === 'connecting') {
      endSession();
      showBanner(message);
    } else if (state.mode === 'listening') {
      showBanner(message);
      stopListening();
    }
  };

  try {
    // Display sharing requires the original click gesture. Request it before
    // waiting for the socket; begin streaming only once both are ready.
    const captured = capture.start(
      (chunk) => { if (audioReady && session === state.session) session.sendAudio(chunk); },
      (rms) => { if (session === state.session) setMeter(el.listenMeter, rms); },
      { source },
    );
    await Promise.all([captured, session.connect()]);
  } catch (err) {
    if (state.session === session) {
      endSession();
      showBanner(err.message);
    }
    return;
  }
  if (state.session !== session) return;

  if (!session.start()) {
    endSession();
    showBanner('The connection closed before listening could start. Please try again.');
    return;
  }
  audioReady = true;

  state.sessionCount++;
  if (state.lines.length) {
    const t = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    const label = `New session · ${t}`;
    const last = state.lines[state.lines.length - 1];
    if (last.kind === 'separator') {
      // The previous session produced nothing: reuse its separator.
      last.label = label;
      el.feed.lastElementChild.textContent = label;
    } else {
      appendLine({ kind: 'separator', label });
    }
  }
  state.mode = 'listening';
  el.speech.classList.remove('active');
  el.speechText.textContent = 'Listening…';
  setMeter(el.listenMeter, 0);
  render();
}

async function stopListening() {
  if (state.mode !== 'listening') return;
  const session = state.session;
  state.mode = 'finishing';
  capture.onended = null;
  el.speech.classList.remove('active');
  el.speechText.textContent = 'Processing the last speech…';
  render();
  await capture.stop(); // flushes the final partial chunk through sendAudio
  if (state.session !== session) return; // socket died meanwhile
  if (!session.stop()) {
    endSession();
    return;
  }
  state.finishDeadline = performance.now() + STOP_MAX_MS;
  armStopTimer(session);
}

/** (Re)start the finishing give-up timer: STOP_IDLE_TIMEOUT_MS of server silence, capped by finishDeadline. */
function armStopTimer(session) {
  clearStopTimer();
  const left = state.finishDeadline - performance.now();
  const hitCap = left <= STOP_IDLE_TIMEOUT_MS;
  state.stopTimer = setTimeout(() => {
    state.stopTimer = null;
    if (state.session === session && state.mode === 'finishing') {
      showBanner(
        hitCap
          ? `The server was still processing ${Math.round(STOP_MAX_MS / 1000)} s after you stopped; gave up waiting. Some speech may be missing.`
          : 'The server stopped responding while finishing; some speech may be missing.',
      );
      endSession();
    }
  }, Math.max(0, Math.min(STOP_IDLE_TIMEOUT_MS, left)));
}

/** Tear down the session (capture, socket, timers) and go back to idle. */
function endSession({ socketGone = false } = {}) {
  clearStopTimer();
  const session = state.session;
  state.session = null;
  capture.onended = null;
  capture.stop();
  if (session && !socketGone) session.close();
  if (session && session.droppedChunks > 0) {
    console.warn(`Dropped ${session.droppedChunks} audio chunks because the connection was backed up.`);
  }
  if (isListeningMode()) state.mode = 'idle';
  el.speech.classList.remove('active');
  setMeter(el.listenMeter, 0);
  render();
}

function onListenClick() {
  if (state.mode === 'idle') startListening();
  else if (state.mode === 'listening') stopListening();
}

// ---------- feed actions ----------

function clearFeed() {
  if (!state.lines.length) return;
  state.lines = [];
  state.seenIds.clear();
  el.feed.replaceChildren();
  el.jump.hidden = true;
  state.stickToBottom = true;
  render();
}

function exportFeed() {
  const out = [];
  for (const l of state.lines) {
    if (l.kind === 'separator') {
      if (out.length && out[out.length - 1] !== '') out.push('');
    } else {
      out.push(`[${fmtTime(l.start)}] ${l.speaker_name}: ${l.text}`);
    }
  }
  while (out.length && out[out.length - 1] === '') out.pop();
  if (!out.length) return;
  const blob = new Blob([out.join('\n') + '\n'], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  const d = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  a.href = url;
  a.download = `conversation-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.txt`;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- health ----------

let healthError = null; // banner text shown for a failed health check

async function loadHealth() {
  try {
    const h = await api.getHealth();
    const parts = [`Speaker model: ${h.embedder}`, `Transcriber: ${h.transcriber}`];
    let note = '';
    if (h.embedder === 'mfcc') note += ' · basic speaker model (install ML extras for better accuracy)';
    if (h.transcriber === 'null') note += ' · speech-to-text disabled (durations only)';
    el.health.textContent = parts.join(' · ') + note;
    if (healthError && el.bannerText.textContent === healthError) hideBanner();
    healthError = null;
    if (h.transcriber !== 'null' && el.bannerText.textContent === TRANSCRIPTION_DISABLED_MESSAGE) {
      hideBanner();
    }
    if (h.transcriber === 'null' && el.banner.hidden) {
      showBanner(TRANSCRIPTION_DISABLED_MESSAGE);
    }
  } catch (err) {
    el.health.textContent = 'Server unreachable.';
    healthError = err.message;
    showBanner(err.message);
  }
}

// ---------- events ----------

el.form.addEventListener('submit', onRecordSubmit);
el.name.addEventListener('input', () => {
  if (!el.enrollError.hidden) setEnrollError('');
});
el.people.addEventListener('click', (e) => {
  const btn = e.target.closest('button[data-action]');
  if (!btn || btn.disabled) return;
  const id = btn.closest('li')?.dataset.id;
  const sp = state.speakers.find((s) => s.id === id);
  if (!sp) return;
  if (btn.dataset.action === 'delete') deleteSpeaker(id);
  else if (btn.dataset.action === 'sample') startRecording({ kind: 'sample', id: sp.id, name: sp.name });
});
el.removeAll.addEventListener('click', deleteAll);
el.listen.addEventListener('click', onListenClick);
el.clear.addEventListener('click', clearFeed);
el.exportBtn.addEventListener('click', exportFeed);
el.bannerClose.addEventListener('click', hideBanner);
el.audioSource.addEventListener('change', () => {
  setEnrollError('');
  setEnrollStatus('');
  render();
});
el.feed.addEventListener('scroll', () => {
  state.stickToBottom = isNearBottom();
  if (state.stickToBottom) el.jump.hidden = true;
}, { passive: true });
el.jump.addEventListener('click', () => {
  el.feed.scrollTop = el.feed.scrollHeight;
  state.stickToBottom = true;
  el.jump.hidden = true;
});
window.addEventListener('focus', () => refreshSpeakers());
window.addEventListener('pagehide', () => {
  // A page can return from the back-forward cache with its JS state intact.
  // Clear active operations as well as resources so it returns ready to start.
  if (state.recording) {
    clearInterval(state.recording.timer);
    state.recording = null;
    state.mode = 'idle';
    setEnrollStatus('');
    setMeter(el.enrollMeter, 0);
  }
  if (state.session) endSession();
  else {
    capture.onended = null;
    capture.stop();
    render();
  }
});

render();
loadHealth();
refreshSpeakers();
