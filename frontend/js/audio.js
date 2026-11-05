// Microphone or shared computer audio via AudioWorklet. Emits 16 kHz mono Int16Array chunks
// (1600 samples = 100 ms) plus an RMS level (0..1) per chunk.

let sharedContext = null;
let workletReady = null; // Promise for addModule on sharedContext

const WORKLET_URL = new URL('./pcm-worklet.js', import.meta.url);

// getUserMedia may sit behind a permission prompt, so it gets a generous limit;
// the rest (worklet load + context resume) should be quick.
const PERMISSION_TIMEOUT_MS = 60000;
const AUDIO_INIT_TIMEOUT_MS = 10000;

/** Reject with `message` if `promise` has not settled within `ms`. */
function withTimeout(promise, ms, message) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(Object.assign(new Error(message), { timeout: true })), ms);
  });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

/** Lazily create the single shared AudioContext. Call from a user gesture. */
export function getAudioContext() {
  if (!sharedContext || sharedContext.state === 'closed') {
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) throw new Error('This browser does not support the Web Audio API.');
    sharedContext = new Ctor({ latencyHint: 'interactive' });
    workletReady = null;
  }
  return sharedContext;
}

function loadWorklet(ctx) {
  if (!workletReady) {
    workletReady = ctx.audioWorklet.addModule(WORKLET_URL).catch((e) => {
      workletReady = null;
      throw e;
    });
  }
  return workletReady;
}

/**
 * Create and resume the shared AudioContext (and start loading the worklet).
 * Call synchronously at the top of a click handler, before any `await`:
 * Safari/iOS only let a context start inside the user gesture. Never throws;
 * problems surface later from AudioCapture.start().
 */
export function primeAudio() {
  try {
    const ctx = getAudioContext();
    if (ctx.state !== 'running') ctx.resume().catch(() => {});
    if (typeof AudioWorkletNode !== 'undefined' && ctx.audioWorklet) loadWorklet(ctx).catch(() => {});
  } catch {
    /* reported by AudioCapture.start() */
  }
}

function friendlyMicError(err) {
  const name = err && err.name;
  switch (name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'Microphone access was denied. Allow microphone access for this site in your browser settings and try again.';
    case 'NotFoundError':
    case 'DevicesNotFoundError':
    case 'OverconstrainedError':
      return 'No microphone was found. Connect a microphone and try again.';
    case 'NotReadableError':
    case 'TrackStartError':
    case 'AbortError':
      return 'The microphone could not be started. It may be in use by another application.';
    default:
      return `Could not start the microphone: ${(err && err.message) || err}`;
  }
}

function friendlyComputerError(err) {
  switch (err && err.name) {
    case 'NotAllowedError':
    case 'PermissionDeniedError':
    case 'SecurityError':
      return 'Computer audio sharing was cancelled or denied. Try again, select a browser tab and enable Share audio in the sharing dialog.';
    case 'InvalidStateError':
      return 'Start computer audio sharing by pressing the capture button while this page is focused.';
    case 'NotFoundError':
      return 'No shareable screen or browser tab was found. Open the tab playing audio and try again.';
    case 'NotReadableError':
      return 'The selected screen or tab could not be shared. Check your operating system screen-recording permissions and try again.';
    default:
      return `Could not start computer audio sharing: ${(err && err.message) || err}`;
  }
}

function stopTracks(stream) {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    try { track.stop(); } catch { /* still release the other tracks */ }
  }
}

function disconnect(node) {
  if (node) {
    try { node.disconnect(); } catch { /* already disconnected */ }
  }
}

export class AudioCapture {
  constructor() {
    this.stream = null;
    this.source = null; // MediaStreamAudioSourceNode, retained for compatibility
    this.node = null;
    this.sink = null;
    this.running = false;
    this.captureSource = 'microphone';
    this._gen = 0;
    this._active = null;
    this._lastStop = null;
  }

  /**
   * Capture microphone or shared computer audio as 16 kHz mono PCM16.
   * Call directly from a user gesture: getDisplayMedia must run before an await.
   * Video is required by the browser sharing API and retained until stop, but is
   * never rendered, encoded, or sent; only audio enters the muted worklet graph.
   */
  async start(onChunk, onLevel, { source = 'microphone' } = {}) {
    if (this.running) throw new Error('Audio capture is already running.');
    if (!['microphone', 'computer'].includes(source)) throw new Error('Choose microphone or computer audio.');
    if (!window.isSecureContext) {
      throw new Error('Audio capture needs a secure page. Open the app via https:// or http://localhost.');
    }
    const devices = navigator.mediaDevices;
    const method = source === 'computer' ? 'getDisplayMedia' : 'getUserMedia';
    if (!devices || typeof devices[method] !== 'function') {
      throw new Error(source === 'computer'
        ? 'This browser does not support computer audio sharing. Use a current Chrome or Edge and share a browser tab with audio.'
        : 'This browser does not support microphone capture.');
    }
    if (typeof AudioWorkletNode === 'undefined') {
      throw new Error('This browser does not support AudioWorklet. Please use a current Chrome, Edge, Firefox or Safari.');
    }

    const session = {
      gen: ++this._gen, stream: null, source: null, node: null, sink: null,
      listeners: [], cancelled: false, ended: false, ready: false,
      flushing: false, disposed: false, stopPromise: null, resolveFlush: null,
    };
    this._active = session;
    this.running = true;
    this.captureSource = source;
    const cancelled = () => session.cancelled || session.gen !== this._gen;
    const cancelError = () => new Error('Audio capture start was cancelled.');
    let ctx, resumed, permission;
    try {
      // All activation-sensitive operations happen synchronously in this call.
      ctx = getAudioContext();
      resumed = ctx.state === 'suspended' ? ctx.resume() : Promise.resolve();
      resumed.catch(() => {});
      permission = source === 'computer'
        ? devices.getDisplayMedia({
          video: true, audio: true, systemAudio: 'include',
          selfBrowserSurface: 'exclude', surfaceSwitching: 'exclude',
        })
        : devices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        });
      // Cancellation cannot dismiss the browser prompt. Release any stream
      // delivered after cancellation/timeout, without touching a newer capture.
      permission = Promise.resolve(permission);
      permission.then((stream) => { if (session.cancelled) stopTracks(stream); }, () => {});
      try {
        session.stream = await withTimeout(permission, PERMISSION_TIMEOUT_MS, 'permission prompt timed out');
      } catch (err) {
        if (cancelled()) throw cancelError();
        if (err && err.timeout) {
          throw new Error(source === 'computer'
            ? 'Computer audio sharing did not start in time. Check the sharing dialog and try again.'
            : 'The microphone did not start in time. Check for a microphone permission prompt and try again.');
        }
        throw new Error(source === 'computer' ? friendlyComputerError(err) : friendlyMicError(err));
      }
      if (cancelled()) throw cancelError();
      const audioTracks = session.stream.getAudioTracks();
      if (!audioTracks.length || audioTracks.every((track) => track.readyState === 'ended')) {
        throw new Error(source === 'computer'
          ? 'No shared audio was received. Share a browser tab playing audio and enable Share audio in the sharing dialog. Screen and system audio availability depends on your browser and operating system.'
          : 'The microphone did not provide an active audio track. Reconnect it and try again.');
      }
      const ended = () => {
        if (session.cancelled || session.ended) return;
        session.ended = true;
        // Stop video and audio together, even if the caller has no onended hook.
        this._stopSession(session, true);
        try { if (this.onended) this.onended(); } catch (err) { console.error('AudioCapture ended handler failed', err); }
      };
      for (const track of session.stream.getTracks()) {
        track.addEventListener('ended', ended);
        session.listeners.push([track, ended]);
      }
      try {
        await withTimeout(Promise.all([loadWorklet(ctx), resumed]), AUDIO_INIT_TIMEOUT_MS,
          'the browser did not start audio in time (tap the button again)');
      } catch (err) {
        throw new Error(`Could not initialise audio processing: ${(err && err.message) || err}`);
      }
      if (cancelled() || session.ended) throw cancelError();
      session.source = ctx.createMediaStreamSource(session.stream);
      session.node = new AudioWorkletNode(ctx, 'pcm-capture', {
        numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [1],
        processorOptions: { sampleRate: ctx.sampleRate },
      });
      session.node.port.onmessage = (event) => {
        const msg = event.data;
        if (!msg || session.disposed) return;
        if (msg.type === 'chunk' && (session.ready || session.flushing)) {
          try {
            onChunk(msg.samples);
            if (onLevel) onLevel(msg.rms);
          } catch (err) { console.error('AudioCapture chunk handler failed', err); }
        } else if (msg.type === 'flushed' && session.resolveFlush) {
          session.resolveFlush();
        }
      };
      session.sink = ctx.createGain();
      session.sink.gain.value = 0; // never play captured computer audio back
      session.source.connect(session.node);
      session.node.connect(session.sink);
      session.sink.connect(ctx.destination);
      session.ready = true;
      this.stream = session.stream;
      this.source = session.source;
      this.node = session.node;
      this.sink = session.sink;
    } catch (err) {
      const wasCancelled = cancelled();
      await this._stopSession(session, false);
      if (wasCancelled) throw cancelError();
      // Browser methods may throw synchronously instead of rejecting a promise.
      if (!permission) throw new Error(source === 'computer' ? friendlyComputerError(err) : friendlyMicError(err));
      throw err;
    }
  }

  _stopSession(session, flush) {
    if (session.stopPromise) return session.stopPromise;
    session.cancelled = true;
    session.ready = false;
    if (this._active === session) {
      this._active = null;
      this.running = false;
      this.stream = this.source = this.node = this.sink = null;
    }
    for (const [track, handler] of session.listeners) {
      if (track.removeEventListener) track.removeEventListener('ended', handler);
    }
    session.listeners = [];
    disconnect(session.source);
    stopTracks(session.stream);
    session.flushing = flush && !!session.node;
    session.stopPromise = (async () => {
      try {
        if (session.flushing) {
          await new Promise((resolve) => {
            const timer = setTimeout(resolve, 500);
            session.resolveFlush = () => { clearTimeout(timer); resolve(); };
            try { session.node.port.postMessage({ type: 'stop' }); }
            catch { session.resolveFlush(); }
          });
        }
      } finally {
        session.resolveFlush = null;
        session.flushing = false;
        session.disposed = true;
        if (session.node) session.node.port.onmessage = null;
        disconnect(session.node);
        disconnect(session.sink);
      }
    })();
    this._lastStop = session.stopPromise;
    return session.stopPromise;
  }

  /** Flush the final partial PCM chunk, then release every audio/video track. */
  async stop() {
    this._gen++;
    if (!this._active) return this._lastStop;
    return this._stopSession(this._active, true);
  }
}

// Existing callers keep microphone behavior without changing their imports.
export class MicCapture extends AudioCapture {}
