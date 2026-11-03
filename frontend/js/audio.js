// Microphone capture via AudioWorklet. Emits 16 kHz mono Int16Array chunks
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
 * problems surface later from MicCapture.start().
 */
export function primeAudio() {
  try {
    const ctx = getAudioContext();
    if (ctx.state !== 'running') ctx.resume().catch(() => {});
    if (typeof AudioWorkletNode !== 'undefined' && ctx.audioWorklet) loadWorklet(ctx).catch(() => {});
  } catch {
    /* reported by MicCapture.start() */
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

export class MicCapture {
  constructor() {
    this.stream = null;
    this.source = null;
    this.node = null;
    this.sink = null;
    this.running = false;
    this._gen = 0; // bumps on every start/stop so a stale start() can bail out
  }

  /**
   * Start capturing. Must be called from a user gesture (click) the first time.
   * @param {(chunk: Int16Array) => void} onChunk
   * @param {(rms: number) => void} [onLevel]
   */
  async start(onChunk, onLevel) {
    if (this.running) throw new Error('Microphone is already running.');
    if (!window.isSecureContext) {
      throw new Error('Microphone access needs a secure page. Open the app via https:// or http://localhost.');
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('This browser does not support microphone capture.');
    }
    if (typeof AudioWorkletNode === 'undefined') {
      throw new Error('This browser does not support AudioWorklet. Please use a current Chrome, Edge, Firefox or Safari.');
    }

    const gen = ++this._gen;
    this.running = true;

    // Create/resume the context synchronously inside the user gesture.
    let ctx;
    try {
      ctx = getAudioContext();
    } catch (e) {
      this.running = false;
      throw e;
    }
    const resumed = ctx.state === 'suspended' ? ctx.resume() : Promise.resolve();
    resumed.catch(() => {}); // awaited (with a timeout) below

    let stream;
    const gum = navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    try {
      stream = await withTimeout(gum, PERMISSION_TIMEOUT_MS, 'getUserMedia timed out');
    } catch (err) {
      if (gen === this._gen) this.running = false;
      if (err && err.timeout) {
        // If the prompt is answered later, release the device straight away.
        gum.then((s) => s.getTracks().forEach((t) => t.stop()), () => {});
        throw new Error('The microphone did not start in time. Check for a microphone permission prompt and try again.');
      }
      throw new Error(friendlyMicError(err));
    }

    try {
      await withTimeout(
        Promise.all([loadWorklet(ctx), resumed]),
        AUDIO_INIT_TIMEOUT_MS,
        'the browser did not start audio in time (tap the button again)',
      );
    } catch (err) {
      stream.getTracks().forEach((t) => t.stop());
      if (gen === this._gen) this.running = false;
      throw new Error(`Could not initialise audio processing: ${(err && err.message) || err}`);
    }

    if (gen !== this._gen) {
      // stop() was called while we were waiting.
      stream.getTracks().forEach((t) => t.stop());
      throw new Error('Microphone start was cancelled.');
    }

    this.stream = stream;
    this.source = ctx.createMediaStreamSource(stream);
    this.node = new AudioWorkletNode(ctx, 'pcm-capture', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [1],
      processorOptions: { sampleRate: ctx.sampleRate },
    });
    this._flushed = null;
    this.node.port.onmessage = (e) => {
      const msg = e.data;
      if (!msg) return;
      if (msg.type === 'chunk') {
        try {
          onChunk(msg.samples);
          if (onLevel) onLevel(msg.rms);
        } catch (err) {
          console.error('MicCapture chunk handler failed', err);
        }
      } else if (msg.type === 'flushed' && this._flushed) {
        this._flushed();
      }
    };
    // Keep the node in the rendered graph (some browsers only pull nodes that
    // reach the destination); the output is silent and muted anyway.
    this.sink = ctx.createGain();
    this.sink.gain.value = 0;
    this.source.connect(this.node);
    this.node.connect(this.sink);
    this.sink.connect(ctx.destination);

    // If the device goes away (unplugged, permission revoked) surface it.
    for (const track of stream.getAudioTracks()) {
      track.addEventListener('ended', () => {
        if (this.stream === stream && this.onended) this.onended();
      });
    }
  }

  /**
   * Stop capturing. Flushes the last partial chunk to onChunk before resolving.
   * Safe to call multiple times.
   */
  async stop() {
    this._gen++;
    if (!this.running) return;
    this.running = false;
    const { stream, source, node, sink } = this;
    this.stream = this.source = this.node = this.sink = null;

    if (source) {
      try { source.disconnect(); } catch { /* already disconnected */ }
    }
    if (stream) stream.getTracks().forEach((t) => t.stop());
    if (node) {
      await new Promise((resolve) => {
        const timer = setTimeout(resolve, 500);
        this._flushed = () => { clearTimeout(timer); resolve(); };
        try {
          node.port.postMessage({ type: 'stop' });
        } catch {
          clearTimeout(timer);
          resolve();
        }
      });
      this._flushed = null;
      node.port.onmessage = null;
      try { node.disconnect(); } catch { /* ignore */ }
    }
    if (sink) {
      try { sink.disconnect(); } catch { /* ignore */ }
    }
  }
}
