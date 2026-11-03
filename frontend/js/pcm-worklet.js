// AudioWorkletProcessor: downmix to mono, low-pass, resample to 16 kHz,
// convert to Int16 and post fixed-size chunks (1600 samples = 100 ms).
//
// Loaded with audioWorklet.addModule(), which evaluates it as an ES module,
// so the pure DSP helpers below are also exported for unit tests in Node.
// The processor itself is only defined when running inside an AudioWorkletGlobalScope.

export const TARGET_RATE = 16000;
export const CHUNK_SAMPLES = 1600;

/** Direct-form-I biquad low-pass (RBJ cookbook). State persists across blocks. */
export class Biquad {
  constructor(sampleRate, cutoff, q) {
    const w0 = (2 * Math.PI * cutoff) / sampleRate;
    const cos = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * q);
    const a0 = 1 + alpha;
    this.b0 = (1 - cos) / 2 / a0;
    this.b1 = (1 - cos) / a0;
    this.b2 = (1 - cos) / 2 / a0;
    this.a1 = (-2 * cos) / a0;
    this.a2 = (1 - alpha) / a0;
    this.x1 = 0; this.x2 = 0; this.y1 = 0; this.y2 = 0;
  }

  /** Filter `buf` in place. */
  process(buf) {
    let { x1, x2, y1, y2 } = this;
    const { b0, b1, b2, a1, a2 } = this;
    for (let i = 0; i < buf.length; i++) {
      const x = buf[i];
      let y = b0 * x + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
      if (Math.abs(y) < 1e-20) y = 0; // flush denormals
      x2 = x1; x1 = x; y2 = y1; y1 = y;
      buf[i] = y;
    }
    this.x1 = x1; this.x2 = x2; this.y1 = y1; this.y2 = y2;
  }
}

/**
 * Streaming resampler: anti-alias low-pass (6th-order Butterworth as three
 * biquads, only when downsampling) followed by linear interpolation.
 * The fractional read position and the last input sample are carried across
 * calls, so block boundaries introduce no drift and no discontinuities.
 */
export class StreamResampler {
  constructor(inRate, outRate = TARGET_RATE) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.step = inRate / outRate; // input samples advanced per output sample
    this.passthrough = inRate === outRate;
    // Position of the next output sample, in input-sample units relative to the
    // start of the next block. Index -1 refers to `this.prev`.
    this.pos = 0;
    this.prev = 0;
    this.filters = [];
    if (inRate > outRate) {
      const cutoff = Math.min(0.45 * outRate, 0.45 * inRate); // ~7.2 kHz for 16 kHz output
      this.filters = [0.5176, 0.7071, 1.9319].map((q) => new Biquad(inRate, cutoff, q));
    }
    this.scratch = new Float32Array(0);
  }

  /**
   * Resample one block of mono float samples. Calls `emit(sample)` for every
   * output sample (float, nominally in [-1, 1]).
   */
  process(input, emit) {
    const n = input.length;
    if (n === 0) return;
    if (this.passthrough) {
      for (let i = 0; i < n; i++) emit(input[i]);
      return;
    }
    let buf = input;
    if (this.filters.length) {
      if (this.scratch.length !== n) this.scratch = new Float32Array(n);
      this.scratch.set(input);
      buf = this.scratch;
      for (const f of this.filters) f.process(buf);
    }
    let pos = this.pos;
    const step = this.step;
    const prev = this.prev;
    // Interpolate between buf[i] and buf[i+1]; need i+1 <= n-1, i.e. pos < n-1.
    while (pos < n - 1) {
      const i = Math.floor(pos);
      const frac = pos - i;
      const s0 = i < 0 ? prev : buf[i];
      const s1 = buf[i + 1];
      emit(s0 + (s1 - s0) * frac);
      pos += step;
    }
    this.pos = pos - n;
    this.prev = buf[n - 1];
  }
}

/** Float [-1, 1] -> Int16 with clamping. */
export function floatToInt16(x) {
  if (!(x === x)) return 0; // NaN
  const s = x < -1 ? -1 : x > 1 ? 1 : x;
  return s < 0 ? Math.round(s * 0x8000) : Math.round(s * 0x7fff);
}

/** Average all channels of one render quantum into `out` (mono). Returns frame count. */
export function downmix(channels, out) {
  const nch = channels.length;
  if (nch === 0) return 0;
  const n = channels[0].length;
  if (nch === 1) {
    out.set(channels[0]);
    return n;
  }
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let c = 0; c < nch; c++) s += channels[c][i];
    out[i] = s / nch;
  }
  return n;
}

/**
 * Collects resampled float samples into Int16 chunks of `size` samples and
 * hands each finished chunk (with its RMS level, 0..1) to `onChunk`.
 */
export class ChunkAccumulator {
  constructor(size, onChunk) {
    this.size = size;
    this.onChunk = onChunk;
    this.reset();
  }

  reset() {
    this.buf = new Int16Array(this.size);
    this.len = 0;
    this.sumSq = 0;
  }

  push(x) {
    const v = floatToInt16(x);
    this.buf[this.len++] = v;
    const f = v / 0x8000;
    this.sumSq += f * f;
    if (this.len === this.size) this._emit();
  }

  /** Emit whatever is buffered (possibly shorter than `size`). */
  flush() {
    if (this.len > 0) this._emit();
  }

  _emit() {
    const len = this.len;
    const out = len === this.size ? this.buf : this.buf.slice(0, len);
    const rms = Math.sqrt(this.sumSq / len);
    this.reset();
    this.onChunk(out, rms);
  }
}

if (typeof AudioWorkletProcessor !== 'undefined' && typeof registerProcessor === 'function') {
  class PcmCaptureProcessor extends AudioWorkletProcessor {
    constructor(options) {
      super();
      const opts = (options && options.processorOptions) || {};
      // `sampleRate` is a global in AudioWorkletGlobalScope.
      const inRate = opts.sampleRate || sampleRate;
      this.resampler = new StreamResampler(inRate, TARGET_RATE);
      this.acc = new ChunkAccumulator(CHUNK_SAMPLES, (samples, rms) => {
        this.port.postMessage({ type: 'chunk', samples, rms }, [samples.buffer]);
      });
      this.mono = new Float32Array(128);
      this.emit = (x) => this.acc.push(x);
      this.stopped = false;
      this.port.onmessage = (e) => {
        if (e.data && e.data.type === 'stop') {
          this.acc.flush();
          this.stopped = true;
          this.port.postMessage({ type: 'flushed' });
        }
      };
    }

    process(inputs) {
      if (this.stopped) return false;
      const input = inputs[0];
      if (input && input.length > 0 && input[0].length > 0) {
        if (this.mono.length !== input[0].length) this.mono = new Float32Array(input[0].length);
        const n = downmix(input, this.mono);
        this.resampler.process(n === this.mono.length ? this.mono : this.mono.subarray(0, n), this.emit);
      }
      return true;
    }
  }

  registerProcessor('pcm-capture', PcmCaptureProcessor);
}
