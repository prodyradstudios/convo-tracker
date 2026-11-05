import assert from 'node:assert/strict';
import test from 'node:test';

let generation = 0;
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function track(kind) {
  const listeners = new Map();
  return {
    kind, readyState: 'live', stops: 0,
    stop() { this.stops++; this.readyState = 'ended'; },
    addEventListener(name, cb) { listeners.set(name, [...(listeners.get(name) ?? []), cb]); },
    removeEventListener(name, cb) { listeners.set(name, (listeners.get(name) ?? []).filter(x => x !== cb)); },
    end() { this.readyState = 'ended'; for (const cb of [...(listeners.get('ended') ?? [])]) cb(); },
  };
}
function stream(tracks) {
  return { getTracks: () => tracks, getAudioTracks: () => tracks.filter(t => t.kind === 'audio'),
    getVideoTracks: () => tracks.filter(t => t.kind === 'video') };
}
async function harness({ request, graphFailure = false, autoFlush = true } = {}) {
  const nodes = [], gains = [], calls = [];
  class Context {
    state = 'running'; sampleRate = 48000; destination = {};
    audioWorklet = { addModule: async () => {} };
    resume() { return Promise.resolve(); }
    createMediaStreamSource(s) {
      if (graphFailure) throw new Error('graph unavailable');
      const source = { stream: s, connect(to) { this.to = to; }, disconnect() {} };
      return source;
    }
    createGain() {
      const gain = { gain: { value: 1 }, connect(to) { this.to = to; }, disconnect() {} };
      gains.push(gain); return gain;
    }
  }
  class Worklet {
    constructor() {
      this.port = { onmessage: null, postMessage: msg => {
        if (msg.type === 'stop' && autoFlush) queueMicrotask(() => this.port.onmessage?.({ data: { type: 'flushed' } }));
      } }; nodes.push(this);
    }
    connect(to) { this.to = to; }
    disconnect() {}
  }
  globalThis.window = { isSecureContext: true, AudioContext: Context };
  globalThis.AudioWorkletNode = Worklet;
  Object.defineProperty(globalThis, 'navigator', { configurable: true, value: { mediaDevices: {
    getDisplayMedia: opts => { calls.push(opts); return request(); },
    getUserMedia: opts => { calls.push(opts); return request(); },
  } } });
  const module = await import(new URL(`../js/audio.js?adversarial=${++generation}`, import.meta.url));
  return { capture: new module.AudioCapture(), nodes, gains, calls };
}

test('computer picker is invoked synchronously in the user gesture', async () => {
  const pending = deferred();
  const h = await harness({ request: () => pending.promise });
  const audio = track('audio'), video = track('video');
  const starting = h.capture.start(() => {}, null, { source: 'computer' });
  assert.equal(h.calls.length, 1, 'picker must be opened before awaiting any work');
  assert.equal(h.calls[0].audio, true);
  assert.ok(h.calls[0].video);
  pending.resolve(stream([audio, video]));
  await starting;
  assert.equal(h.gains[0].gain.value, 0, 'captured playback must never feed speakers');
  await h.capture.stop();
  assert.equal(audio.readyState, 'ended'); assert.equal(video.readyState, 'ended');
});

test('sharing video without audio rejects and releases the entire share', async () => {
  const video = track('video');
  const h = await harness({ request: async () => stream([video]) });
  await assert.rejects(h.capture.start(() => {}, null, { source: 'computer' }), /audio/i);
  assert.equal(video.readyState, 'ended');
  assert.equal(h.capture.running, false);
  assert.equal(h.nodes.length, 0);
});

test('cancelled pending picker cannot resurrect capture or leak any tracks', async () => {
  const pending = deferred();
  const h = await harness({ request: () => pending.promise });
  const audio = track('audio'), video = track('video');
  const starting = h.capture.start(() => {}, null, { source: 'computer' });
  const rejected = assert.rejects(starting, /cancel/i);
  await h.capture.stop();
  pending.resolve(stream([audio, video]));
  await rejected;
  assert.equal(h.capture.running, false);
  assert.equal(audio.readyState, 'ended'); assert.equal(video.readyState, 'ended');
  assert.equal(h.nodes.length, 0);
});

test('audio graph setup failure releases audio and video tracks', async () => {
  const audio = track('audio'), video = track('video');
  const h = await harness({ request: async () => stream([audio, video]), graphFailure: true });
  await assert.rejects(h.capture.start(() => {}, null, { source: 'computer' }), /graph|processing|initial/i);
  assert.equal(h.capture.running, false);
  assert.equal(audio.readyState, 'ended'); assert.equal(video.readyState, 'ended');
});

test('ending the shared display stops audio and notifies only once', async () => {
  const audio = track('audio'), video = track('video');
  const h = await harness({ request: async () => stream([audio, video]) });
  let ended = 0;
  h.capture.onended = () => ended++;
  await h.capture.start(() => {}, null, { source: 'computer' });
  video.end(); audio.end();
  assert.equal(ended, 1);
  await h.capture.stop();
  assert.equal(audio.readyState, 'ended'); assert.equal(video.readyState, 'ended');
});

test('finishing an old flush cannot remove a new capture callback or tracks', async () => {
  const oldAudio = track('audio'), oldVideo = track('video');
  const newAudio = track('audio'), newVideo = track('video');
  let calls = 0;
  const h = await harness({ request: async () => ++calls === 1 ? stream([oldAudio, oldVideo]) : stream([newAudio, newVideo]), autoFlush: false });
  await h.capture.start(() => {}, null, { source: 'computer' });
  const stoppingOld = h.capture.stop();
  const chunks = [];
  await h.capture.start(chunk => chunks.push(chunk), null, { source: 'computer' });
  h.nodes[0].port.onmessage?.({ data: { type: 'flushed' } });
  await stoppingOld;
  h.nodes[1].port.onmessage?.({ data: { type: 'chunk', samples: new Int16Array([1, 2]), rms: .1 } });
  assert.equal(h.capture.running, true);
  assert.equal(chunks.length, 1);
  assert.equal(newAudio.readyState, 'live'); assert.equal(newVideo.readyState, 'live');
  const stoppingNew = h.capture.stop();
  h.nodes[1].port.onmessage?.({ data: { type: 'flushed' } });
  await stoppingNew;
});
