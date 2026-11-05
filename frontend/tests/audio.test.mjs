import assert from 'node:assert/strict';
import test from 'node:test';

let moduleNumber = 0;
function deferred() {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}
function track(kind) {
  const handlers = new Set();
  return { kind, readyState: 'live', stops: 0,
    addEventListener(name, callback) { handlers.add(callback); },
    removeEventListener(name, callback) { handlers.delete(callback); },
    stop() { this.readyState = 'ended'; this.stops++; },
    end() { this.readyState = 'ended'; for (const cb of [...handlers]) cb(); },
  };
}
function stream() {
  const tracks = [track('audio'), track('video')];
  return { tracks, getTracks: () => tracks, getAudioTracks: () => tracks.filter(t => t.kind === 'audio') };
}
async function browser({ permission = null, moduleLoad = null, failConnect = false, manualFlush = false } = {}) {
  const shared = stream();
  const calls = [];
  const nodes = [];
  const graph = [];
  const loadingStarted = deferred();
  function graphNode() {
    const node = { disconnected: false, connections: [],
      connect(other) { if (failConnect) throw new Error('graph failed'); this.connections.push(other); },
      disconnect() { this.disconnected = true; },
    };
    graph.push(node);
    return node;
  }
  class Context {
    constructor() { this.state = 'running'; this.sampleRate = 48000; this.destination = {}; this.audioWorklet = { addModule: () => { loadingStarted.resolve(); return moduleLoad || Promise.resolve(); } }; }
    createMediaStreamSource(input) { assert.equal(input, shared); return graphNode(); }
    createGain() { const node = graphNode(); node.gain = { value: 1 }; return node; }
  }
  class Worklet {
    constructor(ctx, name, options) {
      Object.assign(this, graphNode());
      graph.pop(); graph.push(this);
      this.ctx = ctx; this.name = name; this.options = options;
      this.port = { onmessage: null, postMessage: msg => {
        assert.deepEqual(msg, { type: 'stop' });
        if (!manualFlush) queueMicrotask(() => this.flush());
      } };
      nodes.push(this);
    }
    flush() {
      this.port.onmessage?.({ data: { type: 'chunk', samples: new Int16Array([123, -456]), rms: .25 } });
      this.port.onmessage?.({ data: { type: 'flushed' } });
    }
  }
  const devices = {
    getDisplayMedia(options) { calls.push(['computer', options]); return permission || Promise.resolve(shared); },
    getUserMedia(options) { calls.push(['microphone', options]); return permission || Promise.resolve(shared); },
  };
  globalThis.window = { isSecureContext: true, AudioContext: Context };
  Object.defineProperty(globalThis, 'navigator', { value: { mediaDevices: devices }, configurable: true });
  globalThis.AudioWorkletNode = Worklet;
  const module = await import(new URL(`../js/audio.js?audio-test=${moduleNumber++}`, import.meta.url));
  return { ...module, shared, calls, nodes, graph, loadingStarted: loadingStarted.promise };
}

test('computer request runs synchronously and uses the existing muted PCM graph', async () => {
  const api = await browser();
  const capture = new api.AudioCapture();
  const chunks = [], levels = [];
  const starting = capture.start(c => chunks.push(c), n => levels.push(n), { source: 'computer' });
  assert.equal(api.calls.length, 1, 'request must happen before start returns its promise');
  assert.deepEqual(api.calls[0], ['computer', {
    video: true, audio: true, systemAudio: 'include',
    selfBrowserSurface: 'exclude', surfaceSwitching: 'exclude',
  }]);
  await starting;
  assert.equal(capture.sink.gain.value, 0);
  assert.equal(capture.node.name, 'pcm-capture');
  assert.equal(capture.node.options.processorOptions.sampleRate, 48000);
  assert.equal(api.shared.tracks[1].readyState, 'live', 'keep browser sharing alive until stop');
  await capture.stop();
  assert.deepEqual([...chunks[0]], [123, -456], 'flush delivers final partial PCM chunk');
  assert.deepEqual(levels, [.25]);
  assert.ok(api.shared.tracks.every(t => t.readyState === 'ended'));
  assert.equal(capture.running, false);
});

test('MicCapture retains microphone defaults', async () => {
  const api = await browser();
  const capture = new api.MicCapture();
  await capture.start(() => {});
  assert.equal(api.calls[0][0], 'microphone');
  assert.equal(api.calls[0][1].audio.channelCount, 1);
  assert.equal(api.calls.some(c => c[0] === 'computer'), false);
  await capture.stop();
});

test('all tracks and partial graph are cleaned when connection throws', async () => {
  const api = await browser({ failConnect: true });
  const capture = new api.AudioCapture();
  await assert.rejects(capture.start(() => {}, null, { source: 'computer' }), /graph failed/);
  assert.ok(api.shared.tracks.every(t => t.readyState === 'ended'));
  assert.ok(api.graph.every(n => n.disconnected));
  assert.equal(capture.running, false);
  assert.equal(capture.stream, null);
});

test('stop during sharing prompt releases a later permission result', async () => {
  const permission = deferred();
  const api = await browser({ permission: permission.promise });
  const capture = new api.AudioCapture();
  const starting = capture.start(() => {}, null, { source: 'computer' });
  await capture.stop();
  permission.resolve(api.shared);
  await assert.rejects(starting, /cancelled/);
  assert.ok(api.shared.tracks.every(t => t.readyState === 'ended'));
  assert.equal(api.nodes.length, 0);
});

test('stop while loading processing releases tracks without constructing a graph later', async () => {
  const loading = deferred();
  const api = await browser({ moduleLoad: loading.promise });
  const capture = new api.AudioCapture();
  const starting = capture.start(() => {}, null, { source: 'computer' });
  const cancelled = assert.rejects(starting, /cancelled/);
  await api.loadingStarted;
  await capture.stop();
  assert.ok(api.shared.tracks.every(t => t.readyState === 'ended'));
  loading.resolve();
  await cancelled;
  assert.equal(api.nodes.length, 0);
});

test('an old flush cannot consume a new capture flush callback', async () => {
  const api = await browser({ manualFlush: true });
  const capture = new api.AudioCapture();
  let oldChunks = 0, newChunks = 0;
  await capture.start(() => oldChunks++);
  const oldNode = api.nodes[0];
  const oldStop = capture.stop();
  // A browser supplies a new active stream for each permission grant.
  for (const t of api.shared.tracks) t.readyState = 'live';
  await capture.start(() => newChunks++);
  const newNode = api.nodes[1];
  oldNode.flush();
  await oldStop;
  assert.equal(capture.node, newNode);
  assert.equal(capture.running, true);
  const newStop = capture.stop();
  newNode.flush();
  await newStop;
  assert.equal(oldChunks, 1);
  assert.equal(newChunks, 1);
});
