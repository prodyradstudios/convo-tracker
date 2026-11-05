// Optional real-DOM integration. Browser/media devices and WebSocket transport are
// mocked for lifecycle cases; a separate case uses native AudioWorklet/PCM
// processing. No test automates a real OS/browser sharing permission dialog.
import assert from 'node:assert/strict';
import test from 'node:test';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
let chromium;
try { ({ chromium } = await import('playwright')); }
catch { try { ({ chromium } = await import('/opt/codex/cua_node/lib/node_modules/playwright/index.mjs')); } catch {} }
const executablePath = process.env.CHROMIUM_PATH || '/usr/bin/chromium';
const available = chromium && existsSync(executablePath);

function browserMocks() {
  window.__test = { requests: [], sockets: [], tracks: [], nodes: [], captureMode: 'ready', socketMode: 'manual', sends: [] };
  const state = window.__test;
  function track(kind) {
    const handlers = new Set();
    const t = { kind, readyState: 'live', stop() { this.readyState = 'ended'; },
      addEventListener(name, handler) { if (name === 'ended') handlers.add(handler); },
      removeEventListener(name, handler) { handlers.delete(handler); },
      end() { this.readyState = 'ended'; for (const fn of [...handlers]) fn(); } };
    state.tracks.push(t); return t;
  }
  function stream() {
    const tracks = [track('audio'), track('video')];
    return { getTracks: () => tracks, getAudioTracks: () => tracks.filter(t => t.kind === 'audio') };
  }
  Object.defineProperty(navigator, 'mediaDevices', { configurable: true, value: {
    getDisplayMedia(options) {
      state.requests.push({ options, gesture: navigator.userActivation.isActive });
      if (state.captureMode === 'pending') return new Promise(resolve => { state.resolveShare = () => resolve(stream()); });
      return Promise.resolve(stream());
    },
    getUserMedia() { throw new Error('Test must select computer audio, never microphone'); },
  } });
  window.AudioContext = class {
    state = 'running'; sampleRate = 48000; destination = {};
    audioWorklet = { addModule: async () => {} };
    resume() { return Promise.resolve(); }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createGain() { return { gain: { value: 1 }, connect() {}, disconnect() {} }; }
  };
  window.AudioWorkletNode = class {
    constructor() {
      this.port = { onmessage: null, postMessage: () => queueMicrotask(() => this.port.onmessage?.({ data: { type: 'flushed' } })) };
      state.nodes.push(this);
    }
    connect() {} disconnect() {}
  };
  state.emit = (samples = 1600) => state.nodes.at(-1)?.port.onmessage?.({ data: { type: 'chunk', samples: new Int16Array(samples), rms: .1 } });
  window.WebSocket = class {
    static CONNECTING = 0; static OPEN = 1; static CLOSING = 2; static CLOSED = 3;
    readyState = 0; bufferedAmount = 0;
    constructor() {
      this.displayRequestedFirst = state.requests.length > 0;
      state.sockets.push(this);
      setTimeout(() => {
        if (this.readyState !== 0) return;
        if (state.socketMode === 'fail') { this.readyState = 3; this.onclose?.({ code: 1006, reason: '' }); }
        else if (state.socketMode === 'open') this.open();
      }, 100);
    }
    open() { this.readyState = 1; this.onopen?.(); }
    send(data) {
      state.sends.push(typeof data === 'string' ? JSON.parse(data) : { type: 'audio' });
      if (typeof data === 'string' && JSON.parse(data).type === 'stop') {
        queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ type: 'status', state: 'stopped' }) }));
      }
    }
    close() { this.readyState = 3; this.onclose?.({ code: 1000, reason: '' }); }
  };
}

test('computer audio uses the enrollment and conversation lifecycle in a real DOM', { skip: !available }, async t => {
  const root = fileURLToPath(new URL('../', import.meta.url));
  let enrollmentBody = null;
  const server = createServer(async (req, res) => {
    if (req.url === '/api/health') return res.end(JSON.stringify({ status: 'ok', embedder: 'mfcc', transcriber: 'whisper' }));
    if (req.url === '/api/speakers') {
      if (req.method === 'POST') {
        const chunks = []; for await (const chunk of req) chunks.push(chunk);
        enrollmentBody = Buffer.concat(chunks);
        res.statusCode = 201;
        return res.end(JSON.stringify({ id: 'bob', name: 'Bob', color: '#00ff00', num_samples: 1 }));
      }
      return res.end(JSON.stringify({ speakers: [{ id: 'alice', name: 'Alice', color: '#ff0000', num_samples: 1 }] }));
    }
    const filename = path.resolve(root, '.' + (req.url === '/' ? '/index.html' : req.url.split('?')[0]));
    if (!filename.startsWith(root)) { res.statusCode = 403; return res.end(); }
    try { res.setHeader('content-type', filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.css') ? 'text/css' : 'text/html'); res.end(await readFile(filename)); }
    catch { res.statusCode = 404; res.end(); }
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const browser = await chromium.launch({ executablePath, headless: true, args: ['--no-sandbox'] });
  async function page() {
    const p = await browser.newPage();
    await p.addInitScript(browserMocks);
    await p.goto(`http://127.0.0.1:${server.address().port}`);
    await p.waitForFunction(() => !document.querySelector('#listen-btn').disabled);
    await p.selectOption('#audio-source', 'computer');
    return p;
  }
  try {
    await t.test('conversation requests sharing in the click before socket setup and gates audio until start', async () => {
      const p = await page();
      await p.click('#listen-btn');
      await p.waitForFunction(() => window.__test.nodes.length > 0);
      await p.evaluate(() => window.__test.emit());
      assert.equal(await p.locator('#audio-source').isDisabled(), true);
      assert.equal(await p.evaluate(() => window.__test.sends.length), 0);
      await p.evaluate(() => window.__test.sockets[0].open());
      await p.waitForFunction(() => document.querySelector('#listen-btn').textContent.includes('Stop'));
      const before = await p.evaluate(() => ({ requests: window.__test.requests, first: window.__test.sockets[0].displayRequestedFirst, sends: window.__test.sends }));
      assert.equal(before.requests[0].gesture, true);
      assert.equal(before.first, true);
      assert.equal(before.sends[0].type, 'start');
      assert.equal(before.sends.filter(s => s.type === 'audio').length, 0);
      await p.evaluate(() => window.__test.emit());
      assert.equal(await p.evaluate(() => window.__test.sends.at(-1).type), 'audio');
      await p.evaluate(() => window.__test.tracks.find(t => t.kind === 'video').end());
      await p.waitForFunction(() => !document.querySelector('#audio-source').disabled);
      assert.equal(await p.evaluate(() => window.__test.tracks.every(t => t.readyState === 'ended')), true);
      await p.close();
    });
    await t.test('enrollment records computer PCM as the existing WAV upload', async () => {
      const p = await page();
      await p.fill('#name-input', 'Bob'); await p.click('#record-btn');
      await p.waitForFunction(() => document.querySelector('#enroll-status').textContent.includes('Recording'));
      await p.evaluate(() => window.__test.emit(48000));
      await p.click('#record-btn');
      await p.waitForFunction(() => document.querySelector('#enroll-status').textContent.includes('Enrolled Bob'));
      assert.ok(enrollmentBody?.includes(Buffer.from('RIFF')));
      assert.equal(await p.evaluate(() => window.__test.requests.length), 1);
      assert.equal(await p.evaluate(() => window.__test.tracks.every(t => t.readyState === 'ended')), true);
      await p.close();
    });
    await t.test('socket failure cancels a pending picker and releases a late share', async () => {
      const p = await page();
      await p.evaluate(() => { window.__test.captureMode = 'pending'; window.__test.socketMode = 'fail'; });
      await p.click('#listen-btn');
      await p.waitForFunction(() => !document.querySelector('#audio-source').disabled);
      await p.evaluate(() => window.__test.resolveShare());
      await p.waitForFunction(() => window.__test.tracks.length === 2 && window.__test.tracks.every(t => t.readyState === 'ended'));
      assert.equal(await p.evaluate(() => window.__test.nodes.length), 0);
      await p.close();
    });
    await t.test('native AudioWorklet converts shared-stream audio into muted PCM16 chunks', async () => {
      const p = await browser.newPage();
      await p.goto(`http://127.0.0.1:${server.address().port}`);
      await p.evaluate(async () => {
        const { AudioCapture } = await import('/js/audio.js');
        const ctx = new AudioContext({ sampleRate: 48000 });
        const destination = ctx.createMediaStreamDestination();
        const oscillator = ctx.createOscillator();
        oscillator.frequency.value = 440;
        oscillator.connect(destination); oscillator.start();
        const canvas = document.createElement('canvas');
        const video = canvas.captureStream(1).getVideoTracks()[0];
        const shared = new MediaStream([...destination.stream.getAudioTracks(), video]);
        navigator.mediaDevices.getDisplayMedia = () => Promise.resolve(shared);
        window.__native = { capture: new AudioCapture(), ctx, oscillator, shared, chunks: [], error: null };
        const button = document.createElement('button'); button.id = 'native-capture'; button.textContent = 'Native capture';
        button.onclick = () => {
          ctx.resume();
          window.__native.capture.start((pcm) => window.__native.chunks.push({ length: pcm.length, pcm: pcm instanceof Int16Array, peak: Math.max(...pcm.map(Math.abs)) }), null, { source: 'computer' }).catch(error => { window.__native.error = error.message; });
        };
        document.body.prepend(button);
      });
      try {
        await p.click('#native-capture');
        await p.waitForFunction(() => window.__native.chunks.length >= 2 || window.__native.error);
        const native = await p.evaluate(() => ({ error: window.__native.error, chunks: window.__native.chunks, gain: window.__native.capture.sink?.gain.value }));
        assert.equal(native.error, null);
        assert.equal(native.gain, 0);
        assert.ok(native.chunks.length >= 2);
        assert.ok(native.chunks.every(chunk => chunk.pcm && chunk.length === 1600));
        assert.ok(native.chunks.some(chunk => chunk.peak > 1000));
        await p.evaluate(async () => { await window.__native.capture.stop(); });
        assert.equal(await p.evaluate(() => window.__native.shared.getTracks().every(track => track.readyState === 'ended')), true);
      } finally {
        await p.evaluate(async () => { await window.__native.capture.stop(); window.__native.oscillator.stop(); await window.__native.ctx.close(); });
        await p.close();
      }
    });
    await t.test('pagehide cancels enrollment without uploading and restores idle controls', async () => {
      const p = await page();
      enrollmentBody = null;
      await p.fill('#name-input', 'Bob'); await p.click('#record-btn');
      await p.waitForFunction(() => document.querySelector('#enroll-status').textContent.includes('Recording'));
      await p.evaluate(() => window.dispatchEvent(new Event('pagehide')));
      await p.waitForFunction(() => !document.querySelector('#audio-source').disabled, null, { timeout: 1000 });
      assert.equal(await p.evaluate(() => window.__test.tracks.every(t => t.readyState === 'ended')), true);
      assert.equal(await p.locator('#record-btn').textContent(), 'Record');
      assert.equal(await p.locator('#enroll-status').textContent(), '');
      assert.equal(enrollmentBody, null);
      await p.close();
    });
    await t.test('pagehide stops every capture track and closes the transport', async () => {
      const p = await page(); await p.click('#listen-btn');
      await p.waitForFunction(() => window.__test.nodes.length > 0);
      await p.evaluate(() => window.__test.sockets[0].open());
      await p.waitForFunction(() => document.querySelector('#listen-btn').textContent.includes('Stop'));
      await p.evaluate(() => window.dispatchEvent(new Event('pagehide')));
      assert.equal(await p.evaluate(() => window.__test.tracks.every(t => t.readyState === 'ended')), true);
      assert.equal(await p.evaluate(() => window.__test.sockets[0].readyState), 3);
      await p.waitForFunction(() => !document.querySelector('#audio-source').disabled, null, { timeout: 1000 });
      assert.match(await p.locator('#listen-btn').textContent(), /Start listening/);
      await p.close();
    });
  } finally { await browser.close(); await new Promise(resolve => server.close(resolve)); }
});
