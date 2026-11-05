// Browser integration check: real Web Audio + AudioWorklet, synthetic shared
// MediaStream. This does not exercise the browser picker or native OS capture.
// Run: node frontend/tests/audio-worklet.browser.mjs
// PLAYWRIGHT_MODULE / CHROMIUM_PATH can select locally installed browser tools.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || '/opt/codex/cua_node/lib/node_modules/playwright');
const assets = new Map(await Promise.all(['audio.js', 'pcm-worklet.js'].map(async name =>
  [`/js/${name}`, await readFile(new URL(`../js/${name}`, import.meta.url))])));
const server = createServer((request, response) => {
  if (request.url === '/') {
    response.writeHead(200, { 'content-type': 'text/html' });
    response.end('<!doctype html><button id="capture">Start synthetic shared audio</button>');
  } else if (assets.has(request.url)) {
    response.writeHead(200, { 'content-type': 'text/javascript' });
    response.end(assets.get(request.url));
  } else {
    response.writeHead(404); response.end();
  }
});
await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
let browser;
try {
  browser = await chromium.launch({
    executablePath: process.env.CHROMIUM_PATH || '/usr/bin/chromium',
    headless: true,
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  await page.goto(`http://127.0.0.1:${server.address().port}/`);
  await page.evaluate(async () => {
    const { AudioCapture, getAudioContext } = await import('/js/audio.js');
    const capture = new AudioCapture();
    const synth = new AudioContext({ sampleRate: 48000 });
    const oscillator = synth.createOscillator();
    oscillator.frequency.value = 440;
    const gain = synth.createGain();
    gain.gain.value = .2;
    const destination = synth.createMediaStreamDestination();
    oscillator.connect(gain).connect(destination);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 32;
    canvas.getContext('2d').fillRect(0, 0, 32, 32);
    const video = canvas.captureStream(30);
    const shared = new MediaStream([...destination.stream.getAudioTracks(), ...video.getVideoTracks()]);
    const chunks = [], levels = [];
    let requests = 0, gesture = false, options;
    Object.defineProperty(navigator.mediaDevices, 'getDisplayMedia', {
      configurable: true,
      value: request => {
        requests++;
        gesture = navigator.userActivation.isActive;
        options = request;
        return Promise.resolve(shared);
      },
    });
    document.querySelector('#capture').onclick = () => {
      window.captureFinished = (async () => {
        try {
          synth.resume();
          oscillator.start();
          let resolveFullChunks;
          const fullChunksReady = new Promise(resolve => { resolveFullChunks = resolve; });
          await capture.start(chunk => {
            if (!(chunk instanceof Int16Array)) throw new Error('PCM chunk is not Int16Array');
            chunks.push(Array.from(chunk));
            if (chunks.length === 3) resolveFullChunks();
          }, level => levels.push(level), { source: 'computer' });
          const inputRate = getAudioContext().sampleRate;
          const muted = capture.sink.gain.value === 0;
          await fullChunksReady;
          // Deliberately stop between 100ms worklet chunks, exercising flush.
          await new Promise(resolve => setTimeout(resolve, 37));
          await capture.stop();
          const samples = chunks.flat();
          // MediaStreamAudioSourceNode starts with a little transport silence.
          // Measure pitch on steady full chunks after the first 100ms, while
          // retaining every chunk (including silence) for size/flush checks.
          const steady = chunks.filter(chunk => chunk.length === 1600).slice(1).flat();
          let crossings = 0;
          for (let i = 1; i < steady.length; i++) {
            if (steady[i - 1] <= 0 && steady[i] > 0) crossings++;
          }
          window.captureResult = {
            requests, gesture, options, muted, inputRate,
            chunkSizes: chunks.map(chunk => chunk.length),
            sampleCount: samples.length,
            frequency: crossings / (steady.length / 16000),
            maxLevel: Math.max(...levels),
            tracksEnded: shared.getTracks().every(track => track.readyState === 'ended'),
            running: capture.running,
          };
        } catch (error) {
          window.captureError = error.stack || error.message;
        } finally {
          await capture.stop();
          oscillator.stop();
          for (const track of shared.getTracks()) track.stop();
          await synth.close();
          await getAudioContext().close();
        }
      })();
    };
  });
  await page.click('#capture');
  await page.waitForFunction(() => window.captureResult || window.captureError, null, { timeout: 15000 });
  await page.evaluate(() => window.captureFinished);
  const result = await page.evaluate(() => ({ result: window.captureResult, error: window.captureError }));
  assert.equal(result.error, undefined, result.error);
  assert.deepEqual(errors, []);
  const actual = result.result;
  assert.equal(actual.requests, 1);
  assert.equal(actual.gesture, true);
  assert.equal(actual.options.audio, true);
  assert.equal(actual.options.video, true);
  assert.equal(actual.muted, true);
  assert.ok(actual.chunkSizes.length >= 4, JSON.stringify(actual));
  assert.ok(actual.chunkSizes.slice(0, -1).every(size => size === 1600), JSON.stringify(actual));
  assert.ok(actual.chunkSizes.at(-1) > 0 && actual.chunkSizes.at(-1) < 1600, 'final partial chunk must be flushed');
  assert.ok(actual.frequency > 420 && actual.frequency < 460, `440Hz remains correct after resampling to 16kHz: ${JSON.stringify(actual)}`);
  assert.ok(actual.maxLevel > .08 && actual.maxLevel < .2, 'level reflects real captured oscillator samples');
  assert.equal(actual.tracksEnded, true);
  assert.equal(actual.running, false);
  console.log(JSON.stringify({ status: 'passed', scope: 'real browser AudioWorklet with synthetic shared MediaStream; no native OS sharing', ...actual }));
} finally {
  if (browser) await browser.close();
  await new Promise(resolve => server.close(resolve));
}
