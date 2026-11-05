import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';

const source = await readFile(new URL('../js/session.js', import.meta.url), 'utf8');
const { SessionClient } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

test('routes live speaker changes separately from final transcript lines', () => {
  const speakers = [];
  const utterances = [];
  const client = new SessionClient({
    onSpeaker: (speaker) => speakers.push(speaker),
    onUtterance: (utterance) => utterances.push(utterance),
  });
  const speaker = { type: 'speaker', speaker_id: 'bob', speaker_name: 'Bob', start: 1.2, confidence: 0.9 };
  client._onMessage({ data: JSON.stringify(speaker) });
  assert.deepEqual(speakers, [speaker]);
  assert.deepEqual(utterances, []);
  const utterance = { type: 'utterance', speaker_id: 'alice', text: 'Earlier speech' };
  client._onMessage({ data: JSON.stringify(utterance) });
  assert.deepEqual(utterances, [utterance]);
  assert.deepEqual(speakers, [speaker]);
});

test('clients without a live-speaker handler accept speaker events', () => {
  const client = new SessionClient();
  assert.doesNotThrow(() => client._onMessage({ data: JSON.stringify({ type: 'speaker', speaker_id: null, speaker_name: 'Unknown' }) }));
});
