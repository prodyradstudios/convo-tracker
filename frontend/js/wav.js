// Encode 16-bit PCM samples as a WAV (RIFF) Blob. Mono, little-endian.

/**
 * @param {Int16Array | Int16Array[]} chunks  one Int16Array or a list of them
 * @param {number} [sampleRate=16000]
 * @returns {Blob} audio/wav
 */
export function encodeWav(chunks, sampleRate = 16000) {
  const list = chunks instanceof Int16Array ? [chunks] : Array.from(chunks || []);
  let numSamples = 0;
  for (const c of list) numSamples += c.length;

  const numChannels = 1;
  const bytesPerSample = 2;
  const blockAlign = numChannels * bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = numSamples * bytesPerSample;

  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const writeAscii = (offset, s) => {
    for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
  };

  writeAscii(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeAscii(8, 'WAVE');
  writeAscii(12, 'fmt ');
  view.setUint32(16, 16, true); // fmt chunk size (PCM)
  view.setUint16(20, 1, true); // audio format: PCM
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeAscii(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (const c of list) {
    for (let i = 0; i < c.length; i++, offset += 2) view.setInt16(offset, c[i], true);
  }
  return new Blob([buffer], { type: 'audio/wav' });
}
