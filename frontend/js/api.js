// Thin fetch wrappers for the REST API (see docs/SPEC.md).

const BASE = '/api';

function describeDetail(detail) {
  if (typeof detail === 'string') return detail;
  // FastAPI validation errors: [{loc, msg, type}, ...]
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && d.msg ? `${(d.loc || []).slice(1).join('.') || 'request'}: ${d.msg}` : String(d)))
      .join('; ');
  }
  if (detail != null) return JSON.stringify(detail);
  return '';
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(BASE + path, { headers: { Accept: 'application/json' }, ...options });
  } catch {
    throw new Error('Cannot reach the server. Is it running?');
  }
  if (res.status === 204) return null;
  const text = await res.text();
  let data = null;
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = null;
    }
  }
  if (!res.ok) {
    const msg = (data && describeDetail(data.detail)) || `Request failed (HTTP ${res.status})`;
    const err = new Error(msg);
    err.status = res.status;
    throw err;
  }
  return data;
}

/** @returns {Promise<{status: string, embedder: string, transcriber: string}>} */
export function getHealth() {
  return request('/health');
}

/** @returns {Promise<Array<{id: string, name: string, color: string, num_samples: number, created_at: number}>>} */
export async function listSpeakers() {
  const data = await request('/speakers');
  return (data && data.speakers) || [];
}

/** Enroll a new speaker. @param {string} name @param {Blob} wav */
export function createSpeaker(name, wav) {
  const form = new FormData();
  form.append('name', name);
  form.append('audio', wav, 'sample.wav');
  return request('/speakers', { method: 'POST', body: form });
}

/** Add another voice sample to an existing speaker. */
export function addSample(id, wav) {
  const form = new FormData();
  form.append('audio', wav, 'sample.wav');
  return request(`/speakers/${encodeURIComponent(id)}/samples`, { method: 'POST', body: form });
}

export function deleteSpeaker(id) {
  return request(`/speakers/${encodeURIComponent(id)}`, { method: 'DELETE' });
}

export function deleteAllSpeakers() {
  return request('/speakers', { method: 'DELETE' });
}
