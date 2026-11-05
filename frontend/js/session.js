// WebSocket client for /ws/session (see docs/SPEC.md).

const MAX_BUFFERED_BYTES = 256 * 1024; // ~8 s of 16 kHz PCM16; drop audio beyond this
const CONNECT_TIMEOUT_MS = 10000;

export function sessionUrl() {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  return `${proto}//${location.host}/ws/session`;
}

export class SessionClient {
  /**
   * @param {{
   *   onStatus?: (state: 'listening'|'stopped') => void,
   *   onSpeech?: (active: boolean) => void,
   *   onSpeaker?: (speaker: object) => void,
   *   onUtterance?: (u: object) => void,
   *   onError?: (message: string) => void,
   *   onClose?: (info: {expected: boolean, code: number, reason: string}) => void,
   * }} handlers
   */
  constructor(handlers = {}) {
    this.handlers = handlers;
    this.ws = null;
    this.closing = false; // true once we initiated the close
    this.closed = false;
    this.droppedChunks = 0;
  }

  /** Open the socket. Resolves when open; rejects if it fails (or takes too long) to connect. */
  connect(url = sessionUrl(), timeoutMs = CONNECT_TIMEOUT_MS) {
    return new Promise((resolve, reject) => {
      let opened = false;
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (err) {
        reject(new Error(`Could not open connection: ${err.message || err}`));
        return;
      }
      ws.binaryType = 'arraybuffer';
      this.ws = ws;

      const timer = setTimeout(() => {
        if (opened) return;
        reject(new Error('Timed out connecting to the server for live listening.'));
        this.close();
      }, timeoutMs);

      ws.onopen = () => {
        clearTimeout(timer);
        if (this.closing) return; // gave up (timed out or closed) before it opened
        opened = true;
        resolve();
      };
      ws.onmessage = (e) => this._onMessage(e);
      ws.onerror = () => {
        // A close event always follows; handled there.
      };
      ws.onclose = (e) => {
        clearTimeout(timer);
        this.closed = true;
        if (!opened) {
          reject(new Error('Could not connect to the server for live listening.'));
          return;
        }
        const cb = this.handlers.onClose;
        if (cb) cb({ expected: this.closing, code: e.code, reason: e.reason || '' });
      };
    });
  }

  get isOpen() {
    return !!this.ws && this.ws.readyState === WebSocket.OPEN;
  }

  _sendJson(obj) {
    if (!this.isOpen) return false;
    this.ws.send(JSON.stringify(obj));
    return true;
  }

  start() {
    return this._sendJson({ type: 'start' });
  }

  stop() {
    return this._sendJson({ type: 'stop' });
  }

  /**
   * Send one PCM16 chunk. Returns false if dropped (socket not open or backed up).
   * @param {Int16Array} samples
   */
  sendAudio(samples) {
    if (!this.isOpen) return false;
    if (this.ws.bufferedAmount > MAX_BUFFERED_BYTES) {
      this.droppedChunks++;
      return false;
    }
    // Send exactly this view's bytes (the view may not span its whole buffer).
    const bytes =
      samples.byteOffset === 0 && samples.byteLength === samples.buffer.byteLength
        ? samples.buffer
        : samples.buffer.slice(samples.byteOffset, samples.byteOffset + samples.byteLength);
    this.ws.send(bytes);
    return true;
  }

  /** Close the socket (expected close: onClose gets expected=true). */
  close() {
    this.closing = true;
    if (this.ws && this.ws.readyState <= WebSocket.OPEN) {
      try {
        this.ws.close(1000, 'client done');
      } catch {
        /* ignore */
      }
    }
  }

  _onMessage(e) {
    if (typeof e.data !== 'string') return;
    let msg;
    try {
      msg = JSON.parse(e.data);
    } catch {
      console.warn('Ignoring non-JSON message from server', e.data);
      return;
    }
    const h = this.handlers;
    switch (msg && msg.type) {
      case 'status':
        if (h.onStatus) h.onStatus(msg.state);
        break;
      case 'speech':
        if (h.onSpeech) h.onSpeech(!!msg.active);
        break;
      case 'speaker':
        if (h.onSpeaker) h.onSpeaker(msg);
        break;
      case 'utterance':
        if (h.onUtterance) h.onUtterance(msg);
        break;
      case 'error':
        if (h.onError) h.onError(String(msg.message || 'Unknown server error'));
        break;
      default:
        console.warn('Ignoring unknown message from server', msg);
    }
  }
}
