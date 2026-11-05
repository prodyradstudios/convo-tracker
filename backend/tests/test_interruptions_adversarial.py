"""Independent regressions for fast interruption detection and sample ownership."""
import numpy as np

from app.audio.vad import Segment, Segmenter, VadConfig
from app.speakers.registry import SpeakerRegistry
from app.speakers.streaming import StreamingTurnDetector
from app.speakers.turns import TurnConfig
from conftest import SR, FakeEmbedder, tone


def detector(*frequencies):
    emb = FakeEmbedder()
    registry = SpeakerRegistry(emb.name, emb.default_threshold)
    ids = [registry.enroll(str(f), emb.embed(tone(1, f)))['id'] for f in frequencies]
    return StreamingTurnDetector(emb, registry, TurnConfig(), VadConfig()), ids


def observe_audio(detector, audio):
    changes = []
    for end in range(SR // 10, audio.size + 1, SR // 10):
        change = detector.observe(Segment(audio[:end], 0, end / SR, noise_floor_db=-90))
        if change is not None:
            changes.append((end / SR, change))
    return changes


def test_similar_cheap_fingerprint_cannot_hide_full_model_change_forever():
    """An optimization fingerprint must not veto the selected voice model forever."""
    det, ids = detector(200, 400)
    class ConstantFingerprint:
        def embed(self, audio):
            return np.array([1., 0.], dtype=np.float32)
    det._cheap = ConstantFingerprint()
    changes = observe_audio(det, np.concatenate([tone(2, 200), tone(2, 400)]))
    assert [c.speaker_id for _, c in changes] == ids
    assert changes[-1][0] <= 3.2


def test_gain_changes_do_not_become_speaker_changes():
    det, ids = detector(200, 400)
    changes = observe_audio(det, np.concatenate([
        tone(1.5, 200, amp=.03), tone(1.5, 200, amp=.6), tone(1.5, 200, amp=.03),
    ]))
    assert [c.speaker_id for _, c in changes] == [ids[0]]


def test_split_at_preserves_every_sample_with_pending_partial_frame():
    seg = Segmenter(VadConfig(min_utterance_s=0.0))
    audio = tone(2.0004375, 200)
    assert seg.feed(audio) == []
    prefix = seg.split_at(.81)
    assert prefix is not None
    suffix = seg.flush()
    assert len(suffix) == 1
    joined = np.concatenate([prefix.audio, suffix[0].audio])
    np.testing.assert_array_equal(joined, audio)
    assert prefix.end == suffix[0].start


def test_one_enrolled_speaker_can_be_interrupted_by_unknown():
    det, ids = detector(200)
    changes = observe_audio(det, np.concatenate([tone(2, 200), tone(1, 600)]))
    assert [c.speaker_id for _, c in changes] == [ids[0], None]
    assert changes[-1][0] <= 2.8


def test_ambiguous_identity_never_becomes_a_handoff():
    emb = FakeEmbedder()
    registry = SpeakerRegistry(emb.name, emb.default_threshold)
    # Two voices enrolled as the same pattern: deliberately ambiguous evidence.
    for name in ['A', 'B']:
        registry.enroll(name, emb.embed(tone(1, 200)))
    det = StreamingTurnDetector(emb, registry, TurnConfig(), VadConfig())
    assert observe_audio(det, tone(3, 200)) == []


def test_short_aba_interruptions_preserve_audio_under_irregular_packet_sizes():
    from app.pipeline import SessionPipeline
    from app.transcription.transcriber import NullTranscriber
    audio = np.concatenate([tone(2, 200), tone(.7, 400), tone(2, 200)])
    outputs = []
    for sizes in [[audio.size], [13, 783, 55, 1600, 2323, 347]]:
        det, ids = detector(200, 400)
        pipe = SessionPipeline(det.embedder, NullTranscriber(), det.registry)
        segments, changes = [], []
        pos, i = 0, 0
        while pos < audio.size:
            size = sizes[i % len(sizes)]
            segments.extend(pipe.feed(audio[pos:pos+size]))
            changes.extend(pipe.pop_speaker_events())
            pos += size
            i += 1
        segments.extend(pipe.flush())
        assert [c['speaker_id'] for c in changes] == [ids[0], ids[1], ids[0]]
        np.testing.assert_array_equal(np.concatenate([s.audio for s in segments]), audio)
        assert all(abs(a.end - b.start) < 1e-9 for a, b in zip(segments, segments[1:]))
        outputs.append([s.end for s in segments])
    np.testing.assert_allclose(outputs[0], outputs[1], atol=.09)


def test_live_identity_changes_are_not_queued_behind_slow_transcription():
    import asyncio
    from test_session_handler import GatedTranscriber, make_handler, wait_until

    async def scenario():
        tr = GatedTranscriber()
        handler, ws = make_handler(tr)
        for name, freq in [('A', 200), ('B', 400)]:
            handler.state.registry.enroll(name, handler.state.embedder.embed(tone(1, freq)))
        task = asyncio.create_task(handler.run())
        try:
            ws.text('start')
            ws.audio(np.concatenate([tone(2, 200), tone(.8, 400), tone(1, 200)]))
            await wait_until(lambda: tr.calls >= 1 and ws.inbox.empty(), timeout=4)
            identities = [m['speaker_name'] for m in ws.of_type('speaker')]
            assert identities == ['A', 'B', 'A']
            assert ws.of_type('utterance') == []
            tr.gate.set()
            ws.text('stop')
            await wait_until(lambda: any(m.get('state') == 'stopped' for m in ws.of_type('status')))
            assert not any(m['type'] == 'speaker' for m in ws.sent[ws.sent.index(ws.of_type('status')[-1]) + 1:])
        finally:
            tr.gate.set()
            ws.disconnect()
            await asyncio.wait_for(task, 4)
    asyncio.run(scenario())
