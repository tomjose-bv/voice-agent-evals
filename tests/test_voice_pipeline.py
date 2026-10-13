import asyncio
import json
import os
from pathlib import Path

import pytest

pytest.importorskip("numpy")

from voice_agent_evals.agent import make_agent
from voice_agent_evals.backend import MockBackend
from voice_agent_evals.voice.audio import EnergyVAD, silence, tone, wav_bytes
from voice_agent_evals.voice.engines import FakeSTT, FakeTTS, NullTTS, wav_duration_s
from voice_agent_evals.voice.pipeline import VoiceSession

ROOT = Path(__file__).resolve().parents[1]


class Recorder:
    def __init__(self):
        self.events = []

    async def __call__(self, msg):
        self.events.append(msg)

    def of(self, kind):
        return [e for e in self.events if isinstance(e, dict) and e.get("type") == kind]

    @property
    def audio(self):
        return [e for e in self.events if isinstance(e, bytes)]


def utterance(seconds=1.0):
    return silence(0.3) + tone(seconds) + silence(0.9)


def make_session(transcripts, tts=None, **kw):
    rec = Recorder()
    agent = make_agent("scripted", MockBackend())
    s = VoiceSession(agent, FakeSTT(transcripts), tts or FakeTTS(), rec, **kw)
    return s, rec


async def feed(session, pcm, chunk=1024, pace_s=0.0):
    for i in range(0, len(pcm), chunk):
        await session.feed_audio(pcm[i:i + chunk])
        if pace_s:
            await asyncio.sleep(pace_s)  # let background partial transcription run, as in real time
    if session.response_task:
        await session.response_task


# ---- VAD -------------------------------------------------------------------

def test_vad_detects_one_utterance_and_ignores_silence():
    vad = EnergyVAD()
    events = vad.feed(silence(1.0))
    assert events == []
    events = vad.feed(tone(1.0) + silence(1.0))
    kinds = [e.kind for e in events]
    assert kinds == ["speech_start", "speech_end"]
    dur = len(events[1].audio) / 2 / 16000
    assert 1.0 <= dur <= 2.0  # utterance + pre-roll + hangover


def test_vad_splits_long_speech():
    vad = EnergyVAD(max_utterance_s=2.0)
    kinds = [e.kind for e in vad.feed(tone(4.5))]
    assert kinds.count("speech_end") == 2


def test_wav_helpers():
    w = wav_bytes(tone(0.5))
    assert w[:4] == b"RIFF" and abs(wav_duration_s(w) - 0.5) < 0.01
    assert NullTTS().synthesize("hi") == b""


# ---- pipeline with fakes ----------------------------------------------------

def test_greeting_is_spoken():
    async def go():
        s, rec = make_session([])
        await s.start()
        return rec
    rec = asyncio.run(go())
    assert "automated assistant" in rec.of("ready")[0]["greeting"]
    assert rec.of("audio") and rec.audio and rec.audio[0][:4] == b"RIFF"


def test_full_turn_partial_final_agent_audio():
    async def go():
        s, rec = make_session(["Is there an outage at SE5 9ZZ?"], partial_interval_s=0.3)
        await s.start()
        await feed(s, utterance(1.2), pace_s=0.01)
        if s._partial_task:
            await s._partial_task
        return s, rec
    s, rec = asyncio.run(go())
    assert rec.of("partial"), "expected at least one partial transcript"
    assert rec.of("final")[0]["text"] == "Is there an outage at SE5 9ZZ?"
    agent = rec.of("agent")[0]
    assert "18:00" in agent["text"] and agent["tools"] == ["check_outage"]
    assert len(rec.audio) == 2  # greeting + reply
    assert s.timings and set(s.timings[0]) == {"stt_ms", "agent_ms", "tts_ms"}


def test_escalation_event_with_handoff():
    async def go():
        s, rec = make_session(["Let me talk to a real person"])
        await s.start()
        await feed(s, utterance())
        return rec
    rec = asyncio.run(go())
    esc = rec.of("escalated")
    assert esc and esc[0]["handoff"]["reason"] == "caller asked for a human agent"
    assert "Escalation:" in esc[0]["summary"]


class SlowTTS(FakeTTS):
    def synthesize(self, text):
        import time
        time.sleep(0.5)
        return super().synthesize(text)


def test_barge_in_cancels_pending_response():
    async def go():
        s, rec = make_session(["What's my bill?", "Actually, is there an outage at SE5 9ZZ?"], tts=SlowTTS())
        await s.start()
        # first utterance; don't await the response task so it is still synthesising
        for i in range(0, len(utterance()), 1024):
            await s.feed_audio(utterance()[i:i + 1024])
        await asyncio.sleep(0.05)
        assert s.busy
        first_id = s.response_id
        await feed(s, tone(1.0) + silence(0.9))  # caller talks over the agent
        return s, rec, first_id
    s, rec, first_id = asyncio.run(go())
    cancels = rec.of("cancel")
    assert cancels and cancels[0]["id"] == first_id
    audio_ids = [e["id"] for e in rec.of("audio")]
    assert first_id not in audio_ids  # cancelled reply never reached the client
    assert s.interrupted_turns >= 1
    last = rec.of("agent")[-1]
    assert last["id"] == s.response_id and last["id"] in audio_ids  # the new reply is spoken


def test_client_barge_in_and_typed_text():
    async def go():
        s, rec = make_session([])
        await s.start()
        await s.handle_control({"type": "playback", "state": "playing"})
        await s.handle_control({"type": "barge_in"})
        await s.handle_control({"type": "text", "text": "I want to speak to a manager"})
        await s.response_task
        return s, rec
    s, rec = asyncio.run(go())
    assert rec.of("cancel") and not s.client_playing
    assert rec.of("escalated")


# ---- websocket server ---------------------------------------------------------

def test_websocket_server_end_to_end():
    from fastapi.testclient import TestClient

    from voice_agent_evals.voice.server import create_app

    app = create_app(stt=FakeSTT(["my internet is down, postcode SE5 9ZZ"]), tts=FakeTTS())
    with TestClient(app) as client:
        assert client.get("/health").json()["ok"] is True
        assert "Start call" in client.get("/").text
        with client.websocket_connect("/ws") as ws:
            assert json.loads(ws.receive_text())["type"] == "ready"
            assert json.loads(ws.receive_text())["type"] == "audio"
            assert ws.receive_bytes()[:4] == b"RIFF"
            pcm = utterance()
            for i in range(0, len(pcm), 2048):
                ws.send_bytes(pcm[i:i + 2048])
            seen = {}
            while "audio" not in seen or "agent" not in seen:
                msg = ws.receive()
                if msg.get("text"):
                    m = json.loads(msg["text"])
                    seen[m["type"]] = m
            assert "outage" in seen["agent"]["text"]


# ---- optional real-model round trip ----------------------------------------------

VOICE = ROOT / "models" / "en_US-lessac-medium.onnx"


@pytest.mark.skipif(not VOICE.exists() or os.environ.get("VAE_SKIP_MODEL_TESTS"), reason="needs the Piper voice in models/ and a faster-whisper download (skipped when absent or VAE_SKIP_MODEL_TESTS is set)")
def test_piper_to_whisper_round_trip():
    """Real TTS -> real STT: synthesise a sentence with Piper and transcribe it with faster-whisper."""
    import numpy as np

    from voice_agent_evals.voice.engines import FasterWhisperSTT, PiperTTS, resample_to_16k
    import io
    import wave

    wav = PiperTTS(VOICE).synthesize("My internet is down. My postcode is S E 5, 9 Z Z.")
    with wave.open(io.BytesIO(wav)) as w:
        sr = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768
    pcm = (resample_to_16k(x, sr) * 32767).astype("<i2").tobytes()
    text = FasterWhisperSTT("tiny.en").transcribe(pcm).lower()
    assert "internet" in text and "down" in text
