"""Speech-to-text and text-to-speech engines (all local), plus fakes for tests."""

from __future__ import annotations

import os
import tempfile
import threading
import io
import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from .audio import SAMPLE_RATE, pcm_bytes_to_float, tone, wav_bytes


class STT(Protocol):
    def transcribe(self, pcm: bytes, final: bool = True) -> str:
        """Transcribe 16 kHz mono int16 PCM. ``final=False`` for interim partials."""
        ...


class TTS(Protocol):
    name: str

    def synthesize(self, text: str) -> bytes:
        """Return a complete WAV file (bytes); b"" means "no audio"."""
        ...


# ----------------------------------------------------------------------- STT
class FasterWhisperSTT:
    """faster-whisper on CPU with int8 weights (tiny.en / base.en recommended)."""

    def __init__(self, model: str = "tiny.en", cpu_threads: int = 4) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=cpu_threads)
        self._lock = threading.Lock()

    def transcribe(self, pcm: bytes, final: bool = True) -> str:
        audio = pcm_bytes_to_float(pcm)
        if audio.size < SAMPLE_RATE // 4:
            return ""
        with self._lock:  # the model isn't safe to call concurrently
            segments, _ = self.model.transcribe(
                audio, language="en", beam_size=1, vad_filter=False, condition_on_previous_text=False
            )
            return " ".join(s.text.strip() for s in segments).strip()


class FakeSTT:
    """Returns scripted transcripts. Partials peek at the next one; finals pop it."""

    def __init__(self, transcripts: list[str] | None = None) -> None:
        self.transcripts = list(transcripts or [])
        self.calls = 0

    def peek(self) -> str:
        return self.transcripts[0] if self.transcripts else ""

    def transcribe(self, pcm: bytes, final: bool = True) -> str:
        self.calls += 1
        if final:
            return self.transcripts.pop(0) if self.transcripts else ""
        return self.peek()


# ----------------------------------------------------------------------- TTS
class PiperTTS:
    name = "piper"

    def __init__(self, voice_path: str | Path) -> None:
        from piper import PiperVoice

        p = Path(voice_path)
        if not p.exists():
            raise FileNotFoundError(f"Piper voice not found: {p}")
        self.voice = PiperVoice.load(str(p))
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        buf = io.BytesIO()
        with self._lock, wave.open(buf, "wb") as w:
            self.voice.synthesize_wav(text, w)
        return buf.getvalue()


class SapiTTS:
    """Windows SAPI voices through pyttsx3 (fallback; robotic but always available)."""

    name = "sapi"

    def __init__(self) -> None:
        import pyttsx3  # noqa: F401  (fail fast if missing)

        self._lock = threading.Lock()

    def synthesize(self, text: str) -> bytes:
        import pyttsx3

        with self._lock:
            fd, path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            try:
                engine = pyttsx3.init()
                engine.save_to_file(text, path)
                engine.runAndWait()
                engine.stop()
                return Path(path).read_bytes()
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass


class NullTTS:
    name = "none"

    def synthesize(self, text: str) -> bytes:
        return b""


class FakeTTS:
    """Short tone whose length scales with the text. For tests."""

    name = "fake"

    def __init__(self, seconds_per_char: float = 0.01) -> None:
        self.seconds_per_char = seconds_per_char
        self.spoken: list[str] = []

    def synthesize(self, text: str) -> bytes:
        self.spoken.append(text)
        return wav_bytes(tone(max(0.1, len(text) * self.seconds_per_char)))


def make_tts(engine: str = "auto", piper_voice: str = "") -> TTS:
    """Pick a TTS engine. ``auto`` = Piper if a voice file exists, else SAPI, else none."""
    engine = (engine or "auto").lower()
    if engine in ("auto", "piper"):
        try:
            return PiperTTS(piper_voice or "models/en_US-lessac-medium.onnx")
        except Exception as exc:
            if engine == "piper":
                raise
            print(f"[tts] Piper unavailable ({exc}); trying SAPI")
    if engine in ("auto", "sapi"):
        try:
            return SapiTTS()
        except Exception as exc:
            if engine == "sapi":
                raise
            print(f"[tts] SAPI unavailable ({exc}); running without audio output")
    return NullTTS()


def wav_duration_s(wav: bytes) -> float:
    if not wav:
        return 0.0
    with wave.open(io.BytesIO(wav)) as w:
        return w.getnframes() / float(w.getframerate())


def resample_to_16k(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return x
    n = int(len(x) * SAMPLE_RATE / sr)
    return np.interp(np.linspace(0, len(x), n, endpoint=False), np.arange(len(x)), x).astype(np.float32)
