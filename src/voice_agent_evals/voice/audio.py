"""PCM helpers and a simple energy-based voice activity detector (VAD).

Audio format everywhere: 16 kHz, mono, signed 16-bit little-endian PCM.
"""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def pcm_bytes_to_float(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0


def float_to_pcm_bytes(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def tone(seconds: float, freq: float = 220.0, amp: float = 0.3, sample_rate: int = SAMPLE_RATE) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    return float_to_pcm_bytes(amp * np.sin(2 * np.pi * freq * t))


def silence(seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    return b"\x00\x00" * int(seconds * sample_rate)


@dataclass
class VADEvent:
    kind: str  # "speech_start" | "speech_end"
    audio: bytes = b""  # full utterance for speech_end


@dataclass
class EnergyVAD:
    """Frame-energy VAD with an adaptive noise floor and hangover.

    Deliberately simple and dependency-free. Speech starts after ``start_frames``
    consecutive loud frames and ends after ``end_frames`` quiet frames.
    """

    threshold_db: float = 12.0  # dB above the noise floor that counts as speech
    min_rms: float = 0.01  # absolute floor so digital silence never counts
    start_frames: int = 3  # 90 ms
    end_frames: int = 20  # 600 ms
    pre_roll_frames: int = 10  # keep 300 ms before speech start
    max_utterance_s: float = 15.0
    noise_floor: float = 0.003
    in_speech: bool = False
    _buf: bytearray = field(default_factory=bytearray)
    _utt: bytearray = field(default_factory=bytearray)
    _pre: list[bytes] = field(default_factory=list)
    _loud: int = 0
    _quiet: int = 0

    def reset(self) -> None:
        self.in_speech = False
        self._buf.clear()
        self._utt.clear()
        self._pre.clear()
        self._loud = self._quiet = 0

    @property
    def current_utterance(self) -> bytes:
        return bytes(self._utt)

    def _is_loud(self, frame: bytes) -> bool:
        x = pcm_bytes_to_float(frame)
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        loud = rms > max(self.min_rms, self.noise_floor * (10 ** (self.threshold_db / 20)))
        if not loud and not self.in_speech:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * max(rms, 1e-4)
        return loud

    def feed(self, pcm: bytes) -> list[VADEvent]:
        events: list[VADEvent] = []
        self._buf.extend(pcm)
        nbytes = FRAME_SAMPLES * 2
        while len(self._buf) >= nbytes:
            frame = bytes(self._buf[:nbytes])
            del self._buf[:nbytes]
            loud = self._is_loud(frame)
            if not self.in_speech:
                self._pre.append(frame)
                if len(self._pre) > self.pre_roll_frames:
                    self._pre.pop(0)
                self._loud = self._loud + 1 if loud else 0
                if self._loud >= self.start_frames:
                    self.in_speech = True
                    self._quiet = 0
                    self._utt = bytearray(b"".join(self._pre))
                    self._pre.clear()
                    events.append(VADEvent("speech_start"))
            else:
                self._utt.extend(frame)
                self._quiet = 0 if loud else self._quiet + 1
                too_long = len(self._utt) / 2 / SAMPLE_RATE >= self.max_utterance_s
                if self._quiet >= self.end_frames or too_long:
                    events.append(VADEvent("speech_end", bytes(self._utt)))
                    self.in_speech = False
                    self._utt.clear()
                    self._loud = 0
        return events
