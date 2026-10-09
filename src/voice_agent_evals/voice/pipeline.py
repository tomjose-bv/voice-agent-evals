"""Transport-agnostic voice session: audio in -> VAD -> STT -> agent -> TTS -> audio out.

The websocket server is a thin wrapper around ``VoiceSession``; tests drive it
directly with fake STT/TTS engines.

Events sent to the client (JSON dicts, plus raw WAV bytes for audio):
  {"type": "ready", "greeting": ...}
  {"type": "vad", "state": "speech_start" | "speech_end"}
  {"type": "partial", "text": ...}            interim transcript while speaking
  {"type": "final", "text": ...}              final transcript of an utterance
  {"type": "agent", "id": n, "text": ..., "escalated": bool, "handoff": {...} | None}
  {"type": "audio", "id": n, "bytes": len}    followed by one binary WAV message
  {"type": "cancel", "id": n, "reason": ...}  barge-in: drop response n
  {"type": "escalated", "handoff": {...}, "summary": "..."}
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from ..agent.base import Agent
from ..agent.handoff import handoff_text
from .audio import SAMPLE_RATE, EnergyVAD
from .engines import STT, TTS

Send = Callable[[Any], Awaitable[None]]


class VoiceSession:
    def __init__(
        self,
        agent: Agent,
        stt: STT,
        tts: TTS,
        send: Send,
        vad: EnergyVAD | None = None,
        partial_interval_s: float = 0.8,
    ) -> None:
        self.agent = agent
        self.stt = stt
        self.tts = tts
        self.send = send
        self.vad = vad or EnergyVAD()
        self.partial_interval_s = partial_interval_s
        self.response_id = 0
        self.response_task: asyncio.Task[None] | None = None
        self.client_playing = False
        self.cancelled: list[int] = []
        self.interrupted_turns = 0
        self.timings: list[dict[str, float]] = []
        self._last_partial_samples = 0
        self._partial_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------- lifecycle
    async def start(self) -> None:
        greet = self.agent.greet()
        await self.send({"type": "ready", "greeting": greet.text})
        await self._speak(greet.text, self._next_id())

    def _next_id(self) -> int:
        self.response_id += 1
        return self.response_id

    @property
    def busy(self) -> bool:
        return self.response_task is not None and not self.response_task.done()

    # ----------------------------------------------------------------- input
    async def feed_audio(self, pcm: bytes) -> None:
        for ev in self.vad.feed(pcm):
            if ev.kind == "speech_start":
                await self.send({"type": "vad", "state": "speech_start"})
                self._last_partial_samples = 0
                if self.busy or self.client_playing:
                    await self.barge_in("user started speaking")
            elif ev.kind == "speech_end":
                await self.send({"type": "vad", "state": "speech_end"})
                await self._finalise(ev.audio)
        if self.vad.in_speech:
            samples = len(self.vad.current_utterance) // 2
            if samples - self._last_partial_samples >= self.partial_interval_s * SAMPLE_RATE:
                self._last_partial_samples = samples
                if self._partial_task is None or self._partial_task.done():
                    self._partial_task = asyncio.create_task(self._partial(self.vad.current_utterance))

    async def _partial(self, pcm: bytes) -> None:
        text = await asyncio.to_thread(self.stt.transcribe, pcm, False)
        if text and self.vad.in_speech:
            await self.send({"type": "partial", "text": text})

    async def _finalise(self, pcm: bytes) -> None:
        t0 = time.perf_counter()
        text = await asyncio.to_thread(self.stt.transcribe, pcm, True)
        stt_ms = (time.perf_counter() - t0) * 1000
        text = (text or "").strip()
        if not text:
            return
        await self.send({"type": "final", "text": text, "stt_ms": round(stt_ms, 1)})
        await self.handle_text(text, stt_ms=stt_ms)

    async def handle_text(self, text: str, stt_ms: float = 0.0) -> None:
        """Respond to a finished utterance (from STT or typed)."""
        if self.agent.ended:
            await self.send({"type": "agent", "id": 0, "text": "This call has ended.", "escalated": self.agent.escalated,
                             "handoff": getattr(self.agent, "handoff", None)})
            return
        if self.busy:
            await self.barge_in("new utterance")
        rid = self._next_id()
        self.response_task = asyncio.create_task(self._respond(rid, text, stt_ms))

    async def handle_control(self, msg: dict[str, Any]) -> None:
        kind = msg.get("type")
        if kind == "text":
            await self.handle_text(str(msg.get("text", "")).strip())
        elif kind == "barge_in":
            await self.barge_in("client detected speech during playback")
        elif kind == "playback":
            self.client_playing = msg.get("state") == "playing"

    # ---------------------------------------------------------------- output
    async def _respond(self, rid: int, text: str, stt_ms: float) -> None:
        try:
            t0 = time.perf_counter()
            turn = await asyncio.to_thread(self.agent.respond, text)
            agent_ms = (time.perf_counter() - t0) * 1000
            handoff = getattr(self.agent, "handoff", None)
            await self.send({"type": "agent", "id": rid, "text": turn.text, "escalated": turn.escalated,
                             "handoff": handoff, "tools": [c.name for c in turn.tool_calls],
                             "agent_ms": round(agent_ms, 1)})
            tts_ms = await self._speak(turn.text, rid)
            self.timings.append({"stt_ms": stt_ms, "agent_ms": agent_ms, "tts_ms": tts_ms})
            if turn.escalated and handoff:
                await self.send({"type": "escalated", "handoff": handoff, "summary": handoff_text(handoff)})
        except asyncio.CancelledError:
            self.interrupted_turns += 1
            raise

    async def _speak(self, text: str, rid: int) -> float:
        t0 = time.perf_counter()
        wav = await asyncio.to_thread(self.tts.synthesize, text)
        tts_ms = (time.perf_counter() - t0) * 1000
        if wav and rid not in self.cancelled:
            await self.send({"type": "audio", "id": rid, "bytes": len(wav), "tts_ms": round(tts_ms, 1)})
            await self.send(wav)
        return tts_ms

    async def barge_in(self, reason: str) -> None:
        """Stop the current response: cancel pending work and tell the client to stop playback."""
        rid = self.response_id
        if rid not in self.cancelled:
            self.cancelled.append(rid)
        if self.busy:
            assert self.response_task is not None
            self.response_task.cancel()
            try:
                await self.response_task
            except (asyncio.CancelledError, Exception):
                pass
        self.client_playing = False
        await self.send({"type": "cancel", "id": rid, "reason": reason})
