"""FastAPI app: serves the browser client and one websocket per call."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from ..agent import make_agent
from ..backend import MockBackend
from ..config import Settings
from .engines import STT, TTS, FakeSTT, FasterWhisperSTT, make_tts
from .pipeline import VoiceSession

STATIC = Path(__file__).parent / "static"


def create_app(
    agent_kind: str = "scripted",
    stt: str | STT = "whisper",
    tts: str | TTS | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    settings = settings or Settings.load()
    engines: dict[str, Any] = {}

    def get_stt() -> STT:
        if "stt" not in engines:
            if not isinstance(stt, str):
                engines["stt"] = stt
            elif stt == "fake":
                engines["stt"] = FakeSTT()
            else:
                print(f"[stt] loading faster-whisper {settings.stt_model} (first run downloads the model)")
                engines["stt"] = FasterWhisperSTT(settings.stt_model)
        return engines["stt"]

    def get_tts() -> TTS:
        if "tts" not in engines:
            engines["tts"] = tts if (tts is not None and not isinstance(tts, str)) else make_tts(
                tts or settings.tts_engine, settings.piper_voice
            )
            print(f"[tts] using {engines['tts'].name}")
        return engines["tts"]

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        # Load models before the first call connects.
        await asyncio.to_thread(get_stt)
        await asyncio.to_thread(get_tts)
        yield

    app = FastAPI(title="voice-agent-evals demo", lifespan=lifespan)

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({
            "ok": True,
            "agent": agent_kind,
            "stt": getattr(engines.get("stt"), "model_name", type(engines.get("stt")).__name__),
            "tts": getattr(engines.get("tts"), "name", None),
        })

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        lock = asyncio.Lock()

        async def send(msg: Any) -> None:
            async with lock:
                if isinstance(msg, (bytes, bytearray)):
                    await websocket.send_bytes(bytes(msg))
                else:
                    await websocket.send_text(json.dumps(msg, default=str))

        agent = make_agent(agent_kind, MockBackend(credit_limit=settings.credit_limit), settings)
        session = VoiceSession(agent, get_stt(), get_tts(), send)
        await session.start()
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    await session.feed_audio(msg["bytes"])
                elif msg.get("text"):
                    try:
                        await session.handle_control(json.loads(msg["text"]))
                    except json.JSONDecodeError:
                        pass
        except WebSocketDisconnect:
            pass
        finally:
            if session.busy and session.response_task:
                session.response_task.cancel()

    return app
