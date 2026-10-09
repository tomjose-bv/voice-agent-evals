"""Settings loaded from environment variables / a local .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Settings:
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.1:8b"
    llm_api_key: str = "ollama"
    caller_model: str = ""
    judge_model: str = ""
    llm_timeout_s: float = 120.0
    credit_limit: float = 50.0
    stt_model: str = "tiny.en"
    tts_engine: str = "auto"
    piper_voice: str = ""

    @classmethod
    def load(cls) -> "Settings":
        load_dotenv(override=False)
        model = _env("VAE_LLM_MODEL", cls.llm_model)
        return cls(
            llm_base_url=_env("VAE_LLM_BASE_URL", cls.llm_base_url),
            llm_model=model,
            llm_api_key=_env("VAE_LLM_API_KEY", cls.llm_api_key),
            caller_model=_env("VAE_CALLER_MODEL", "") or model,
            judge_model=_env("VAE_JUDGE_MODEL", "") or model,
            llm_timeout_s=float(_env("VAE_LLM_TIMEOUT_S", str(cls.llm_timeout_s))),
            credit_limit=float(_env("VAE_CREDIT_LIMIT", str(cls.credit_limit))),
            stt_model=_env("VAE_STT_MODEL", cls.stt_model),
            tts_engine=_env("VAE_TTS_ENGINE", cls.tts_engine),
            piper_voice=_env("VAE_PIPER_VOICE", cls.piper_voice),
        )


def enable_system_trust_store() -> None:
    """Use the OS certificate store (works behind corporate TLS inspection).

    Verification stays fully enabled; we only change *which* roots are trusted.
    """
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception:  # pragma: no cover - truststore is a hard dependency
        pass
