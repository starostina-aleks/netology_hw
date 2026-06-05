"""Конфигурация приложения.

Загружает настройки из переменных окружения (.env).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(slots=True)
class Settings:
    service_name: str
    # Общие настройки
    request_timeout_seconds: float
    retry_attempts: int
    history_limit: int
    log_path: Path
    redis_host: str
    redis_port: int
    redis_ttl: int
    openrouter_api_key:str
    open_api_key:str
    ans_language:str

    # Провайдеры в порядке приоритета
    

    @classmethod
    def from_env(cls) -> "Settings":
        base_dir = Path(__file__).resolve().parent.parent
        return cls(
            service_name=os.getenv("SUPPORT_SERVICE_NAME", "CloudBox"),
            request_timeout_seconds=float(os.getenv("SUPPORT_TIMEOUT_SECONDS", "30")),
            retry_attempts=int(os.getenv("SUPPORT_RETRY_ATTEMPTS", "3")),
            history_limit=int(os.getenv("SUPPORT_HISTORY_LIMIT", "10")),
            log_path=Path(os.getenv("SUPPORT_LOG_PATH", base_dir / "assistant.log")),
            redis_host=os.getenv("REDIS_HOST", "localhost"),
            redis_port=int(os.getenv("REDIS_PORT", "6379")),
            redis_ttl=int(os.getenv("REDIS_TTL", "3600")),
            openrouter_api_key=os.getenv("OPENROUTER_API_KEY"), 
            open_api_key=os.getenv("OPENAI_API_KEY"),
            ans_language=os.getenv("LANGUAGE")
            
        )