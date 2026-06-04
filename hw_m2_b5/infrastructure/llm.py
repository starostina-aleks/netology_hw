from __future__ import annotations

from .budget_tracker import BudgetTracker
from openai import OpenAI
from config import Settings
from tenacity import retry, retry_if_exception,wait_exponential_jitter,stop_after_attempt

from openai import RateLimitError, APIStatusError, APITimeoutError
from typing import Any
from loguru import logger

FALLBACK_ANSWER = "Передаю вопрос специалисту."

class RobustLLMClient:
    """LLM-клиент с retry, fallback и логированием.

    Объединяет все паттерны надёжности в одном классе:
    - Exponential backoff через tenacity при 429 (rate limit)
    - Fallback-цепочка: если основной провайдер упал — переключаемся на резервный
    - Логирование успехов и ошибок для мониторинга
    """

    def __init__(self,tracker:BudgetTracker, settings: Settings) -> None:
        
        self.settings = settings
        # Провайдеры в порядке приоритета
        self.providers = [

            {
                "name": "OpenAI",
                "client": OpenAI(
                    base_url="https://api.openai.com/v1",
                    api_key=settings.open_api_key,
                ),
                "model": "gpt-4o-mini",
            },
            {
                "name": "OpenRouter",
                "client": OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                ),
                "model": "gpt-4o-mini",
            },
            {
                "name": "Ollama (локальный)",                 
                 "client": OpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                ),
                "model": "gemma3:1b",               
            }
        ]

        self.tracker=tracker

    def _call_provider(
            self, 
            client: Any, 
            model: str, 
            messages: list[dict],
            temperature: float = 0.2,
            max_tokens: int = 250,) -> Any:
        
        """Вызов одного провайдера с retry через tenacity."""
        
        def should_retry(error: BaseException) -> bool:
            if isinstance(error, (RateLimitError,APITimeoutError)):
                return True
            if isinstance(error, APIStatusError) and error.status_code >= 500:
                return True
            return False

        # кастомная функцию для логирования
        def log_retry(state):
            exc = state.outcome.exception()
            status = getattr(exc, "status_code", None)  # None для APITimeoutError
            sleep_for = state.next_action.sleep if state.next_action else 0.0
            logger.warning(
             "retry attempt=%d exc=%s status=%s sleep=%.2fs",
             state.attempt_number, type(exc).__name__, status, sleep_for,
             )    
        @retry(
            wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
            stop=stop_after_attempt(5),
            retry=retry_if_exception(should_retry),
            before_sleep=log_retry,
        )
        def _do_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=self.settings.request_timeout_seconds,
            )

        return _do_call()

    def chat(self, messages: list[dict]) -> str:
        """Отправляет сообщения, пробуя провайдеров по порядку.
        Возвращает текст ответа от первого успешного провайдера.
        Если все провайдеры упали — бросает RuntimeError.
        """
        for provider in self.providers:
            try:
                response = self._call_provider(
                    provider["client"],
                    provider["model"],
                    messages,
                )
                if response.usage:
                    usage = response.usage
                    self.tracker.track_call(provider["model"], usage.prompt_tokens, usage.completion_tokens)
                
                return (response.choices[0].message.content or "").strip()
            except Exception as e:
                logger.warning(f"{provider['name']} failed: {e}")
                continue

        raise RuntimeError("Сервис временно недоступен")
    
    