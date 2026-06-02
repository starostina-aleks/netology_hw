PRICES_PER_1M_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4.1": {"input": 2.00, "output": 8.00},
    "gpt-4.1-mini": {"input": 0.40, "output": 1.60},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}

class BudgetTracker:
    """Трекер бюджета: накапливает расходы и предупреждает при превышении порога.

    В production можно хранить расходы в Redis/БД, а алерты отправлять
    в Slack, Telegram или email.
    """

    def __init__(self, daily_budget: float = 10.0, alert_threshold: float = 0.8):
        self.daily_budget = daily_budget
        self.alert_threshold = alert_threshold  # процент бюджета для алерта
        self.daily_total = 0.0
        self.calls_today = 0

    def track_call(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Записывает стоимость вызова и проверяет бюджет."""
        price = PRICES_PER_1M_TOKENS.get(model, {"input": 2.00, "output": 8.00})
        cost = (
            prompt_tokens / 1_000_000 * price["input"]
            + completion_tokens / 1_000_000 * price["output"]
        )

        self.daily_total += cost
        self.calls_today += 1

        # Проверяем пороги
        if self.daily_total > self.daily_budget:
            self._send_alert(
                f"ПРЕВЫШЕН бюджет! ${self.daily_total:.4f} / ${self.daily_budget:.2f}"
            )
        elif self.daily_total > self.daily_budget * self.alert_threshold:
            self._send_alert(
                f"Бюджет на {self.daily_total / self.daily_budget * 100:.0f}%: "
                f"${self.daily_total:.4f} / ${self.daily_budget:.2f}"
            )

    def _send_alert(self, message: str) -> None:
        """Отправляет алерт. В production — Slack/Telegram/email."""
        logger.warning(f"[ALERT] {message}")

    def report(self) -> None:
        """Печатает сводку расходов за день."""
        print(f"\n--- Сводка ---")
        print(f"Вызовов сегодня: {self.calls_today}")
        print(f"Потрачено: ${self.daily_total:.6f} / ${self.daily_budget:.2f}")
        print(f"Осталось: ${self.daily_budget - self.daily_total:.6f}")

# Slide: Собираем всё вместе: надёжный клиент

import os
import logging
from typing import Any
import random
from google.colab import userdata
from datetime import datetime


logging.basicConfig(
    filename='log_llmclient.log',
    level=logging.WARNING,
    format="%(asctime)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True
)
logger = logging.getLogger("api_logger")


class RobustLLMClient:
    """LLM-клиент с retry, fallback и логированием.

    Объединяет все паттерны надёжности в одном классе:
    - Exponential backoff через tenacity при 429 (rate limit)
    - Fallback-цепочка: если основной провайдер упал — переключаемся на резервный
    - Логирование успехов и ошибок для мониторинга
    """

    def __init__(self,tracker:BudgetTracker) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit(
                "Установите зависимость openai: pip install openai"
            ) from exc

        # Провайдеры в порядке приоритета
        self.providers = [

            {
                "name": "OpenAI",
                "client": OpenAI(
                    base_url="https://api.openai.com/v1",
                    api_key=userdata.get('OPENAI_API_KEY'),
                ),
                "model": "gpt-4o-mini",
            },
            {
                "name": "OpenRouter",
                "client": OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=userdata.get('OPENROUTER_API_KEY'),
                ),
                "model": "gpt-4o-mini",
            },

        ]

        self.tracker=tracker

    def _call_provider(self, client: Any, model: str, messages: list[dict]) -> Any:
        """Вызов одного провайдера с retry через tenacity."""
        try:
            from tenacity import (
                retry,
                wait_exponential,
                stop_after_attempt,
                retry_if_exception_type,
                before_sleep_log,
                wait_exponential_jitter,
                wait_random

            )
            from openai import RateLimitError, APIStatusError
        except ImportError as exc:
            raise SystemExit(
                "Установите зависимость tenacity: pip install tenacity"
            ) from exc

        # кастомная функцию для логирования
        def custom_before_sleep_log(retry_state):
            # Номер текущей попытки
            attempt_num = retry_state.attempt_number

            # Сколько секунд будем спать перед следующим шагом
            next_action_time = retry_state.next_action.sleep

            # Какое исключение вызвало сбой
            exception = retry_state.outcome.exception()

            # timestamp, код ошибки, номер попытки
            logger.warning(
            f"Ошибка: {exception}. [Номер попытки #{attempt_num}]. "
            f"Ждем {next_action_time} сек. перед повтором..."
            )
        @retry(
            wait=wait_exponential_jitter(initial=1, max=60, jitter=2),
            stop=stop_after_attempt(5),
            retry=retry_if_exception_type((RateLimitError,APIStatusError)),#перехват ошибок 429, 500
            before_sleep=custom_before_sleep_log,
        )
        def _do_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=messages,
                timeout=30,
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
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"{provider['name']} failed: {e}")
                continue

        raise RuntimeError("Сервис временно недоступен")


def main() -> None:
    """Демонстрация надёжного LLM-клиента с retry и fallback."""
    tracker = BudgetTracker(daily_budget=0.01, alert_threshold=0.8)  # маленький бюджет для демо
    llm = RobustLLMClient(tracker)
    quest="что такое Circuit Breaker pattern и напиши пример на python"
    print(f"\nВопрос: {quest}")
    print("Отправляю запрос через RobustLLMClient...\n")
    answer = llm.chat([
        {"role": "user", "content": quest}
    ])

    print(f"Ответ: {answer}")
    tracker.report()


if __name__ == "__main__":
    main()
