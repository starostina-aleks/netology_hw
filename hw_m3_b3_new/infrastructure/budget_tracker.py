from __future__ import annotations
from loguru import logger

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
