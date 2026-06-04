"""Главный модуль бизнес-логики ассистента поддержки.

Класс ``SupportAssistantApp`` оркестрирует весь цикл обработки запроса:
проверка кеша → вызов LLM (с retry/fallback) → ведение истории → логирование.
"""

from __future__ import annotations

import time
from uuid import uuid4

from loguru import logger

from config import Settings
from models import AssistantResponse, SessionStats
from infrastructure.redis_cache import RedisLLMCache
from infrastructure.llm import FALLBACK_ANSWER, RobustLLMClient
from prompts.loader import build_answer_messages, build_system_prompt 
from  infrastructure.budget_tracker import BudgetTracker

class SupportAssistantApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.system_prompt = build_system_prompt(settings.service_name,settings.ans_language)
        
        self.history: list[dict[str, str]] = []
        self.failed_attempts = 0
        self.cache = RedisLLMCache(settings.redis_host, settings.redis_port, settings.redis_ttl)
        self.tracker = BudgetTracker(daily_budget=0.01, alert_threshold=0.8)
        self.client = RobustLLMClient(self.tracker,settings)

        # Логирование только в файл (убираем дефолтный вывод в stderr)
        logger.remove()
        logger.add(settings.log_path, format="{time} {message}", rotation="10 MB")

    def handle_command(self, command: str) -> str | None:
        if command == "/clear":
            self.history.clear()
            self.failed_attempts = 0
            return "История очищена."
        if command == "/clear_cache":
            deleted = self.cache.clear()
            return f"Кеш очищен. Удалено ключей: {deleted}."
        if command == "/reset_redis_stats":
            self.cache.reset_stats()
            return "Статистика Redis сброшена."
        if command == "/stats":
            cache_info = self.cache.stats()
            return (
                f"Redis: {cache_info['keys']} ключей, "
                f"hit rate: {cache_info['hit_rate']} "
                f"({cache_info['hits']}/{cache_info['hits'] + cache_info['misses']})"
            )
        if command == "/quit":
            return None
        return "Доступные команды: /clear, /clear_cache, /reset_redis_stats, /stats, /quit"

    def respond(self, user_message: str) -> AssistantResponse:
        started_at = time.perf_counter()        
        cached = self.cache.get(user_message)
        
        
        if cached is not None:
            self._remember_turn(user_message, cached)
            latency = time.perf_counter() - started_at
            self._log(user_message, cached, latency, True)
            return cached
        
        messages = build_answer_messages(self.system_prompt, self.history, user_message)
       
        result = self.client.chat(messages)
        latency = time.perf_counter() - started_at
        self.cache.set(user_message, result)  
        self._remember_turn(user_message, result)
        
        self._log(
            user_message,  result,latency, False
        )
        return result
        
    def _remember_turn(self, user_message: str, answer: str) -> None:
        self.history.append({"role": "user", "content": user_message})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.settings.history_limit:
            self.history = self.history[-self.settings.history_limit :]

    def _log(
        self,
        user_message: str,
        answer: str,
        latency_seconds: float,
        from_cache: bool
    ) -> None:
        logger.info(
            "{lat:.3f}s | cache={cache} | Q: {msg} | A: {ans}",
            lat=latency_seconds,
            cache=from_cache,
            msg=user_message[:100],
            ans=answer[:100],
        )        

    