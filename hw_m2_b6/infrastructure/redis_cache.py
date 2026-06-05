from __future__ import annotations

import hashlib
import json

import redis
from loguru import logger


class RedisLLMCache:
    """Production-кеш на Redis."""

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600) -> None:
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def _make_key(self, messages: list[dict]) -> str:
        data = json.dumps(
            { "messages": messages},
            sort_keys=True,
            ensure_ascii=False,
        )
        return f"llm:{hashlib.sha256(data.encode()).hexdigest()}"
    
    def get(self,  messages: list[dict]) -> str | None:
        key = self._make_key( messages)
        value = self.client.get(key)
        return value

    def set(self,  messages: list[dict], response: str) -> None:
        key = self._make_key( messages)
        self.client.setex(key, self.ttl, response)

    def stats(self) -> dict[str, str | int]:
        info = self.client.info("stats")
        hits = info.get("keyspace_hits", 0)
        misses = info.get("keyspace_misses", 0)
        total = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": f"{hits / total * 100:.1f}%" if total else "N/A",
            "keys": self.client.dbsize(),
        }
    
    def clear(self) -> int:
        """Удаляет все ключи support:* из Redis. Возвращает количество удалённых."""
        deleted = 0
        for key in self.client.scan_iter("llm:*"):
            self.client.delete(key)
            deleted += 1
        return deleted

    def reset_stats(self) -> None:
        """Сбрасывает накопленную статистику Redis (hits/misses и др.)."""
        self.client.config_resetstat()