import hashlib
import json

import redis
from loguru import logger


class RedisLLMCache:
    """Production-кеш на Redis."""

    def __init__(self, host: str = "localhost", port: int = 6379, ttl: int = 3600) -> None:
        self.client = redis.Redis(host=host, port=port, decode_responses=True)
        self.ttl = ttl

    def _make_key(self, model: str, messages: list[dict], temperature: float) -> str:
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return f"llm:{hashlib.sha256(data.encode()).hexdigest()}"

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str | None:
        key = self._make_key(model, messages, temperature)
        value = self.client.get(key)
        if value is not None:
            logger.debug("Cache hit: {}", key)
        else:
            logger.debug("Cache miss: {}", key)
        return value

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self.client.setex(key, self.ttl, response)
        logger.debug("Cached with TTL={}s: {}", self.ttl, key)

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