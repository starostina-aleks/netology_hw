import hashlib
import json
import time
from loguru import logger

class LLMCache:
    """In-memory кеш с TTL."""

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._cache: dict[str, tuple[str, float]] = {}
        self.ttl = ttl_seconds
        self.hits = 0
        self.misses = 0

    def _make_key(
        self, model: str, messages: list[dict], temperature: float = 0
    ) -> str:
        """Ключ = хеш(модель + параметры + промпт)."""
        data = json.dumps(
            {"model": model, "messages": messages, "temperature": temperature},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(data.encode()).hexdigest()

    def get(self, model: str, messages: list[dict], temperature: float = 0) -> str | None:
        key = self._make_key(model, messages, temperature)
        if key in self._cache:
            value, created_at = self._cache[key]
            if time.time() - created_at < self.ttl:
                self.hits += 1
                return value
            del self._cache[key]  # TTL истёк
        self.misses += 1
        return None

    def set(self, model: str, messages: list[dict], temperature: float, response: str) -> None:
        key = self._make_key(model, messages, temperature)
        self._cache[key] = (response, time.time())

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total * 100 if total > 0 else 0.0

    def stats(self) -> dict[str, str | int]:

        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hit_rate,
            "keys": len(self._cache),
        }

from loguru import logger
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from google.colab import userdata




def build_client() -> OpenAI:
    """Создаёт клиент OpenAI с проверкой API-ключа."""
    return OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=userdata.get('OPENROUTER_API_KEY'),
                )


def chat_with_cache(
    client: Any,
    messages: list[dict],
    model: str = "gpt-4o-mini",
    temperature: float = 0,
    cache: Any | None = None,
) -> str:
    """Запрос к LLM с кешированием."""
    # 1. Проверяем кеш
    if cache:
        cached = cache.get(model, messages, temperature)
        if cached:
            logger.info("Ответ из кеша")
            return cached

    # 2. Запрос к API
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=60,
    )
    answer = response.choices[0].message.content

    # 3. Сохраняем в кеш (только детерминированные ответы)
    if cache and temperature == 0:
        cache.set(model, messages, temperature, answer)
        logger.info("Сохранено в кеш (tokens: {:d})", response.usage.total_tokens)

    return answer

user_prompts=[
    "Что такое REST API?",
    '''что значит @property
    def hit_rate(self) -> float:''',
    "Что такое REST API?",
    "Как начать использовать Redis",
    "time.time() python"


]
def main() -> None:
    """Демонстрация интеграции кеша с LLM-клиентом."""

    client = build_client()
    cache = LLMCache(ttl_seconds=3600)  # 1 час



    for user_prompt in user_prompts:
      logger.info(f"Вопрос: {user_prompt}")
      messages = [
        {"role": "system", "content": "Ты сеньор python разработчик. Отвечай на русском"},
        {"role": "user", "content": user_prompt},
        ]

      answer = chat_with_cache(client, messages, cache=cache)  # API
      logger.info(f"\nОтвет: {answer}")
      stats = cache.stats()
      logger.info(
            "Cache stats: keys={}, hits={}, misses={}, hit_rate={:.1f}%",
            stats["keys"], stats["hits"], stats["misses"], stats["hit_rate"],
        )

if __name__ == "__main__":
    main()

'''
2026-06-02 12:06:15.624 | INFO     | __main__:main:70 - Вопрос: Что такое REST API?
2026-06-02 12:06:17.495 | INFO     | __main__:chat_with_cache:47 - Сохранено в кеш (tokens: 89)
2026-06-02 12:06:17.496 | INFO     | __main__:main:77 -
Ответ: REST API (Representational State Transfer Application Programming Interface) — это архитектурный стиль для разработки веб-сервисов, который использует стандартные HTTP-протоколы для обмена данными между клиентом и сервером. Основные принципы REST включают:

1. **Стат
2026-06-02 12:06:17.498 | INFO     | __main__:main:79 - Cache stats: keys=1, hits=0, misses=1, hit_rate=0.0%
2026-06-02 12:06:17.500 | INFO     | __main__:main:70 - Вопрос: что значит @property
    def hit_rate(self) -> float:
2026-06-02 12:06:19.116 | INFO     | __main__:chat_with_cache:47 - Сохранено в кеш (tokens: 98)
2026-06-02 12:06:19.117 | INFO     | __main__:main:77 -
Ответ: Декоратор `@property` в Python используется для создания свойств (properties) в классах. Это позволяет вам определять методы, которые можно вызывать как атрибуты, что делает код более чистым и удобным для чтения.

Когда вы используете `@property`,
2026-06-02 12:06:19.119 | INFO     | __main__:main:79 - Cache stats: keys=2, hits=0, misses=2, hit_rate=0.0%
2026-06-02 12:06:19.119 | INFO     | __main__:main:70 - Вопрос: Что такое REST API?
2026-06-02 12:06:19.121 | INFO     | __main__:chat_with_cache:32 - Ответ из кеша
2026-06-02 12:06:19.121 | INFO     | __main__:main:77 -
Ответ: REST API (Representational State Transfer Application Programming Interface) — это архитектурный стиль для разработки веб-сервисов, который использует стандартные HTTP-протоколы для обмена данными между клиентом и сервером. Основные принципы REST включают:

1. **Стат
2026-06-02 12:06:19.132 | INFO     | __main__:main:79 - Cache stats: keys=2, hits=1, misses=2, hit_rate=33.3%
2026-06-02 12:06:19.133 | INFO     | __main__:main:70 - Вопрос: Как начать использовать Redis
2026-06-02 12:06:20.810 | INFO     | __main__:chat_with_cache:47 - Сохранено в кеш (tokens: 88)
2026-06-02 12:06:20.811 | INFO     | __main__:main:77 -
Ответ: Чтобы начать использовать Redis, выполните следующие шаги:

### 1. Установка Redis

#### На Windows:
- Вы можете использовать WSL (Windows Subsystem for Linux) и установить Redis через пакетный менеджер, например, с помощью `apt`:
  ```bash
  sudo apt
2026-06-02 12:06:20.813 | INFO     | __main__:main:79 - Cache stats: keys=3, hits=1, misses=3, hit_rate=25.0%
2026-06-02 12:06:20.814 | INFO     | __main__:main:70 - Вопрос: time.time() python
2026-06-02 12:06:21.979 | INFO     | __main__:chat_with_cache:47 - Сохранено в кеш (tokens: 88)
2026-06-02 12:06:21.981 | INFO     | __main__:main:77 -
Ответ: `time.time()` — это функция из модуля `time` в Python, которая возвращает текущее время в секундах с начала эпохи (обычно это 1 января 1970 года, 00:00:00 UTC). Значение возвращается в виде числа с
2026-06-02 12:06:21.983 | INFO     | __main__:main:79 - Cache stats: keys=4, hits=1, misses=4, hit_rate=20.0%
'''