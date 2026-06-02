from loguru import logger
import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from redis_cache import RedisLLMCache




def build_client() -> OpenAI:
    """Создаёт клиент OpenAI с проверкой API-ключа."""
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise SystemExit("Не найден OPENAI_API_KEY в переменных окружения или .env")

    return OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv('OPENROUTER_API_KEY'),
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
    print('build_client')
    cache = RedisLLMCache(ttl=3600)  # 1 час



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
            "Cache stats: keys={}, hits={}, misses={}, hit_rate={}",
            stats["keys"], stats["hits"], stats["misses"], stats["hit_rate"],
        )

if __name__ == "__main__":
    main()