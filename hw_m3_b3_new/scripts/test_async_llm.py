import asyncio
from loguru import logger
# Импортируем ваш класс из соседнего файла client.py
from infrastructure.async_llm import AsyncLLMClient
from  infrastructure.budget_tracker import BudgetTracker
from config import Settings

# Настройка логов
logger.add(
    "llm_calls.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    rotation="10 MB",
    encoding="utf-8"
)

async def main():
    tracker = BudgetTracker(daily_budget=20, alert_threshold=0.8)
    settings = Settings.from_env()
    # Инициализация клиента (выберите тот вариант init, который вы оставили)
    client = AsyncLLMClient(tracker=tracker,settings=settings)

    try:
        print("Отправка запроса...")
        prompt = "Почему кровь красная? Ответь в одном предложении."
        messages=[{"role": "user", "content": prompt}]
        #response = await client.chat(messages=messages)

        print("\nОтвет от модели:")
        #print(response)

        async for token in client.stream_chat(messages):

            print(token,flush=True,end='')

    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")



if __name__ == "__main__":
    asyncio.run(main())
