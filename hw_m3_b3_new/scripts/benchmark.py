from __future__ import annotations
import asyncio
from loguru import logger
import time
from infrastructure.async_llm import AsyncLLMClient
from infrastructure.llm import RobustLLMClient
from  infrastructure.budget_tracker import BudgetTracker
from config import Settings


logger.add(
    "llm_call.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
    rotation="10 MB",
    encoding="utf-8"
)
prompts = [
    "Объясни принцип работы блокчейна на примере обычной школьной библиотеки, избегая сложных технических терминов.",
    "Напиши краткий гайд для новичка о том, как нейросети генерируют изображения, уложившись в один абзац.",
    "Опиши концепцию теории относительности Эйнштейна так, чтобы её суть понял десятилетний ребёнок.",
    "Сравни квантовые компьютеры с обычными кремниевыми, выделив три главных технологических отличия в виде списка.",
    "Объясни феномен «зловещей долины» в робототехнике и приведи один яркий пример из кинематографа.",
    "Придумай пять креативных и недорогих идей для маркетинговой кампании локальной кофейни в спальном районе.",
    "Объясни разницу между акциями и облигациями для человека, который впервые решил заняться инвестированием.",
    "Напиши план идеального сопроводительного письма (cover letter) для отклика на вакансию frontend-разработчика.",
    "Сформулируй три главных правила эффективного делегирования задач для начинающего руководителя стартапа.",
    "Объясни концепцию воронки продаж, используя метафору знакомства и построения личных отношений.",
    "Опиши технику тайм-менеджмента «Помодоро» и объясни, почему она помогает бороться с прокрастинацией.",
    "Объясни разницу между синдромом самозванца и объективной неуверенностью в своих силах из-за нехватки опыта.",
    "Напиши три практических совета, как вежливо, но твердо отказывать коллегам, не портя с ними отношения.",
    "Объясни концепцию эмоционального интеллекта и назови один маркер, по которому можно распознать его высокий уровень.",
    "Составь пошаговый чек-лист для проведения вечерней рефлексии, которая поможет снизить уровень стресса перед сном.",
    "Напиши краткую рецензию на твой любимый классический роман, выделив его скрытый философский подтекст.",
    "Объясни суть архитектурного стиля баухаус и назови три его главных признака в интерьере.",
    "Составь рецепт быстрого и полезного завтрака из пяти ингредиентов, которые обычно есть в любом холодильнике.",
    "Расскажи историю появления чая матча и объясни, почему он стал так популярен в современной велнес-культуре.",
    "Предложи концепцию для фантастического рассказа, где человечество полностью лишилось возможности спать."
]

async def test():
    tracker = BudgetTracker(daily_budget=20, alert_threshold=0.8)
    settings = Settings.from_env()
    # Инициализация клиента (выберите тот вариант init, который вы оставили)
    client = AsyncLLMClient(tracker=tracker,settings=settings)
    sync_client = RobustLLMClient(tracker=tracker, settings=settings)

    try:

        concurrency_options = [1, 5, 10]

        logger.info(f"Запуск бенчмарка для {len(prompts)} запросов...\n")

        for concurrency in concurrency_options:
            logger.info(f"=== Выполнение с concurrency = {concurrency} ===")
            start_time = time.perf_counter()

            # Вызываем метод batch (в вашем классе он называется batch)
            results = await client.batch(prompts, concurrency=concurrency)
            #results = await client.batch_chat_strict(prompts, concurrency=concurrency)

            total_time = time.perf_counter() - start_time
            # Считаем сколько успешных ответов получили
            success_count = sum(1 for res in results if not isinstance(res, Exception))

            logger.info(f"Успешно выполнено: {success_count}/{len(prompts)}")
            logger.info(f"Общее время на все запросы: {total_time:.2f} сек.\n")

        logger.info(f"=== Выполнение sync_client ===")

        start_time = time.perf_counter()
        for prompt in prompts:
            messages=[{"role": "user", "content": prompt}]
            sync_client.chat(messages=messages)



        total_time = time.perf_counter() - start_time

        #logger.info(f"Успешно выполнено: {success_count}/{len(prompts)}")
        logger.info(f"Общее время на все запросы: {total_time:.2f} сек.\n")


    except Exception as e:
        logger.error(f"\nПроизошла ошибка: {e}")

async def test_batch():
    tracker = BudgetTracker(daily_budget=20, alert_threshold=0.8)
    settings = Settings.from_env()
    # Инициализация клиента (выберите тот вариант init, который вы оставили)
    client = AsyncLLMClient(tracker=tracker,settings=settings)
    concurrency=20
    try:
            start_time = time.perf_counter()
            # Вызываем метод batch (в вашем классе он называется batch)
            results = await client.batch(prompts,20)
            total_time = time.perf_counter() - start_time
            # Считаем сколько успешных ответов получили
            success_count = sum(1 for res in results if not isinstance(res, Exception))

            logger.info(f"Успешно выполнено: {success_count}/{len(prompts)}")
            logger.info(f"Общее время на все запросы: {total_time:.2f} сек.\n")
    except Exception as e:
        logger.error(f"\nПроизошла ошибка: {e}")

async def test_batch_chat_strict():
    tracker = BudgetTracker(daily_budget=20, alert_threshold=0.8)
    settings = Settings.from_env()
    # Инициализация клиента (выберите тот вариант init, который вы оставили)
    client = AsyncLLMClient(tracker=tracker,settings=settings)
    concurrency=20
    try:
            start_time = time.perf_counter()
            # Вызываем метод batch (в вашем классе он называется batch)

            results = await client.batch_chat_strict(prompts, concurrency=concurrency)

            total_time = time.perf_counter() - start_time
            # Считаем сколько успешных ответов получили
            success_count = sum(1 for res in results if not isinstance(res, Exception))

            logger.info(f"Успешно выполнено: {success_count}/{len(prompts)}")
            logger.info(f"Общее время на все запросы: {total_time:.2f} сек.\n")
    except Exception as e:
        logger.error(f"\nПроизошла ошибка: {e}")

#asyncio.run(test())
asyncio.run(test_batch())
asyncio.run(test_batch_chat_strict())
