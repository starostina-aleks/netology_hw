from __future__ import annotations

from .budget_tracker import BudgetTracker
from openai import OpenAI
from config import Settings

from openai import RateLimitError, APIStatusError, APITimeoutError
from typing import Any
from loguru import logger
from models import LLMResult
import asyncio
import time
from typing import Any, Dict, List, Optional
from openai import AsyncOpenAI, OpenAIError
from openai import APIError


class AsyncLLMClient:

    def __init__(self,tracker:BudgetTracker, settings: Settings, max_concurrency=8) -> None:
        
        self.settings = settings
        self._sem=asyncio.Semaphore(max_concurrency)
        self.state = {"call_count": 0}
        # Провайдеры в порядке приоритета
        self.providers = [
            {
                "name": "OpenAI",
                "client": AsyncOpenAI(
                    base_url="https://api.vsegpt.ru/v1",  # "https://openrouter.ai/api/v1",
                    api_key=settings.open_api_key,
                    max_retries=3,
                    timeout=30,
                ),
                "model": "gpt-4o-mini"
            },
            {
                "name": "OpenRouter",
                "client": AsyncOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=settings.openrouter_api_key,
                    max_retries=3,
                    timeout=30,
                ),
                "model": "openai/gpt-oss-120b:free",
            },


            {
                "name": "Ollama (локальный)",                 
                 "client": AsyncOpenAI(
                    base_url="http://localhost:11434/v1",
                    api_key="ollama",
                    max_retries=3,
                    timeout=30,
                ),
                "model": "gemma3:1b",               
            }
        ]

        self.tracker=tracker

    async def chat(self, messages: list[dict],idx=0) -> LLMResult:
        start_time = time.perf_counter()
        status = "success"
        current_model = "unknown"
        try:
            # Ограничиваем общее время выполнения метода в 15 секунд
            async with asyncio.timeout(30):
                # Ограничиваем количество одновременных вызовов
                async with self._sem:
                    for provider in self.providers:
                        try:
                            current_model = provider["model"]
                            #if idx==3:
                                #current_model="123"
                            # Асинхронный вызов через await
                            response = await provider["client"].chat.completions.create(
                            model=current_model,
                            messages=messages,
                            max_tokens=250
                            )
                            if response.usage:
                                usage = response.usage
                                self.tracker.track_call(provider["model"], usage.prompt_tokens, usage.completion_tokens)
                    
                            text = (response.choices[0].message.content or "").strip()
                            tokens = getattr(response.usage, "total_tokens", 0) if response.usage else 0
                    
                            return LLMResult(text,tokens,provider['name'],provider["model"] )
                    
                        except Exception as e:
                            logger.warning(f"№{idx}: {provider['name']} failed: {e}")
                            continue
                    raise RuntimeError("Сервис временно недоступен")
        except (TimeoutError, asyncio.TimeoutError):
            status = "timeout"
            logger.error("LLM chat call timed out after 15 seconds")
            raise RuntimeError("Превышено время ожидания ответа от сервиса")
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000
            prompt_chars = sum(len(str(msg.get("content", ""))) for msg in messages)
            logger.info(
                f"№{idx}: llm.call | duration_ms={duration_ms:.2f} | model={current_model} | "
                f"prompt_chars={prompt_chars} | status={status}"
            )
    


    async def stream_chat(self, messages: list[dict]):
        async with self._sem:
            for provider in self.providers:
                try:
                    # Асинхронный вызов через await
                    stream = await provider["client"].chat.completions.create(
                        model=provider["model"],
                        messages=messages,
                        max_tokens=500,
                        stream=True,
                        stream_options={"include_usage": True},
                    )
                    async for chunk in stream:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content
                        if chunk.usage:
                            logger.info(f"tokens={chunk.usage.total_tokens}")
                    return

                except Exception as e:
                    logger.error(f"{provider['name']} failed: {e}")
                    continue

            raise RuntimeError("Сервис временно недоступен")


    async def batch_chat_strict(self,prompts:list[str],concurrency:int=5)->list[str]|None:
        self._sem = asyncio.Semaphore(concurrency)
        try:
            async with asyncio.TaskGroup() as tg:
                tasks=[tg.create_task(self.chat([{"role": "user", "content": prompt}],idx)) for idx,prompt in enumerate(prompts)]
            return  [task.result().text for task in tasks]
        except *APIError as eg:
            for error in eg.exceptions:
                logger.error(f"Детали ошибки API: {error}")

        except* TimeoutError as eg:
            for error in eg.exceptions:
                logger.error(f"Детали ошибки TimeoutError: {error}")

    async def batch(self,prompts:list[str],concurrency:int=5)->list[LLMResult|Exception]:
        self._sem = asyncio.Semaphore(concurrency)
        coros = [
            self.chat([{"role": "user", "content": prompt}], idx) for idx, prompt in enumerate(prompts)
        ]
        return await asyncio.gather(*coros,return_exceptions=True, )
