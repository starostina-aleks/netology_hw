from __future__ import annotations

import json
from openai import OpenAI
from dotenv import load_dotenv
from ..tools.schemas import tools_list, validate_tools_list
from ..tools.handlers import search_documents
from loguru import logger

import os
from app.prompts.loader import render_system_prompt

load_dotenv()

available_functions = {
    "search_documents": search_documents,

}

class LLMClient:
    def __init__(self) -> None :
        validate_tools_list()
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

    def answer(self, question: str) -> tuple[str, int]:
        messages = [
            {"role": "system", "content": render_system_prompt(product_name="Acme Cloud")},
            {"role": "user", "content": question},
        ]

        response =self.client.chat.completions.create(
                model="openai/gpt-oss-120b:free",
                messages=messages,
                tools=tools_list,
            )
        logger.info(f"Ответ модели (шаг 1): {response.choices[0].message.content}. Всего токенов : {response.usage.total_tokens}")
        msg=response.choices[0].message
        messages.append(msg)
        if msg.tool_calls:
            for tc in msg.tool_calls:
                logger.info(f"Функция: {tc.function.name} Аргументы: {tc.function.arguments}")
                function_name = tc.function.name
                # Получаем функцию из словаря. Если её нет — вернется None
                function_to_call = available_functions.get(function_name)
                if function_to_call:
                    try:
                        result = function_to_call(**json.loads(tc.function.arguments))
                        logger.info(f"Результат функции {function_name:} {result}")

                    except Exception as e:
                        result=f"Ошибка при выполнении функции {function_name}: {e}"
                        logger.error(result)
                else:
                    # Обработка случая, когда модель вызвала несуществующую функцию
                    result=f"Функция '{function_name}' не найдена в доступных инструментах"
                    logger.warning(result)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": str(result)})

            response = self.client.chat.completions.create(
                 model="openai/gpt-oss-120b:free",
                 messages=messages,
                 tools=tools_list,
            )
            logger.info( f"Ответ модели (шаг 2):")
        else:
            logger.info("Модель ответила без вызова инструментов:")


        return response.choices[0].message.content, response.usage.total_tokens


