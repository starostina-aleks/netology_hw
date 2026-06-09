from __future__ import annotations

from app.llm.client import LLMClient
from loguru import logger
import os

logger.remove()
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"run_tool_call.log")

logger.add(log_path, format="{time} {message}", rotation="10 MB")
# Запуск проверки

assistant = LLMClient()
while True:
    user_input = input("\nВы: ").strip()

    if not user_input:
        continue
    if user_input=='quit':
        print("До свидания!")
        break
    logger.info(f"Вопрос: {user_input}")
    response,total_tokens = assistant.answer(user_input)
    logger.info(f"{response}. Всего токенов : {total_tokens}")
    print(f"Ответ: {response}")




