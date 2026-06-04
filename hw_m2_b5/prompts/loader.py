"""Загрузка и сборка промптов для LLM.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any
from jinja2 import Template


def _read_prompt_file(filename: str) -> str:
    return resources.files(__package__).joinpath(filename).read_text(encoding="utf-8").strip()


SERVICE_FACTS = _read_prompt_file("service_facts.txt")
SYSTEM_PROMPT_TEMPLATE = _read_prompt_file("system_prompt.txt")
SYSTEM_FEW_SHOTS = json.loads(_read_prompt_file("system_few_shots.json"))


def build_system_prompt(service_name: str,ans_language:str) -> str:
    template = Template(SYSTEM_PROMPT_TEMPLATE)

    return template.render(
        language=ans_language,
        service_name=service_name,
        service_facts=SERVICE_FACTS,
    )


def build_answer_messages(system_prompt: str, history: list[dict[str, str]], user_message: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for entry in SYSTEM_FEW_SHOTS:
          # Добавляем вопрос от пользователя
          messages.append({"role": "user", "content": entry["question"]})
          # Добавляем ответ от ассистента
          messages.append({"role": "assistant", "content": entry["answer"]})
    messages.extend(history)
    messages.append({"role": "user", "content": user_message})
    return messages


