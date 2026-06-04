"""Интерактивный CLI (REPL) для общения с ассистентом поддержки.

Принимает пользовательский ввод, обрабатывает команды CLI
и выводит ответ ассистента.
"""

from __future__ import annotations

from config import Settings
from core.assistant import SupportAssistantApp


def main():
    settings = Settings.from_env()
    
    assistant = SupportAssistantApp(settings)

    print(f"=== {settings.service_name} Support CLI ===")
    print("Команды: /clear, /clear_cache, /reset_redis_stats, /stats, /quit")
    
    while True:
        try:
            user_input = input("\nВы: ").strip()
            
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания!")
            return None

        if not user_input:
            continue
        
        if user_input.startswith("/"):
            command_result = assistant.handle_command(user_input)
            if command_result is None:
                print("До свидания!")
                return None
            print(command_result)
            continue
        
        response = assistant.respond(user_input)
        print(response)
        
