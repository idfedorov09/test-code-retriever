#!/usr/bin/env python3
"""
Интерактивный CLI интерфейс для RAG системы.
Позволяет задавать вопросы в интерактивном режиме.
"""

import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

def print_banner():
    print("🤖 RAG СИСТЕМА ДЛЯ АНАЛИЗА PYTHON КОДА")
    print("=" * 50)
    print("Доступные команды:")
    print("  /help     - показать справку")
    print("  /tool     - сменить инструмент")
    print("  /status   - показать текущие настройки")
    print("  /quit     - выйти")
    print("=" * 50)

def print_help():
    print("\n📖 СПРАВКА:")
    print("  • Задавайте вопросы на естественном языке")
    print("  • Примеры вопросов:")
    print("    - Как используется класс PrefixedDBModel?")
    print("    - Какие функции вызывают метод render?")
    print("    - Покажи архитектуру проекта")
    print("    - Где определен класс User?")
    print("  • Используйте /tool для смены инструмента")
    print()

def select_tool():
    print("\n🛠️  ВЫБОР ИНСТРУМЕНТА:")
    print("1. tool_llm_encoder - лучше для анализа классов и наследования")
    print("2. tool_bge_code - лучше для анализа вызовов функций")
    print("3. auto - автоматический выбор")
    
    while True:
        choice = input("Выберите (1-3): ").strip()
        if choice == "1":
            return "tool_llm_encoder", "LLM Embedder"
        elif choice == "2":
            return "tool_bge_code", "BGE Code"
        elif choice == "3":
            return "auto", "Автоматический"
        else:
            print("❌ Неверный выбор. Введите 1, 2 или 3.")

def auto_select_tool(question):
    """Автоматический выбор инструмента на основе вопроса."""
    question_lower = question.lower()
    
    # Ключевые слова для анализа классов
    class_keywords = ["класс", "class", "наследование", "inheritance", "использует", "inherited", "extends"]
    
    if any(keyword in question_lower for keyword in class_keywords):
        return "tool_llm_encoder", "LLM Embedder (auto: class analysis)"
    else:
        return "tool_bge_code", "BGE Code (auto: function analysis)"

def main():
    print_banner()
    
    try:
        # Импортируем инструменты
        print("📚 Загружаем RAG инструменты...")
        from rag import tool_llm_encoder, tool_bge_code
        print("✅ Инструменты загружены успешно!\n")
        
        # Настройки по умолчанию
        current_tool_key = "auto"
        current_tool_name = "Автоматический"
        
        while True:
            try:
                # Показываем текущие настройки
                print(f"🎯 Текущий инструмент: {current_tool_name}")
                
                # Получаем вопрос от пользователя
                question = input("\n❓ Ваш вопрос (или команда): ").strip()
                
                if not question:
                    continue
                
                # Обработка команд
                if question.startswith('/'):
                    command = question.lower()
                    
                    if command == '/help':
                        print_help()
                        continue
                    
                    elif command == '/tool':
                        current_tool_key, current_tool_name = select_tool()
                        print(f"✅ Выбран инструмент: {current_tool_name}")
                        continue
                    
                    elif command == '/status':
                        print(f"\n📊 ТЕКУЩИЕ НАСТРОЙКИ:")
                        print(f"  Инструмент: {current_tool_name}")
                        print(f"  Ключ: {current_tool_key}")
                        continue
                    
                    elif command in ['/quit', '/exit']:
                        print("👋 До свидания!")
                        break
                    
                    else:
                        print(f"❌ Неизвестная команда: {question}")
                        print("💡 Используйте /help для справки")
                        continue
                
                # Выбираем инструмент
                if current_tool_key == "auto":
                    selected_tool_key, tool_display_name = auto_select_tool(question)
                else:
                    selected_tool_key = current_tool_key
                    tool_display_name = current_tool_name
                
                # Получаем инструмент
                if selected_tool_key == "tool_llm_encoder":
                    selected_tool = tool_llm_encoder
                else:
                    selected_tool = tool_bge_code
                
                print(f"🔄 Обрабатываем запрос с помощью {tool_display_name}...")
                print("=" * 50)
                
                # Выполняем запрос
                result = selected_tool.invoke(question)
                
                # Выводим результат
                print(result)
                print("=" * 50)
                
            except KeyboardInterrupt:
                print("\n\n👋 До свидания!")
                break
            except Exception as e:
                print(f"\n❌ Ошибка: {e}")
                print("💡 Попробуйте переформулировать вопрос")
                
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что файл rag.py находится в той же директории")
        print("💡 И что все зависимости установлены")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
