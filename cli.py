#!/usr/bin/env python3
"""
CLI интерфейс для RAG системы анализа кода.
Позволяет задавать вопросы к уже построенному индексу.

Использование:
    python cli.py "Как используется класс PrefixedDBModel?" --tool=tool_llm_encoder
    python cli.py "Какие функции вызывают метод render?" --tool=tool_bge_code
"""

import argparse
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="RAG система для анализа Python кода",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s "Как используется класс PrefixedDBModel?" --tool=tool_llm_encoder
  %(prog)s "Какие функции вызывают метод render?" --tool=tool_bge_code
  %(prog)s "Покажи архитектуру проекта" --tool=auto
        """
    )
    
    parser.add_argument(
        "question",
        help="Вопрос для анализа кода"
    )
    
    parser.add_argument(
        "--tool", "-t",
        choices=["tool_llm_encoder", "tool_bge_code", "auto"],
        default="auto",
        help="Выбор инструмента RAG (по умолчанию: auto)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"🔍 Вопрос: {args.question}")
        print(f"🛠️  Инструмент: {args.tool}")
        print("=" * 60)
    
    try:
        # Импортируем инструменты из основного модуля
        print("📚 Загружаем RAG инструменты...")
        from rag import tool_llm_encoder, tool_bge_code
        
        # Выбираем инструмент
        if args.tool == "tool_llm_encoder":
            selected_tool = tool_llm_encoder
            tool_name = "LLM Embedder"
        elif args.tool == "tool_bge_code":
            selected_tool = tool_bge_code
            tool_name = "BGE Code"
        else:  # auto
            # Автоматический выбор на основе вопроса
            question_lower = args.question.lower()
            if any(keyword in question_lower for keyword in ["класс", "class", "наследование", "inheritance", "использует"]):
                selected_tool = tool_llm_encoder
                tool_name = "LLM Embedder (auto-selected for class analysis)"
            else:
                selected_tool = tool_bge_code
                tool_name = "BGE Code (auto-selected)"
        
        if args.verbose:
            print(f"🎯 Используем: {tool_name}")
            print("🔄 Обрабатываем запрос...\n")
        
        # Выполняем запрос
        result = selected_tool.invoke(args.question)
        
        # Выводим результат
        print("📋 РЕЗУЛЬТАТ:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        if args.verbose:
            print(f"✅ Запрос выполнен успешно с помощью {tool_name}")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что файл rag.py находится в той же директории")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
