#!/usr/bin/env python3
"""
Обновленный CLI интерфейс для новой абстрактной RAG системы.
Поддерживает разные типы проектов и автоматический выбор.

Использование:
    python cli_new.py "Как используется класс PrefixedDBModel?" --type=python
    python cli_new.py "Какие есть Docker файлы?" --type=universal
    python cli_new.py "Покажи архитектуру проекта" --type=auto
"""

import argparse
import sys
import os
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

def main():
    parser = argparse.ArgumentParser(
        description="Новая абстрактная RAG система для анализа проектов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s "Как используется класс PrefixedDBModel?" --type=python
  %(prog)s "Какие есть Docker файлы?" --type=universal  
  %(prog)s "Покажи архитектуру проекта" --type=auto
  %(prog)s "Есть ли проблемы с безопасностью?" --type=auto --gpu
        """
    )
    
    parser.add_argument(
        "question",
        help="Вопрос для анализа проекта"
    )
    
    parser.add_argument(
        "--type", "-t",
        choices=["python", "universal", "auto"],
        default="auto",
        help="Тип RAG системы (по умолчанию: auto)"
    )
    
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Принудительно использовать GPU"
    )
    
    parser.add_argument(
        "--no-gpu",
        action="store_true", 
        help="Принудительно использовать CPU"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Подробный вывод"
    )
    
    parser.add_argument(
        "--project-path", "-p",
        help="Путь к проекту (по умолчанию из TEST_PROJ_PATH)"
    )
    
    args = parser.parse_args()
    
    # Определяем настройки GPU
    use_gpu = None
    if args.gpu:
        use_gpu = True
    elif args.no_gpu:
        use_gpu = False
    
    # Определяем путь к проекту
    project_path = args.project_path or os.getenv('TEST_PROJ_PATH')
    if not project_path:
        print("❌ Укажите путь к проекту через --project-path или переменную TEST_PROJ_PATH")
        sys.exit(1)
    
    if not os.path.exists(project_path):
        print(f"❌ Проект не найден: {project_path}")
        sys.exit(1)
    
    if args.verbose:
        print(f"🔍 Вопрос: {args.question}")
        print(f"🛠️  Тип RAG: {args.type}")
        print(f"📁 Проект: {project_path}")
        print(f"🎮 GPU: {use_gpu}")
        print("=" * 60)
    
    try:
        # Импортируем функцию создания RAG инструмента
        print("📚 Загружаем RAG систему...")
        from rag_main import create_rag_tool
        
        # Создаем инструмент
        tool = create_rag_tool(
            project_path=project_path,
            rag_type=args.type,
            use_gpu=use_gpu
        )
        
        if args.verbose:
            print("🔄 Обрабатываем запрос...\n")
        
        # Выполняем запрос
        result = tool.invoke(args.question)
        
        # Выводим результат
        print("📋 РЕЗУЛЬТАТ:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        if args.verbose:
            print(f"✅ Запрос выполнен успешно с помощью {args.type.upper()} RAG")
            
    except ImportError as e:
        print(f"❌ Ошибка импорта: {e}")
        print("💡 Убедитесь, что все файлы RAG системы находятся в той же директории")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка выполнения: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
