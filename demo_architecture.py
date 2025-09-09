#!/usr/bin/env python3
"""
Демонстрация новой абстрактной архитектуры RAG системы.
Показывает как работают разные реализации на одном проекте.
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

from rag_base import RAGSystemFactory

def demo_architecture():
    """Демонстрирует возможности новой архитектуры"""
    
    print("🏗️  ДЕМОНСТРАЦИЯ АБСТРАКТНОЙ RAG АРХИТЕКТУРЫ")
    print("=" * 60)
    
    # Показываем доступные RAG системы
    print("\n📋 ДОСТУПНЫЕ RAG СИСТЕМЫ:")
    available_systems = RAGSystemFactory.list_available()
    for system in available_systems:
        print(f"  ✅ {system}")
    
    # Демонстрируем автоматическое определение типа проекта
    print("\n🎯 АВТОМАТИЧЕСКОЕ ОПРЕДЕЛЕНИЕ ТИПА ПРОЕКТА:")
    
    test_projects = [
        "/Users/idfedorov09/my_prog/trash/rag",  # Текущий проект
        "/tmp/test_python",  # Гипотетический Python проект
        "/tmp/test_js",      # Гипотетический JS проект
    ]
    
    for project_path in test_projects:
        try:
            detected_type = RAGSystemFactory.detect_project_type(project_path)
            print(f"  📁 {project_path} → {detected_type}")
        except Exception as e:
            print(f"  ❌ {project_path} → ошибка: {e}")
    
    # Показываем различия между реализациями
    print("\n🔍 РАЗЛИЧИЯ МЕЖДУ РЕАЛИЗАЦИЯМИ:")
    
    print("\n  🐍 PYTHON RAG:")
    print("    • AST анализ Python кода")
    print("    • Граф вызовов функций (calls/called_by)")
    print("    • Граф наследования классов (inherited_by)")
    print("    • Специализированные промпты для Python")
    print("    • Модель: BAAI/llm-embedder")
    
    print("\n  🌐 UNIVERSAL RAG:")
    print("    • Текстовый анализ любых файлов")
    print("    • Поддержка JS, Docker, YAML, SQL, etc.")
    print("    • Контекстные промпты для каждого типа")
    print("    • Умное разбиение на чанки")
    print("    • Модель: BAAI/bge-code-v1")
    
    print("\n  🤖 AUTO RAG:")
    print("    • Автоматический выбор подходящей реализации")
    print("    • Анализ файлов проекта для определения типа")
    print("    • Fallback на универсальную реализацию")
    
    # Показываем примеры использования
    print("\n💡 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:")
    
    examples = [
        ("Python проект", "python", "Как используется класс User?"),
        ("Фуллстек проект", "universal", "Какие есть Docker файлы?"),
        ("Любой проект", "auto", "Покажи архитектуру проекта"),
        ("Конфигурация", "universal", "Какие переменные окружения используются?"),
        ("Безопасность", "auto", "Есть ли проблемы с безопасностью?"),
    ]
    
    for project_type, rag_type, question in examples:
        print(f"\n  📋 {project_type}:")
        print(f"     python cli_new.py \"{question}\" --type={rag_type}")
    
    # Показываем архитектурные преимущества
    print("\n🎯 АРХИТЕКТУРНЫЕ ПРЕИМУЩЕСТВА:")
    
    print("\n  ✅ РАСШИРЯЕМОСТЬ:")
    print("    • Легко добавить новые языки/технологии")
    print("    • Просто создать специализированные анализаторы")
    print("    • Модульная архитектура")
    
    print("\n  ✅ ПЕРЕИСПОЛЬЗОВАНИЕ:")
    print("    • Общие базовые классы")
    print("    • Единый интерфейс для всех реализаций")
    print("    • Общие утилиты и паттерны")
    
    print("\n  ✅ ГИБКОСТЬ:")
    print("    • Можно комбинировать разные подходы")
    print("    • Настраиваемые парсеры и анализаторы")
    print("    • Автоматический выбор оптимальной стратегии")
    
    print("\n🚀 ГОТОВО К ИСПОЛЬЗОВАНИЮ!")
    print("   Выберите подходящую реализацию и начинайте анализировать проекты!")
    
    return True

def demo_extensibility():
    """Демонстрирует как легко добавить новую реализацию"""
    
    print("\n" + "="*60)
    print("🔧 КАК ДОБАВИТЬ НОВУЮ РЕАЛИЗАЦИЮ")
    print("="*60)
    
    example_code = '''
# Пример: добавление поддержки Go проектов

from rag_base import BaseRAGSystem, BaseFileMap
from rag_base import RAGSystemFactory

class GoFileMap(BaseFileMap):
    def __init__(self, path, functions, structs, interfaces):
        self.functions = functions
        self.structs = structs  
        self.interfaces = interfaces
        super().__init__(path, "go", len(content.splitlines()))
    
    def to_text(self):
        # Формат для Go файлов
        pass

class GoRAGSystem(BaseRAGSystem):
    def get_file_patterns(self):
        return {"go": "**/*.go"}
    
    def get_evidence_prompt_template(self):
        return "Анализируй Go код с учетом goroutines, channels, interfaces..."
    
    # ... остальные методы

# Регистрируем новую систему
RAGSystemFactory.register('go', GoRAGSystem)

# Теперь можно использовать:
# python cli_new.py "Вопрос?" --type=go
'''
    
    print(example_code)
    
    print("\n✨ ВСЁ! Новая реализация готова к использованию!")
    print("   Фабрика автоматически подхватит её и сделает доступной")

if __name__ == "__main__":
    demo_architecture()
    demo_extensibility()
