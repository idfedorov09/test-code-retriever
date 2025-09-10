#!/usr/bin/env python3
"""
Главный модуль новой абстрактной RAG системы.
Автоматически выбирает подходящую реализацию в зависимости от типа проекта.
"""

import os
import gc
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения
load_dotenv()

# Импортируем систему управления памятью
try:
    from gpu_utils import GPUMemoryManager, cleanup_gpu, monitor_gpu
    GPU_UTILS_AVAILABLE = True
    gpu_manager = GPUMemoryManager()
except ImportError:
    GPU_UTILS_AVAILABLE = False
    print("⚠️  GPU утилиты недоступны")

# GPU detection and configuration
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
    print(f"🚀 Device: {DEVICE}")
    if GPU_AVAILABLE:
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"
    print("⚠️  PyTorch не найден, используем CPU")

# Импортируем базовые классы и реализации
from rag_base import RAGSystemFactory
from rag_python import PythonRAGSystem
from rag_universal import UniversalRAGSystem
from rag_javascript import JavaScriptRAGSystem

# LangChain imports
from langchain_community.llms import YandexGPT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def load_model(model_name, local_dir="./models", wrapper_cls=HuggingFaceEmbeddings, use_gpu=None):
    """
    Загружает модель эмбеддингов с поддержкой GPU.
    """
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    local_path = Path(local_dir) / model_name.replace("/", "_")
    
    # Конфигурация для GPU
    model_kwargs = {}
    encode_kwargs = {}
    
    if use_gpu and GPU_AVAILABLE:
        model_kwargs.update({
            'device': DEVICE,
            'trust_remote_code': True,
        })
        encode_kwargs.update({
            'batch_size': 32,  # Увеличиваем batch_size для GPU
        })
        print(f"🔥 Загружаем {model_name} на GPU")
    else:
        model_kwargs.update({
            'device': 'cpu',
        })
        encode_kwargs.update({
            'batch_size': 8,  # Меньший batch_size для CPU
        })
        print(f"🐌 Загружаем {model_name} на CPU")
    
    if local_path.exists():
        return wrapper_cls(
            model_name=str(local_path),
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        emb = wrapper_cls(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        try:
            # Сохраняем модель локально
            if hasattr(emb, 'client'):
                emb.client.save(str(local_path))
            elif hasattr(emb, '_client'):
                emb._client.save(str(local_path))
        except Exception as e:
            print(f"⚠️  Не удалось сохранить модель локально: {e}")
        return emb


def create_rag_tool(
    project_path: str,
    rag_type: str = "auto",
    use_gpu: bool = None,
    **config
):
    """
    Создает RAG инструмент для анализа проекта.
    
    Args:
        project_path: Путь к проекту
        rag_type: Тип RAG системы ("python", "universal", "auto")
        use_gpu: Использовать GPU (None = автоопределение)
        **config: Дополнительные параметры конфигурации
    """
    
    # Инициализация LLM
    llm = YandexGPT(
        iam_token=os.getenv('YANDEX_GPT_API_KEY'),
        folder_id=os.getenv('YANDEX_GPT_FOLDER_ID'),
        model_name="yandexgpt"
    )
    
    # Автоматическое определение типа проекта
    if rag_type == "auto":
        rag_type = RAGSystemFactory.detect_project_type(project_path)
        print(f"🎯 Автоматически определен тип проекта: {rag_type}")
    
    # Проверяем память перед загрузкой модели
    if GPU_UTILS_AVAILABLE and use_gpu:
        print("📊 Проверка памяти перед созданием RAG системы:")
        monitor_gpu()
        if gpu_manager.check_memory_threshold(75):
            print("⚠️  Высокое использование памяти, выполняем очистку...")
            cleanup_gpu()
    
    # Выбор модели эмбеддингов в зависимости от типа
    if rag_type == "python":
        # Для Python проектов используем специализированную модель
        embeddings = load_model(
            model_name="BAAI/llm-embedder",
            wrapper_cls=HuggingFaceBgeEmbeddings,
            use_gpu=use_gpu
        )
    else:
        # Для универсальных проектов используем кодовую модель
        embeddings = load_model(
            model_name="BAAI/bge-code-v1",
            use_gpu=use_gpu
        )
    
    # Создание RAG системы
    print(f"🔧 Создаем {rag_type.upper()} RAG систему...")
    
    rag_system = RAGSystemFactory.create(
        rag_type,
        llm=llm,
        embeddings=embeddings,
        **config
    )
    
    # Создание инструмента
    tool = rag_system.create_tool(project_path)
    
    print(f"✅ {rag_type.upper()} RAG инструмент готов!")
    return tool


def main():
    """Основная функция для демонстрации"""
    
    project_path = os.getenv('TEST_PROJ_PATH')
    if not project_path:
        print("❌ Установите переменную окружения TEST_PROJ_PATH")
        return
    
    # Начальная очистка памяти
    if GPU_UTILS_AVAILABLE:
        print("🧹 Начальная очистка GPU памяти...")
        cleanup_gpu()
        monitor_gpu()
    
    print("🤖 Создание RAG инструментов...")
    
    # Создаем Python RAG инструмент
    print("\n" + "="*60)
    python_tool = create_rag_tool(
        project_path=project_path,
        rag_type="python",
        use_gpu=True
    )
    
    # Очистка памяти между созданиями инструментов
    if GPU_UTILS_AVAILABLE:
        print("🧹 Очистка памяти между инструментами...")
        cleanup_gpu()
        gc.collect()  # Дополнительная сборка мусора
    
    # Создаем универсальный RAG инструмент
    print("\n" + "="*60)
    universal_tool = create_rag_tool(
        project_path=project_path,
        rag_type="universal",
        use_gpu=True
    )
    
    # Очистка памяти между созданиями инструментов
    if GPU_UTILS_AVAILABLE:
        print("🧹 Очистка памяти между инструментами...")
        cleanup_gpu()
        gc.collect()
    
    # Создаем автоматический RAG инструмент
    print("\n" + "="*60)
    auto_tool = create_rag_tool(
        project_path=project_path,
        rag_type="auto",
        use_gpu=True
    )
    
    # Тестовые запросы
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ ИНСТРУМЕНТОВ")
    print("="*60)
    
    test_questions = [
        "Как используется класс PrefixedDBModel?",
        "Какие есть конфигурационные файлы в проекте?",
        "Есть ли Docker файлы и как они настроены?",
    ]
    
    tools = {
        "Python RAG": python_tool,
        "Universal RAG": universal_tool,
        "Auto RAG": auto_tool
    }
    
    for question in test_questions:
        print(f"\n❓ Вопрос: {question}")
        print("-" * 50)
        
        for tool_name, tool in tools.items():
            try:
                print(f"\n🔍 {tool_name}:")
                result = tool.invoke(question)
                print(result[:200] + "..." if len(result) > 200 else result)
            except Exception as e:
                print(f"❌ Ошибка в {tool_name}: {e}")
        
        print("\n" + "="*60)


if __name__ == "__main__":
    import signal
    import atexit
    
    def cleanup_on_exit():
        """Очистка ресурсов при завершении работы"""
        print("\n🧹 Очистка ресурсов...")
        if GPU_UTILS_AVAILABLE:
            try:
                from gpu_utils import aggressive_cleanup_gpu
                aggressive_cleanup_gpu()
                print("✅ GPU память очищена")
            except Exception as e:
                print(f"⚠️  Ошибка при очистке GPU: {e}")
        gc.collect()
        print("✅ Ресурсы освобождены")
    
    def signal_handler(sig, frame):
        print("\n🛑 Получен сигнал завершения...")
        cleanup_on_exit()
        exit(0)
    
    # Регистрируем обработчики
    atexit.register(cleanup_on_exit)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Завершение работы...")
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        if GPU_UTILS_AVAILABLE:
            cleanup_gpu()
    finally:
        cleanup_on_exit()
