#!/usr/bin/env python3
"""
Безопасная версия главного модуля RAG системы.
Создает только один инструмент за раз для предотвращения проблем с памятью.
"""

import os
import gc
import sys
from dotenv import load_dotenv
from pathlib import Path

# Загружаем переменные окружения
load_dotenv()

# Импортируем систему управления памятью
try:
    from gpu_utils import GPUMemoryManager, cleanup_gpu, monitor_gpu, aggressive_cleanup_gpu
    GPU_UTILS_AVAILABLE = True
    gpu_manager = GPUMemoryManager()
    print("✅ GPU утилиты загружены")
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
        
        # Устанавливаем переменную окружения для лучшего управления памятью
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        print("⚙️  Установлено PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"
    print("⚠️  PyTorch не найден, используем CPU")

# Импортируем базовые классы и реализации
from rag_base import RAGSystemFactory
from rag_python import PythonRAGSystem
from rag_universal import UniversalRAGSystem

# LangChain imports
from langchain_community.llms import YandexGPT
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings


def load_model_safe(model_name, local_dir="./models", wrapper_cls=HuggingFaceEmbeddings, use_gpu=None):
    """
    Безопасная загрузка модели с управлением памяти.
    """
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    # Агрессивная очистка памяти перед загрузкой
    if GPU_UTILS_AVAILABLE and use_gpu:
        print("🧹 Агрессивная очистка памяти перед загрузкой модели...")
        aggressive_cleanup_gpu()
        monitor_gpu()
        
        # Проверяем, достаточно ли памяти
        memory_info = gpu_manager.get_gpu_memory_info()
        if memory_info.get('free_memory_gb', 0) < 10:
            print("⚠️  Недостаточно свободной памяти, используем CPU")
            use_gpu = False
    
    local_path = Path(local_dir) / model_name.replace("/", "_")
    
    # Конфигурация с учетом доступной памяти
    model_kwargs = {}
    encode_kwargs = {}
    
    if use_gpu and GPU_AVAILABLE:
        # Консервативные настройки для GPU
        batch_size = 16  # Уменьшенный размер батча
        if GPU_UTILS_AVAILABLE:
            optimal_batch = gpu_manager.get_optimal_batch_size(16)
            batch_size = min(optimal_batch, 16)  # Не больше 16
            print(f"⚡ Консервативный размер батча: {batch_size}")
        
        model_kwargs.update({
            'device': DEVICE,
            'trust_remote_code': True,
        })
        encode_kwargs.update({
            'batch_size': batch_size,
        })
        print(f"🔥 Загружаем {model_name} на GPU (консервативно)")
    else:
        model_kwargs.update({
            'device': 'cpu',
        })
        encode_kwargs.update({
            'batch_size': 8,
        })
        print(f"🐌 Загружаем {model_name} на CPU")
    
    try:
        if local_path.exists():
            model = wrapper_cls(
                model_name=str(local_path),
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
        else:
            model = wrapper_cls(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            # Пытаемся сохранить модель локально
            try:
                if hasattr(model, 'client'):
                    model.client.save(str(local_path))
                elif hasattr(model, '_client'):
                    model._client.save(str(local_path))
            except Exception as e:
                print(f"⚠️  Не удалось сохранить модель локально: {e}")
        
        # Мониторинг после загрузки
        if GPU_UTILS_AVAILABLE and use_gpu:
            print("📊 Состояние памяти после загрузки:")
            monitor_gpu()
        
        return model
        
    except Exception as e:
        print(f"❌ Ошибка загрузки модели {model_name}: {e}")
        if GPU_UTILS_AVAILABLE:
            aggressive_cleanup_gpu()
        raise


def create_single_rag_tool(rag_type: str, project_path: str, use_gpu: bool = None):
    """
    Создает один RAG инструмент с максимальной безопасностью для памяти.
    """
    print(f"\n🔧 СОЗДАНИЕ {rag_type.upper()} RAG ИНСТРУМЕНТА")
    print("=" * 60)
    
    # Начальная очистка
    if GPU_UTILS_AVAILABLE:
        print("🧹 Начальная очистка памяти...")
        aggressive_cleanup_gpu()
        monitor_gpu()
    
    try:
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
        
        # Выбор модели эмбеддингов
        if rag_type == "python":
            embeddings = load_model_safe(
                model_name="BAAI/llm-embedder",
                wrapper_cls=HuggingFaceBgeEmbeddings,
                use_gpu=use_gpu
            )
        else:
            embeddings = load_model_safe(
                model_name="BAAI/bge-code-v1",
                use_gpu=use_gpu
            )
        
        # Создание RAG системы
        print(f"🏗️  Создаем {rag_type.upper()} RAG систему...")
        
        rag_system = RAGSystemFactory.create(
            rag_type,
            llm=llm,
            embeddings=embeddings,
        )
        
        # Создание инструмента
        tool = rag_system.create_tool(project_path)
        
        print(f"✅ {rag_type.upper()} RAG инструмент готов!")
        
        # Финальный мониторинг
        if GPU_UTILS_AVAILABLE:
            print("📊 Финальное состояние памяти:")
            monitor_gpu()
        
        return tool
        
    except Exception as e:
        print(f"❌ Ошибка создания {rag_type} инструмента: {e}")
        if GPU_UTILS_AVAILABLE:
            aggressive_cleanup_gpu()
        raise


def main():
    """Безопасная основная функция"""
    
    project_path = os.getenv('TEST_PROJ_PATH')
    if not project_path:
        print("❌ Установите переменную окружения TEST_PROJ_PATH")
        return
    
    print("🛡️  БЕЗОПАСНЫЙ РЕЖИМ СОЗДАНИЯ RAG ИНСТРУМЕНТОВ")
    print("=" * 60)
    
    if len(sys.argv) > 1:
        # Создаем только указанный тип
        rag_type = sys.argv[1].lower()
        if rag_type not in ['python', 'universal', 'auto']:
            print("❌ Поддерживаемые типы: python, universal, auto")
            return
        
        try:
            tool = create_single_rag_tool(rag_type, project_path, use_gpu=True)
            
            # Простой тест
            print(f"\n🧪 ТЕСТ {rag_type.upper()} ИНСТРУМЕНТА")
            print("=" * 40)
            
            test_question = "Покажи структуру проекта"
            print(f"❓ Вопрос: {test_question}")
            
            result = tool.invoke(test_question)
            print(f"✅ Ответ получен ({len(result)} символов)")
            print(f"📄 Первые 200 символов: {result[:200]}...")
            
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    else:
        print("💡 Использование:")
        print("   python rag_main_safe.py python     # Создать Python RAG")
        print("   python rag_main_safe.py universal  # Создать Universal RAG")
        print("   python rag_main_safe.py auto       # Автовыбор типа")
        print("\n🎯 Этот режим создает только один инструмент за раз")
        print("   для предотвращения проблем с памятью GPU")


def cleanup_on_exit():
    """Очистка ресурсов при завершении работы"""
    print("\n🧹 Очистка ресурсов...")
    if GPU_UTILS_AVAILABLE:
        try:
            aggressive_cleanup_gpu()
            print("✅ GPU память очищена")
        except Exception as e:
            print(f"⚠️  Ошибка при очистке GPU: {e}")
    gc.collect()
    print("✅ Ресурсы освобождены")


if __name__ == "__main__":
    import signal
    import atexit
    
    def signal_handler(sig, frame):
        print("\n🛑 Получен сигнал завершения...")
        cleanup_on_exit()
        sys.exit(0)
    
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
            aggressive_cleanup_gpu()
    finally:
        cleanup_on_exit()
