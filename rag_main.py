#!/usr/bin/env python3
"""
Главный модуль новой абстрактной RAG системы.
Автоматически выбирает подходящую реализацию в зависимости от типа проекта.
"""

import os
import gc
from dotenv import load_dotenv
from pathlib import Path

from langchain_core.embeddings import Embeddings

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
from rag_architecture import ArchitectureRAGSystem

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
    embeddings: Embeddings,
    rag_type: str = "auto",
    use_gpu: bool = None,
    **config
):
    """
    Создает RAG инструмент для анализа проекта.
    
    Args:
        project_path: Путь к проекту
        rag_type: Тип RAG системы ("python", "universal", "auto")
        embeddings: Модель эмбеддингов
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