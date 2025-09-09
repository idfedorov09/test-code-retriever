#!/usr/bin/env python3
"""
Тестовый скрипт для проверки GPU поддержки в RAG системе.
"""

import os
import sys
from pathlib import Path

def test_gpu_availability():
    """Проверяет доступность GPU."""
    print("🔍 Проверяем доступность GPU...")
    
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        device = "cuda" if gpu_available else "cpu"
        
        print(f"🚀 Device: {device}")
        
        if gpu_available:
            print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            print(f"🔢 CUDA Version: {torch.version.cuda}")
            
            # Тест простых операций на GPU
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            z = torch.mm(x, y)
            print("✅ Базовые операции PyTorch на GPU работают")
            
        else:
            print("⚠️  GPU недоступно, будет использоваться CPU")
            
        return gpu_available
        
    except ImportError:
        print("❌ PyTorch не установлен")
        return False
    except Exception as e:
        print(f"❌ Ошибка при проверке GPU: {e}")
        return False

def test_faiss_gpu():
    """Проверяет поддержку FAISS-GPU."""
    print("\n🔍 Проверяем FAISS GPU поддержку...")
    
    try:
        import faiss
        print(f"📦 FAISS версия: {faiss.__version__}")
        
        # Проверяем наличие GPU функций
        if hasattr(faiss, 'StandardGpuResources'):
            print("✅ FAISS-GPU доступен")
            
            # Тест создания GPU индекса
            try:
                res = faiss.StandardGpuResources()
                dimension = 384  # Размерность для тестирования
                
                # Создаем простой индекс на GPU
                cpu_index = faiss.IndexFlatIP(dimension)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                print("✅ Создание GPU индекса работает")
                
            except Exception as e:
                print(f"⚠️  Ошибка создания GPU индекса: {e}")
                
        else:
            print("❌ FAISS-GPU не найден, установлена CPU версия")
            
    except ImportError:
        print("❌ FAISS не установлен")
    except Exception as e:
        print(f"❌ Ошибка при проверке FAISS: {e}")

def test_sentence_transformers_gpu():
    """Проверяет работу sentence-transformers на GPU."""
    print("\n🔍 Проверяем Sentence Transformers GPU поддержку...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Загружаем легкую модель для тестирования
        model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
        
        print(f"🎯 Модель загружена на: {model.device}")
        
        # Тест энкодинга
        sentences = ["This is a test sentence.", "Another test sentence."]
        embeddings = model.encode(sentences)
        
        print(f"✅ Энкодинг работает, размерность: {embeddings.shape}")
        
    except ImportError:
        print("❌ sentence-transformers не установлен")
    except Exception as e:
        print(f"❌ Ошибка при тестировании sentence-transformers: {e}")

def main():
    """Основная функция тестирования."""
    print("🧪 Тестирование GPU поддержки для RAG системы")
    print("=" * 50)
    
    gpu_available = test_gpu_availability()
    test_faiss_gpu()
    test_sentence_transformers_gpu()
    
    print("\n" + "=" * 50)
    print("📋 Рекомендации по установке:")
    
    if not gpu_available:
        print("1. Установите CUDA Toolkit")
        print("2. Переустановите PyTorch с CUDA поддержкой:")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("3. Установите зависимости для GPU:")
    print("   pip install faiss-gpu sentence-transformers accelerate")
    
    print("\n✨ Готово! Теперь можно запускать RAG на GPU")

if __name__ == "__main__":
    main()
