#!/usr/bin/env python3
"""
Утилиты для управления GPU памятью в RAG системе
"""

import gc
import os
import psutil
from typing import Optional, Dict, Any
import warnings

# Подавляем предупреждения
warnings.filterwarnings("ignore")

class GPUMemoryManager:
    """Менеджер GPU памяти для предотвращения утечек"""
    
    def __init__(self):
        self.torch_available = False
        self.cuda_available = False
        self.device = None
        self._init_torch()
    
    def _init_torch(self):
        """Инициализация PyTorch и CUDA"""
        try:
            import torch
            self.torch_available = True
            self.cuda_available = torch.cuda.is_available()
            
            if self.cuda_available:
                self.device = torch.device('cuda')
                # Устанавливаем настройки для лучшего управления памятью
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            else:
                self.device = torch.device('cpu')
                
        except ImportError:
            self.torch_available = False
            self.cuda_available = False
            self.device = None
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """Получает информацию о состоянии GPU памяти"""
        if not self.cuda_available:
            return {"gpu_available": False}
        
        try:
            import torch
            
            # Получаем информацию о памяти
            total_memory = torch.cuda.get_device_properties(0).total_memory
            reserved_memory = torch.cuda.memory_reserved(0)
            allocated_memory = torch.cuda.memory_allocated(0)
            free_memory = total_memory - reserved_memory
            
            return {
                "gpu_available": True,
                "device_name": torch.cuda.get_device_name(0),
                "total_memory_gb": total_memory / (1024**3),
                "allocated_memory_gb": allocated_memory / (1024**3),
                "reserved_memory_gb": reserved_memory / (1024**3),
                "free_memory_gb": free_memory / (1024**3),
                "memory_usage_percent": (reserved_memory / total_memory) * 100
            }
        except Exception as e:
            return {"gpu_available": True, "error": str(e)}
    
    def cleanup_gpu_memory(self, aggressive: bool = False):
        """Очищает GPU память"""
        if not self.cuda_available:
            return
        
        try:
            import torch
            
            # Принудительная сборка мусора
            gc.collect()
            
            # Очистка кэша PyTorch
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            if aggressive:
                # Агрессивная очистка - синхронизация и повторная очистка
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                
                # Попытка освободить неиспользуемые сегменты памяти
                if hasattr(torch.cuda, 'memory_snapshot'):
                    try:
                        torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                        os.remove("memory_snapshot.pickle")  # Удаляем файл снимка
                    except:
                        pass
                
        except Exception as e:
            print(f"⚠️  Ошибка при очистке GPU памяти: {e}")
    
    def check_memory_threshold(self, threshold_percent: float = 85.0) -> bool:
        """Проверяет, превышен ли порог использования памяти"""
        if not self.cuda_available:
            return False
        
        memory_info = self.get_gpu_memory_info()
        if "memory_usage_percent" in memory_info:
            return memory_info["memory_usage_percent"] > threshold_percent
        
        return False
    
    def get_optimal_batch_size(self, base_batch_size: int = 32) -> int:
        """Определяет оптимальный размер батча на основе доступной памяти"""
        if not self.cuda_available:
            return base_batch_size
        
        memory_info = self.get_gpu_memory_info()
        if "free_memory_gb" not in memory_info:
            return base_batch_size
        
        free_memory_gb = memory_info["free_memory_gb"]
        
        # Эвристика для определения размера батча
        if free_memory_gb > 10:
            return base_batch_size * 2
        elif free_memory_gb > 5:
            return base_batch_size
        elif free_memory_gb > 2:
            return max(base_batch_size // 2, 8)
        else:
            return max(base_batch_size // 4, 4)
    
    def monitor_memory_usage(self):
        """Выводит текущее состояние памяти"""
        memory_info = self.get_gpu_memory_info()
        
        if not memory_info.get("gpu_available", False):
            print("🐌 GPU недоступен, используется CPU")
            return
        
        if "error" in memory_info:
            print(f"⚠️  Ошибка получения информации о GPU: {memory_info['error']}")
            return
        
        print(f"🔥 GPU: {memory_info.get('device_name', 'Unknown')}")
        print(f"📊 Память: {memory_info['allocated_memory_gb']:.1f}GB / {memory_info['total_memory_gb']:.1f}GB "
              f"({memory_info['memory_usage_percent']:.1f}%)")
        print(f"🆓 Свободно: {memory_info['free_memory_gb']:.1f}GB")
        
        # Предупреждения
        if memory_info['memory_usage_percent'] > 90:
            print("🚨 КРИТИЧЕСКОЕ: Память GPU почти исчерпана!")
        elif memory_info['memory_usage_percent'] > 75:
            print("⚠️  ВНИМАНИЕ: Высокое использование GPU памяти")

# Глобальный экземпляр менеджера
gpu_manager = GPUMemoryManager()

def cleanup_gpu():
    """Быстрая очистка GPU памяти"""
    gpu_manager.cleanup_gpu_memory()

def aggressive_cleanup_gpu():
    """Агрессивная очистка GPU памяти"""
    gpu_manager.cleanup_gpu_memory(aggressive=True)

def monitor_gpu():
    """Мониторинг GPU памяти"""
    gpu_manager.monitor_memory_usage()

def get_gpu_info():
    """Получить информацию о GPU"""
    return gpu_manager.get_gpu_memory_info()

def check_gpu_health():
    """Проверка состояния GPU"""
    return not gpu_manager.check_memory_threshold()

if __name__ == "__main__":
    print("🔧 GPU Memory Manager - Тест")
    print("=" * 50)
    
    monitor_gpu()
    
    print("\n🧹 Выполняем очистку памяти...")
    cleanup_gpu()
    
    print("\n📊 Состояние после очистки:")
    monitor_gpu()
