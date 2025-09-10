#!/usr/bin/env python3
"""
Тестирование системы управления GPU памятью
"""

import os
import sys
import time
from gpu_utils import GPUMemoryManager, cleanup_gpu, monitor_gpu, get_gpu_info

def test_memory_management():
    """Тест управления памятью"""
    print("🧪 ТЕСТ УПРАВЛЕНИЯ GPU ПАМЯТЬЮ")
    print("=" * 50)
    
    # Создаем менеджер
    manager = GPUMemoryManager()
    
    print("\n📊 Начальное состояние:")
    monitor_gpu()
    
    if not manager.cuda_available:
        print("\n❌ CUDA недоступен, тест невозможен")
        return
    
    print("\n🔧 Тестируем очистку памяти...")
    
    # Симулируем использование памяти
    try:
        import torch
        
        # Создаем большие тензоры
        print("📈 Создаем тензоры для имитации нагрузки...")
        tensors = []
        
        for i in range(5):
            tensor = torch.randn(1000, 1000, device='cuda')
            tensors.append(tensor)
            print(f"   Тензор {i+1} создан")
            
            # Проверяем память
            info = get_gpu_info()
            if info.get('memory_usage_percent', 0) > 50:
                print(f"   💾 Использование памяти: {info['memory_usage_percent']:.1f}%")
        
        print("\n📊 Состояние после создания тензоров:")
        monitor_gpu()
        
        # Очищаем тензоры
        print("\n🧹 Удаляем тензоры...")
        del tensors
        
        # Тестируем очистку
        print("🧹 Выполняем cleanup_gpu()...")
        cleanup_gpu()
        
        print("\n📊 Состояние после очистки:")
        monitor_gpu()
        
        print("\n✅ Тест управления памятью завершен успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время теста: {e}")
        cleanup_gpu()

def test_memory_monitoring():
    """Тест мониторинга памяти"""
    print("\n🔍 ТЕСТ МОНИТОРИНГА ПАМЯТИ")
    print("=" * 30)
    
    manager = GPUMemoryManager()
    
    # Получаем детальную информацию
    info = manager.get_gpu_memory_info()
    
    print("\n📋 Детальная информация о GPU:")
    for key, value in info.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Тестируем пороги
    print(f"\n🚨 Проверка порогов:")
    print(f"   Критический порог (85%): {'⚠️  ПРЕВЫШЕН' if manager.check_memory_threshold(85) else '✅ ОК'}")
    print(f"   Высокий порог (75%): {'⚠️  ПРЕВЫШЕН' if manager.check_memory_threshold(75) else '✅ ОК'}")
    
    # Тестируем оптимальный размер батча
    optimal_batch = manager.get_optimal_batch_size()
    print(f"\n⚡ Оптимальный размер батча: {optimal_batch}")

def stress_test():
    """Стресс-тест для проверки устойчивости"""
    print("\n💪 СТРЕСС-ТЕСТ")
    print("=" * 20)
    
    if not GPUMemoryManager().cuda_available:
        print("❌ CUDA недоступен, стресс-тест пропущен")
        return
    
    try:
        import torch
        
        print("🔥 Создаем множество тензоров...")
        
        for cycle in range(3):
            print(f"\n🔄 Цикл {cycle + 1}/3")
            
            # Создаем много тензоров
            tensors = []
            for i in range(10):
                try:
                    tensor = torch.randn(500, 500, device='cuda')
                    tensors.append(tensor)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("   ⚠️  Достигнут лимит памяти, выполняем очистку...")
                        cleanup_gpu()
                        break
                    else:
                        raise
            
            # Показываем состояние
            info = get_gpu_info()
            if 'memory_usage_percent' in info:
                print(f"   📊 Использование памяти: {info['memory_usage_percent']:.1f}%")
            
            # Очищаем
            del tensors
            cleanup_gpu()
            
            time.sleep(1)  # Небольшая пауза
        
        print("\n✅ Стресс-тест завершен успешно!")
        
    except Exception as e:
        print(f"\n❌ Ошибка во время стресс-теста: {e}")
        cleanup_gpu()

if __name__ == "__main__":
    print("🚀 СИСТЕМА УПРАВЛЕНИЯ GPU ПАМЯТЬЮ - ТЕСТИРОВАНИЕ")
    print("=" * 60)
    
    try:
        # Основной тест
        test_memory_management()
        
        # Тест мониторинга
        test_memory_monitoring()
        
        # Стресс-тест (опционально)
        if "--stress" in sys.argv:
            stress_test()
        
        print(f"\n🎉 ВСЕ ТЕСТЫ ЗАВЕРШЕНЫ!")
        print(f"💡 Для стресс-теста запустите: python {sys.argv[0]} --stress")
        
    except KeyboardInterrupt:
        print("\n🛑 Тестирование прервано пользователем")
        cleanup_gpu()
    except Exception as e:
        print(f"\n💥 Критическая ошибка: {e}")
        cleanup_gpu()
    finally:
        print("\n🧹 Финальная очистка...")
        cleanup_gpu()
