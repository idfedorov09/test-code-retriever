#!/usr/bin/env python3
"""
Веб-интерфейс для RAG системы анализа кода.
Простой Flask-приложение для удобного использования через браузер.
"""

import sys
import os
from pathlib import Path
from flask import Flask, render_template_string, request, jsonify
import json

# Добавляем текущую директорию в путь для импорта
sys.path.insert(0, str(Path(__file__).parent))

app = Flask(__name__)

# HTML шаблон
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG - Анализ Python Кода</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            background: white;
            border-radius: 10px;
            padding: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #555;
        }
        input[type="text"], select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #ddd;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #007bff;
        }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            transition: background 0.3s;
        }
        button:hover {
            background: #0056b3;
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .result {
            margin-top: 30px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #007bff;
            white-space: pre-wrap;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        .error {
            border-left-color: #dc3545;
            background: #f8d7da;
            color: #721c24;
        }
        .loading {
            text-align: center;
            color: #666;
        }
        .examples {
            margin-top: 20px;
            padding: 15px;
            background: #e9ecef;
            border-radius: 8px;
        }
        .examples h3 {
            margin-top: 0;
            color: #495057;
        }
        .example {
            background: white;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 4px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .example:hover {
            background: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RAG - Анализ Python Кода</h1>
        
        <form id="questionForm">
            <div class="form-group">
                <label for="question">Ваш вопрос:</label>
                <input type="text" id="question" name="question" placeholder="Как используется класс PrefixedDBModel?" required>
            </div>
            
            <div class="form-group">
                <label for="rag_type">Тип RAG системы:</label>
                <select id="rag_type" name="rag_type">
                    <option value="auto">🤖 Автоматический выбор</option>
                    <option value="python">🐍 Python (AST анализ)</option>
                    <option value="universal">🌐 Универсальный (все файлы)</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="use_gpu">Использование GPU:</label>
                <select id="use_gpu" name="use_gpu">
                    <option value="auto">⚡ Автоматически</option>
                    <option value="true">🔥 Принудительно GPU</option>
                    <option value="false">🐌 Только CPU</option>
                </select>
            </div>
            
            <button type="submit">🔍 Анализировать</button>
        </form>
        
        <div class="examples">
            <h3>💡 Примеры вопросов:</h3>
            <div class="example" onclick="setQuestion('Как используется класс PrefixedDBModel?')">
                Как используется класс PrefixedDBModel?
            </div>
            <div class="example" onclick="setQuestion('Какие функции вызывают метод render?')">
                Какие функции вызывают метод render?
            </div>
            <div class="example" onclick="setQuestion('Покажи архитектуру проекта')">
                Покажи архитектуру проекта
            </div>
            <div class="example" onclick="setQuestion('Какие есть Docker файлы?')">
                Какие есть Docker файлы?
            </div>
            <div class="example" onclick="setQuestion('Есть ли проблемы с безопасностью?')">
                Есть ли проблемы с безопасностью?
            </div>
        </div>
        
        <div id="project-info" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; display: none;">
            <h3>📊 Информация о проекте</h3>
            <div id="project-details"></div>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        function setQuestion(question) {
            document.getElementById('question').value = question;
        }
        
        // Загружаем информацию о проекте при загрузке страницы
        window.addEventListener('load', async function() {
            try {
                const response = await fetch('/project-info');
                const projectInfo = await response.json();
                
                if (projectInfo && !projectInfo.error) {
                    const projectInfoDiv = document.getElementById('project-info');
                    const projectDetailsDiv = document.getElementById('project-details');
                    
                    let detailsHtml = `
                        <p><strong>🎯 Основной тип:</strong> ${projectInfo.main_type.toUpperCase()}</p>
                        <p><strong>🔧 Обнаруженные технологии:</strong> ${projectInfo.detected_types.join(', ')}</p>
                    `;
                    
                    if (projectInfo.is_multi_tech) {
                        detailsHtml += '<p><strong>📦 Мульти-технологичный проект</strong></p>';
                    }
                    
                    if (projectInfo.file_stats) {
                        const topExtensions = Object.entries(projectInfo.file_stats)
                            .sort((a, b) => b[1] - a[1])
                            .slice(0, 5);
                        
                        detailsHtml += '<p><strong>📁 Топ файлов:</strong> ';
                        detailsHtml += topExtensions.map(([ext, count]) => `${ext} (${count})`).join(', ');
                        detailsHtml += '</p>';
                    }
                    
                    projectDetailsDiv.innerHTML = detailsHtml;
                    projectInfoDiv.style.display = 'block';
                }
            } catch (error) {
                console.log('Не удалось загрузить информацию о проекте:', error);
            }
        });
        
        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const question = formData.get('question');
            const rag_type = formData.get('rag_type');
            const use_gpu = formData.get('use_gpu');
            
            const resultDiv = document.getElementById('result');
            const submitButton = document.querySelector('button[type="submit"]');
            
            // Показываем загрузку
            resultDiv.style.display = 'block';
            resultDiv.className = 'result loading';
            resultDiv.textContent = '🔄 Анализируем код... Это может занять несколько секунд.';
            submitButton.disabled = true;
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ question, rag_type, use_gpu })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.className = 'result';
                    let resultText = data.result;
                    
                    // Добавляем информацию о проекте если есть
                    if (data.project_info) {
                        const info = data.project_info;
                        resultText = `🎯 Проект: ${info.main_type.toUpperCase()}` + 
                                   (info.is_multi_tech ? ` (мульти-технологии: ${info.detected_types.join(', ')})` : '') + 
                                   `\\n📊 Использована система: ${data.rag_system}\\n\\n` + resultText;
                    }
                    
                    resultDiv.textContent = resultText;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.textContent = '❌ Ошибка: ' + data.error;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.textContent = '❌ Ошибка соединения: ' + error.message;
            } finally {
                submitButton.disabled = false;
            }
        });
    </script>
</body>
</html>
"""

# Глобальные переменные
rag_tools = {}
project_info = None

# Импортируем менеджер GPU памяти
try:
    from gpu_utils import gpu_manager, cleanup_gpu, monitor_gpu
    GPU_UTILS_AVAILABLE = True
except ImportError:
    GPU_UTILS_AVAILABLE = False
    print("⚠️  GPU утилиты недоступны")

def init_tools():
    """Инициализация RAG инструментов с новой архитектурой."""
    global rag_tools, project_info
    
    try:
        import os
        from rag_main import create_rag_tool
        from rag_base import RAGSystemFactory
        
        # Очищаем GPU память перед инициализацией
        if GPU_UTILS_AVAILABLE:
            print("🧹 Очистка GPU памяти перед инициализацией...")
            cleanup_gpu()
            monitor_gpu()
        
        project_path = os.getenv('TEST_PROJ_PATH')
        if not project_path:
            print("❌ Установите переменную окружения TEST_PROJ_PATH")
            return False
        
        # Получаем информацию о проекте
        project_info = RAGSystemFactory.get_project_info(project_path)
        print(f"🎯 Обнаружены технологии: {', '.join(project_info['detected_types'])}")
        print(f"📊 Основной тип: {project_info['main_type']}")
        
        # Создаем инструменты для разных типов
        available_types = ['python', 'universal', 'auto']
        
        for rag_type in available_types:
            try:
                print(f"🔧 Создаем {rag_type.upper()} RAG инструмент...")
                
                # Проверяем память перед созданием каждого инструмента
                if GPU_UTILS_AVAILABLE and gpu_manager.check_memory_threshold(80):
                    print("⚠️  Высокое использование памяти, выполняем очистку...")
                    cleanup_gpu()
                
                tool = create_rag_tool(
                    project_path=project_path,
                    rag_type=rag_type,
                    use_gpu=None  # Автоопределение
                )
                rag_tools[rag_type] = tool
                print(f"✅ {rag_type.upper()} инструмент готов")
                
                # Мониторинг после создания
                if GPU_UTILS_AVAILABLE:
                    monitor_gpu()
                
            except Exception as e:
                print(f"⚠️  Не удалось создать {rag_type} инструмент: {e}")
                if GPU_UTILS_AVAILABLE:
                    cleanup_gpu()  # Очищаем память при ошибке
        
        return len(rag_tools) > 0
        
    except ImportError as e:
        print(f"❌ Ошибка импорта новой RAG архитектуры: {e}")
        return False
    except Exception as e:
        print(f"❌ Ошибка инициализации: {e}")
        if GPU_UTILS_AVAILABLE:
            cleanup_gpu()
        return False

def select_rag_tool(rag_type, question):
    """Выбор RAG инструмента."""
    if rag_type == "auto":
        # Автоматический выбор на основе проекта
        if project_info:
            main_type = project_info['main_type']
            if main_type in rag_tools:
                return rag_tools[main_type], f"Auto-selected {main_type.upper()}"
        
        # Fallback на универсальный
        return rag_tools.get('universal'), "Auto-selected UNIVERSAL"
    
    elif rag_type in rag_tools:
        return rag_tools[rag_type], rag_type.upper()
    
    else:
        # Fallback на любой доступный
        if rag_tools:
            fallback_type = list(rag_tools.keys())[0]
            return rag_tools[fallback_type], f"Fallback to {fallback_type.upper()}"
    
    return None, "No tool available"

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        question = data.get('question', '').strip()
        rag_type = data.get('rag_type', 'auto')
        use_gpu = data.get('use_gpu', 'auto')
        
        if not question:
            return jsonify({'success': False, 'error': 'Вопрос не может быть пустым'})
        
        # Проверяем состояние памяти перед выполнением
        if GPU_UTILS_AVAILABLE:
            if gpu_manager.check_memory_threshold(85):
                print("⚠️  Критическое использование памяти, выполняем очистку...")
                cleanup_gpu()
        
        # Выбираем RAG инструмент
        selected_tool, rag_system_name = select_rag_tool(rag_type, question)
        
        if not selected_tool:
            return jsonify({
                'success': False, 
                'error': 'Не удалось найти подходящий RAG инструмент'
            })
        
        # Выполняем анализ
        result = selected_tool.invoke(question)
        
        # Очищаем память после выполнения
        if GPU_UTILS_AVAILABLE:
            cleanup_gpu()
        
        # Получаем информацию о памяти для ответа
        memory_info = {}
        if GPU_UTILS_AVAILABLE:
            memory_info = gpu_manager.get_gpu_memory_info()
        
        return jsonify({
            'success': True,
            'result': result,
            'rag_system': rag_system_name,
            'project_info': project_info,
            'memory_info': memory_info,
            'settings': {
                'rag_type': rag_type,
                'use_gpu': use_gpu
            }
        })
        
    except Exception as e:
        # Очищаем память при ошибке
        if GPU_UTILS_AVAILABLE:
            cleanup_gpu()
        
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 
        'tools_loaded': len(rag_tools) > 0,
        'available_rag_systems': list(rag_tools.keys()),
        'project_info': project_info
    })

@app.route('/project-info')
def get_project_info():
    """Возвращает информацию о проекте"""
    return jsonify(project_info if project_info else {'error': 'Project not analyzed'})

@app.route('/memory-info')
def get_memory_info():
    """Возвращает информацию о состоянии GPU памяти"""
    if GPU_UTILS_AVAILABLE:
        return jsonify(gpu_manager.get_gpu_memory_info())
    else:
        return jsonify({'gpu_available': False, 'error': 'GPU utilities not available'})

@app.route('/cleanup-memory', methods=['POST'])
def cleanup_memory():
    """Принудительная очистка GPU памяти"""
    if GPU_UTILS_AVAILABLE:
        try:
            cleanup_gpu()
            memory_info = gpu_manager.get_gpu_memory_info()
            return jsonify({
                'success': True, 
                'message': 'Память очищена',
                'memory_info': memory_info
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)})
    else:
        return jsonify({'success': False, 'error': 'GPU utilities not available'})

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
    
    # Очищаем глобальные переменные
    global rag_tools
    rag_tools.clear()
    
    import gc
    gc.collect()
    print("✅ Ресурсы освобождены")

if __name__ == '__main__':
    import signal
    import atexit
    import sys
    
    # Регистрируем обработчики для корректного завершения
    atexit.register(cleanup_on_exit)
    
    def signal_handler(sig, frame):
        print("\n🛑 Получен сигнал завершения...")
        cleanup_on_exit()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Запуск веб-интерфейса RAG системы...")
    
    # Инициализируем инструменты
    if not init_tools():
        print("❌ Не удалось загрузить RAG инструменты")
        sys.exit(1)
    
    print("✅ RAG инструменты загружены успешно!")
    print("🌐 Веб-интерфейс будет доступен по адресу: http://localhost:5000")
    print("💡 Нажмите Ctrl+C для остановки")
    
    try:
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n🛑 Завершение работы...")
    finally:
        cleanup_on_exit()
