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
    <title>RAG - Анализ Кода</title>
    <!-- Подключаем marked.js для рендеринга markdown -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <!-- Подключаем highlight.js для подсветки синтаксиса -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            backdrop-filter: blur(10px);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.2em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .form-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #555;
            font-size: 14px;
        }
        input[type="text"], select, textarea {
            width: 100%;
            padding: 14px 16px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
            background: #fafbfc;
        }
        input[type="text"]:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
            background: white;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            padding: 14px 32px;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
        }
        button:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
            box-shadow: none;
        }
        .result {
            margin-top: 30px;
            background: white;
            border-radius: 12px;
            border: 1px solid #e1e5e9;
            overflow: hidden;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .result-header {
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            padding: 15px 20px;
            font-weight: 600;
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .result-content {
            padding: 25px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 15px;
            line-height: 1.6;
            color: #333;
        }
        .result-content h1, .result-content h2, .result-content h3, .result-content h4, .result-content h5, .result-content h6 {
            color: #2c3e50;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 600;
        }
        .result-content h1 { font-size: 24px; border-bottom: 2px solid #667eea; padding-bottom: 8px; }
        .result-content h2 { font-size: 20px; border-bottom: 1px solid #e1e5e9; padding-bottom: 5px; }
        .result-content h3 { font-size: 18px; }
        .result-content p {
            margin-bottom: 16px;
        }
        .result-content ul, .result-content ol {
            margin-bottom: 16px;
            padding-left: 25px;
        }
        .result-content li {
            margin-bottom: 8px;
        }
        .result-content blockquote {
            border-left: 4px solid #667eea;
            padding: 15px 20px;
            margin: 20px 0;
            background: #f8f9fc;
            border-radius: 0 8px 8px 0;
            font-style: italic;
        }
        .result-content code {
            background: #f1f3f4;
            padding: 3px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 14px;
            color: #d63384;
        }
        .result-content pre {
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            overflow-x: auto;
            margin: 20px 0;
            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
            font-size: 14px;
            line-height: 1.5;
        }
        .result-content pre code {
            background: none;
            padding: 0;
            color: inherit;
        }
        .result-content table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .result-content th, .result-content td {
            padding: 12px 16px;
            text-align: left;
            border-bottom: 1px solid #e9ecef;
        }
        .result-content th {
            background: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }
        .result-content strong {
            color: #2c3e50;
            font-weight: 600;
        }
        .result-content em {
            color: #6c757d;
            font-style: italic;
        }
        .project-info-badge {
            display: inline-block;
            background: linear-gradient(45deg, #28a745, #20c997);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }
        .rag-system-badge {
            display: inline-block;
            background: linear-gradient(45deg, #ffc107, #fd7e14);
            color: white;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        .error {
            border: 1px solid #dc3545;
        }
        .error .result-header {
            background: linear-gradient(45deg, #dc3545, #c82333);
        }
        .error .result-content {
            color: #721c24;
            background: #f8d7da;
        }
        .loading {
            text-align: center;
            color: #666;
            padding: 40px;
            font-size: 16px;
        }
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-left: 10px;
            border: 3px solid #667eea;
            border-radius: 50%;
            border-top-color: transparent;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        .examples {
            margin-top: 25px;
            padding: 20px;
            background: linear-gradient(135deg, #f8f9fc 0%, #e9ecef 100%);
            border-radius: 12px;
            border: 1px solid #e1e5e9;
        }
        .examples h3 {
            margin-top: 0;
            margin-bottom: 15px;
            color: #495057;
            font-size: 16px;
            font-weight: 600;
        }
        .example {
            background: white;
            padding: 12px 16px;
            margin: 8px 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
            border: 1px solid #e9ecef;
            font-size: 14px;
        }
        .example:hover {
            background: #667eea;
            color: white;
            transform: translateX(5px);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 RAG - Анализ Кода</h1>
        
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
                    <option value="javascript">JS</option>
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
            <div class="example" onclick="setQuestion('Покажи мне код любого файла Docker')">
                🐳 Покажи мне код любого файла Docker
            </div>
            <div class="example" onclick="setQuestion('Какие есть конфигурационные файлы?')">
                ⚙️ Какие есть конфигурационные файлы?
            </div>
            <div class="example" onclick="setQuestion('Покажи архитектуру проекта')">
                🏗️ Покажи архитектуру проекта
            </div>
            <div class="example" onclick="setQuestion('Как используется класс PrefixedDBModel?')">
                🐍 Как используется класс PrefixedDBModel?
            </div>
            <div class="example" onclick="setQuestion('Есть ли проблемы с безопасностью?')">
                🔒 Есть ли проблемы с безопасностью?
            </div>
            <div class="example" onclick="setQuestion('Какие JavaScript файлы есть в проекте?')">
                ⚡ Какие JavaScript файлы есть в проекте?
            </div>
            <div class="example" onclick="setQuestion('Покажи содержимое package.json')">
                📦 Покажи содержимое package.json
            </div>
            <div class="example" onclick="setQuestion('Какие переменные окружения используются?')">
                🌍 Какие переменные окружения используются?
            </div>
        </div>
        
        <div id="project-info" style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; display: none;">
            <h3>📊 Информация о проекте</h3>
            <div id="project-details"></div>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <div class="result-header">
                <span id="result-header-text">📊 Результат анализа</span>
            </div>
            <div id="result-content" class="result-content"></div>
        </div>
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
        
        // Настройка marked.js для рендеринга markdown
        if (typeof marked !== 'undefined') {
            marked.setOptions({
                highlight: function(code, lang) {
                    if (typeof hljs !== 'undefined' && lang && hljs.getLanguage(lang)) {
                        try {
                            return hljs.highlight(code, { language: lang }).value;
                        } catch (err) {}
                    }
                    return code;
                },
                breaks: true,
                gfm: true
            });
        }

        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const question = formData.get('question');
            const rag_type = formData.get('rag_type');
            const use_gpu = formData.get('use_gpu');
            
            const resultDiv = document.getElementById('result');
            const resultHeaderText = document.getElementById('result-header-text');
            const resultContent = document.getElementById('result-content');
            const submitButton = document.querySelector('button[type="submit"]');
            
            // Показываем загрузку
            resultDiv.style.display = 'block';
            resultDiv.className = 'result';
            resultHeaderText.innerHTML = '🔄 Анализ в процессе...';
            resultContent.className = 'result-content loading';
            resultContent.innerHTML = 'Анализируем код... Это может занять несколько секунд.';
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
                    
                    // Создаем заголовок с информацией о проекте и системе
                    let headerInfo = '📊 Результат анализа';
                    if (data.project_info && data.rag_system) {
                        const info = data.project_info;
                        const projectBadge = `<span class="project-info-badge">🎯 ${info.main_type.toUpperCase()}${info.is_multi_tech ? ' (мульти-технологии)' : ''}</span>`;
                        const ragBadge = `<span class="rag-system-badge">📊 ${data.rag_system}</span>`;
                        headerInfo = projectBadge + ragBadge;
                    }
                    resultHeaderText.innerHTML = headerInfo;
                    
                    // Рендерим markdown ответ
                    resultContent.className = 'result-content';
                    if (typeof marked !== 'undefined') {
                        try {
                            const htmlContent = marked.parse(data.result);
                            resultContent.innerHTML = htmlContent;
                            
                            // Применяем подсветку синтаксиса
                            if (typeof hljs !== 'undefined') {
                                resultContent.querySelectorAll('pre code').forEach((block) => {
                                    hljs.highlightElement(block);
                                });
                            }
                        } catch (err) {
                            console.error('Ошибка рендеринга markdown:', err);
                            resultContent.textContent = data.result;
                        }
                    } else {
                        // Fallback если marked.js не загружен
                        resultContent.innerHTML = data.result.replace(/\\n/g, '<br>');
                    }
                } else {
                    resultDiv.className = 'result error';
                    resultHeaderText.innerHTML = '❌ Ошибка анализа';
                    resultContent.className = 'result-content';
                    resultContent.textContent = data.error;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultHeaderText.innerHTML = '❌ Ошибка соединения';
                resultContent.className = 'result-content';
                resultContent.textContent = error.message;
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
        available_types = ['python', 'javascript', 'universal', 'auto']
        
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
        
        # Убираем из ответа информацию о проекте и системе, так как она отображается в заголовке
        lines = result.split('\n')
        filtered_lines = []
        for line in lines:
            if not (line.startswith('🎯 Проект:') or 
                   line.startswith('📊 Использована система:') or
                   (line.strip() == '' and len(filtered_lines) == 0)):  # Убираем пустые строки в начале
                filtered_lines.append(line)
        result = '\n'.join(filtered_lines)
        
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
