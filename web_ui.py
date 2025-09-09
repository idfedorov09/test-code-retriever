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
                <label for="tool">Инструмент анализа:</label>
                <select id="tool" name="tool">
                    <option value="auto">Автоматический выбор</option>
                    <option value="tool_llm_encoder">LLM Embedder (лучше для классов)</option>
                    <option value="tool_bge_code">BGE Code (лучше для функций)</option>
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
            <div class="example" onclick="setQuestion('Где определен класс User?')">
                Где определен класс User?
            </div>
        </div>
        
        <div id="result" class="result" style="display: none;"></div>
    </div>

    <script>
        function setQuestion(question) {
            document.getElementById('question').value = question;
        }
        
        document.getElementById('questionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const question = formData.get('question');
            const tool = formData.get('tool');
            
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
                    body: JSON.stringify({ question, tool })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.className = 'result';
                    resultDiv.textContent = data.result;
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

# Глобальные переменные для инструментов
tool_llm_encoder = None
tool_bge_code = None

def init_tools():
    """Инициализация RAG инструментов."""
    global tool_llm_encoder, tool_bge_code
    
    try:
        from rag import tool_llm_encoder as tle, tool_bge_code as tbc
        tool_llm_encoder = tle
        tool_bge_code = tbc
        return True
    except ImportError as e:
        print(f"❌ Ошибка импорта RAG инструментов: {e}")
        return False

def auto_select_tool(question):
    """Автоматический выбор инструмента."""
    question_lower = question.lower()
    class_keywords = ["класс", "class", "наследование", "inheritance", "использует", "inherited", "extends"]
    
    if any(keyword in question_lower for keyword in class_keywords):
        return tool_llm_encoder, "LLM Embedder"
    else:
        return tool_bge_code, "BGE Code"

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        question = data.get('question', '').strip()
        tool_choice = data.get('tool', 'auto')
        
        if not question:
            return jsonify({'success': False, 'error': 'Вопрос не может быть пустым'})
        
        # Выбираем инструмент
        if tool_choice == 'tool_llm_encoder':
            selected_tool = tool_llm_encoder
            tool_name = "LLM Embedder"
        elif tool_choice == 'tool_bge_code':
            selected_tool = tool_bge_code
            tool_name = "BGE Code"
        else:  # auto
            selected_tool, tool_name = auto_select_tool(question)
        
        # Выполняем анализ
        result = selected_tool.invoke(question)
        
        return jsonify({
            'success': True,
            'result': result,
            'tool_used': tool_name
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'tools_loaded': tool_llm_encoder is not None})

if __name__ == '__main__':
    print("🚀 Запуск веб-интерфейса RAG системы...")
    
    # Инициализируем инструменты
    if not init_tools():
        print("❌ Не удалось загрузить RAG инструменты")
        sys.exit(1)
    
    print("✅ RAG инструменты загружены успешно!")
    print("🌐 Веб-интерфейс будет доступен по адресу: http://localhost:5000")
    print("💡 Нажмите Ctrl+C для остановки")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
