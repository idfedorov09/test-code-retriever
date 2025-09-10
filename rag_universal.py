#!/usr/bin/env python3
"""
Универсальная RAG система для анализа любых текстовых файлов.
Использует простой текстовый анализ вместо сложного AST парсинга.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from rag_base import (
    BaseRAGSystem, BaseFileMap, BaseCodeElement, FileParser, DependencyAnalyzer,
    ProjectIndex, RAGSystemFactory, _relpath, _read_text
)


# -----------------------------------------------------------------------------
# Универсальные структуры данных
# -----------------------------------------------------------------------------

@dataclass
class UniversalCodeChunk(BaseCodeElement):
    """Универсальный фрагмент кода"""
    content: str
    chunk_type: str  # "function", "class", "config", "generic"
    keywords: List[str] = field(default_factory=list)
    
    def to_line(self) -> str:
        return f"{self.chunk_type}: {self.name}  # L{self.lineno}-{self.end_lineno}"


@dataclass
class UniversalFileMap(BaseFileMap):
    """Универсальная карта файла"""
    chunks: List[UniversalCodeChunk]
    keywords: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    def to_text(self) -> str:
        parts = [f"FILE: {self.path} (type: {self.file_type})"]
        
        if self.keywords:
            parts.append("KEYWORDS: " + ", ".join(self.keywords[:10]))
        
        if self.dependencies:
            parts.append("DEPENDENCIES: " + ", ".join(self.dependencies[:5]))
        
        if self.chunks:
            parts.append("CHUNKS:")
            for chunk in self.chunks[:5]:  # Показываем только первые 5 чанков
                parts.append(f"  - {chunk.to_line()}")
            if len(self.chunks) > 5:
                parts.append(f"  ... and {len(self.chunks) - 5} more chunks")
        
        parts.append(f"LOC: {self.loc}")
        return "\n".join(parts)
    
    def get_searchable_content(self) -> List[str]:
        content = []
        for chunk in self.chunks:
            content.append(f"{chunk.chunk_type}:{chunk.name}")
            content.extend([f"keyword:{kw}" for kw in chunk.keywords])
        return content


# -----------------------------------------------------------------------------
# Универсальные парсеры
# -----------------------------------------------------------------------------

class UniversalFileParser(FileParser):
    """Универсальный парсер для текстовых файлов"""
    
    # Паттерны для разных типов файлов
    FILE_TYPE_PATTERNS = {
        '.py': 'python',
        '.js': 'javascript', '.ts': 'typescript', '.jsx': 'react', '.tsx': 'react',
        '.html': 'html', '.css': 'css', '.scss': 'sass',
        '.sql': 'sql',
        '.yml': 'yaml', '.yaml': 'yaml',
        '.json': 'json',
        '.md': 'markdown', '.rst': 'rst',
        '.txt': 'text', '.log': 'log',
        '.env': 'env',
        '.toml': 'toml', '.ini': 'ini',
        '.sh': 'shell', '.bash': 'shell',
    }
    
    # Контекстные промпты для разных типов
    CONTEXT_PROMPTS = {
        'python': "Python код: функции, классы, импорты, PEP8 стандарты",
        'javascript': "JavaScript/TypeScript: функции, классы, модули, современные практики",
        'react': "React компоненты: JSX, хуки, состояние, props",
        'html': "HTML разметка: семантика, доступность, SEO",
        'css': "CSS стили: селекторы, flexbox, grid, адаптивность",
        'sql': "SQL запросы: таблицы, индексы, производительность",
        'yaml': "YAML конфигурация: структура, переменные, безопасность",
        'json': "JSON данные: структура, валидация",
        'dockerfile': "Docker: образы, безопасность, оптимизация размера",
        'markdown': "Документация: структура, ссылки, форматирование",
        'shell': "Shell скрипты: команды, переменные, безопасность",
        'env': "Переменные окружения: секреты, конфигурация"
    }
    
    def can_parse(self, file_path: str) -> bool:
        """Проверяет, является ли файл текстовым"""
        # Пропускаем бинарные файлы и ненужные директории
        path_parts = Path(file_path).parts
        skip_dirs = {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'env', '.env', 'dist', 'build'}
        
        if any(part in skip_dirs for part in path_parts):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                f.read(1024)  # Пробуем прочитать первые 1KB
            return True
        except:
            return False
    
    def parse(self, file_path: str, root: str) -> Optional[UniversalFileMap]:
        try:
            content = _read_text(file_path)
            file_type = self._detect_file_type(file_path)
            
            chunks = self._split_into_chunks(content, file_type)
            keywords = self._extract_keywords(content, file_type)
            dependencies = self._extract_dependencies(content, file_type)
            
            return UniversalFileMap(
                path=_relpath(file_path, root),
                file_type=file_type,
                chunks=chunks,
                keywords=keywords,
                dependencies=dependencies,
                loc=len(content.splitlines())
            )
        except Exception:
            return None
    
    def _detect_file_type(self, path: str) -> str:
        """Определяет тип файла по расширению и содержимому"""
        ext = Path(path).suffix.lower()
        filename = Path(path).name.lower()
        
        # Специальные случаи по имени файла
        if 'dockerfile' in filename or filename == 'dockerfile':
            return 'dockerfile'
        if 'docker-compose' in filename:
            return 'dockerfile'  # Тоже Docker
        if filename in {'makefile', 'cmake', 'cmakelists.txt', 'rakefile'}:
            return 'build'
        if filename in {'requirements.txt', 'package.json', 'composer.json', 'gemfile', 'cargo.toml', 'go.mod'}:
            return 'config'
        if filename.startswith('.env'):
            return 'env'
        if filename in {'readme.md', 'readme.rst', 'readme.txt', 'readme'}:
            return 'markdown'
        if filename.startswith('.') and not filename.endswith(('.py', '.js', '.ts')):
            return 'config'
        
        # По расширению
        file_type = self.FILE_TYPE_PATTERNS.get(ext, 'text')
        
        # Дополнительные проверки для конфигурационных файлов
        if file_type == 'text' and ext in {'.conf', '.cfg', '.ini', '.properties'}:
            return 'config'
        
        return file_type
    
    def _split_into_chunks(self, content: str, file_type: str) -> List[UniversalCodeChunk]:
        """Разбивает контент на логические части"""
        if file_type == 'python':
            return self._split_python_chunks(content)
        elif file_type in ['javascript', 'typescript', 'react']:
            return self._split_js_chunks(content)
        elif file_type == 'dockerfile':
            return self._split_dockerfile_chunks(content)
        elif file_type in ['yaml', 'json']:
            return self._split_config_chunks(content)
        elif file_type == 'sql':
            return self._split_sql_chunks(content)
        else:
            return self._split_generic_chunks(content)
    
    def _split_python_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Разбиение Python кода на функции и классы"""
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_name = "module_level"
        current_type = "generic"
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith(('def ', 'class ', 'async def ')):
                # Сохраняем предыдущий чанк
                if current_chunk:
                    chunks.append(UniversalCodeChunk(
                        name=current_name,
                        content='\n'.join(current_chunk),
                        chunk_type=current_type,
                        lineno=start_line,
                        end_lineno=i-1,
                        keywords=self._extract_python_keywords('\n'.join(current_chunk))
                    ))
                
                # Начинаем новый чанк
                current_chunk = [line]
                start_line = i
                
                if stripped.startswith('class '):
                    current_type = "class"
                    current_name = stripped.split()[1].split('(')[0].rstrip(':')
                else:
                    current_type = "function"
                    current_name = stripped.split()[1].split('(')[0]
            else:
                current_chunk.append(line)
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append(UniversalCodeChunk(
                name=current_name,
                content='\n'.join(current_chunk),
                chunk_type=current_type,
                lineno=start_line,
                end_lineno=len(lines),
                keywords=self._extract_python_keywords('\n'.join(current_chunk))
            ))
        
        return chunks
    
    def _split_js_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Разбиение JavaScript/React кода"""
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_name = "module_level"
        current_type = "generic"
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            # Функции и классы
            if (stripped.startswith(('function ', 'const ', 'let ', 'var ', 'class ', 'export function')) or
                'function' in stripped or '=>' in stripped):
                
                if current_chunk:
                    chunks.append(UniversalCodeChunk(
                        name=current_name,
                        content='\n'.join(current_chunk),
                        chunk_type=current_type,
                        lineno=start_line,
                        end_lineno=i-1,
                        keywords=self._extract_js_keywords('\n'.join(current_chunk))
                    ))
                
                current_chunk = [line]
                start_line = i
                
                if 'class ' in stripped:
                    current_type = "class"
                    current_name = re.search(r'class\s+(\w+)', stripped)
                    current_name = current_name.group(1) if current_name else "unknown_class"
                else:
                    current_type = "function"
                    # Пытаемся извлечь имя функции
                    match = re.search(r'(?:function\s+(\w+)|const\s+(\w+)|let\s+(\w+)|var\s+(\w+))', stripped)
                    current_name = next((g for g in match.groups() if g), "anonymous") if match else "anonymous"
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append(UniversalCodeChunk(
                name=current_name,
                content='\n'.join(current_chunk),
                chunk_type=current_type,
                lineno=start_line,
                end_lineno=len(lines),
                keywords=self._extract_js_keywords('\n'.join(current_chunk))
            ))
        
        return chunks
    
    def _split_dockerfile_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Разбиение Dockerfile на логические блоки"""
        chunks = []
        lines = content.splitlines()
        current_chunk = []
        current_name = "dockerfile_block"
        start_line = 1
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            
            if stripped.startswith(('FROM ', 'RUN ', 'COPY ', 'ADD ', 'WORKDIR ', 'EXPOSE ')):
                if current_chunk:
                    chunks.append(UniversalCodeChunk(
                        name=current_name,
                        content='\n'.join(current_chunk),
                        chunk_type="docker_command",
                        lineno=start_line,
                        end_lineno=i-1,
                        keywords=self._extract_docker_keywords('\n'.join(current_chunk))
                    ))
                
                current_chunk = [line]
                start_line = i
                current_name = stripped.split()[0].lower() + "_block"
            else:
                current_chunk.append(line)
        
        if current_chunk:
            chunks.append(UniversalCodeChunk(
                name=current_name,
                content='\n'.join(current_chunk),
                chunk_type="docker_command",
                lineno=start_line,
                end_lineno=len(lines),
                keywords=self._extract_docker_keywords('\n'.join(current_chunk))
            ))
        
        return chunks
    
    def _split_config_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Разбиение конфигурационных файлов"""
        chunks = []
        
        try:
            import json
            data = json.loads(content)
            
            if isinstance(data, dict):
                # Создаем отдельные чанки для разных секций
                
                # Основная информация
                main_info = {}
                for key in ['name', 'version', 'description', 'author', 'license']:
                    if key in data:
                        main_info[key] = data[key]
                
                if main_info:
                    chunks.append(UniversalCodeChunk(
                        name="main_info",
                        content=json.dumps(main_info, indent=2),
                        chunk_type="package_info",
                        lineno=1,
                        end_lineno=10,
                        keywords=list(main_info.keys())
                    ))
                
                # Dependencies
                if 'dependencies' in data:
                    deps = data['dependencies']
                    chunks.append(UniversalCodeChunk(
                        name="dependencies",
                        content=json.dumps(deps, indent=2),
                        chunk_type="dependencies",
                        lineno=1,
                        end_lineno=len(str(deps).splitlines()),
                        keywords=list(deps.keys())[:20]  # Первые 20 зависимостей как ключевые слова
                    ))
                
                # Dev Dependencies
                if 'devDependencies' in data:
                    dev_deps = data['devDependencies']
                    chunks.append(UniversalCodeChunk(
                        name="devDependencies",
                        content=json.dumps(dev_deps, indent=2),
                        chunk_type="dev_dependencies",
                        lineno=1,
                        end_lineno=len(str(dev_deps).splitlines()),
                        keywords=list(dev_deps.keys())[:10]
                    ))
                
                # Scripts
                if 'scripts' in data:
                    scripts = data['scripts']
                    chunks.append(UniversalCodeChunk(
                        name="scripts",
                        content=json.dumps(scripts, indent=2),
                        chunk_type="scripts",
                        lineno=1,
                        end_lineno=len(str(scripts).splitlines()),
                        keywords=list(scripts.keys())
                    ))
                
                # Если нет специальных секций, создаем общий чанк
                if not chunks:
                    chunks.append(UniversalCodeChunk(
                        name="config",
                        content=content,
                        chunk_type="configuration",
                        lineno=1,
                        end_lineno=len(content.splitlines()),
                        keywords=self._extract_config_keywords(content)
                    ))
            else:
                # Не словарь - создаем обычный чанк
                chunks.append(UniversalCodeChunk(
                    name="config",
                    content=content,
                    chunk_type="configuration",
                    lineno=1,
                    end_lineno=len(content.splitlines()),
                    keywords=self._extract_config_keywords(content)
                ))
                
        except (json.JSONDecodeError, Exception):
            # Если не JSON или ошибка парсинга - создаем обычный чанк
            chunks.append(UniversalCodeChunk(
                name="config",
                content=content,
                chunk_type="configuration",
                lineno=1,
                end_lineno=len(content.splitlines()),
                keywords=self._extract_config_keywords(content)
            ))
        
        return chunks
    
    def _split_sql_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Разбиение SQL файлов"""
        chunks = []
        statements = re.split(r';\s*\n', content)
        
        for i, stmt in enumerate(statements):
            if stmt.strip():
                stmt_type = "query"
                stmt_name = f"statement_{i+1}"
                
                # Определяем тип SQL выражения
                first_word = stmt.strip().split()[0].upper() if stmt.strip() else ""
                if first_word in ['CREATE', 'ALTER', 'DROP']:
                    stmt_type = "ddl"
                elif first_word in ['SELECT', 'INSERT', 'UPDATE', 'DELETE']:
                    stmt_type = "dml"
                
                chunks.append(UniversalCodeChunk(
                    name=stmt_name,
                    content=stmt,
                    chunk_type=stmt_type,
                    lineno=1,  # Упрощенно
                    end_lineno=len(stmt.splitlines()),
                    keywords=self._extract_sql_keywords(stmt)
                ))
        
        return chunks
    
    def _split_generic_chunks(self, content: str) -> List[UniversalCodeChunk]:
        """Универсальное разбиение на параграфы"""
        paragraphs = re.split(r'\n\s*\n', content)
        chunks = []
        
        for i, para in enumerate(paragraphs):
            if para.strip():
                chunks.append(UniversalCodeChunk(
                    name=f"section_{i+1}",
                    content=para,
                    chunk_type="text_section",
                    lineno=1,  # Упрощенно
                    end_lineno=len(para.splitlines()),
                    keywords=self._extract_text_keywords(para)
                ))
        
        return chunks
    
    def _extract_keywords(self, content: str, file_type: str) -> List[str]:
        """Извлекает ключевые слова в зависимости от типа файла"""
        if file_type == 'python':
            return self._extract_python_keywords(content)
        elif file_type in ['javascript', 'typescript', 'react']:
            return self._extract_js_keywords(content)
        elif file_type == 'dockerfile':
            return self._extract_docker_keywords(content)
        elif file_type in ['yaml', 'json']:
            return self._extract_config_keywords(content)
        elif file_type == 'sql':
            return self._extract_sql_keywords(content)
        else:
            return self._extract_text_keywords(content)
    
    def _extract_python_keywords(self, content: str) -> List[str]:
        keywords = []
        # Импорты
        keywords.extend(re.findall(r'import\s+(\w+)', content))
        keywords.extend(re.findall(r'from\s+(\w+)', content))
        # Функции и классы
        keywords.extend(re.findall(r'def\s+(\w+)', content))
        keywords.extend(re.findall(r'class\s+(\w+)', content))
        return list(set(keywords))
    
    def _extract_js_keywords(self, content: str) -> List[str]:
        keywords = []
        # Импорты и экспорты
        keywords.extend(re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', content))
        keywords.extend(re.findall(r'export\s+(?:function\s+)?(\w+)', content))
        # Функции
        keywords.extend(re.findall(r'function\s+(\w+)', content))
        keywords.extend(re.findall(r'const\s+(\w+)\s*=', content))
        return list(set(keywords))
    
    def _extract_docker_keywords(self, content: str) -> List[str]:
        keywords = []
        # Docker команды
        keywords.extend(re.findall(r'FROM\s+([^\s]+)', content))
        keywords.extend(re.findall(r'COPY\s+([^\s]+)', content))
        keywords.extend(re.findall(r'RUN\s+(\w+)', content))
        return list(set(keywords))
    
    def _extract_config_keywords(self, content: str) -> List[str]:
        """Извлекает ключевые слова из конфигурационных файлов"""
        keywords = []
        
        try:
            import json
            data = json.loads(content)
            
            if isinstance(data, dict):
                # Извлекаем ключи верхнего уровня
                keywords.extend(data.keys())
                
                # Специально для package.json/package-lock.json
                if 'dependencies' in data and isinstance(data['dependencies'], dict):
                    keywords.extend(list(data['dependencies'].keys())[:30])  # Первые 30 зависимостей
                
                if 'devDependencies' in data and isinstance(data['devDependencies'], dict):
                    keywords.extend(list(data['devDependencies'].keys())[:15])  # Первые 15 dev зависимостей
                
                if 'scripts' in data and isinstance(data['scripts'], dict):
                    keywords.extend(data['scripts'].keys())
                
                # Извлекаем важные значения
                for key in ['name', 'version', 'description']:
                    if key in data and isinstance(data[key], str):
                        keywords.append(data[key])
            
        except (json.JSONDecodeError, Exception):
            # Fallback: простое извлечение ключей из текста
            keywords = re.findall(r'"(\w+)"\s*:', content)
        
        # Убираем дубликаты и ограничиваем количество
        return list(set(keywords))[:50]
    
    def _extract_sql_keywords(self, content: str) -> List[str]:
        keywords = []
        # Таблицы
        keywords.extend(re.findall(r'FROM\s+(\w+)', content, re.IGNORECASE))
        keywords.extend(re.findall(r'JOIN\s+(\w+)', content, re.IGNORECASE))
        keywords.extend(re.findall(r'TABLE\s+(\w+)', content, re.IGNORECASE))
        return list(set(keywords))
    
    def _extract_text_keywords(self, content: str) -> List[str]:
        # Простое извлечение слов
        words = re.findall(r'\b[A-Za-z_]\w+\b', content)
        # Фильтруем слишком короткие и общие слова
        keywords = [w for w in words if len(w) > 3 and w.lower() not in {'this', 'that', 'with', 'from', 'have', 'will', 'been', 'were'}]
        return list(set(keywords[:20]))  # Ограничиваем количество
    
    def _extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Извлекает зависимости из файла"""
        deps = []
        
        if file_type == 'python':
            # Python импорты
            deps.extend(re.findall(r'import\s+([^\s,]+)', content))
            deps.extend(re.findall(r'from\s+([^\s]+)\s+import', content))
        elif file_type in ['javascript', 'typescript']:
            # JS/TS импорты
            deps.extend(re.findall(r'import.*from\s+[\'"]([^\'"]+)[\'"]', content))
            deps.extend(re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', content))
        elif file_type == 'dockerfile':
            # Docker образы
            deps.extend(re.findall(r'FROM\s+([^\s]+)', content))
        
        return list(set(deps))


# -----------------------------------------------------------------------------
# Универсальная RAG система
# -----------------------------------------------------------------------------

class UniversalRAGSystem(BaseRAGSystem):
    """Универсальная RAG система для анализа любых текстовых файлов"""
    
    def __init__(self, llm, embeddings, **kwargs):
        parsers = [UniversalFileParser()]
        analyzers = []  # Пока без специальных анализаторов
        
        # Консервативная конфигурация по умолчанию для предотвращения проблем с памятью
        default_config = {
            'max_documents': 1000,          # Максимум документов для индексации
            'max_chunks_per_file': 30,      # Максимум чанков на файл
            'max_file_size_kb': 1000,        # Максимальный размер файла в KB
            'faiss_batch_size': 20,        # Размер батча для FAISS
            'bm25_k': 8,                   # Количество документов для BM25
            'dense_k': 8,                  # Количество документов для dense retrieval
            'ensemble_weights': (0.6, 0.4), # BM25 приоритетнее
            'map_char_budget': 20000,      # Бюджет символов для контекста
            'evidence_char_budget': 15000, # Бюджет для evidence
            'max_evidence_items': 6,       # Максимум evidence items
            'answer_language': 'ru',
            'use_compression': False       # Отключаем компрессию для универсального RAG
        }
        
        # Объединяем с пользовательской конфигурацией
        merged_config = {**default_config, **kwargs}
        
        super().__init__(llm, embeddings, parsers, analyzers, **merged_config)
        
        print(f"⚙️  Universal RAG конфигурация:")
        print(f"   📄 Макс документов: {self.config.get('max_documents')}")
        print(f"   📦 Размер FAISS батча: {self.config.get('faiss_batch_size')}")
        print(f"   🔍 Чанков на файл: {self.config.get('max_chunks_per_file')}")
    
    def get_file_patterns(self) -> Dict[str, str]:
        return {"all": "**/*"}  # Все файлы
    
    def get_evidence_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
("system", """
You are a code reviewer analyzing a multi-language project. You will see a compact map of the repository with different file types and their contents.

The map shows:
- Files of different types (Python, JavaScript, Docker, YAML, etc.)
- Code chunks and their keywords
- Dependencies and relationships
- Configuration files and their settings

Given the user question, propose up to {max_items} precise evidence items (files/chunks to inspect) in JSON.
Each item must be an object with keys: file (relative path), symbol (chunk name or "*" for whole file), and reason.

IMPORTANT: 
- Paths must be repository-relative (e.g., `src/app.py`) — NEVER absolute
- Copy the exact path shown after `FILE:` in the context
- Consider the file type when determining relevance
- For configuration questions, focus on config files (YAML, JSON, ENV)
- For architecture questions, look at multiple file types
- For questions about specific files (like "package.json", "Dockerfile"), always request the whole file with symbol "*"
- For questions about file statistics (lines, size), always request the whole file with symbol "*"
- For questions about dependencies in JSON files, request both the whole file and specific chunks like "dependencies"
- For questions about specific line ranges (e.g., "lines 100-200", "строки 100-200"), use the line range as symbol (e.g., "100-200")
- For questions about specific line numbers (e.g., "line 11799-11833"), use the line range as symbol (e.g., "11799-11833")
- For questions about "what's in lines X-Y of file Z", use the line range as symbol (e.g., "11799-11833")
- Include both specific chunks and whole files for complete analysis

Respond ONLY with JSON array, no prose.
"""),
    ("human", "Question:\n{question}\n\nContext (map snippets):\n{context}\n")
])
    
    def get_answer_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
    ("system", """
You are a senior software architect analyzing a multi-technology project for code review. Using the provided repository map and evidence, answer the question comprehensively.

The repository includes files of different types:
- Source code (Python, JavaScript, etc.)
- Configuration files (YAML, JSON, ENV)
- Infrastructure files (Dockerfile, docker-compose)
- Documentation (Markdown, README)
- Build files (Makefile, package.json)

For questions about:
- Architecture: Consider relationships between different components and technologies
- Configuration: Focus on settings, environment variables, and deployment setup
- Dependencies: Look at imports, package files, and external services
- Security: Check for hardcoded secrets, permissions, and best practices
- Performance: Analyze code patterns and configuration optimizations

Provide specific file:line references from the evidence and explain relationships clearly.
Be precise, cite concrete references as `file:line`, and provide actionable insights. 
Consider the multi-technology nature of the project in your analysis.
Reply in {answer_language}.
"""),
    ("human", "Question:\n{question}\n\nRepo map (summaries):\n{map_text}\n\nEvidence (code bodies):\n{evidence_text}\n")
])
    
    def _create_documents(self, file_maps: List[BaseFileMap]) -> List[Document]:
        docs = []
        universal_maps = [fm for fm in file_maps if isinstance(fm, UniversalFileMap)]
        
        MAX_DOCS = self.config.get('max_documents', 10000)  # Увеличено для лучшего покрытия
        MAX_CHUNKS_PER_FILE = self.config.get('max_chunks_per_file', 100)  # Увеличено
        MAX_FILE_SIZE = self.config.get('max_file_size_kb', 1000)  # Увеличено
        
        print(f"📊 Создание документов: {len(universal_maps)} файлов")
        print(f"⚙️  Ограничения: макс {MAX_DOCS} документов, {MAX_CHUNKS_PER_FILE} чанков/файл, макс {MAX_FILE_SIZE}KB на файл")
        
        doc_count = 0
        
        # Сортируем файлы по приоритету и размеру
        universal_maps.sort(key=lambda fm: (self._get_file_priority(fm), fm.loc))
        
        # Create documents from file maps
        for fm in universal_maps:
            if doc_count >= MAX_DOCS:
                print(f"⚠️  Достигнут лимит документов ({MAX_DOCS}), пропускаем остальные файлы")
                break
            
            # Пропускаем слишком большие файлы
            file_size_kb = fm.loc * 0.05  # Примерная оценка размера
            if file_size_kb > MAX_FILE_SIZE:
                print(f"⚠️  Пропускаем большой файл: {fm.path} (~{file_size_kb:.1f}KB)")
                continue
            
            # Пропускаем файлы с малым количеством полезного контента
            if not fm.chunks and fm.loc < 10:
                continue
            
            # Создаем документ для всего файла только для очень маленьких и важных файлов
            if fm.loc < 100 and self._is_important_file(fm):  # Увеличено до 100 строк
                file_content = fm.to_text()
                if len(file_content) < 3000:  # Увеличено до 3000 символов
                    docs.append(Document(
                        page_content=file_content,
                        metadata={"source": fm.path, "type": f"{fm.file_type}-map", "loc": fm.loc}
                    ))
                    doc_count += 1
            
            # Создаем документы для ключевых чанков
            if fm.chunks:
                context_prompt = UniversalFileParser.CONTEXT_PROMPTS.get(fm.file_type, "")
                
                # Выбираем наиболее важные чанки
                important_chunks = self._select_important_chunks(fm.chunks, MAX_CHUNKS_PER_FILE)
                
                for chunk in important_chunks:
                    if doc_count >= MAX_DOCS:
                        break
                    
                    # Ограничиваем размер контента чанка
                    chunk_content = chunk.content
                    if len(chunk_content) > 2000:  # Увеличено до 2000 символов
                        chunk_content = chunk_content[:2000] + "...[truncated]"
                    
                    doc_content = f"""FILE: {fm.path} (type: {fm.file_type})
CONTEXT: {context_prompt}
CHUNK: {chunk.name} ({chunk.chunk_type})

{chunk_content}

KEYWORDS: {', '.join(chunk.keywords[:5])}"""  # Увеличено до 5 ключевых слов
                    
                    docs.append(Document(
                        page_content=doc_content,
                        metadata={
                            "source": fm.path,
                            "type": f"{fm.file_type}-chunk",
                            "chunk_name": chunk.name,
                            "chunk_type": chunk.chunk_type
                        }
                    ))
                    doc_count += 1
            
            # Для больших файлов создаем специальные документы с метаданными
            if fm.loc > 50:  # Файлы больше 50 строк
                file_stats = f"""FILE: {fm.path} (type: {fm.file_type})
SIZE: {fm.loc} lines
KEYWORDS: {', '.join(fm.keywords[:10])}
DEPENDENCIES: {', '.join(fm.dependencies[:5])}

This is a large file with {fm.loc} lines. Use specific queries to get detailed information about its contents."""
                
                docs.append(Document(
                    page_content=file_stats,
                    metadata={
                        "source": fm.path,
                        "type": f"{fm.file_type}-stats",
                        "loc": fm.loc,
                        "is_large_file": True
                    }
                ))
                doc_count += 1
        
        # Создаем компактный сводный документ по типам файлов
        if universal_maps:
            file_types_summary = ["PROJECT OVERVIEW:"]
            type_counts = {}
            files_by_type = {}
            
            for fm in universal_maps:
                type_counts[fm.file_type] = type_counts.get(fm.file_type, 0) + 1
                if fm.file_type not in files_by_type:
                    files_by_type[fm.file_type] = []
                files_by_type[fm.file_type].append(fm.path)
            
            # Ограничиваем количество типов в сводке
            for file_type, count in sorted(type_counts.items())[:15]:  # Увеличено до 15 типов
                context = UniversalFileParser.CONTEXT_PROMPTS.get(file_type, "")[:50]
                file_types_summary.append(f"  {file_type}: {count} files - {context}")
                
                # Добавляем конкретные имена файлов для важных типов
                if file_type in ['dockerfile', 'yaml', 'json', 'python', 'javascript', 'typescript', 'react'] and count <= 10:
                    files_list = files_by_type[file_type][:8]  # Увеличено до 8 файлов
                    file_types_summary.append(f"    Files: {', '.join(files_list)}")
                    if len(files_by_type[file_type]) > 8:
                        file_types_summary.append(f"    ... and {len(files_by_type[file_type]) - 8} more")
            
            docs.append(Document(
                page_content="\n".join(file_types_summary),
                metadata={"source": "__overview__", "type": "project-summary"}
            ))
            doc_count += 1
            
            # Создаем специальные документы для важных типов файлов с содержимым
            important_types = ['dockerfile', 'yaml', 'json', 'shell', 'config', 'env']
            for file_type in important_types:
                if file_type in files_by_type and len(files_by_type[file_type]) <= 12:
                    if doc_count >= MAX_DOCS - 5:  # Оставляем место для других документов
                        break
                        
                    type_summary = [f"{file_type.upper()} FILES IN PROJECT:"]
                    for fm in [f for f in universal_maps if f.file_type == file_type][:8]:  # Максимум 8 файлов этого типа
                        type_summary.append(f"  FILE: {fm.path} ({fm.loc} lines)")
                        
                        # Добавляем ключевые слова и зависимости
                        if fm.keywords:
                            key_words = fm.keywords[:5]
                            type_summary.append(f"    Keywords: {', '.join(key_words)}")
                        if fm.dependencies:
                            deps = fm.dependencies[:3]
                            type_summary.append(f"    Dependencies: {', '.join(deps)}")
                        
                        # Для очень важных типов добавляем краткое содержимое
                        if file_type in ['dockerfile', 'env'] and fm.chunks:
                            first_chunk = fm.chunks[0]
                            if len(first_chunk.content) < 500:  # Только короткие файлы
                                type_summary.append(f"    Content preview: {first_chunk.content[:200]}...")
                    
                    docs.append(Document(
                        page_content="\n".join(type_summary),
                        metadata={"source": f"__{file_type}_files__", "type": f"{file_type}-files-summary"}
                    ))
                    doc_count += 1
        
        print(f"✅ Создано {doc_count} документов (из {len(universal_maps)} файлов)")
        return docs
    
    def _get_file_priority(self, fm: UniversalFileMap) -> int:
        """Возвращает приоритет файла (меньше = важнее)"""
        # Важные типы файлов получают высокий приоритет
        priority_map = {
            'python': 1,
            'javascript': 1, 'typescript': 1, 'react': 1,
            'dockerfile': 2,
            'yaml': 2, 'json': 2,
            'sql': 3,
            'markdown': 4,
            'text': 5
        }
        return priority_map.get(fm.file_type, 6)
    
    def _is_important_file(self, fm: UniversalFileMap) -> bool:
        """Проверяет, является ли файл важным для индексации"""
        important_types = {'python', 'javascript', 'typescript', 'react', 'dockerfile', 'yaml', 'json'}
        important_names = {'readme', 'config', 'settings', 'requirements', 'package'}
        
        if fm.file_type in important_types:
            return True
        
        filename = fm.path.lower()
        return any(name in filename for name in important_names)
    
    def _select_important_chunks(self, chunks: List[UniversalCodeChunk], max_chunks: int) -> List[UniversalCodeChunk]:
        """Выбирает наиболее важные чанки для индексации"""
        if len(chunks) <= max_chunks:
            return chunks
        
        # Приоритет по типу чанка
        priority_order = {
            'class': 3,
            'function': 2,
            'configuration': 2,
            'docker_command': 1,
            'generic': 0,
            'text_section': 0
        }
        
        # Сортируем по приоритету и размеру
        sorted_chunks = sorted(
            chunks,
            key=lambda c: (
                priority_order.get(c.chunk_type, 0),
                len(c.keywords),  # Больше ключевых слов = важнее
                len(c.content)    # Больше контента = важнее
            ),
            reverse=True
        )
        
        return sorted_chunks[:max_chunks]
    
    def _build_search_index(
        self, 
        docs: List[Document], 
        project_path: str, 
        file_maps: List[BaseFileMap]
    ) -> ProjectIndex:
        print(f"🏗️  Построение поискового индекса для {len(docs)} документов...")
        
        # Импортируем GPU утилиты если доступны
        try:
            from gpu_utils import cleanup_gpu, monitor_gpu, gpu_manager
            GPU_UTILS_AVAILABLE = True
        except ImportError:
            GPU_UTILS_AVAILABLE = False
        
        # BM25 retriever (быстрый, не использует GPU)
        print("📝 Создание BM25 индекса...")
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.get('bm25_k', 12)
        
        # FAISS retriever с батчевой обработкой
        print("🔍 Создание FAISS индекса...")
        
        # Проверяем память перед созданием FAISS
        if GPU_UTILS_AVAILABLE:
            if gpu_manager.check_memory_threshold(70):
                print("⚠️  Высокое использование памяти, выполняем очистку...")
                cleanup_gpu()
            monitor_gpu()
        
        try:
            # Батчевая обработка для больших наборов документов
            batch_size = self.config.get('faiss_batch_size', 50)  # Консервативный размер
            
            if len(docs) > batch_size:
                print(f"📦 Батчевая обработка: {len(docs)} документов по {batch_size}")
                
                # Создаем FAISS индекс по частям
                first_batch = docs[:batch_size]
                faiss = FAISS.from_documents(
                    first_batch, 
                    self.embeddings, 
                    distance_strategy=DistanceStrategy.COSINE
                )
                
                # Добавляем остальные документы батчами
                remaining_docs = docs[batch_size:]
                for i in range(0, len(remaining_docs), batch_size):
                    batch = remaining_docs[i:i + batch_size]
                    batch_num = i // batch_size + 2
                    total_batches = (len(docs) - 1) // batch_size + 1
                    print(f"📦 Обработка батча {batch_num}/{total_batches} ({len(batch)} документов)")
                    
                    # Очистка памяти между батчами
                    if GPU_UTILS_AVAILABLE:
                        cleanup_gpu()
                    
                    # Добавляем батч к существующему индексу
                    batch_texts = [doc.page_content for doc in batch]
                    batch_metadatas = [doc.metadata for doc in batch]
                    faiss.add_texts(batch_texts, batch_metadatas)
            else:
                # Обычная обработка для небольших наборов
                print(f"📄 Обычная обработка для {len(docs)} документов")
                faiss = FAISS.from_documents(
                    docs, 
                    self.embeddings, 
                    distance_strategy=DistanceStrategy.COSINE
                )
            
            print("✅ FAISS индекс создан успешно")
            
        except Exception as e:
            print(f"❌ Ошибка создания FAISS индекса: {e}")
            if GPU_UTILS_AVAILABLE:
                cleanup_gpu()
            
            # Fallback: используем только BM25
            print("🔄 Переключаемся на BM25-only режим")
            return ProjectIndex(
                root=project_path,
                docs=docs,
                retriever=bm25,
                file_maps=file_maps
            )
        
        # Очистка памяти после создания индекса
        if GPU_UTILS_AVAILABLE:
            cleanup_gpu()
            print("📊 Финальное состояние памяти:")
            monitor_gpu()
        
        dense_retriever = faiss.as_retriever(search_kwargs={"k": self.config.get('dense_k', 12)})
        
        # Ensemble retriever
        print("🤝 Создание ensemble retriever...")
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense_retriever], 
            weights=list(self.config.get('ensemble_weights', (0.5, 0.5)))
        )
        
        print("✅ Поисковый индекс готов!")
        return ProjectIndex(
            root=project_path,
            docs=docs,
            retriever=ensemble,
            file_maps=file_maps
        )
    
    def _answer_question(self, question: str, index: ProjectIndex) -> str:
        # Retrieve relevant documents
        evidence_char_budget = self.config.get('evidence_char_budget', 15000)  # Уменьшено для универсального RAG
        max_evidence_items = self.config.get('max_evidence_items', 6)  # Уменьшено
        map_char_budget = self.config.get('map_char_budget', 20000)  # Уменьшено

        retrieved = index.retriever.invoke(question)
        
        # Gather context
        map_text = self._gather_map_snippets(retrieved, map_char_budget)
        
        # Generate evidence plan
        evidence_prompt = self.get_evidence_prompt_template()
        evidence_chain = evidence_prompt | self.llm | StrOutputParser()

        raw_plan = evidence_chain.invoke({
            "question": question,
            "context": map_text,
            "max_items": max_evidence_items,
        })
        
        plan_json = raw_plan.strip()
        plan_json = plan_json[plan_json.find("[") : plan_json.rfind("]") + 1] if "[" in plan_json and "]" in plan_json else "[]"
        try:
            plan = json.loads(plan_json)
            if not isinstance(plan, list):
                plan = []
        except Exception:
            plan = []

        # Fetch requested bodies
        evidence_pairs = self._extract_bodies(index.root, plan[:max_evidence_items]) if plan else []
        evidence_text = "\n\n".join([f"### {lbl}\n" + code for (lbl, code) in evidence_pairs])
        evidence_text = self._trim_to_chars(evidence_text, evidence_char_budget)
        
        # Generate final answer
        answer_prompt = self.get_answer_prompt_template()
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": self.config.get('answer_language', 'ru'),
            "evidence_text": evidence_text if evidence_text else "(no additional bodies requested)",
        })
        
        return final
    
    def _extract_bodies(self, root: str, requests: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Return list of (label, code_block) for requested {file,symbol}."""
        out: List[Tuple[str, str]] = []
        
        # Сначала собираем все доступные файлы для лучшего поиска
        available_files = {}
        for root_dir, dirs, files in os.walk(root):
            # Пропускаем ненужные директории
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}]
            for file in files:
                file_path = os.path.join(root_dir, file)
                rel_path = os.path.relpath(file_path, root)
                available_files[file.lower()] = rel_path
                available_files[rel_path.lower()] = rel_path
        
        for req in requests:
            rel = req.get("file", "").strip()
            sym = req.get("symbol", "").strip()
            
            if not rel:
                continue
            
            # Пытаемся найти файл разными способами
            target_path = None
            
            # 1. Прямой путь
            direct_path = os.path.join(root, rel)
            if os.path.isfile(direct_path):
                target_path = direct_path
            
            # 2. Поиск по имени файла (case-insensitive)
            elif rel.lower() in available_files:
                target_path = os.path.join(root, available_files[rel.lower()])
            
            # 3. Поиск по частичному совпадению
            else:
                rel_lower = rel.lower()
                for filename, filepath in available_files.items():
                    if (rel_lower in filename or 
                        filename in rel_lower or
                        rel_lower in filepath.lower() or
                        filepath.lower().endswith(rel_lower)):
                        target_path = os.path.join(root, filepath)
                        break
            
            if not target_path or not os.path.isfile(target_path):
                continue
            
            try:
                src = _read_text(target_path)
                rel_for_output = os.path.relpath(target_path, root)
                file_size = len(src)
                line_count = len(src.splitlines())
            except Exception:
                continue

            # Специальная обработка для разных типов файлов
            if sym == "*" or not sym:
                # Для больших файлов используем специальную логику
                if file_size > 10000:  # Больше 10KB
                    content = self._handle_large_file(src, rel_for_output, file_size, line_count)
                else:
                    content = src
                out.append((f"{rel_for_output}:1", content))
            else:
                # Проверяем, не запрашивается ли диапазон строк
                line_range = self._extract_line_range_from_symbol(sym)
                if line_range:
                    start_line, end_line = line_range
                    if start_line <= line_count and end_line <= line_count:
                        # Извлекаем запрашиваемые строки
                        lines = src.splitlines()
                        requested_lines = lines[start_line-1:end_line]
                        
                        # Создаем результат с номерами строк
                        result = f"# Строки {start_line}-{end_line} из {rel_for_output}\n\n"
                        result += "```\n"
                        for i, line in enumerate(requested_lines, start_line):
                            result += f"{i:4d}: {line}\n"
                        result += "```\n"
                        
                        out.append((f"{rel_for_output}:{start_line}-{end_line}", result))
                    else:
                        # Строки не найдены
                        out.append((f"{rel_for_output}:{start_line}-{end_line}", 
                                  f"# Строки {start_line}-{end_line} не найдены в файле {rel_for_output}\n"
                                  f"Файл содержит {line_count} строк"))
                else:
                    # Ищем конкретный чанк или символ
                    found = self._find_symbol_in_file(src, sym, rel_for_output)
                    if found:
                        out.append(found)
                    else:
                        # Fallback: возвращаем начало файла с указанием что символ не найден
                        content = f"# Symbol '{sym}' not found in file, showing full content:\n\n"
                        content += src[:3000] + "...[truncated]" if len(src) > 3000 else src
                        out.append((f"{rel_for_output}:1", content))

        return out
    
    def _handle_large_file(self, content: str, file_path: str, file_size: int, line_count: int) -> str:
        """Обрабатывает большие файлы с умным извлечением контента"""
        file_ext = file_path.split('.')[-1].lower()
        
        # Статистика файла
        stats = f"# Файл: {file_path}\n"
        stats += f"# Размер: {file_size:,} байт\n"
        stats += f"# Строк: {line_count:,}\n\n"
        
        # Универсальная обработка для всех типов файлов
        result = stats
        
        # Анализируем структуру файла
        lines = content.splitlines()
        
        # Показываем начало файла
        result += "## 📄 Начало файла\n\n"
        result += "```\n"
        for i, line in enumerate(lines[:20], 1):
            result += f"{i:4d}: {line}\n"
        result += "```\n\n"
        
        # Анализируем содержимое в зависимости от типа
        if file_ext == 'json':
            result += self._analyze_json_structure(content)
        elif file_ext in ['py', 'js', 'ts', 'jsx', 'tsx']:
            result += self._analyze_code_structure(content, file_ext)
        elif file_ext in ['md', 'txt', 'rst']:
            result += self._analyze_text_structure(content)
        elif file_ext in ['yaml', 'yml']:
            result += self._analyze_yaml_structure(content)
        elif file_ext in ['xml', 'html']:
            result += self._analyze_xml_structure(content)
        else:
            result += self._analyze_generic_structure(content)
        
        # Показываем конец файла
        if line_count > 40:
            result += "## 📄 Конец файла\n\n"
            result += "```\n"
            for i, line in enumerate(lines[-10:], line_count - 9):
                result += f"{i:4d}: {line}\n"
            result += "```\n\n"
        
        # Добавляем полный контент (обрезанный)
        result += "## 📄 Полный контент (первые 3000 символов)\n\n"
        result += "```\n"
        result += content[:3000]
        if len(content) > 3000:
            result += "\n... [truncated]"
        result += "\n```\n"
        
        return result
    
    def _analyze_json_structure(self, content: str) -> str:
        """Анализирует структуру JSON файла"""
        try:
            import json
            data = json.loads(content)
            
            result = "## 📊 JSON Структура\n\n"
            
            if isinstance(data, dict):
                result += f"**Корневые ключи:** {', '.join(data.keys())}\n\n"
                
                # Анализируем зависимости
                if 'dependencies' in data:
                    deps = data['dependencies']
                    result += f"**Dependencies:** {len(deps)} пакетов\n"
                    for name, version in list(deps.items())[:10]:
                        result += f"- {name}: {version}\n"
                    if len(deps) > 10:
                        result += f"... и еще {len(deps) - 10} зависимостей\n"
                    result += "\n"
                
                # Анализируем другие важные ключи
                for key in ['name', 'version', 'description', 'scripts', 'engines']:
                    if key in data:
                        result += f"**{key.title()}:** {data[key]}\n"
            else:
                result += f"**Тип данных:** {type(data).__name__}\n"
            
            return result
            
        except json.JSONDecodeError:
            return "## ❌ Ошибка парсинга JSON\n\n"
        except Exception as e:
            return f"## ❌ Ошибка анализа JSON: {e}\n\n"
    
    def _analyze_code_structure(self, content: str, file_ext: str) -> str:
        """Анализирует структуру файла с кодом"""
        lines = content.splitlines()
        
        result = f"## 📝 Анализ {file_ext.upper()} файла\n\n"
        
        # Считаем функции, классы, импорты
        functions = []
        classes = []
        imports = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith(('def ', 'function ', 'const ', 'let ', 'var ')):
                functions.append(f"Строка {i}: {stripped}")
            elif stripped.startswith(('class ', 'interface ')):
                classes.append(f"Строка {i}: {stripped}")
            elif stripped.startswith(('import ', 'from ', 'require(')):
                imports.append(f"Строка {i}: {stripped}")
        
        if functions:
            result += f"**Функции ({len(functions)}):**\n"
            for func in functions[:10]:
                result += f"- {func}\n"
            if len(functions) > 10:
                result += f"... и еще {len(functions) - 10} функций\n"
            result += "\n"
        
        if classes:
            result += f"**Классы ({len(classes)}):**\n"
            for cls in classes[:10]:
                result += f"- {cls}\n"
            if len(classes) > 10:
                result += f"... и еще {len(classes) - 10} классов\n"
            result += "\n"
        
        if imports:
            result += f"**Импорты ({len(imports)}):**\n"
            for imp in imports[:10]:
                result += f"- {imp}\n"
            if len(imports) > 10:
                result += f"... и еще {len(imports) - 10} импортов\n"
            result += "\n"
        
        return result
    
    def _analyze_text_structure(self, content: str) -> str:
        """Анализирует структуру текстового файла"""
        lines = content.splitlines()
        
        result = "## 📄 Анализ текстового файла\n\n"
        
        # Считаем заголовки, ссылки, списки
        headers = []
        links = []
        lists = []
        
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('#'):
                headers.append(f"Строка {i}: {stripped}")
            elif 'http' in stripped or 'www.' in stripped:
                links.append(f"Строка {i}: {stripped}")
            elif stripped.startswith(('-', '*', '+', '1.', '2.', '3.')):
                lists.append(f"Строка {i}: {stripped}")
        
        if headers:
            result += f"**Заголовки ({len(headers)}):**\n"
            for header in headers[:10]:
                result += f"- {header}\n"
            result += "\n"
        
        if links:
            result += f"**Ссылки ({len(links)}):**\n"
            for link in links[:5]:
                result += f"- {link}\n"
            result += "\n"
        
        if lists:
            result += f"**Списки ({len(lists)}):**\n"
            for item in lists[:5]:
                result += f"- {item}\n"
            result += "\n"
        
        return result
    
    def _analyze_yaml_structure(self, content: str) -> str:
        """Анализирует структуру YAML файла"""
        lines = content.splitlines()
        
        result = "## 📄 Анализ YAML файла\n\n"
        
        # Считаем секции (ключи верхнего уровня)
        sections = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and not stripped.startswith('-') and ':' in stripped:
                key = stripped.split(':')[0].strip()
                if not key.startswith(' '):  # Только ключи верхнего уровня
                    sections.append(f"Строка {i}: {key}")
        
        if sections:
            result += f"**Секции ({len(sections)}):**\n"
            for section in sections[:10]:
                result += f"- {section}\n"
            result += "\n"
        
        return result
    
    def _analyze_xml_structure(self, content: str) -> str:
        """Анализирует структуру XML/HTML файла"""
        lines = content.splitlines()
        
        result = "## 📄 Анализ XML/HTML файла\n\n"
        
        # Считаем теги
        tags = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if '<' in stripped and '>' in stripped:
                # Простое извлечение тегов
                import re
                found_tags = re.findall(r'<(\w+)', stripped)
                for tag in found_tags:
                    tags.append(f"Строка {i}: <{tag}>")
        
        if tags:
            result += f"**Теги ({len(tags)}):**\n"
            for tag in tags[:10]:
                result += f"- {tag}\n"
            result += "\n"
        
        return result
    
    def _analyze_generic_structure(self, content: str) -> str:
        """Анализирует структуру обычного файла"""
        lines = content.splitlines()
        
        result = "## 📄 Анализ файла\n\n"
        
        # Считаем непустые строки
        non_empty = [i for i, line in enumerate(lines, 1) if line.strip()]
        
        result += f"**Статистика:**\n"
        result += f"- Всего строк: {len(lines)}\n"
        result += f"- Непустых строк: {len(non_empty)}\n"
        result += f"- Пустых строк: {len(lines) - len(non_empty)}\n\n"
        
        # Показываем первые непустые строки
        if non_empty:
            result += f"**Первые непустые строки:**\n"
            for i in non_empty[:10]:
                result += f"Строка {i}: {lines[i-1].strip()}\n"
            result += "\n"
        
        return result
    
    def _extract_line_range_from_symbol(self, symbol: str) -> Optional[Tuple[int, int]]:
        """Извлекает диапазон строк из символа"""
        import re
        
        # Ищем паттерны типа "11799-11833" или "11799:11833"
        patterns = [
            r'(\d+)-(\d+)',  # 11799-11833
            r'(\d+):(\d+)',  # 11799:11833
            r'строк[иа]?\s+(\d+)-(\d+)',  # строки 11799-11833
            r'строк[иа]?\s+(\d+):(\d+)',  # строки 11799:11833
            r'line[s]?\s+(\d+)-(\d+)',  # lines 11799-11833
            r'line[s]?\s+(\d+):(\d+)',  # lines 11799:11833
        ]
        
        for pattern in patterns:
            match = re.search(pattern, symbol, re.IGNORECASE)
            if match:
                start_line = int(match.group(1))
                end_line = int(match.group(2))
                return (start_line, end_line)
        
        return None
    
    def _find_symbol_in_file(self, content: str, symbol: str, file_path: str) -> Optional[Tuple[str, str]]:
        """Ищет символ в файле и возвращает соответствующий фрагмент"""
        lines = content.splitlines()
        
        # Ищем по имени функции/класса/секции
        for i, line in enumerate(lines):
            if (symbol in line and 
                any(keyword in line.lower() for keyword in ['def ', 'class ', 'function', 'const ', 'let ', 'var '])):
                # Найдена строка с определением, извлекаем блок
                start_line = i + 1
                
                # Пытаемся найти конец блока (простая эвристика)
                end_line = min(i + 50, len(lines))  # Максимум 50 строк
                for j in range(i + 1, len(lines)):
                    if lines[j].strip() and not lines[j].startswith((' ', '\t')):
                        end_line = j
                        break
                
                block_content = '\n'.join(lines[i:end_line])
                return (f"{file_path}:{start_line}", block_content)
        
        return None
    
    def _gather_map_snippets(self, docs: List[Document], max_chars: int = 20000) -> str:
        pieces = []
        
        # Приоритизируем документы по важности
        priority_types = [
            "project-summary",
            "dockerfile-files-summary", "yaml-files-summary", "json-files-summary", 
            "shell-files-summary", "config-files-summary", "env-files-summary",
            "python-map", "javascript-map", "typescript-map", "react-map",
            "dockerfile-map", "yaml-map", "json-map", "text-map", "markdown-map"
        ]
        
        # Сначала добавляем приоритетные документы
        for priority_type in priority_types:
            for d in docs:
                if d.metadata.get("type") == priority_type:
                    pieces.append(f"---\n{d.page_content}\n")
                    if sum(len(p) for p in pieces) > max_chars:
                        break
            if sum(len(p) for p in pieces) > max_chars:
                break
        
        # Затем добавляем чанки, если есть место
        if sum(len(p) for p in pieces) < max_chars * 0.8:  # Если использовано меньше 80%
            for d in docs:
                doc_type = d.metadata.get("type", "")
                if doc_type.endswith("-chunk") and doc_type not in [p.split("\n")[1] for p in pieces]:
                    pieces.append(f"---\n{d.page_content}\n")
                    if sum(len(p) for p in pieces) > max_chars:
                        break
        
        return self._trim_to_chars("\n".join(pieces), max_chars)
    
    def _trim_to_chars(self, text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + "\n...\n" + text[-tail:]
    
    def _get_tool_description(self) -> str:
        return (
            "Analyze multi-technology projects including Python, JavaScript, Docker, configs, etc. "
            "Uses universal text analysis for comprehensive project understanding. "
            "Input: a natural-language question about any aspect of the project."
        )


# Регистрируем универсальную RAG систему
RAGSystemFactory.register('universal', UniversalRAGSystem)
