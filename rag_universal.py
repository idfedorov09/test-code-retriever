#!/usr/bin/env python3
"""
Универсальная RAG система для анализа любых текстовых файлов.
Использует простой текстовый анализ вместо сложного AST парсинга.
"""

from __future__ import annotations

import os
import re
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

class UniversalFileParser:
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
        
        # Специальные случаи
        filename = Path(path).name.lower()
        if 'dockerfile' in filename:
            return 'dockerfile'
        if filename in {'makefile', 'cmake', 'cmakelists.txt'}:
            return 'build'
        if filename.startswith('.'):
            return 'config'
        
        return self.FILE_TYPE_PATTERNS.get(ext, 'text')
    
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
        # Для YAML/JSON создаем один большой чанк с ключевыми словами
        return [UniversalCodeChunk(
            name="config",
            content=content,
            chunk_type="configuration",
            lineno=1,
            end_lineno=len(content.splitlines()),
            keywords=self._extract_config_keywords(content)
        )]
    
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
        # Простое извлечение ключей из YAML/JSON
        keywords = re.findall(r'(\w+):', content)
        return list(set(keywords))
    
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
        super().__init__(llm, embeddings, parsers, analyzers, **kwargs)
    
    def get_file_patterns(self) -> Dict[str, str]:
        return {"all": "**/*"}  # Все файлы
    
    def get_evidence_prompt_template(self) -> str:
        return """
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
- Include both specific chunks and whole files for complete analysis

Respond ONLY with JSON array, no prose.
"""
    
    def get_answer_prompt_template(self) -> str:
        return """
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
"""
    
    def _create_documents(self, file_maps: List[BaseFileMap]) -> List[Document]:
        docs = []
        universal_maps = [fm for fm in file_maps if isinstance(fm, UniversalFileMap)]
        
        # Create documents from file maps
        for fm in universal_maps:
            # Создаем документ для всего файла
            file_content = fm.to_text()
            docs.append(Document(
                page_content=file_content,
                metadata={"source": fm.path, "type": f"{fm.file_type}-map", "loc": fm.loc}
            ))
            
            # Создаем документы для каждого чанка с контекстом
            context_prompt = UniversalFileParser.CONTEXT_PROMPTS.get(fm.file_type, "")
            
            for chunk in fm.chunks:
                chunk_content = f"""
FILE: {fm.path} (type: {fm.file_type})
CONTEXT: {context_prompt}
CHUNK: {chunk.name} ({chunk.chunk_type})

CONTENT:
{chunk.content}

KEYWORDS: {', '.join(chunk.keywords)}
"""
                docs.append(Document(
                    page_content=chunk_content,
                    metadata={
                        "source": fm.path,
                        "type": f"{fm.file_type}-chunk",
                        "chunk_name": chunk.name,
                        "chunk_type": chunk.chunk_type
                    }
                ))
        
        # Создаем сводный документ по типам файлов
        file_types_summary = ["PROJECT FILE TYPES SUMMARY:"]
        type_counts = {}
        for fm in universal_maps:
            type_counts[fm.file_type] = type_counts.get(fm.file_type, 0) + 1
        
        for file_type, count in sorted(type_counts.items()):
            context = UniversalFileParser.CONTEXT_PROMPTS.get(file_type, "")
            file_types_summary.append(f"  {file_type}: {count} files - {context}")
        
        docs.append(Document(
            page_content="\n".join(file_types_summary),
            metadata={"source": "__file_types__", "type": "project-summary"}
        ))
        
        return docs
    
    def _build_search_index(
        self, 
        docs: List[Document], 
        project_path: str, 
        file_maps: List[BaseFileMap]
    ) -> ProjectIndex:
        # BM25 retriever
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.get('bm25_k', 12)
        
        # FAISS retriever
        faiss = FAISS.from_documents(docs, self.embeddings, distance_strategy=DistanceStrategy.COSINE)
        dense_retriever = faiss.as_retriever(search_kwargs={"k": self.config.get('dense_k', 12)})
        
        # Ensemble retriever
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense_retriever], 
            weights=list(self.config.get('ensemble_weights', (0.5, 0.5)))
        )
        
        return ProjectIndex(
            root=project_path,
            docs=docs,
            retriever=ensemble,
            file_maps=file_maps
        )
    
    def _answer_question(self, question: str, index: ProjectIndex) -> str:
        # Retrieve relevant documents
        retrieved = index.retriever.invoke(question)
        
        # Gather context (упрощенная версия)
        pieces = []
        for d in retrieved[:10]:  # Берем первые 10 документов
            pieces.append(f"---\n{d.page_content}\n")
        
        map_text = "\n".join(pieces)[:self.config.get('map_char_budget', 24000)]
        
        # Generate answer directly (упрощенная версия без evidence planning)
        answer_prompt = ChatPromptTemplate.from_template(self.get_answer_prompt_template())
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": self.config.get('answer_language', 'ru'),
            "evidence_text": "(using retrieved context)",
        })
        
        return final
    
    def _get_tool_description(self) -> str:
        return (
            "Analyze multi-technology projects including Python, JavaScript, Docker, configs, etc. "
            "Uses universal text analysis for comprehensive project understanding. "
            "Input: a natural-language question about any aspect of the project."
        )


# Регистрируем универсальную RAG систему
RAGSystemFactory.register('universal', UniversalRAGSystem)
