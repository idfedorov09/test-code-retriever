#!/usr/bin/env python3
"""
Абстрактная базовая архитектура для RAG системы анализа кода.
Поддерживает разные языки программирования и типы файлов.
"""

from __future__ import annotations

import os
import glob
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Protocol
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import StructuredTool
from langchain_core.language_models import BaseChatModel
from langchain_core.embeddings import Embeddings


# -----------------------------------------------------------------------------
# Абстрактные базовые классы для представления кода
# -----------------------------------------------------------------------------

@dataclass
class BaseFileMap(ABC):
    """Базовый класс для представления файла любого типа"""
    path: str
    file_type: str
    loc: int
    
    @abstractmethod
    def to_text(self) -> str:
        """Преобразует файл в текстовое представление для индексирования"""
        pass
    
    @abstractmethod
    def get_searchable_content(self) -> List[str]:
        """Возвращает список поисковых элементов (функции, классы, etc.)"""
        pass


@dataclass 
class BaseCodeElement(ABC):
    """Базовый класс для элемента кода (функция, класс, etc.)"""
    name: str
    lineno: int
    end_lineno: int
    
    @abstractmethod
    def to_line(self) -> str:
        """Строковое представление элемента"""
        pass


# -----------------------------------------------------------------------------
# Протоколы для парсеров и анализаторов
# -----------------------------------------------------------------------------

class FileParser(Protocol):
    """Протокол для парсеров файлов"""
    
    def can_parse(self, file_path: str) -> bool:
        """Проверяет, может ли парсер обработать данный файл"""
        ...
    
    def parse(self, file_path: str, root: str) -> Optional[BaseFileMap]:
        """Парсит файл и возвращает его представление"""
        ...


class DependencyAnalyzer(Protocol):
    """Протокол для анализа зависимостей между элементами кода"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        """Строит граф зависимостей между элементами"""
        ...


# -----------------------------------------------------------------------------
# Абстрактная RAG система
# -----------------------------------------------------------------------------

class BaseRAGSystem(ABC):
    """Абстрактная базовая RAG система"""
    
    def __init__(
        self,
        llm: BaseChatModel,
        embeddings: Embeddings,
        parsers: List[FileParser],
        analyzers: List[DependencyAnalyzer],
        **kwargs
    ):
        self.llm = llm
        self.embeddings = embeddings
        self.parsers = parsers
        self.analyzers = analyzers
        self.config = kwargs
    
    @abstractmethod
    def get_file_patterns(self) -> Dict[str, str]:
        """Возвращает паттерны файлов для данной технологии"""
        pass
    
    @abstractmethod
    def get_evidence_prompt_template(self) -> str:
        """Возвращает шаблон промпта для поиска доказательств"""
        pass
    
    @abstractmethod
    def get_answer_prompt_template(self) -> str:
        """Возвращает шаблон промпта для ответа"""
        pass
    
    def build_index(self, project_path: str) -> 'ProjectIndex':
        """Строит индекс проекта"""
        project_path = os.path.abspath(project_path)
        
        # 1. Находим все подходящие файлы
        all_files = self._find_project_files(project_path)
        
        # 2. Парсим файлы
        file_maps = self._parse_files(all_files, project_path)
        
        # 3. Анализируем зависимости
        self._analyze_dependencies(file_maps)
        
        # 4. Создаем документы для индексирования
        docs = self._create_documents(file_maps)
        
        # 5. Строим поисковый индекс
        return self._build_search_index(docs, project_path, file_maps)
    
    def _find_project_files(self, project_path: str) -> List[str]:
        """Находит все файлы проекта по паттернам"""
        all_files = []
        patterns = self.get_file_patterns()
        
        for pattern in patterns.values():
            files = glob.glob(os.path.join(project_path, pattern), recursive=True)
            all_files.extend([f for f in files if os.path.isfile(f)])
        
        return list(set(all_files))  # Убираем дубликаты
    
    def _parse_files(self, file_paths: List[str], root: str) -> List[BaseFileMap]:
        """Парсит файлы с помощью подходящих парсеров"""
        file_maps = []
        
        for file_path in file_paths:
            for parser in self.parsers:
                if parser.can_parse(file_path):
                    file_map = parser.parse(file_path, root)
                    if file_map:
                        file_maps.append(file_map)
                    break
        
        return file_maps
    
    def _analyze_dependencies(self, file_maps: List[BaseFileMap]) -> None:
        """Анализирует зависимости с помощью анализаторов"""
        for analyzer in self.analyzers:
            analyzer.build_dependency_graph(file_maps)
    
    @abstractmethod
    def _create_documents(self, file_maps: List[BaseFileMap]) -> List[Document]:
        """Создает документы для индексирования"""
        pass
    
    @abstractmethod
    def _build_search_index(
        self, 
        docs: List[Document], 
        project_path: str, 
        file_maps: List[BaseFileMap]
    ) -> 'ProjectIndex':
        """Строит поисковый индекс"""
        pass
    
    def create_tool(self, project_path: str, **tool_config) -> StructuredTool:
        """Создает LangChain инструмент для данной RAG системы"""
        index = self.build_index(project_path)
        
        def _run(question: str) -> str:
            return self._answer_question(question, index)
        
        return StructuredTool.from_function(
            func=_run,
            name=f"{self.__class__.__name__.lower()}_tool",
            description=self._get_tool_description(),
        )
    
    @abstractmethod
    def _answer_question(self, question: str, index: 'ProjectIndex') -> str:
        """Отвечает на вопрос используя индекс"""
        pass
    
    @abstractmethod
    def _get_tool_description(self) -> str:
        """Возвращает описание инструмента"""
        pass


# -----------------------------------------------------------------------------
# Вспомогательные классы
# -----------------------------------------------------------------------------

@dataclass
class ProjectIndex:
    """Индекс проекта с поисковыми возможностями"""
    root: str
    docs: List[Document]
    retriever: Any
    file_maps: List[BaseFileMap]
    metadata: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Фабрика RAG систем
# -----------------------------------------------------------------------------

class RAGSystemFactory:
    """Фабрика для создания RAG систем под разные технологии"""
    
    _systems: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str, system_class: type):
        """Регистрирует RAG систему"""
        cls._systems[name] = system_class
    
    @classmethod
    def create(cls, name: str, **kwargs) -> BaseRAGSystem:
        """Создает RAG систему по имени"""
        if name not in cls._systems:
            raise ValueError(f"Unknown RAG system: {name}")
        
        return cls._systems[name](**kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """Возвращает список доступных RAG систем"""
        return list(cls._systems.keys())
    
    @classmethod
    def detect_project_type(cls, project_path: str) -> str:
        """Автоматически определяет основной тип проекта из множества технологий"""
        detected_types = cls.detect_project_types(project_path)
        
        # Приоритет выбора основного типа
        priority_order = ['python', 'javascript', 'react', 'docker', 'universal']
        
        for preferred_type in priority_order:
            if preferred_type in detected_types:
                return preferred_type
        
        return 'universal'
    
    @classmethod
    def detect_project_types(cls, project_path: str) -> List[str]:
        """Определяет все типы технологий в проекте"""
        detected_types = []
        
        try:
            # Рекурсивно проходим по всем файлам проекта
            all_files = []
            for root, dirs, files in os.walk(project_path):
                # Исключаем ненужные директории
                dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', '.venv', 'venv', 'dist', 'build'}]
                all_files.extend(files)
            
            file_extensions = set()
            special_files = set()
            
            for file in all_files:
                # Собираем расширения
                if '.' in file:
                    ext = file.split('.')[-1].lower()
                    file_extensions.add(ext)
                
                # Собираем специальные файлы
                special_files.add(file.lower())
            
            # Python проект
            if 'py' in file_extensions:
                if any(f in special_files for f in ['requirements.txt', 'pyproject.toml', 'setup.py', 'pipfile']):
                    detected_types.append('python')
            
            # JavaScript/Node.js проект
            if any(ext in file_extensions for ext in ['js', 'ts', 'mjs']):
                if 'package.json' in special_files:
                    detected_types.append('javascript')
            
            # React проект
            if any(ext in file_extensions for ext in ['jsx', 'tsx']):
                detected_types.append('react')
            
            # Docker проект
            if any(f.startswith('dockerfile') for f in special_files) or 'dockerfile' in special_files:
                detected_types.append('docker')
            
            # Go проект
            if 'go' in file_extensions and any(f in special_files for f in ['go.mod', 'go.sum']):
                detected_types.append('go')
            
            # Rust проект
            if 'rs' in file_extensions and 'cargo.toml' in special_files:
                detected_types.append('rust')
            
            # Java проект
            if 'java' in file_extensions and any(f in special_files for f in ['pom.xml', 'build.gradle', 'gradle.build']):
                detected_types.append('java')
            
            # C/C++ проект
            if any(ext in file_extensions for ext in ['c', 'cpp', 'cc', 'cxx', 'h', 'hpp']):
                if any(f in special_files for f in ['makefile', 'cmake', 'cmakelists.txt']):
                    detected_types.append('cpp')
            
            # PHP проект
            if 'php' in file_extensions and 'composer.json' in special_files:
                detected_types.append('php')
            
            # Ruby проект
            if 'rb' in file_extensions and any(f in special_files for f in ['gemfile', 'rakefile']):
                detected_types.append('ruby')
            
            # .NET проект
            if any(ext in file_extensions for ext in ['cs', 'vb', 'fs']):
                if any(f.endswith('.csproj') or f.endswith('.sln') for f in special_files):
                    detected_types.append('dotnet')
            
            # Всегда добавляем универсальный тип если есть текстовые файлы
            if file_extensions:
                detected_types.append('universal')
            
        except Exception as e:
            print(f"⚠️  Ошибка при определении типов проекта: {e}")
            detected_types = ['universal']
        
        return detected_types if detected_types else ['universal']
    
    @classmethod
    def get_project_info(cls, project_path: str) -> Dict[str, Any]:
        """Возвращает подробную информацию о проекте"""
        detected_types = cls.detect_project_types(project_path)
        main_type = cls.detect_project_type(project_path)
        
        # Подсчет файлов по типам
        file_stats = {}
        try:
            for root, dirs, files in os.walk(project_path):
                dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', '.venv'}]
                for file in files:
                    if '.' in file:
                        ext = file.split('.')[-1].lower()
                        file_stats[ext] = file_stats.get(ext, 0) + 1
        except Exception:
            pass
        
        return {
            'detected_types': detected_types,
            'main_type': main_type,
            'file_stats': file_stats,
            'is_multi_tech': len(detected_types) > 1
        }


# -----------------------------------------------------------------------------
# Утилиты
# -----------------------------------------------------------------------------

def _relpath(path: str, root: str) -> str:
    """Получает относительный путь"""
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except Exception:
        return path


def _read_text(path: str) -> str:
    """Читает текстовый файл"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _safe_get_lines(text: str, start: int, end: int) -> str:
    """Безопасно извлекает строки из текста"""
    lines = text.splitlines()
    start = max(start - 1, 0)
    end = min(end, len(lines))
    return "\n".join(lines[start:end])
