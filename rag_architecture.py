#!/usr/bin/env python3
"""
RAG система для анализа архитектуры проекта.
Анализирует структуру файлов, директорий, конфигурации и архитектурные паттерны.
"""

from __future__ import annotations

import os
import json
import yaml
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
    ProjectIndex, RAGSystemFactory, _relpath, _read_text, _safe_get_lines
)


# -----------------------------------------------------------------------------
# Архитектурные структуры данных
# -----------------------------------------------------------------------------

@dataclass
class ArchitectureElement(BaseCodeElement):
    """Базовый элемент архитектуры"""
    element_type: str  # 'file', 'directory', 'config', 'service', 'module'
    description: str
    dependencies: List[str] = field(default_factory=list)
    used_by: List[str] = field(default_factory=list)
    
    def to_line(self) -> str:
        deps = f" deps: {', '.join(self.dependencies[:3])}" if self.dependencies else ""
        if len(self.dependencies) > 3:
            deps += f" (+{len(self.dependencies)-3} more)"
        used = f" used_by: {', '.join(self.used_by[:3])}" if self.used_by else ""
        if len(self.used_by) > 3:
            used += f" (+{len(self.used_by)-3} more)"
        return f"{self.element_type}: {self.name} - {self.description}{deps}{used}  # L{self.lineno}-{self.end_lineno}"


@dataclass
class ArchitectureFileMap(BaseFileMap):
    """Карта архитектурного файла/директории"""
    elements: List[ArchitectureElement]
    config_data: Dict[str, Any] = field(default_factory=dict)
    file_size: int = 0
    is_config: bool = False
    is_docker: bool = False
    is_documentation: bool = False
    
    def to_text(self) -> str:
        parts = [f"ARCHITECTURE: {self.path}"]
        parts.append(f"TYPE: {self.file_type}")
        parts.append(f"SIZE: {self.file_size} bytes")
        
        if self.is_config:
            parts.append("CONFIG:")
            if isinstance(self.config_data, dict):
                # Ограничиваем количество ключей для больших конфигов
                items_shown = 0
                max_config_items = 8
                for key, value in self.config_data.items():
                    if items_shown >= max_config_items:
                        remaining = len(self.config_data) - max_config_items
                        parts.append(f"  ... and {remaining} more config items")
                        break
                    
                    if isinstance(value, (dict, list)):
                        # Ограничиваем размер сложных структур
                        value_str = json.dumps(value, indent=2)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        parts.append(f"  {key}: {value_str}")
                    else:
                        parts.append(f"  {key}: {str(value)[:80]}")
                    items_shown += 1
            elif isinstance(self.config_data, list):
                parts.append(f"  List with {len(self.config_data)} items")
                # Показываем только первые 3 элемента для экономии места
                for i, item in enumerate(self.config_data[:3]):
                    if isinstance(item, (dict, list)):
                        item_str = json.dumps(item, indent=2)
                        if len(item_str) > 150:
                            item_str = item_str[:150] + "..."
                        parts.append(f"  [{i}]: {item_str}")
                    else:
                        parts.append(f"  [{i}]: {str(item)[:80]}")
                if len(self.config_data) > 3:
                    parts.append(f"  ... and {len(self.config_data) - 3} more items")
            else:
                parts.append(f"  {str(self.config_data)[:150]}")
        
        if self.elements:
            parts.append("ELEMENTS:")
            # Ограничиваем количество элементов
            max_elements = 10
            for i, elem in enumerate(self.elements):
                if i >= max_elements:
                    remaining = len(self.elements) - max_elements
                    parts.append(f"  ... and {remaining} more elements")
                    break
                parts.append(f"  - {elem.to_line()}")
        
        parts.append(f"LOC: {self.loc}")
        return "\n".join(parts)
    
    def get_searchable_content(self) -> List[str]:
        content = []
        for elem in self.elements:
            content.append(f"{elem.element_type}:{elem.name}")
            content.append(f"desc:{elem.description}")
        return content


# -----------------------------------------------------------------------------
# Архитектурный парсер
# -----------------------------------------------------------------------------

class ArchitectureFileParser(FileParser):
    """Парсер для анализа архитектуры проекта"""
    
    def can_parse(self, file_path: str) -> bool:
        # Анализируем все файлы для архитектурного понимания
        return True
    
    def parse(self, file_path: str, root: str) -> Optional[ArchitectureFileMap]:
        try:
            rel_path = _relpath(file_path, root)
            file_size = os.path.getsize(file_path)
            
            # Определяем тип файла
            file_type = self._get_file_type(file_path)
            is_config = self._is_config_file(file_path)
            is_docker = self._is_docker_file(file_path)
            is_documentation = self._is_documentation_file(file_path)
            
            elements = []
            config_data = {}
            
            # Анализируем содержимое файла
            if is_config:
                config_data = self._parse_config_file(file_path)
                elements.extend(self._extract_config_elements(rel_path, config_data))
            elif is_docker:
                elements.extend(self._extract_docker_elements(rel_path, file_path))
            elif is_documentation:
                elements.extend(self._extract_documentation_elements(rel_path, file_path))
            else:
                elements.extend(self._extract_code_elements(rel_path, file_path))
            
            # Анализируем структуру директории
            if os.path.isdir(file_path):
                elements.extend(self._extract_directory_elements(rel_path, file_path))
            
            return ArchitectureFileMap(
                path=rel_path,
                file_type=file_type,
                elements=elements,
                config_data=config_data,
                file_size=file_size,
                is_config=is_config,
                is_docker=is_docker,
                is_documentation=is_documentation,
                loc=self._count_lines(file_path)
            )
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def _get_file_type(self, file_path: str) -> str:
        """Определяет тип файла"""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        if name.startswith('dockerfile'):
            return 'dockerfile'
        elif name in ['package.json', 'requirements.txt', 'pyproject.toml', 'setup.py']:
            return 'dependencies'
        elif name in ['docker-compose.yml', 'docker-compose.yaml']:
            return 'docker-compose'
        elif name in ['makefile', 'makefile.am']:
            return 'makefile'
        elif name in ['readme.md', 'readme.rst', 'readme.txt']:
            return 'readme'
        elif ext in ['.md', '.rst', '.txt']:
            return 'documentation'
        elif ext in ['.json', '.yaml', '.yml', '.toml', '.ini', '.cfg']:
            return 'config'
        elif ext in ['.py']:
            return 'python'
        elif ext in ['.js', '.ts', '.jsx', '.tsx']:
            return 'javascript'
        elif ext in ['.go']:
            return 'go'
        elif ext in ['.rs']:
            return 'rust'
        elif ext in ['.java']:
            return 'java'
        elif ext in ['.cpp', '.c', '.h', '.hpp']:
            return 'cpp'
        else:
            return 'other'
    
    def _is_config_file(self, file_path: str) -> bool:
        """Проверяет, является ли файл конфигурационным"""
        name = Path(file_path).name.lower()
        ext = Path(file_path).suffix.lower()
        
        config_files = {
            'package.json', 'requirements.txt', 'pyproject.toml', 'setup.py',
            'docker-compose.yml', 'docker-compose.yaml', 'makefile',
            'pom.xml', 'build.gradle', 'cargo.toml', 'go.mod',
            'composer.json', 'gemfile', 'rakefile'
        }
        
        config_exts = {'.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.xml'}
        
        return name in config_files or ext in config_exts
    
    def _is_docker_file(self, file_path: str) -> bool:
        """Проверяет, является ли файл Docker-файлом"""
        name = Path(file_path).name.lower()
        return name.startswith('dockerfile') or name in ['docker-compose.yml', 'docker-compose.yaml']
    
    def _is_documentation_file(self, file_path: str) -> bool:
        """Проверяет, является ли файл документацией"""
        name = Path(file_path).name.lower()
        ext = Path(file_path).suffix.lower()
        
        doc_files = {'readme.md', 'readme.rst', 'readme.txt', 'changelog.md', 'license'}
        doc_exts = {'.md', '.rst', '.txt'}
        
        return name in doc_files or ext in doc_exts
    
    def _parse_config_file(self, file_path: str) -> Dict[str, Any]:
        """Парсит конфигурационный файл"""
        try:
            ext = Path(file_path).suffix.lower()
            content = _read_text(file_path)
            
            if ext == '.json':
                return json.loads(content)
            elif ext in ['.yaml', '.yml']:
                return yaml.safe_load(content)
            elif ext == '.toml':
                try:
                    import toml
                    return toml.loads(content)
                except ImportError:
                    return {'content': content[:500], 'format': 'toml'}
            else:
                # Для других форматов возвращаем базовую информацию
                return {'content': content[:500]}
        except Exception:
            return {}
    
    def _extract_config_elements(self, rel_path: str, config_data: Dict[str, Any]) -> List[ArchitectureElement]:
        """Извлекает элементы из конфигурационного файла"""
        elements = []
        
        # Анализируем зависимости
        if 'dependencies' in config_data:
            deps = config_data['dependencies']
            if isinstance(deps, dict):
                elements.append(ArchitectureElement(
                    name="dependencies",
                    element_type="dependencies",
                    description=f"External dependencies: {len(deps)} packages",
                    lineno=1,
                    end_lineno=1
                ))
            elif isinstance(deps, list):
                elements.append(ArchitectureElement(
                    name="dependencies",
                    element_type="dependencies",
                    description=f"External dependencies: {len(deps)} packages (list format)",
                    lineno=1,
                    end_lineno=1
                ))
        
        # Анализируем скрипты
        if 'scripts' in config_data:
            scripts = config_data['scripts']
            if isinstance(scripts, dict):
                elements.append(ArchitectureElement(
                    name="scripts",
                    element_type="scripts",
                    description=f"Build/run scripts: {', '.join(scripts.keys())}",
                    lineno=1,
                    end_lineno=1
                ))
            elif isinstance(scripts, list):
                elements.append(ArchitectureElement(
                    name="scripts",
                    element_type="scripts",
                    description=f"Build/run scripts: {len(scripts)} scripts (list format)",
                    lineno=1,
                    end_lineno=1
                ))
        
        # Анализируем сервисы (для docker-compose)
        if 'services' in config_data:
            services = config_data['services']
            if isinstance(services, dict):
                for service_name, service_config in services.items():
                    elements.append(ArchitectureElement(
                        name=service_name,
                        element_type="service",
                        description=f"Docker service: {service_config.get('image', 'custom') if isinstance(service_config, dict) else 'custom'}",
                        lineno=1,
                        end_lineno=1
                    ))
            elif isinstance(services, list):
                # Если services - это список
                for i, service in enumerate(services):
                    if isinstance(service, dict):
                        service_name = service.get('name', f'service_{i}')
                        elements.append(ArchitectureElement(
                            name=service_name,
                            element_type="service",
                            description=f"Docker service: {service.get('image', 'custom')}",
                            lineno=1,
                            end_lineno=1
                        ))
        
        return elements
    
    def _extract_docker_elements(self, rel_path: str, file_path: str) -> List[ArchitectureElement]:
        """Извлекает элементы из Docker файлов"""
        elements = []
        
        try:
            content = _read_text(file_path)
            lines = content.splitlines()
            
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('FROM '):
                    base_image = line[5:].split()[0]
                    elements.append(ArchitectureElement(
                        name="base_image",
                        element_type="docker",
                        description=f"Base image: {base_image}",
                        lineno=i,
                        end_lineno=i
                    ))
                elif line.startswith('RUN '):
                    elements.append(ArchitectureElement(
                        name=f"run_command_{i}",
                        element_type="docker",
                        description=f"Run command: {line[4:][:50]}...",
                        lineno=i,
                        end_lineno=i
                    ))
                elif line.startswith('COPY ') or line.startswith('ADD '):
                    elements.append(ArchitectureElement(
                        name=f"copy_command_{i}",
                        element_type="docker",
                        description=f"Copy/Add: {line.split()[1:3]}",
                        lineno=i,
                        end_lineno=i
                    ))
        except Exception:
            pass
        
        return elements
    
    def _extract_documentation_elements(self, rel_path: str, file_path: str) -> List[ArchitectureElement]:
        """Извлекает элементы из документации"""
        elements = []
        
        try:
            content = _read_text(file_path)
            lines = content.splitlines()
            
            # Ищем заголовки
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('# '):
                    title = line[2:].strip()
                    elements.append(ArchitectureElement(
                        name=f"section_{i}",
                        element_type="documentation",
                        description=f"Section: {title}",
                        lineno=i,
                        end_lineno=i
                    ))
        except Exception:
            pass
        
        return elements
    
    def _extract_code_elements(self, rel_path: str, file_path: str) -> List[ArchitectureElement]:
        """Извлекает базовые элементы из кода"""
        elements = []
        
        try:
            content = _read_text(file_path)
            lines = content.splitlines()
            
            # Простой анализ для определения основных элементов
            for i, line in enumerate(lines, 1):
                line = line.strip()
                
                # Ищем импорты
                if line.startswith(('import ', 'from ', 'require(', 'include ')):
                    elements.append(ArchitectureElement(
                        name=f"import_{i}",
                        element_type="import",
                        description=f"Import: {line[:50]}...",
                        lineno=i,
                        end_lineno=i
                    ))
                
                # Ищем определения функций/классов
                elif line.startswith(('def ', 'class ', 'function ', 'export ')):
                    elements.append(ArchitectureElement(
                        name=f"definition_{i}",
                        element_type="definition",
                        description=f"Definition: {line[:50]}...",
                        lineno=i,
                        end_lineno=i
                    ))
        except Exception:
            pass
        
        return elements
    
    def _extract_directory_elements(self, rel_path: str, dir_path: str) -> List[ArchitectureElement]:
        """Извлекает элементы из директории"""
        elements = []
        
        try:
            # Анализируем содержимое директории
            items = os.listdir(dir_path)
            
            # Подсчитываем типы файлов
            file_types = {}
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isfile(item_path):
                    ext = Path(item).suffix.lower()
                    file_types[ext] = file_types.get(ext, 0) + 1
            
            if file_types:
                type_desc = ", ".join([f"{ext or 'no-ext'}: {count}" for ext, count in file_types.items()])
                elements.append(ArchitectureElement(
                    name="file_types",
                    element_type="directory",
                    description=f"File types: {type_desc}",
                    lineno=1,
                    end_lineno=1
                ))
            
            # Ищем специальные директории
            special_dirs = {'src', 'lib', 'app', 'api', 'models', 'views', 'controllers', 'tests', 'docs'}
            for item in items:
                if item.lower() in special_dirs and os.path.isdir(os.path.join(dir_path, item)):
                    elements.append(ArchitectureElement(
                        name=item,
                        element_type="directory",
                        description=f"Special directory: {item}",
                        lineno=1,
                        end_lineno=1
                    ))
                    
        except Exception:
            pass
        
        return elements
    
    def _count_lines(self, file_path: str) -> int:
        """Подсчитывает количество строк в файле"""
        try:
            if os.path.isfile(file_path):
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return len(f.readlines())
            elif os.path.isdir(file_path):
                # Для директорий считаем общее количество файлов
                count = 0
                for root, dirs, files in os.walk(file_path):
                    count += len(files)
                return count
        except Exception:
            pass
        return 0


# -----------------------------------------------------------------------------
# Анализаторы архитектуры
# -----------------------------------------------------------------------------

class ArchitectureDependencyAnalyzer(DependencyAnalyzer):
    """Анализатор архитектурных зависимостей"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        arch_maps = [fm for fm in file_maps if isinstance(fm, ArchitectureFileMap)]
        
        # Строим граф зависимостей между файлами
        for file_map in arch_maps:
            for element in file_map.elements:
                if element.element_type == "import":
                    # Анализируем импорты для построения зависимостей
                    self._analyze_import_dependencies(element, arch_maps)
                elif element.element_type == "service":
                    # Анализируем зависимости сервисов
                    self._analyze_service_dependencies(element, arch_maps)
    
    def _analyze_import_dependencies(self, element: ArchitectureElement, file_maps: List[ArchitectureFileMap]) -> None:
        """Анализирует зависимости импортов"""
        # Простая эвристика для определения зависимостей
        desc = element.description.lower()
        
        for file_map in file_maps:
            if file_map.path != element.name:  # Не сам файл
                # Проверяем, может ли этот файл быть целью импорта
                if any(keyword in desc for keyword in ['import', 'from', 'require']):
                    # Добавляем потенциальную зависимость
                    if file_map.path not in element.dependencies:
                        element.dependencies.append(file_map.path)
    
    def _analyze_service_dependencies(self, element: ArchitectureElement, file_maps: List[ArchitectureFileMap]) -> None:
        """Анализирует зависимости сервисов"""
        # Для Docker сервисов ищем связанные файлы
        for file_map in file_maps:
            if file_map.is_docker or file_map.is_config:
                if file_map.path not in element.dependencies:
                    element.dependencies.append(file_map.path)


class ArchitecturePatternAnalyzer(DependencyAnalyzer):
    """Анализатор архитектурных паттернов"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        arch_maps = [fm for fm in file_maps if isinstance(fm, ArchitectureFileMap)]
        
        # Выявляем архитектурные паттерны
        self._detect_mvc_pattern(arch_maps)
        self._detect_microservices_pattern(arch_maps)
        self._detect_layered_architecture(arch_maps)
        self._detect_plugin_architecture(arch_maps)
    
    def _detect_mvc_pattern(self, file_maps: List[ArchitectureFileMap]) -> None:
        """Выявляет MVC паттерн"""
        mvc_dirs = {'models', 'views', 'controllers', 'templates'}
        
        for file_map in file_maps:
            path_parts = file_map.path.lower().split(os.sep)
            if any(part in mvc_dirs for part in path_parts):
                # Добавляем элемент, указывающий на MVC паттерн
                mvc_element = ArchitectureElement(
                    name="mvc_pattern",
                    element_type="pattern",
                    description="MVC architectural pattern detected",
                    lineno=1,
                    end_lineno=1
                )
                file_map.elements.append(mvc_element)
    
    def _detect_microservices_pattern(self, file_maps: List[ArchitectureFileMap]) -> None:
        """Выявляет микросервисную архитектуру"""
        service_indicators = ['service', 'api', 'gateway', 'auth']
        
        for file_map in file_maps:
            if file_map.is_docker or 'docker-compose' in file_map.path:
                # Ищем множественные сервисы
                service_count = len([e for e in file_map.elements if e.element_type == "service"])
                if service_count > 1:
                    micro_element = ArchitectureElement(
                        name="microservices_pattern",
                        element_type="pattern",
                        description=f"Microservices pattern detected: {service_count} services",
                        lineno=1,
                        end_lineno=1
                    )
                    file_map.elements.append(micro_element)
    
    def _detect_layered_architecture(self, file_maps: List[ArchitectureFileMap]) -> None:
        """Выявляет слоистую архитектуру"""
        layer_indicators = {
            'presentation': ['ui', 'frontend', 'web', 'api'],
            'business': ['service', 'business', 'logic', 'core'],
            'data': ['data', 'model', 'entity', 'repository', 'dao']
        }
        
        detected_layers = set()
        for file_map in file_maps:
            path_lower = file_map.path.lower()
            for layer, indicators in layer_indicators.items():
                if any(indicator in path_lower for indicator in indicators):
                    detected_layers.add(layer)
        
        if len(detected_layers) >= 2:
            for file_map in file_maps:
                layer_element = ArchitectureElement(
                    name="layered_architecture",
                    element_type="pattern",
                    description=f"Layered architecture detected: {', '.join(detected_layers)}",
                    lineno=1,
                    end_lineno=1
                )
                file_map.elements.append(layer_element)
    
    def _detect_plugin_architecture(self, file_maps: List[ArchitectureFileMap]) -> None:
        """Выявляет плагинную архитектуру"""
        plugin_indicators = ['plugin', 'extension', 'module', 'addon']
        
        for file_map in file_maps:
            path_lower = file_map.path.lower()
            if any(indicator in path_lower for indicator in plugin_indicators):
                plugin_element = ArchitectureElement(
                    name="plugin_architecture",
                    element_type="pattern",
                    description="Plugin architecture pattern detected",
                    lineno=1,
                    end_lineno=1
                )
                file_map.elements.append(plugin_element)


# -----------------------------------------------------------------------------
# Архитектурная RAG система
# -----------------------------------------------------------------------------

class ArchitectureRAGSystem(BaseRAGSystem):
    """RAG система для анализа архитектуры проекта"""
    
    def __init__(self, llm, embeddings, **kwargs):
        parsers = [ArchitectureFileParser()]
        analyzers = [ArchitectureDependencyAnalyzer(), ArchitecturePatternAnalyzer()]
        super().__init__(llm, embeddings, parsers, analyzers, **kwargs)
    
    def get_file_patterns(self) -> Dict[str, str]:
        return {
            "all_files": "**/*",
            "config_files": "**/*.{json,yaml,yml,toml,ini,cfg,xml}",
            "docker_files": "**/Dockerfile*",
            "docker_compose": "**/docker-compose.{yml,yaml}",
            "documentation": "**/*.{md,rst,txt}",
            "code_files": "**/*.{py,js,ts,go,rs,java,cpp,c,h}"
        }
    
    def get_evidence_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
You are an architectural analyst examining a project structure. You will see a comprehensive map of the repository including:

- File and directory structure with types and sizes
- Configuration files and their contents (dependencies, scripts, services)
- Docker configurations and services
- Documentation structure
- Architectural patterns detected (MVC, microservices, layered, plugin)
- Dependencies and relationships between components

Given the user question about project architecture, propose up to {max_items} precise evidence items to examine in JSON format.
Each item must be an object with keys: file (relative path), symbol (specific element or "*" for entire file), and reason.

IMPORTANT:
- Paths must be repository-relative (e.g., `package.json`, `src/api/`)
- For architectural questions, focus on config files, Docker files, and directory structure
- For dependency questions, examine package.json, requirements.txt, docker-compose files
- For pattern questions, look at directory structure and file organization
- Use "*" as symbol to examine entire files when needed

Respond ONLY with JSON array, no prose.
"""),
            ("human", "Question:\n{question}\n\nArchitecture map:\n{context}\n")
        ])
    
    def get_answer_prompt_template(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
You are a senior software architect analyzing project structure and architecture. Using the provided architecture map and evidence, answer the question comprehensively.

The architecture map includes:
- File and directory structure with types and sizes
- Configuration files (package.json, requirements.txt, docker-compose, etc.)
- Docker configurations and service definitions
- Documentation structure
- Detected architectural patterns (MVC, microservices, layered, plugin)
- Dependencies and relationships between components

For architectural questions:
- Analyze directory structure and file organization
- Examine configuration files for dependencies and services
- Look for architectural patterns and design decisions
- Consider scalability, maintainability, and separation of concerns

For dependency questions:
- Check package.json, requirements.txt, go.mod, etc.
- Examine docker-compose for service dependencies
- Look at import statements and module relationships

Provide specific file references and explain architectural decisions clearly.
Be precise, cite concrete references as `file:line`, and provide actionable insights.
Reply in {answer_language}.
"""),
            ("human", "Question:\n{question}\n\nArchitecture map:\n{map_text}\n\nEvidence:\n{evidence_text}\n")
        ])
    
    def _create_documents(self, file_maps: List[ArchitectureFileMap]) -> List[Document]:
        docs = []
        arch_maps = [fm for fm in file_maps if isinstance(fm, ArchitectureFileMap)]
        
        # Создаем документы из карт файлов с чанкингом
        for fm in arch_maps:
            content = fm.to_text()
            
            # Если документ слишком большой, разбиваем на чанки
            max_chunk_size = self.config.get('max_chunk_size', 4000)
            if len(content) > max_chunk_size:
                chunks = self._split_into_chunks(content, max_chunk_size)
                for i, chunk in enumerate(chunks):
                    meta = {
                        "source": fm.path, 
                        "type": "arch-map", 
                        "file_type": fm.file_type,
                        "size": fm.file_size,
                        "is_config": fm.is_config,
                        "is_docker": fm.is_docker,
                        "is_documentation": fm.is_documentation,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                    docs.append(Document(page_content=chunk, metadata=meta))
            else:
                meta = {
                    "source": fm.path, 
                    "type": "arch-map", 
                    "file_type": fm.file_type,
                    "size": fm.file_size,
                    "is_config": fm.is_config,
                    "is_docker": fm.is_docker,
                    "is_documentation": fm.is_documentation
                }
                docs.append(Document(page_content=content, metadata=meta))
        
        # Добавляем сводку по архитектурным паттернам
        patterns = self._extract_architectural_patterns(arch_maps)
        if patterns:
            docs.append(Document(
                page_content=f"ARCHITECTURAL PATTERNS DETECTED:\n{patterns}",
                metadata={"source": "__patterns__", "type": "pattern-summary"}
            ))
        
        # Добавляем сводку по зависимостям
        dependencies = self._extract_dependency_summary(arch_maps)
        if dependencies:
            docs.append(Document(
                page_content=f"DEPENDENCY SUMMARY:\n{dependencies}",
                metadata={"source": "__dependencies__", "type": "dependency-summary"}
            ))
        
        # Добавляем сводку по структуре проекта
        structure = self._extract_project_structure(arch_maps)
        if structure:
            docs.append(Document(
                page_content=f"PROJECT STRUCTURE:\n{structure}",
                metadata={"source": "__structure__", "type": "structure-summary"}
            ))
        
        return docs
    
    def _extract_architectural_patterns(self, file_maps: List[ArchitectureFileMap]) -> str:
        """Извлекает обнаруженные архитектурные паттерны"""
        patterns = []
        
        for file_map in file_maps:
            for element in file_map.elements:
                if element.element_type == "pattern":
                    patterns.append(f"  {element.name}: {element.description} (in {file_map.path})")
        
        return "\n".join(patterns) if patterns else ""
    
    def _extract_dependency_summary(self, file_maps: List[ArchitectureFileMap]) -> str:
        """Извлекает сводку по зависимостям"""
        deps_info = []
        
        for file_map in file_maps:
            if file_map.is_config and file_map.config_data:
                if 'dependencies' in file_map.config_data:
                    deps = file_map.config_data['dependencies']
                    if isinstance(deps, dict):
                        deps_info.append(f"  {file_map.path}: {len(deps)} dependencies")
                    elif isinstance(deps, list):
                        deps_info.append(f"  {file_map.path}: {len(deps)} dependencies (list)")
                elif 'services' in file_map.config_data:
                    services = file_map.config_data['services']
                    if isinstance(services, dict):
                        deps_info.append(f"  {file_map.path}: {len(services)} services")
                    elif isinstance(services, list):
                        deps_info.append(f"  {file_map.path}: {len(services)} services (list)")
        
        return "\n".join(deps_info) if deps_info else ""
    
    def _extract_project_structure(self, file_maps: List[ArchitectureFileMap]) -> str:
        """Извлекает структуру проекта"""
        structure_info = []
        
        # Группируем файлы по типам
        file_types = {}
        total_size = 0
        
        for file_map in file_maps:
            file_type = file_map.file_type
            if file_type not in file_types:
                file_types[file_type] = {'count': 0, 'size': 0}
            file_types[file_type]['count'] += 1
            file_types[file_type]['size'] += file_map.file_size
            total_size += file_map.file_size
        
        structure_info.append(f"Total files: {len(file_maps)}")
        structure_info.append(f"Total size: {total_size:,} bytes")
        structure_info.append("File types:")
        
        for file_type, stats in sorted(file_types.items()):
            structure_info.append(f"  {file_type}: {stats['count']} files, {stats['size']:,} bytes")
        
        return "\n".join(structure_info)
    
    def _split_into_chunks(self, text: str, max_size: int) -> List[str]:
        """Разбивает текст на чанки по строкам, сохраняя контекст"""
        lines = text.split('\n')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for line in lines:
            line_size = len(line) + 1  # +1 для символа новой строки
            
            # Если добавление этой строки превысит лимит, сохраняем текущий чанк
            if current_size + line_size > max_size and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_size = line_size
            else:
                current_chunk.append(line)
                current_size += line_size
        
        # Добавляем последний чанк
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _build_search_index(
        self, 
        docs: List[Document], 
        project_path: str, 
        file_maps: List[BaseFileMap]
    ) -> ProjectIndex:
        # BM25 retriever
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.get('bm25_k', 15)
        
        # Для больших проектов используем только BM25, чтобы избежать проблем с памятью
        max_docs_for_embeddings = self.config.get('max_docs_for_embeddings', 1000)
        if len(docs) > max_docs_for_embeddings:
            print(f"⚠️  Большой проект ({len(docs)} документов), используем только BM25 для экономии памяти")
            retriever = bm25
        else:
            try:
                # Ограничиваем размер документов для эмбеддингов
                max_chars_per_doc = self.config.get('max_chars_per_doc_for_embeddings', 6000)
                processed_docs = []
                
                for doc in docs:
                    if len(doc.page_content) > max_chars_per_doc:
                        # Обрезаем длинные документы
                        truncated_content = doc.page_content[:max_chars_per_doc] + "\n... [truncated]"
                        processed_doc = Document(
                            page_content=truncated_content,
                            metadata=doc.metadata
                        )
                        processed_docs.append(processed_doc)
                    else:
                        processed_docs.append(doc)
                
                # FAISS retriever с обработкой ошибок памяти
                faiss = FAISS.from_documents(processed_docs, self.embeddings, distance_strategy=DistanceStrategy.COSINE)
                dense_retriever = faiss.as_retriever(search_kwargs={"k": self.config.get('dense_k', 15)})
                
                # Ensemble retriever
                ensemble = EnsembleRetriever(
                    retrievers=[bm25, dense_retriever], 
                    weights=list(self.config.get('ensemble_weights', (0.4, 0.6)))
                )
                retriever = ensemble
                
            except Exception as e:
                print(f"⚠️  Ошибка создания эмбеддингов: {e}")
                print("🔄 Используем только BM25 retriever")
                retriever = bm25
        
        # Optional compression (только если не было ошибок с памятью)
        if self.config.get('use_compression', True) and retriever != bm25:
            try:
                compressor = LLMChainExtractor.from_llm(self.llm)
                retriever = ContextualCompressionRetriever(
                    base_compressor=compressor, 
                    base_retriever=retriever
                )
            except Exception as e:
                print(f"⚠️  Ошибка создания компрессора: {e}")
                print("🔄 Используем retriever без компрессии")
        
        return ProjectIndex(
            root=project_path,
            docs=docs,
            retriever=retriever,
            file_maps=file_maps
        )
    
    def _answer_question(self, question: str, index: ProjectIndex) -> str:
        # Retrieve relevant documents
        evidence_char_budget = self.config.get('evidence_char_budget', 25000)
        max_evidence_items = self.config.get('max_evidence_items', 10)
        map_char_budget = self.config.get('map_char_budget', 30000)

        retrieved = index.retriever.invoke(question)
        
        # Gather context
        map_text = self._gather_architecture_snippets(retrieved, map_char_budget)
        
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
        evidence_pairs = self._extract_architecture_bodies(index.root, plan[:max_evidence_items]) if plan else []
        evidence_text = "\n\n".join([f"### {lbl}\n" + code for (lbl, code) in evidence_pairs])
        evidence_text = self._trim_to_chars(evidence_text, evidence_char_budget)
        
        # Generate final answer
        answer_prompt = self.get_answer_prompt_template()
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": self.config.get('answer_language', 'ru'),
            "evidence_text": evidence_text if evidence_text else "(no additional files requested)",
        })
        
        return final
    
    def _extract_architecture_bodies(self, root: str, requests: List[Dict[str, str]]) -> List[Tuple[str, str]]:
        """Извлекает содержимое файлов для архитектурного анализа"""
        out: List[Tuple[str, str]] = []
        
        for req in requests:
            rel = req.get("file", "")
            sym = req.get("symbol", "")
            target = os.path.join(root, rel)
            
            if not os.path.exists(target):
                continue
            
            try:
                if sym == "*" or not sym:
                    # Читаем весь файл
                    if os.path.isfile(target):
                        content = _read_text(target)
                        out.append((f"{rel}:*", content))
                    elif os.path.isdir(target):
                        # Для директорий показываем структуру
                        structure = self._get_directory_structure(target, root)
                        out.append((f"{rel}:*", structure))
                else:
                    # Ищем конкретный элемент
                    content = _read_text(target) if os.path.isfile(target) else ""
                    if content:
                        # Простой поиск по содержимому
                        lines = content.splitlines()
                        for i, line in enumerate(lines, 1):
                            if sym.lower() in line.lower():
                                start = max(1, i - 2)
                                end = min(len(lines), i + 3)
                                snippet = "\n".join(lines[start-1:end])
                                out.append((f"{rel}:{i}", snippet))
                                break
            except Exception as e:
                print(f"Error extracting {target}: {e}")
                continue
        
        return out
    
    def _get_directory_structure(self, dir_path: str, root: str) -> str:
        """Получает структуру директории"""
        try:
            structure = []
            rel_path = _relpath(dir_path, root)
            structure.append(f"Directory: {rel_path}")
            
            items = sorted(os.listdir(dir_path))
            for item in items:
                item_path = os.path.join(dir_path, item)
                if os.path.isdir(item_path):
                    structure.append(f"  📁 {item}/")
                else:
                    size = os.path.getsize(item_path)
                    structure.append(f"  📄 {item} ({size:,} bytes)")
            
            return "\n".join(structure)
        except Exception:
            return f"Directory: {rel_path} (access error)"
    
    def _gather_architecture_snippets(self, docs: List[Document], max_chars: int = 30000) -> str:
        """Собирает архитектурные фрагменты"""
        pieces = []
        for d in docs:
            if d.metadata.get("type") in ("arch-map", "pattern-summary", "dependency-summary", "structure-summary"):
                pieces.append(f"---\n{d.page_content}\n")
            if sum(len(p) for p in pieces) > max_chars:
                break
        
        return self._trim_to_chars("\n".join(pieces), max_chars)
    
    def _trim_to_chars(self, text: str, limit: int) -> str:
        """Обрезает текст до указанного количества символов"""
        if len(text) <= limit:
            return text
        head = limit // 2
        tail = limit - head
        return text[:head] + "\n...\n" + text[-tail:]
    
    def _get_tool_description(self) -> str:
        return (
            "Analyze project architecture and structure. "
            "Examines file organization, configuration files, Docker setup, "
            "dependencies, and architectural patterns. "
            "Input: a natural-language question about project architecture."
        )


RAGSystemFactory.register('architecture', ArchitectureRAGSystem)
