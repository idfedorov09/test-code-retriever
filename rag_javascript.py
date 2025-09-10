#!/usr/bin/env python3
"""
JavaScript-специфичная реализация RAG системы.
Анализирует JavaScript/TypeScript код, строит графы вызовов и зависимостей.
"""

from __future__ import annotations

import os
import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set, Any

import networkx as nx

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
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
# JavaScript-специфичные структуры данных
# -----------------------------------------------------------------------------

@dataclass
class JSFunctionSig(BaseCodeElement):
    """Сигнатура JavaScript функции"""
    qname: str
    args: str
    returns: Optional[str]
    is_async: bool = False
    is_arrow: bool = False
    is_constructor: bool = False
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # module.exports, export default, etc.

    def to_line(self) -> str:
        ret = f" -> {self.returns}" if self.returns else ""
        async_str = "async " if self.is_async else ""
        arrow_str = " =>" if self.is_arrow else ""
        constructor_str = " [constructor]" if self.is_constructor else ""
        
        calls_info = f" calls: {', '.join(self.calls[:3])}" if self.calls else ""
        if len(self.calls) > 3:
            calls_info += f" (+{len(self.calls)-3} more)"
        
        called_by_info = f" called_by: {', '.join(self.called_by[:3])}" if self.called_by else ""
        if len(self.called_by) > 3:
            called_by_info += f" (+{len(self.called_by)-3} more)"
        
        exports_info = f" exports: {', '.join(self.exports)}" if self.exports else ""
        
        return f"{async_str}{self.qname}({self.args}){arrow_str}{ret}{constructor_str}{calls_info}{called_by_info}{exports_info}  # L{self.lineno}-{self.end_lineno}"


@dataclass
class JSClassSig(BaseCodeElement):
    """Сигнатура JavaScript класса"""
    extends: Optional[str]
    methods: List[JSFunctionSig] = field(default_factory=list)
    properties: List[str] = field(default_factory=list)
    inherited_by: List[str] = field(default_factory=list)

    def to_line(self) -> str:
        extends_str = f" extends {self.extends}" if self.extends else ""
        header = f"class {self.name}{extends_str}  # L{self.lineno}-{self.end_lineno}"
        
        if self.inherited_by:
            inheritance_info = f" inherited_by: {', '.join(self.inherited_by[:3])}"
            if len(self.inherited_by) > 3:
                inheritance_info += f" (+{len(self.inherited_by)-3} more)"
            header += inheritance_info
        
        return header

    def to_block(self) -> str:
        header = self.to_line()
        if self.properties:
            props = "\n".join([f"  - {prop}" for prop in self.properties])
            header += f"\n  Properties:\n{props}"
        if self.methods:
            methods = "\n".join(["  - " + m.to_line() for m in self.methods])
            header += f"\n  Methods:\n{methods}"
        return header


@dataclass
class JSVariableSig(BaseCodeElement):
    """Сигнатура JavaScript переменной"""
    var_type: str  # const, let, var
    value_type: Optional[str]  # inferred type
    is_exported: bool = False
    is_imported: bool = False
    source: Optional[str] = None  # for imports

    def to_line(self) -> str:
        type_info = f": {self.value_type}" if self.value_type else ""
        export_info = " [exported]" if self.is_exported else ""
        import_info = f" [imported from {self.source}]" if self.is_imported and self.source else ""
        return f"{self.var_type} {self.name}{type_info}{export_info}{import_info}  # L{self.lineno}"


@dataclass
class JSFileMap(BaseFileMap):
    """Карта JavaScript файла"""
    imports: List[str]
    exports: List[str]
    functions: List[JSFunctionSig]
    classes: List[JSClassSig]
    variables: List[JSVariableSig]
    dependencies: List[str] = field(default_factory=list)

    def to_text(self) -> str:
        parts = [f"FILE: {self.path}"]
        if self.imports:
            parts.append("IMPORTS:\n" + "\n".join([f"  - {imp}" for imp in sorted(self.imports) if imp]))
        if self.exports:
            parts.append("EXPORTS:\n" + "\n".join([f"  - {exp}" for exp in sorted(self.exports) if exp]))
        if self.classes:
            parts.append("CLASSES:\n" + "\n".join([c.to_block() for c in self.classes]))
        if self.functions:
            parts.append("FUNCTIONS:\n" + "\n".join([f"  - {fn.to_line()}" for fn in self.functions]))
        if self.variables:
            parts.append("VARIABLES:\n" + "\n".join([f"  - {var.to_line()}" for var in self.variables]))
        parts.append(f"LOC: {self.loc}")
        return "\n".join(parts)

    def get_searchable_content(self) -> List[str]:
        content = []
        content.extend(self.imports)
        content.extend(self.exports)
        content.extend([f.name for f in self.functions])
        content.extend([c.name for c in self.classes])
        content.extend([v.name for v in self.variables])
        content.extend(self.dependencies)
        return [item for item in content if item]


# -----------------------------------------------------------------------------
# JavaScript парсер
# -----------------------------------------------------------------------------

class JavaScriptFileParser(FileParser):
    """Парсер JavaScript/TypeScript файлов"""
    
    # Контекстные подсказки для разных типов JS файлов
    CONTEXT_PROMPTS = {
        'javascript': "JavaScript файл с функциями, классами и модулями",
        'typescript': "TypeScript файл с типизированными функциями, классами и интерфейсами",
        'jsx': "React JSX файл с компонентами и хуками",
        'tsx': "React TypeScript файл с типизированными компонентами",
        'vue': "Vue.js файл с компонентами и композицией",
        'svelte': "Svelte файл с компонентами и реактивностью",
        'node': "Node.js файл с серверной логикой и модулями",
        'express': "Express.js файл с маршрутами и middleware",
        'react': "React компонент с хуками и состоянием",
        'next': "Next.js файл с страницами и API routes",
        'nuxt': "Nuxt.js файл с страницами и компонентами",
        'angular': "Angular файл с компонентами и сервисами",
        'config': "Конфигурационный файл (webpack, babel, etc.)",
        'test': "Тестовый файл (Jest, Mocha, etc.)",
        'storybook': "Storybook файл с историями компонентов"
    }

    def can_parse(self, file_path: str) -> bool:
        """Проверяет, может ли парсер обработать данный файл"""
        path_lower = file_path.lower()
        
        # JavaScript/TypeScript файлы
        if path_lower.endswith(('.js', '.ts', '.jsx', '.tsx', '.mjs', '.cjs')):
            return True
        
        # Vue файлы
        if path_lower.endswith('.vue'):
            return True
        
        # Svelte файлы
        if path_lower.endswith('.svelte'):
            return True
        
        # Конфигурационные файлы
        if any(name in path_lower for name in [
            'webpack.config', 'babel.config', 'rollup.config', 
            'vite.config', 'next.config', 'nuxt.config',
            'tailwind.config', 'postcss.config'
        ]):
            return True
        
        # Package.json и другие JSON конфиги
        if path_lower.endswith(('.json', '.jsonc')) and any(name in path_lower for name in [
            'package', 'tsconfig', 'jsconfig', 'babel', 'eslint'
        ]):
            return True
        
        # Тестовые файлы
        if 'test' in path_lower or 'spec' in path_lower:
            return True
        
        # Storybook файлы
        if 'story' in path_lower or 'stories' in path_lower:
            return True
        
        return False

    def detect_file_type(self, file_path: str) -> str:
        """Определяет тип JavaScript файла"""
        path_lower = file_path.lower()
        
        if 'test' in path_lower or 'spec' in path_lower:
            return 'test'
        if 'story' in path_lower or 'stories' in path_lower:
            return 'storybook'
        if 'config' in path_lower or 'webpack' in path_lower or 'babel' in path_lower:
            return 'config'
        if path_lower.endswith('.tsx'):
            return 'tsx'
        if path_lower.endswith('.jsx'):
            return 'jsx'
        if path_lower.endswith('.ts'):
            return 'typescript'
        if path_lower.endswith('.vue'):
            return 'vue'
        if path_lower.endswith('.svelte'):
            return 'svelte'
        if 'next' in path_lower or 'pages' in path_lower or 'app' in path_lower:
            return 'next'
        if 'nuxt' in path_lower:
            return 'nuxt'
        if 'angular' in path_lower:
            return 'angular'
        if 'express' in path_lower or 'server' in path_lower or 'api' in path_lower:
            return 'express'
        if 'node' in path_lower or 'server' in path_lower:
            return 'node'
        if 'react' in path_lower or 'component' in path_lower:
            return 'react'
        if path_lower.endswith('.js'):
            return 'javascript'
        
        return 'javascript'

    def parse(self, file_path: str, root: str) -> Optional[JSFileMap]:
        """Парсит файл и возвращает его представление (протокол FileParser)"""
        return self.parse_file(file_path)

    def parse_file(self, file_path: str) -> JSFileMap:
        """Парсит JavaScript файл и создает карту"""
        try:
            content = _read_text(file_path)
            lines = content.splitlines()
            loc = len(lines)
            
            # Определяем тип файла
            file_type = self.detect_file_type(file_path)
            
            # Парсим содержимое
            imports = self._extract_imports(content)
            exports = self._extract_exports(content)
            functions = self._extract_functions(content, lines)
            classes = self._extract_classes(content, lines)
            variables = self._extract_variables(content, lines)
            dependencies = self._extract_dependencies(content, file_type)
            
            # Анализируем вызовы функций
            self._analyze_function_calls(functions, content)
            
            return JSFileMap(
                path=file_path,
                file_type=file_type,
                loc=loc,
                imports=imports,
                exports=exports,
                functions=functions,
                classes=classes,
                variables=variables,
                dependencies=dependencies
            )
            
        except Exception as e:
            print(f"⚠️  Ошибка парсинга {file_path}: {e}")
            return JSFileMap(
                path=file_path,
                file_type='javascript',
                loc=0,
                imports=[],
                exports=[],
                functions=[],
                classes=[],
                variables=[],
                dependencies=[]
            )

    def _extract_imports(self, content: str) -> List[str]:
        """Извлекает импорты из JavaScript файла"""
        imports = []
        
        # ES6 imports
        import_patterns = [
            r'import\s+(\{[^}]*\})\s+from\s+[\'"]([^\'"]+)[\'"]',  # named imports
            r'import\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # default import
            r'import\s+[\'"]([^\'"]+)[\'"]',  # side-effect import
            r'import\s+\*\s+as\s+(\w+)\s+from\s+[\'"]([^\'"]+)[\'"]',  # namespace import
        ]
        
        for pattern in import_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        imports.append(f"{match[0]} from {match[1]}")
                    else:
                        imports.append(match[0])
                else:
                    imports.append(match)
        
        # CommonJS requires
        require_patterns = [
            r'require\([\'"]([^\'"]+)[\'"]\)',
            r'const\s+(\w+)\s*=\s*require\([\'"]([^\'"]+)[\'"]\)',
        ]
        
        for pattern in require_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if isinstance(match, tuple):
                    imports.append(f"{match[0]} = require({match[1]})")
                else:
                    imports.append(f"require({match})")
        
        return list(set(imports))

    def _extract_exports(self, content: str) -> List[str]:
        """Извлекает экспорты из JavaScript файла"""
        exports = []
        
        # ES6 exports
        export_patterns = [
            r'export\s+default\s+(\w+)',  # default export
            r'export\s+const\s+(\w+)',  # named const export
            r'export\s+function\s+(\w+)',  # named function export
            r'export\s+class\s+(\w+)',  # named class export
            r'export\s+\{([^}]+)\}',  # named exports
        ]
        
        for pattern in export_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            for match in matches:
                if '{' in match:  # named exports
                    # Разбираем { a, b, c }
                    items = [item.strip() for item in match.split(',')]
                    exports.extend(items)
                else:
                    exports.append(match)
        
        # CommonJS exports
        module_exports_patterns = [
            r'module\.exports\s*=\s*(\w+)',
            r'module\.exports\.(\w+)\s*=',
            r'exports\.(\w+)\s*=',
        ]
        
        for pattern in module_exports_patterns:
            matches = re.findall(pattern, content, re.MULTILINE)
            exports.extend(matches)
        
        return list(set(exports))

    def _extract_functions(self, content: str, lines: List[str]) -> List[JSFunctionSig]:
        """Извлекает функции из JavaScript файла"""
        functions = []
        
        # Паттерны для разных типов функций
        function_patterns = [
            # function declarations
            (r'function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?', 'function'),
            # arrow functions
            (r'const\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>', 'arrow'),
            (r'let\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>', 'arrow'),
            (r'var\s+(\w+)\s*=\s*\(([^)]*)\)\s*=>', 'arrow'),
            # async functions
            (r'async\s+function\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?', 'async_function'),
            (r'const\s+(\w+)\s*=\s*async\s*\(([^)]*)\)\s*=>', 'async_arrow'),
            # method definitions
            (r'(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?\s*\{', 'method'),
            (r'async\s+(\w+)\s*\(([^)]*)\)\s*(?::\s*(\w+))?\s*\{', 'async_method'),
        ]
        
        for pattern, func_type in function_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                name = match.group(1)
                args = match.group(2) if len(match.groups()) > 1 else ""
                returns = match.group(3) if len(match.groups()) > 2 else None
                
                # Находим номер строки
                lineno = content[:match.start()].count('\n') + 1
                
                # Определяем тип функции
                is_async = 'async' in func_type
                is_arrow = 'arrow' in func_type
                is_constructor = name[0].isupper() and func_type == 'function'
                
                # Находим конец функции (приблизительно)
                end_lineno = self._find_function_end(content, match.start())
                
                functions.append(JSFunctionSig(
                    name=name,
                    qname=name,
                    lineno=lineno,
                    end_lineno=end_lineno,
                    args=args,
                    returns=returns,
                    is_async=is_async,
                    is_arrow=is_arrow,
                    is_constructor=is_constructor
                ))
        
        return functions

    def _extract_classes(self, content: str, lines: List[str]) -> List[JSClassSig]:
        """Извлекает классы из JavaScript файла"""
        classes = []
        
        # Паттерн для классов
        class_pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        
        matches = re.finditer(class_pattern, content, re.MULTILINE)
        for match in matches:
            name = match.group(1)
            extends = match.group(2) if match.group(2) else None
            
            # Находим номер строки
            lineno = content[:match.start()].count('\n') + 1
            
            # Находим конец класса (приблизительно)
            end_lineno = self._find_class_end(content, match.start())
            
            classes.append(JSClassSig(
                name=name,
                lineno=lineno,
                end_lineno=end_lineno,
                extends=extends
            ))
        
        return classes

    def _extract_variables(self, content: str, lines: List[str]) -> List[JSVariableSig]:
        """Извлекает переменные из JavaScript файла"""
        variables = []
        
        # Паттерны для переменных
        var_patterns = [
            (r'const\s+(\w+)(?::\s*(\w+))?\s*=', 'const'),
            (r'let\s+(\w+)(?::\s*(\w+))?\s*=', 'let'),
            (r'var\s+(\w+)(?::\s*(\w+))?\s*=', 'var'),
        ]
        
        for pattern, var_type in var_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                name = match.group(1)
                value_type = match.group(2) if len(match.groups()) > 1 else None
                
                # Находим номер строки
                lineno = content[:match.start()].count('\n') + 1
                
                # Проверяем, является ли переменная экспортом
                is_exported = 'export' in content[max(0, match.start()-50):match.start()]
                
                variables.append(JSVariableSig(
                    name=name,
                    lineno=lineno,
                    end_lineno=lineno,
                    var_type=var_type,
                    value_type=value_type,
                    is_exported=is_exported
                ))
        
        return variables

    def _extract_dependencies(self, content: str, file_type: str) -> List[str]:
        """Извлекает зависимости из JavaScript файла"""
        dependencies = []
        
        # Извлекаем из импортов
        imports = self._extract_imports(content)
        for imp in imports:
            if 'from' in imp:
                # ES6 import: extract module name
                module = imp.split('from')[-1].strip().strip('\'"')
                dependencies.append(module)
            elif 'require' in imp:
                # CommonJS require: extract module name
                module = imp.split('require(')[-1].split(')')[0].strip().strip('\'"')
                dependencies.append(module)
        
        # Специфичные зависимости для разных типов файлов
        if file_type in ['react', 'jsx', 'tsx']:
            dependencies.extend(['react', 'react-dom'])
        elif file_type == 'vue':
            dependencies.extend(['vue'])
        elif file_type == 'angular':
            dependencies.extend(['@angular/core', '@angular/common'])
        elif file_type == 'express':
            dependencies.extend(['express'])
        elif file_type == 'next':
            dependencies.extend(['next', 'react'])
        
        return list(set(dependencies))

    def _find_function_end(self, content: str, start_pos: int) -> int:
        """Находит конец функции (приблизительно)"""
        # Простой поиск закрывающей скобки
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i, char in enumerate(content[start_pos:], start_pos):
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
                continue
                
            if char in ['"', "'", '`'] and not in_string:
                in_string = True
                continue
            elif char in ['"', "'", '`'] and in_string:
                in_string = False
                continue
                
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return content[:i].count('\n') + 1
        
        return content.count('\n') + 1

    def _find_class_end(self, content: str, start_pos: int) -> int:
        """Находит конец класса (приблизительно)"""
        return self._find_function_end(content, start_pos)

    def _analyze_function_calls(self, functions: List[JSFunctionSig], content: str) -> None:
        """Анализирует вызовы функций и заполняет поля calls/called_by"""
        # Создаем словарь функций для быстрого поиска
        func_dict = {func.name: func for func in functions}
        
        # Анализируем каждую функцию
        for func in functions:
            # Находим тело функции
            func_start = content.find(func.name)
            if func_start == -1:
                continue
                
            # Ищем начало функции (скобка после имени)
            paren_pos = content.find('(', func_start)
            if paren_pos == -1:
                continue
                
            # Находим конец функции
            func_end = self._find_function_end(content, paren_pos)
            if func_end == -1:
                continue
                
            # Извлекаем тело функции
            func_body = content[func_start:func_end * 100]  # Примерно 100 символов на строку
            
            # Ищем вызовы других функций
            calls = self._extract_function_calls_from_body(func_body, func_dict)
            func.calls = calls
            
            # Обновляем called_by для вызываемых функций
            for call in calls:
                if call in func_dict:
                    if func.name not in func_dict[call].called_by:
                        func_dict[call].called_by.append(func.name)

    def _extract_function_calls_from_body(self, body: str, func_dict: Dict[str, JSFunctionSig]) -> List[str]:
        """Извлекает вызовы функций из тела функции"""
        calls = []
        
        # Паттерны для вызовов функций
        call_patterns = [
            r'(\w+)\s*\(',  # functionName(
            r'(\w+)\.(\w+)\s*\(',  # object.method(
            r'(\w+)\[(\w+)\]\s*\(',  # object[method](
        ]
        
        for pattern in call_patterns:
            matches = re.finditer(pattern, body)
            for match in matches:
                if len(match.groups()) == 1:
                    # Простой вызов функции
                    func_name = match.group(1)
                    if func_name in func_dict and func_name != 'function':  # Исключаем ключевое слово
                        calls.append(func_name)
                elif len(match.groups()) == 2:
                    # Вызов метода
                    obj_name = match.group(1)
                    method_name = match.group(2)
                    full_name = f"{obj_name}.{method_name}"
                    calls.append(full_name)
        
        return list(set(calls))  # Убираем дубликаты


# -----------------------------------------------------------------------------
# JavaScript анализатор зависимостей
# -----------------------------------------------------------------------------

class JavaScriptDependencyAnalyzer(DependencyAnalyzer):
    """Анализатор зависимостей для JavaScript проектов"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        """Строит граф зависимостей между элементами (протокол DependencyAnalyzer)"""
        js_maps = [fm for fm in file_maps if isinstance(fm, JSFileMap)]
        if not js_maps:
            return
        
        # Анализируем зависимости
        analysis = self.analyze_dependencies(js_maps)
        
        # Обновляем связи между функциями
        self._update_function_relationships(js_maps, analysis)
    
    def analyze_dependencies(self, file_maps: List[JSFileMap]) -> Dict[str, Any]:
        """Анализирует зависимости между JavaScript файлами"""
        # Строим граф вызовов
        call_graph = self._build_call_graph(file_maps)
        
        # Строим граф наследования
        inheritance_graph = self._build_inheritance_graph(file_maps)
        
        # Анализируем импорты/экспорты
        import_export_graph = self._build_import_export_graph(file_maps)
        
        return {
            'call_graph': call_graph,
            'inheritance_graph': inheritance_graph,
            'import_export_graph': import_export_graph,
            'file_dependencies': self._analyze_file_dependencies(file_maps)
        }

    def _build_call_graph(self, file_maps: List[JSFileMap]) -> nx.DiGraph:
        """Строит граф вызовов функций"""
        G = nx.DiGraph()
        
        # Создаем словарь всех функций для поиска
        all_functions = {}
        for fm in file_maps:
            for func in fm.functions:
                func_id = f"{fm.path}::{func.name}"
                all_functions[func_id] = func
                all_functions[func.name] = func_id  # Для поиска по имени
        
        # Строим граф
        for fm in file_maps:
            for func in fm.functions:
                func_id = f"{fm.path}::{func.name}"
                G.add_node(func_id, **func.__dict__)
                
                # Добавляем вызовы
                for call in func.calls:
                    # Ищем вызываемую функцию
                    if call in all_functions:
                        called_func_id = all_functions[call]
                        G.add_edge(func_id, called_func_id)
                    else:
                        # Добавляем как внешний вызов
                        G.add_node(call, type='external')
                        G.add_edge(func_id, call)
        
        return G

    def _build_inheritance_graph(self, file_maps: List[JSFileMap]) -> nx.DiGraph:
        """Строит граф наследования классов"""
        G = nx.DiGraph()
        
        for fm in file_maps:
            for cls in fm.classes:
                class_id = f"{fm.path}::{cls.name}"
                G.add_node(class_id, **cls.__dict__)
                
                # Добавляем наследование
                if cls.extends:
                    G.add_edge(class_id, cls.extends)
        
        return G

    def _build_import_export_graph(self, file_maps: List[JSFileMap]) -> nx.DiGraph:
        """Строит граф импортов/экспортов"""
        G = nx.DiGraph()
        
        for fm in file_maps:
            # Добавляем экспорты
            for export in fm.exports:
                export_id = f"{fm.path}::{export}"
                G.add_node(export_id, type='export', file=fm.path)
            
            # Добавляем импорты
            for imp in fm.imports:
                if 'from' in imp:
                    module = imp.split('from')[-1].strip().strip('\'"')
                    G.add_edge(fm.path, module)
        
        return G

    def _analyze_file_dependencies(self, file_maps: List[JSFileMap]) -> Dict[str, List[str]]:
        """Анализирует зависимости между файлами"""
        dependencies = {}
        
        for fm in file_maps:
            deps = []
            for imp in fm.imports:
                if 'from' in imp:
                    module = imp.split('from')[-1].strip().strip('\'"')
                    # Ищем соответствующий файл
                    for other_fm in file_maps:
                        if module in other_fm.path or any(exp in module for exp in other_fm.exports):
                            deps.append(other_fm.path)
            dependencies[fm.path] = deps
        
        return dependencies

    def _update_function_relationships(self, file_maps: List[JSFileMap], analysis: Dict[str, Any]) -> None:
        """Обновляет связи между функциями на основе анализа зависимостей"""
        call_graph = analysis.get('call_graph', nx.DiGraph())
        
        # Обновляем called_by для каждой функции
        for fm in file_maps:
            for func in fm.functions:
                func_id = f"{fm.path}::{func.name}"
                
                # Находим функции, которые вызывают эту функцию
                callers = list(call_graph.predecessors(func_id))
                func.called_by = [caller.split('::')[-1] for caller in callers]
                
                # Находим функции, которые вызывает эта функция
                callees = list(call_graph.successors(func_id))
                func.calls = [callee.split('::')[-1] for callee in callees]


# -----------------------------------------------------------------------------
# JavaScript RAG система
# -----------------------------------------------------------------------------

class JavaScriptRAGSystem(BaseRAGSystem):
    """RAG система для анализа JavaScript/TypeScript кода"""
    
    def __init__(self, 
                 llm: BaseChatModel, 
                 embeddings: Embeddings,
                 **kwargs):
        super().__init__(llm, embeddings, [JavaScriptFileParser()], [JavaScriptDependencyAnalyzer()], **kwargs)

    def get_file_patterns(self) -> Dict[str, str]:
        """Возвращает паттерны файлов для JavaScript/TypeScript проектов"""
        return {
            "javascript": "**/*.js",
            "typescript": "**/*.ts", 
            "jsx": "**/*.jsx",
            "tsx": "**/*.tsx",
            "vue": "**/*.vue",
            "svelte": "**/*.svelte",
            "config": "**/*.config.js",
            "json": "**/package.json",
            "test": "**/*.test.js",
            "spec": "**/*.spec.js"
        }

    def get_evidence_prompt_template(self) -> ChatPromptTemplate:
        """Возвращает шаблон для планирования evidence"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a JavaScript code analysis expert. Given a question about JavaScript/TypeScript code, determine what evidence to gather.

Available file types: javascript, typescript, jsx, tsx, vue, svelte, node, express, react, next, nuxt, angular, config, test, storybook

Evidence planning rules:
- For function/class questions, request specific symbols
- For dependency questions, request files with imports/exports
- For architecture questions, request multiple file types
- For configuration questions, focus on config files
- For React questions, look for components and hooks
- For Node.js questions, look for server files and APIs
- For testing questions, look for test files
- For build questions, look for config files (webpack, babel, etc.)

Respond ONLY with JSON array, no prose.
Example: ["file.js::functionName", "component.tsx::*", "config.webpack.js::*"]"""),
            ("human", "Question:\n{question}\n\nContext (map snippets):\n{context}\n")
        ])

    def get_answer_prompt_template(self) -> ChatPromptTemplate:
        """Возвращает шаблон для генерации ответа"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a JavaScript/TypeScript expert. Answer questions about JavaScript code using the provided evidence.

Key areas to focus on:
- Function signatures, parameters, and return types
- Class inheritance and method relationships
- Import/export relationships and module dependencies
- React components, hooks, and lifecycle methods
- Node.js server architecture and API endpoints
- Build configuration and tooling setup
- Testing strategies and test coverage
- Performance optimizations and best practices

Provide detailed, accurate answers with code examples when relevant.
Reply in {answer_language}."""),
            ("human", "Question:\n{question}\n\nRepo map (summaries):\n{map_text}\n\nEvidence (code bodies):\n{evidence_text}\n")
        ])

    def _create_documents(self, file_maps: List[BaseFileMap]) -> List[Document]:
        """Создает документы для индексирования из карт файлов"""
        docs = []
        js_maps = [fm for fm in file_maps if isinstance(fm, JSFileMap)]
        
        print(f"📊 Создание документов: {len(js_maps)} JavaScript файлов")
        
        for fm in js_maps:
            # Создаем документ для всего файла
            file_content = fm.to_text()
            docs.append(Document(
                page_content=file_content,
                metadata={"source": fm.path, "type": f"{fm.file_type}-map", "loc": fm.loc}
            ))
            
            # Создаем документы для функций
            for func in fm.functions:
                func_doc = f"""FUNCTION: {func.qname}
FILE: {fm.path}
TYPE: {fm.file_type}
SIGNATURE: {func.to_line()}

CALLS: {', '.join(func.calls) if func.calls else 'None'}
CALLED BY: {', '.join(func.called_by) if func.called_by else 'None'}
EXPORTS: {', '.join(func.exports) if func.exports else 'None'}"""
                
                docs.append(Document(
                    page_content=func_doc,
                    metadata={
                        "source": fm.path,
                        "type": f"{fm.file_type}-function",
                        "function_name": func.name,
                        "lineno": func.lineno
                    }
                ))
            
            # Создаем документы для классов
            for cls in fm.classes:
                cls_doc = f"""CLASS: {cls.name}
FILE: {fm.path}
TYPE: {fm.file_type}
SIGNATURE: {cls.to_line()}

EXTENDS: {cls.extends or 'None'}
INHERITED BY: {', '.join(cls.inherited_by) if cls.inherited_by else 'None'}
METHODS: {len(cls.methods)} methods
PROPERTIES: {', '.join(cls.properties) if cls.properties else 'None'}"""
                
                docs.append(Document(
                    page_content=cls_doc,
                    metadata={
                        "source": fm.path,
                        "type": f"{fm.file_type}-class",
                        "class_name": cls.name,
                        "lineno": cls.lineno
                    }
                ))
            
            # Создаем документы для переменных
            for var in fm.variables:
                if var.is_exported or var.is_imported:
                    var_doc = f"""VARIABLE: {var.name}
FILE: {fm.path}
TYPE: {fm.file_type}
SIGNATURE: {var.to_line()}

VALUE_TYPE: {var.value_type or 'inferred'}
SOURCE: {var.source or 'local'}"""
                    
                    docs.append(Document(
                        page_content=var_doc,
                        metadata={
                            "source": fm.path,
                            "type": f"{fm.file_type}-variable",
                            "variable_name": var.name,
                            "lineno": var.lineno
                        }
                    ))
        
        print(f"✅ Создано {len(docs)} документов")
        return docs

    def _extract_bodies(self, symbols: List[str], project_path: str) -> List[Tuple[str, str]]:
        """Извлекает тела функций/классов по символам"""
        out = []
        
        for sym in symbols:
            if '::' not in sym:
                continue
                
            file_path, symbol = sym.split('::', 1)
            full_path = os.path.join(project_path, file_path)
            
            if not os.path.exists(full_path):
                continue
            
            try:
                content = _read_text(full_path)
                lines = content.splitlines()
                
                if symbol == '*':
                    # Весь файл
                    out.append((f"{file_path}:1", content))
                else:
                    # Ищем конкретный символ
                    found = self._find_symbol_in_file(content, symbol, file_path)
                    if found:
                        out.append(found)
                    else:
                        # Fallback: показываем начало файла
                        out.append((f"{file_path}:1", content[:2000] + "...[truncated]" if len(content) > 2000 else content))
                        
            except Exception as e:
                print(f"⚠️  Ошибка чтения {file_path}: {e}")
                continue
        
        return out

    def _build_search_index(
        self, 
        docs: List[Document], 
        project_path: str, 
        file_maps: List[BaseFileMap]
    ) -> ProjectIndex:
        """Строит поисковый индекс для JavaScript проекта"""
        print(f"🔍 Строим поисковый индекс для {len(docs)} документов...")
        
        # BM25 retriever для текстового поиска
        bm25 = BM25Retriever.from_documents(docs)
        bm25.k = self.config.get('bm25_k', 12)
        
        # FAISS retriever для семантического поиска
        faiss = FAISS.from_documents(docs, self.embeddings, distance_strategy=DistanceStrategy.COSINE)
        dense_retriever = faiss.as_retriever(search_kwargs={"k": self.config.get('dense_k', 12)})
        
        # Ensemble retriever - комбинируем BM25 и FAISS
        ensemble = EnsembleRetriever(
            retrievers=[bm25, dense_retriever], 
            weights=list(self.config.get('ensemble_weights', (0.5, 0.5)))
        )
        
        # Опциональная компрессия для релевантности
        retriever = ensemble
        if self.config.get('use_compression', True):
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=ensemble
            )
        
        print("✅ Поисковый индекс готов!")
        
        return ProjectIndex(
            root=project_path,
            docs=docs,
            retriever=retriever,
            file_maps=file_maps
        )

    def _answer_question(self, question: str, index: ProjectIndex) -> str:
        """Отвечает на вопрос о JavaScript коде"""
        # Получаем релевантные документы
        evidence_char_budget = self.config.get('evidence_char_budget', 20000)
        max_evidence_items = self.config.get('max_evidence_items', 8)
        map_char_budget = self.config.get('map_char_budget', 24000)

        retrieved = index.retriever.invoke(question)
        
        # Собираем контекст
        map_text = self._gather_map_snippets(retrieved, map_char_budget)
        
        # Генерируем план поиска доказательств
        evidence_prompt = self.get_evidence_prompt_template()
        evidence_chain = evidence_prompt | self.llm | StrOutputParser()

        raw_plan = evidence_chain.invoke({
            "question": question,
            "context": map_text,
            "max_items": self.config.get('max_evidence_items', 8),
        })
        
        # Парсим JSON план
        plan_json = raw_plan.strip()
        plan_json = plan_json[plan_json.find("[") : plan_json.rfind("]") + 1] if "[" in plan_json and "]" in plan_json else "[]"
        try:
            plan = json.loads(plan_json)
            if not isinstance(plan, list):
                plan = []
        except Exception:
            plan = []

        # Извлекаем запрошенные тела функций/классов
        evidence_pairs = self._extract_bodies(index.root, plan[:max_evidence_items]) if plan else []
        evidence_text = "\n\n".join([f"### {lbl}\n" + code for (lbl, code) in evidence_pairs])
        evidence_text = self._trim_to_chars(evidence_text, evidence_char_budget)
        
        # Генерируем финальный ответ
        answer_prompt = self.get_answer_prompt_template()
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": self.config.get('answer_language', 'ru'),
            "evidence_text": evidence_text if evidence_text else "(no additional bodies requested)",
        })
        
        return final

    def _gather_map_snippets(self, retrieved: List[Document], max_chars: int) -> str:
        """Собирает фрагменты карты из релевантных документов"""
        snippets = []
        current_chars = 0
        
        for doc in retrieved:
            if current_chars + len(doc.page_content) > max_chars:
                break
            snippets.append(doc.page_content)
            current_chars += len(doc.page_content)
        
        return "\n\n".join(snippets)

    def _trim_to_chars(self, text: str, max_chars: int) -> str:
        """Обрезает текст до максимального количества символов"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n... [truncated]"

    def _find_symbol_in_file(self, content: str, symbol: str, file_path: str) -> Optional[Tuple[str, str]]:
        """Ищет символ в файле и возвращает соответствующий фрагмент"""
        lines = content.splitlines()
        
        # Ищем по имени функции/класса/переменной
        for i, line in enumerate(lines):
            if symbol in line:
                # Находим начало блока
                start_line = max(0, i - 2)
                end_line = min(len(lines), i + 20)
                
                result = f"# {symbol} in {file_path}\n\n"
                result += "```javascript\n"
                for j in range(start_line, end_line):
                    result += f"{j+1:4d}: {lines[j]}\n"
                result += "```\n"
                
                return (f"{file_path}:{start_line+1}-{end_line}", result)
        
        return None

    def _get_tool_description(self) -> str:
        """Возвращает описание инструмента"""
        return (
            "Analyze JavaScript/TypeScript code architecture and design. "
            "Uses AST analysis, call graphs, inheritance analysis, and React/Node.js specific patterns. "
            "Supports React components, hooks, Express APIs, testing frameworks, and modern JS features. "
            "Input: a natural-language question about JavaScript/TypeScript codebase."
        )


RAGSystemFactory.register('javascript', JavaScriptRAGSystem)
