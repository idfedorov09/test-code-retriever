#!/usr/bin/env python3
"""
Python-специфичная реализация RAG системы.
Анализирует Python код с помощью AST, строит графы вызовов и наследования.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

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
# Python-специфичные структуры данных
# -----------------------------------------------------------------------------

@dataclass
class PythonFunctionSig(BaseCodeElement):
    """Сигнатура Python функции"""
    qname: str
    args: str
    returns: Optional[str]
    decorators: List[str]
    calls: List[str] = field(default_factory=list)
    called_by: List[str] = field(default_factory=list)

    def to_line(self) -> str:
        ret = f" -> {self.returns}" if self.returns else ""
        dec = f" [@{' @'.join(self.decorators)}]" if self.decorators else ""
        calls_info = f" calls: {', '.join(self.calls[:3])}" if self.calls else ""
        if len(self.calls) > 3:
            calls_info += f" (+{len(self.calls)-3} more)"
        called_by_info = f" called_by: {', '.join(self.called_by[:3])}" if self.called_by else ""
        if len(self.called_by) > 3:
            called_by_info += f" (+{len(self.called_by)-3} more)"
        return f"def {self.qname}({self.args}){ret}{dec}{calls_info}{called_by_info}  # L{self.lineno}-{self.end_lineno}"


@dataclass
class PythonClassSig(BaseCodeElement):
    """Сигнатура Python класса"""
    bases: List[str]
    methods: List[PythonFunctionSig] = field(default_factory=list)
    inherited_by: List[str] = field(default_factory=list)

    def to_line(self) -> str:
        bases = f"({', '.join(self.bases)})" if self.bases else ""
        header = f"class {self.name}{bases}  # L{self.lineno}-{self.end_lineno}"
        
        if self.inherited_by:
            inheritance_info = f" inherited_by: {', '.join(self.inherited_by[:3])}"
            if len(self.inherited_by) > 3:
                inheritance_info += f" (+{len(self.inherited_by)-3} more)"
            header += inheritance_info
        
        return header

    def to_block(self) -> str:
        header = self.to_line()
        body = "\n".join(["  - " + m.to_line() for m in self.methods])
        return f"{header}\n{body}" if body else header


@dataclass
class PythonFileMap(BaseFileMap):
    """Карта Python файла"""
    imports: List[str]
    functions: List[PythonFunctionSig]
    classes: List[PythonClassSig]

    def to_text(self) -> str:
        parts = [f"FILE: {self.path}"]
        if self.imports:
            parts.append("IMPORTS:\n" + "\n".join([f"  - {imp}" for imp in sorted(self.imports) if imp]))
        if self.classes:
            parts.append("CLASSES:\n" + "\n".join([c.to_block() for c in self.classes]))
        if self.functions:
            parts.append("FUNCTIONS:\n" + "\n".join([f"  - {fn.to_line()}" for fn in self.functions]))
        parts.append(f"LOC: {self.loc}")
        return "\n".join(parts)

    def get_searchable_content(self) -> List[str]:
        content = []
        for func in self.functions:
            content.append(f"function:{func.qname}")
        for cls in self.classes:
            content.append(f"class:{cls.name}")
            for method in cls.methods:
                content.append(f"method:{cls.name}.{method.name}")
        return content


# -----------------------------------------------------------------------------
# Python парсер
# -----------------------------------------------------------------------------

class PythonFileParser:
    """Парсер Python файлов с использованием AST"""
    
    def can_parse(self, file_path: str) -> bool:
        return file_path.endswith('.py')
    
    def parse(self, file_path: str, root: str) -> Optional[PythonFileMap]:
        try:
            source = _read_text(file_path)
        except Exception:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        rel = _relpath(file_path, root)
        imports: List[str] = []
        functions: List[PythonFunctionSig] = []
        classes: List[PythonClassSig] = []

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.class_stack: List[str] = []
                self.current_function: Optional[PythonFunctionSig] = None

            def visit_Import(self, node: ast.Import) -> None:  # type: ignore[override]
                for alias in node.names:
                    if alias.name:
                        imports.append(alias.name)

            def visit_ImportFrom(self, node: ast.ImportFrom) -> None:  # type: ignore[override]
                mod = node.module or ""
                for alias in node.names:
                    name = alias.name
                    if name:
                        imports.append(f"{mod}.{name}" if mod else name)

            def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
                """Extract function calls within current function."""
                if self.current_function is not None:
                    call_name = self._extract_call_name(node.func)
                    if call_name and isinstance(call_name, str):
                        self.current_function.calls.append(call_name)
                self.generic_visit(node)

            def _extract_call_name(self, node: ast.AST) -> Optional[str]:
                """Extract the name of a function call from AST node."""
                try:
                    if isinstance(node, ast.Name):
                        return node.id
                    elif isinstance(node, ast.Attribute):
                        value_name = self._extract_call_name(node.value)
                        if value_name:
                            return f"{value_name}.{node.attr}"
                        else:
                            return node.attr
                    elif isinstance(node, ast.Subscript):
                        return self._extract_call_name(node.value)
                except Exception:
                    pass
                return None

            def _args(self, node: ast.AST) -> str:
                if not isinstance(node, ast.arguments):
                    return ""
                parts = []
                for a in node.posonlyargs:
                    parts.append(a.arg)
                if node.posonlyargs:
                    parts.append("/")
                parts += [a.arg for a in node.args]
                if node.vararg:
                    parts.append("*" + node.vararg.arg)
                elif node.kwonlyargs:
                    parts.append("*")
                parts += [f"{a.arg}" for a in node.kwonlyargs]
                if node.kwarg:
                    parts.append("**" + node.kwarg.arg)
                return ", ".join(parts)

            def _returns(self, node: Optional[ast.AST]) -> Optional[str]:
                if not node:
                    return None
                try:
                    return ast.unparse(node)
                except Exception:
                    return None

            def _decorators(self, node: ast.AST) -> List[str]:
                decs = getattr(node, "decorator_list", [])
                out = []
                for d in decs:
                    try:
                        out.append(ast.unparse(d))
                    except Exception:
                        pass
                return out

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
                qn = ".".join(self.class_stack + [node.name]) if self.class_stack else node.name
                fs = PythonFunctionSig(
                    name=node.name,
                    qname=qn,
                    args=self._args(node.args),
                    returns=self._returns(node.returns),
                    decorators=self._decorators(node),
                    lineno=getattr(node, "lineno", 0),
                    end_lineno=getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                )
                
                # Set current function context to track calls
                old_function = self.current_function
                self.current_function = fs
                
                # Visit function body to collect calls
                self.generic_visit(node)
                
                # Restore previous function context
                self.current_function = old_function
                
                if self.class_stack:
                    classes[-1].methods.append(fs)
                else:
                    functions.append(fs)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                self.visit_FunctionDef(node)

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
                bases = []
                for b in node.bases:
                    try:
                        bases.append(ast.unparse(b))
                    except Exception:
                        pass
                
                c = PythonClassSig(
                    name=node.name, 
                    bases=bases, 
                    lineno=node.lineno, 
                    end_lineno=getattr(node, "end_lineno", node.lineno)
                )
                classes.append(c)
                self.class_stack.append(node.name)
                self.generic_visit(node)
                self.class_stack.pop()

        Visitor().visit(tree)
        loc = len(source.splitlines())

        return PythonFileMap(
            path=rel,
            file_type="python",
            imports=imports,
            functions=functions,
            classes=classes,
            loc=loc
        )


# -----------------------------------------------------------------------------
# Анализаторы зависимостей
# -----------------------------------------------------------------------------

class PythonCallGraphAnalyzer:
    """Анализатор графа вызовов Python функций"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        python_maps = [fm for fm in file_maps if isinstance(fm, PythonFileMap)]
        
        # Create a mapping from function name to function object
        all_functions: Dict[str, List[PythonFunctionSig]] = {}
        
        # Collect all functions (including methods)
        for file_map in python_maps:
            for func in file_map.functions:
                if func.qname not in all_functions:
                    all_functions[func.qname] = []
                all_functions[func.qname].append(func)
            
            for cls in file_map.classes:
                for method in cls.methods:
                    full_name = f"{cls.name}.{method.qname.split('.')[-1]}"
                    if full_name not in all_functions:
                        all_functions[full_name] = []
                    all_functions[full_name].append(method)
        
        # Build reverse call graph
        for file_map in python_maps:
            # Process standalone functions
            for func in file_map.functions:
                for call_name in func.calls:
                    if not call_name:
                        continue
                        
                    candidates = []
                    
                    # Direct match
                    if call_name in all_functions:
                        candidates.extend(all_functions[call_name])
                    
                    # Try partial matches for method calls
                    if '.' in call_name:
                        method_name = call_name.split('.')[-1]
                        if method_name:
                            for func_name, funcs in all_functions.items():
                                if func_name.endswith(f'.{method_name}'):
                                    candidates.extend(funcs)
                    
                    # Add caller to called_by lists
                    for candidate in candidates:
                        if func.qname and func.qname not in candidate.called_by:
                            candidate.called_by.append(func.qname)
            
            # Process class methods
            for cls in file_map.classes:
                for method in cls.methods:
                    for call_name in method.calls:
                        if not call_name:
                            continue
                            
                        candidates = []
                        
                        if call_name in all_functions:
                            candidates.extend(all_functions[call_name])
                        
                        if '.' in call_name:
                            method_name = call_name.split('.')[-1]
                            if method_name:
                                for func_name, funcs in all_functions.items():
                                    if func_name.endswith(f'.{method_name}'):
                                        candidates.extend(funcs)
                        
                        if method.qname:
                            caller_name = f"{cls.name}.{method.qname.split('.')[-1]}"
                            for candidate in candidates:
                                if caller_name not in candidate.called_by:
                                    candidate.called_by.append(caller_name)


class PythonInheritanceAnalyzer:
    """Анализатор наследования Python классов"""
    
    def build_dependency_graph(self, file_maps: List[BaseFileMap]) -> None:
        python_maps = [fm for fm in file_maps if isinstance(fm, PythonFileMap)]
        
        # Create a mapping from class name to class objects
        all_classes: Dict[str, List[PythonClassSig]] = {}
        
        # Collect all classes
        for file_map in python_maps:
            for cls in file_map.classes:
                if cls.name not in all_classes:
                    all_classes[cls.name] = []
                all_classes[cls.name].append(cls)
        
        # Build inheritance relationships
        for file_map in python_maps:
            for cls in file_map.classes:
                for base_name in cls.bases:
                    if not base_name:
                        continue
                        
                    # Clean base class name (remove generics, etc.)
                    clean_base = base_name.split('[')[0].split('.').pop()
                    
                    if not clean_base:
                        continue
                    
                    # Find base class and add this class as its inheritor
                    if clean_base in all_classes:
                        for base_cls in all_classes[clean_base]:
                            if cls.name and cls.name not in base_cls.inherited_by:
                                base_cls.inherited_by.append(cls.name)


# -----------------------------------------------------------------------------
# Python RAG система
# -----------------------------------------------------------------------------

class PythonRAGSystem(BaseRAGSystem):
    """RAG система для анализа Python проектов"""
    
    def __init__(self, llm, embeddings, **kwargs):
        parsers = [PythonFileParser()]
        analyzers = [PythonCallGraphAnalyzer(), PythonInheritanceAnalyzer()]
        super().__init__(llm, embeddings, parsers, analyzers, **kwargs)
    
    def get_file_patterns(self) -> Dict[str, str]:
        return {"python": "**/*.py"}
    
    def get_evidence_prompt_template(self) -> str:
        return """
You are an architectural code reviewer analyzing Python code. You will see a compact map of a repository with function signatures, imports, classes, call relationships, and inheritance relationships.

The map shows:
- Function signatures with their calls and called_by relationships
- Class methods and their call patterns
- Class inheritance with inherited_by relationships (subclasses)
- Import dependencies

Given the user question, propose up to {max_items} precise evidence items (code bodies to inspect) in JSON.
Each item must be an object with keys: file (relative path), symbol (function or Class.method), and reason.

IMPORTANT: 
- Paths must be repository-relative (e.g., `app/api.py`) — NEVER absolute
- Copy the exact path shown after `FILE:` in the context
- For questions about "what calls X" or "where is X used", look at the `called_by` field
- For questions about "what does X call" or "dependencies of X", look at the `calls` field
- For questions about "how is class X used" or "what uses class X", look at the `inherited_by` field (subclasses)
- For questions about class inheritance, check both `bases` (parent classes) and `inherited_by` (child classes)
- Include both the target class/function and its relationships for complete analysis

Respond ONLY with JSON array, no prose.
"""
    
    def get_answer_prompt_template(self) -> str:
        return """
You are a senior software architect analyzing Python code for code review. Using the provided repository map and evidence, answer the question comprehensively.

The repository map includes:
- Function signatures with call relationships (calls/called_by fields)
- Class hierarchies with inheritance relationships (bases/inherited_by fields)
- Method relationships and class usage patterns
- Import dependencies and module structure

For questions about function usage:
- Use "called_by" information to identify where functions are used
- Use "calls" information to understand what a function depends on

For questions about class usage:
- Use "inherited_by" information to identify subclasses (how class is used through inheritance)
- Use "bases" information to understand parent classes
- Consider inheritance as a form of class usage - subclasses are users of the parent class

Provide specific file:line references from the evidence and explain relationships clearly.
Be precise, cite concrete references as `file:line`, and provide actionable insights. 
If analyzing risks or issues, include specific remediation steps.
Reply in {answer_language}.
"""
    
    def _create_documents(self, file_maps: List[BaseFileMap]) -> List[Document]:
        docs = []
        python_maps = [fm for fm in file_maps if isinstance(fm, PythonFileMap)]
        
        # Create documents from file maps
        for fm in python_maps:
            content = fm.to_text()
            meta = {"source": fm.path, "type": "py-map", "loc": fm.loc}
            docs.append(Document(page_content=content, metadata=meta))
        
        # Add call graph summary
        call_graph_lines = ["FUNCTION CALL RELATIONSHIPS:"]
        for file_map in python_maps:
            for func in file_map.functions:
                if func.called_by:
                    call_graph_lines.append(f"  {func.qname} ({file_map.path}:{func.lineno}) called by: {', '.join(func.called_by)}")
            for cls in file_map.classes:
                for method in cls.methods:
                    if method.called_by:
                        full_name = f"{cls.name}.{method.qname.split('.')[-1]}"
                        call_graph_lines.append(f"  {full_name} ({file_map.path}:{method.lineno}) called by: {', '.join(method.called_by)}")
        
        if len(call_graph_lines) > 1:
            docs.append(Document(
                page_content="\n".join(call_graph_lines), 
                metadata={"source": "__call_graph__", "type": "call-graph-summary"}
            ))
        
        # Add inheritance graph summary
        inheritance_lines = ["CLASS INHERITANCE RELATIONSHIPS:"]
        for file_map in python_maps:
            for cls in file_map.classes:
                if cls.inherited_by:
                    inheritance_lines.append(f"  {cls.name} ({file_map.path}:{cls.lineno}) inherited by: {', '.join(cls.inherited_by)}")
                if cls.bases:
                    inheritance_lines.append(f"  {cls.name} ({file_map.path}:{cls.lineno}) inherits from: {', '.join(cls.bases)}")
        
        if len(inheritance_lines) > 1:
            docs.append(Document(
                page_content="\n".join(inheritance_lines), 
                metadata={"source": "__inheritance_graph__", "type": "inheritance-graph-summary"}
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
        
        # Optional compression
        retriever = ensemble
        if self.config.get('use_compression', True):
            compressor = LLMChainExtractor.from_llm(self.llm)
            retriever = ContextualCompressionRetriever(
                base_compressor=compressor, 
                base_retriever=ensemble
            )
        
        return ProjectIndex(
            root=project_path,
            docs=docs,
            retriever=retriever,
            file_maps=file_maps
        )
    
    def _answer_question(self, question: str, index: ProjectIndex) -> str:
        # Retrieve relevant documents
        retrieved = index.retriever.invoke(question)
        
        # Gather context
        map_text = self._gather_map_snippets(retrieved, self.config.get('map_char_budget', 24000))
        
        # Generate evidence plan
        evidence_prompt = ChatPromptTemplate.from_template(self.get_evidence_prompt_template())
        evidence_chain = evidence_prompt | self.llm | StrOutputParser()
        
        raw_plan = evidence_chain.invoke({
            "question": question,
            "context": map_text,
            "max_items": self.config.get('max_evidence_items', 8),
        })
        
        # Parse evidence plan (simplified)
        evidence_text = "(no additional bodies requested)"  # Placeholder
        
        # Generate final answer
        answer_prompt = ChatPromptTemplate.from_template(self.get_answer_prompt_template())
        answer_chain = answer_prompt | self.llm | StrOutputParser()
        
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": self.config.get('answer_language', 'ru'),
            "evidence_text": evidence_text,
        })
        
        return final
    
    def _gather_map_snippets(self, docs: List[Document], max_chars: int = 20000) -> str:
        pieces = []
        for d in docs:
            if d.metadata.get("type") in ("inheritance-graph-summary", "call-graph-summary", "py-map", "graph-summary"):
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
            "Analyze Python code architecture and design. "
            "Uses AST analysis, call graphs, and inheritance analysis. "
            "Input: a natural-language question about Python codebase."
        )


# Регистрируем Python RAG систему
RAGSystemFactory.register('python', PythonRAGSystem)
