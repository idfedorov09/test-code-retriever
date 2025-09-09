from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEndpointEmbeddings, HuggingFaceEmbeddings

load_dotenv()

import os
import re
import json
import ast
import glob
import textwrap
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any

# GPU detection and configuration
try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
    print(f"🚀 Device: {DEVICE}")
    if GPU_AVAILABLE:
        print(f"📱 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
except ImportError:
    GPU_AVAILABLE = False
    DEVICE = "cpu"
    print("⚠️  PyTorch не найден, используем CPU")

# LangChain core & community
from langchain_core.tools import StructuredTool
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS, DistanceStrategy
from langchain_community.llms import YandexGPT

from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import networkx as nx
import pyan

try:
    from radon.complexity import cc_visit  # type: ignore
except Exception:
    cc_visit = None  # pragma: no cover


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

PY_FILE_GLOB = "**/*.py"


def _relpath(path: str, root: str) -> str:
    try:
        return os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    except Exception:
        return path


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _safe_get_lines(text: str, start: int, end: int) -> str:
    lines = text.splitlines()
    start = max(start - 1, 0)
    end = min(end, len(lines))
    return "\n".join(lines[start:end])


# -----------------------------------------------------------------------------
# AST-based extraction of structure (signatures, imports, classes)
# -----------------------------------------------------------------------------

@dataclass
class FunctionSig:
    qname: str
    args: str
    returns: Optional[str]
    decorators: List[str]
    lineno: int
    end_lineno: int
    calls: List[str] = field(default_factory=list)  # Functions/methods called by this function
    called_by: List[str] = field(default_factory=list)  # Functions that call this function

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
class ClassSig:
    name: str
    bases: List[str]
    methods: List[FunctionSig] = field(default_factory=list)
    inherited_by: List[str] = field(default_factory=list)  # Classes that inherit from this class
    lineno: int = 0
    end_lineno: int = 0

    def to_block(self) -> str:
        bases = f"({', '.join(self.bases)})" if self.bases else ""
        header = f"class {self.name}{bases}  # L{self.lineno}-{self.end_lineno}"
        
        # Add inheritance info
        if self.inherited_by:
            inheritance_info = f" inherited_by: {', '.join(self.inherited_by[:3])}"
            if len(self.inherited_by) > 3:
                inheritance_info += f" (+{len(self.inherited_by)-3} more)"
            header += inheritance_info
        
        body = "\n".join(["  - " + m.to_line() for m in self.methods])
        return f"{header}\n{body}" if body else header


@dataclass
class FileMap:
    path: str
    imports: List[str]
    functions: List[FunctionSig]
    classes: List[ClassSig]
    loc: int

    def to_text(self) -> str:
        parts = [f"FILE: {self.path}"]
        if self.imports:
            parts.append("IMPORTS:\n" + "\n".join([f"  - {imp}" for imp in sorted(self.imports)]))
        if self.classes:
            parts.append("CLASSES:\n" + "\n".join([c.to_block() for c in self.classes]))
        if self.functions:
            parts.append("FUNCTIONS:\n" + "\n".join([f"  - {fn.to_line()}" for fn in self.functions]))
        parts.append(f"LOC: {self.loc}")
        return "\n".join(parts)


def parse_python_file(path: str, root: str) -> Optional[FileMap]:
    try:
        source = _read_text(path)
    except Exception:
        return None

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    rel = _relpath(path, root)
    imports: List[str] = []
    functions: List[FunctionSig] = []
    classes: List[ClassSig] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.class_stack: List[str] = []
            self.current_function: Optional[FunctionSig] = None

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
                    # For method calls like obj.method() or module.function()
                    value_name = self._extract_call_name(node.value)
                    if value_name:
                        return f"{value_name}.{node.attr}"
                    else:
                        return node.attr
                elif isinstance(node, ast.Subscript):
                    # For calls like obj[key]()
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
                return ast.unparse(node)  # py>=3.9
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
            fs = FunctionSig(
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
                # append to last class
                classes[-1].methods.append(fs)
            else:
                functions.append(fs)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
            self.visit_FunctionDef(node)  # treat as normal for mapping

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
            bases = []
            for b in node.bases:
                try:
                    bases.append(ast.unparse(b))
                except Exception:
                    pass
            c = ClassSig(name=node.name, bases=bases, lineno=node.lineno, end_lineno=getattr(node, "end_lineno", node.lineno))
            classes.append(c)
            self.class_stack.append(node.name)
            self.generic_visit(node)
            self.class_stack.pop()

    Visitor().visit(tree)
    loc = len(source.splitlines())

    return FileMap(path=rel, imports=imports, functions=functions, classes=classes, loc=loc)


def build_call_graph(file_maps: List[FileMap]) -> None:
    """Build reverse call graph: populate called_by fields for all functions."""
    # Create a mapping from function name to function object
    all_functions: Dict[str, List[FunctionSig]] = {}
    
    # Collect all functions (including methods)
    for file_map in file_maps:
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
    for file_map in file_maps:
        # Process standalone functions
        for func in file_map.functions:
            for call_name in func.calls:
                if not call_name:  # Skip empty or None call names
                    continue
                    
                # Try to find the called function
                candidates = []
                
                # Direct match
                if call_name in all_functions:
                    candidates.extend(all_functions[call_name])
                
                # Try partial matches for method calls
                if '.' in call_name:
                    method_name = call_name.split('.')[-1]
                    if method_name:  # Ensure method_name is not empty
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
                    if not call_name:  # Skip empty or None call names
                        continue
                        
                    candidates = []
                    
                    # Direct match
                    if call_name in all_functions:
                        candidates.extend(all_functions[call_name])
                    
                    # Try partial matches
                    if '.' in call_name:
                        method_name = call_name.split('.')[-1]
                        if method_name:  # Ensure method_name is not empty
                            for func_name, funcs in all_functions.items():
                                if func_name.endswith(f'.{method_name}'):
                                    candidates.extend(funcs)
                    
                    # Add caller to called_by lists
                    if method.qname:
                        caller_name = f"{cls.name}.{method.qname.split('.')[-1]}"
                        for candidate in candidates:
                            if caller_name not in candidate.called_by:
                                candidate.called_by.append(caller_name)


def build_inheritance_graph(file_maps: List[FileMap]) -> None:
    """Build inheritance graph: populate inherited_by fields for all classes."""
    # Create a mapping from class name to class objects
    all_classes: Dict[str, List[ClassSig]] = {}
    
    # Collect all classes
    for file_map in file_maps:
        for cls in file_map.classes:
            if cls.name not in all_classes:
                all_classes[cls.name] = []
            all_classes[cls.name].append(cls)
    
    # Build inheritance relationships
    for file_map in file_maps:
        for cls in file_map.classes:
            for base_name in cls.bases:
                if not base_name:  # Skip empty or None base names
                    continue
                    
                # Clean base class name (remove generics, etc.)
                clean_base = base_name.split('[')[0].split('.').pop()  # Take last part after dots
                
                if not clean_base:  # Skip if clean_base is empty
                    continue
                
                # Find base class and add this class as its inheritor
                if clean_base in all_classes:
                    for base_cls in all_classes[clean_base]:
                        if cls.name and cls.name not in base_cls.inherited_by:
                            base_cls.inherited_by.append(cls.name)


# -----------------------------------------------------------------------------
# Index construction: documents + (optional) graphs/metrics
# -----------------------------------------------------------------------------

@dataclass
class RepoIndex:
    root: str
    docs: List[Document]
    bm25: BM25Retriever
    vect: Any  # FAISS
    retriever: Any  # Ensemble + compression
    files: List[str]  # repo-relative file paths


def build_repo_index(
        project_path: str,
        embeddings: Embeddings,
        bm25_k: int = 10,
        dense_k: int = 10,
        ensemble_weights: Tuple[float, float] = (0.5, 0.5),
        llm: Optional[BaseChatModel] = None,
) -> RepoIndex:
    project_path = os.path.abspath(project_path)
    py_files = [p for p in glob.glob(os.path.join(project_path, PY_FILE_GLOB), recursive=True) if os.path.isfile(p)]

    file_maps: List[FileMap] = []
    for p in py_files:
        m = parse_python_file(p, project_path)
        if m:
            file_maps.append(m)

    # Build call graph to populate called_by relationships
    build_call_graph(file_maps)
    
    # Build inheritance graph to populate inherited_by relationships
    build_inheritance_graph(file_maps)

    # Create LangChain Documents from maps
    docs: List[Document] = []
    for m in file_maps:
        content = m.to_text()
        meta = {"source": m.path, "type": "py-map", "loc": m.loc}
        docs.append(Document(page_content=content, metadata=meta))

    # Add call graph summary document for better function usage queries
    call_graph_lines = ["FUNCTION CALL RELATIONSHIPS:"]
    for file_map in file_maps:
        for func in file_map.functions:
            if func.called_by:
                call_graph_lines.append(f"  {func.qname} ({file_map.path}:{func.lineno}) called by: {', '.join(func.called_by)}")
        for cls in file_map.classes:
            for method in cls.methods:
                if method.called_by:
                    full_name = f"{cls.name}.{method.qname.split('.')[-1]}"
                    call_graph_lines.append(f"  {full_name} ({file_map.path}:{method.lineno}) called by: {', '.join(method.called_by)}")
    
    if len(call_graph_lines) > 1:  # More than just the header
        docs.append(Document(
            page_content="\n".join(call_graph_lines), 
            metadata={"source": "__call_graph__", "type": "call-graph-summary"}
        ))

    # Add inheritance graph summary document for better class usage queries
    inheritance_lines = ["CLASS INHERITANCE RELATIONSHIPS:"]
    for file_map in file_maps:
        for cls in file_map.classes:
            if cls.inherited_by:
                inheritance_lines.append(f"  {cls.name} ({file_map.path}:{cls.lineno}) inherited by: {', '.join(cls.inherited_by)}")
            if cls.bases:
                inheritance_lines.append(f"  {cls.name} ({file_map.path}:{cls.lineno}) inherits from: {', '.join(cls.bases)}")
    
    if len(inheritance_lines) > 1:  # More than just the header
        docs.append(Document(
            page_content="\n".join(inheritance_lines), 
            metadata={"source": "__inheritance_graph__", "type": "inheritance-graph-summary"}
        ))

    # Optional: add a coarse import-graph summary to help with cross-file questions
    if nx is not None:
        G = nx.DiGraph()
        for m in file_maps:
            mod = m.path.replace(os.sep, ".").rstrip(".py")
            G.add_node(mod, loc=m.loc)
            for imp in m.imports:
                if imp and ("/" in imp or "." in imp):
                    # attempt to map to local module name (best-effort)
                    tgt = imp.replace("/", ".")
                    G.add_edge(mod, tgt)
        # Compute simple centrality to identify hotspots
        try:
            cent = nx.pagerank(G)
        except Exception:
            cent = {n: 0.0 for n in G.nodes}
        # Serialize top hubs
        hubs = sorted(cent.items(), key=lambda kv: kv[1], reverse=True)[:30]
        lines = ["IMPORT GRAPH HUBS:"] + [f"  - {n} (score={s:.4f})" for n, s in hubs]
        docs.append(Document(page_content="\n".join(lines), metadata={"source": "__graph__", "type": "graph-summary"}))

    # BM25 retriever (in-memory)
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = bm25_k

    # Создаем FAISS индекс с GPU поддержкой если доступно
    faiss = FAISS.from_documents(docs, embeddings, distance_strategy=DistanceStrategy.COSINE)
    
    # Переносим FAISS индекс на GPU если доступно
    if GPU_AVAILABLE:
        try:
            import faiss as faiss_lib
            if hasattr(faiss_lib, 'StandardGpuResources'):
                print("🚀 Переносим FAISS индекс на GPU...")
                # Получаем размерность векторов
                dimension = len(embeddings.embed_query("test"))
                
                # Создаем GPU ресурсы
                res = faiss_lib.StandardGpuResources()
                
                # Конфигурируем GPU индекс
                gpu_config = faiss_lib.GpuIndexFlatConfig()
                gpu_config.device = 0  # Используем первую GPU
                
                # Переносим на GPU (если индекс поддерживает)
                try:
                    cpu_index = faiss.index
                    gpu_index = faiss_lib.index_cpu_to_gpu(res, 0, cpu_index)
                    faiss.index = gpu_index
                    print("✅ FAISS индекс успешно перенесен на GPU")
                except Exception as gpu_error:
                    print(f"⚠️  Не удалось перенести FAISS на GPU: {gpu_error}")
                    print("📝 Используем CPU версию FAISS")
        except ImportError:
            print("⚠️  faiss-gpu не найден, используем CPU версию")
    
    dense_retriever = faiss.as_retriever(search_kwargs={"k": dense_k})

    # Hybrid ensemble
    ensemble = EnsembleRetriever(retrievers=[bm25, dense_retriever], weights=list(ensemble_weights))

    # Contextual compression (optional but recommended)
    if llm is not None:
        compressor = LLMChainExtractor.from_llm(llm)  # extracts only query-relevant spans
        retriever: Any = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=ensemble)
    else:
        retriever = ensemble

    rel_files = [m.path for m in file_maps]
    return RepoIndex(root=project_path, docs=docs, bm25=bm25, vect=faiss, retriever=retriever, files=rel_files)


# -----------------------------------------------------------------------------
# Evidence planning: ask the LLM which bodies/lines to fetch
# -----------------------------------------------------------------------------

EVIDENCE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
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
"""),
    ("human", "Question:\n{question}\n\nContext (map snippets):\n{context}\n")
])

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """
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
"""),
    ("human", "Question:\n{question}\n\nRepo map (summaries):\n{map_text}\n\nEvidence (code bodies):\n{evidence_text}\n")
])


def _trim_to_chars(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = limit // 2
    tail = limit - head
    return text[:head] + "\n...\n" + text[-tail:]


def _gather_map_snippets(docs: List[Document], max_chars: int = 20000) -> str:
    # Prioritize inheritance graph, call graph, file maps and graph summary
    pieces = []
    for d in docs:
        if d.metadata.get("type") in ("inheritance-graph-summary", "call-graph-summary", "py-map", "graph-summary"):
            pieces.append(f"---\n{d.page_content}\n")
        if sum(len(p) for p in pieces) > max_chars:
            break
    return _trim_to_chars("\n".join(pieces), max_chars)


def _find_symbol_span(source: str, node: ast.AST) -> Tuple[int, int]:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    # expand decorators upward if present
    if hasattr(node, "decorator_list") and node.decorator_list:
        dec0 = node.decorator_list[0]
        start = min(start, getattr(dec0, "lineno", start))
    return start, end


def _coerce_repo_relative_path(root: str, candidate: str, known_rel_paths: List[str]) -> str:
    """Return repo-relative path for various candidate formats.
    Handles absolute paths inside repo, mixed slashes, ./ prefixes, and bare basenames.
    If ambiguous, prefers shortest path depth.
    Returns empty string if cannot resolve reliably.
    """
    s = (candidate or "").strip().strip('"').strip("'")
    if not s:
        return ""
    s = s.replace("\\", os.sep).replace("//", "/")
    s = s.lstrip("./")
    # Make relative if absolute under root
    try:
        if os.path.isabs(s):
            s = os.path.relpath(os.path.normpath(s), root)
    except Exception:
        pass
    s = os.path.normpath(s)

    # Direct hit
    if s in known_rel_paths:
        return s

    # Try endswith match (e.g., src/pkg/file.py -> pkg/file.py)
    ends = [p for p in known_rel_paths if p.endswith(s)]
    if len(ends) == 1:
        return ends[0]

    # Try basename match
    base = os.path.basename(s)
    cands = [p for p in known_rel_paths if os.path.basename(p) == base]
    if len(cands) == 1:
        return cands[0]

    pool = ends or cands
    if pool:
        # pick the shallowest path as a heuristic
        return sorted(pool, key=lambda x: (x.count(os.sep), len(x)))[0]

    # Last resort: file physically exists
    if os.path.isfile(os.path.join(root, s)):
        return s

    return ""


def _normalize_plan_paths(plan: List[Dict[str, str]], root: str, known_rel_paths: List[str]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in plan:
        rel = _coerce_repo_relative_path(root, item.get("file", ""), known_rel_paths)
        if rel:
            item["file"] = rel
            normalized.append(item)
    return normalized


def _extract_bodies(root: str, requests: List[Dict[str, str]]) -> List[Tuple[str, str]]:
    """Return list of (label, code_block) for requested {file,symbol}."""
    out: List[Tuple[str, str]] = []
    for req in requests:
        rel = req.get("file", "")
        sym = req.get("symbol", "")
        target = os.path.join(root, rel)
        if not os.path.isfile(target):
            continue
        try:
            src = _read_text(target)
            tree = ast.parse(src)
        except Exception:
            continue

        wanted_method: Optional[Tuple[str, Optional[str]]] = None
        if "." in sym:
            cls, meth = sym.split(".", 1)
            wanted_method = (cls, meth)
        else:
            wanted_method = (None, sym)

        found = False

        class F(ast.NodeVisitor):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # type: ignore[override]
                nonlocal found
                if wanted_method and wanted_method[0] is None and node.name == wanted_method[1]:
                    s, e = _find_symbol_span(src, node)
                    code = _safe_get_lines(src, s, e)
                    out.append((f"{rel}:{s}", code))
                    found = True
                self.generic_visit(node)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # type: ignore[override]
                self.visit_FunctionDef(node)  # same handling

            def visit_ClassDef(self, node: ast.ClassDef) -> None:  # type: ignore[override]
                nonlocal found
                if wanted_method and wanted_method[0] == node.name:
                    # scan methods
                    for ch in node.body:
                        if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef)) and ch.name == wanted_method[1]:
                            s, e = _find_symbol_span(src, ch)
                            code = _safe_get_lines(src, s, e)
                            out.append((f"{rel}:{s}", code))
                            found = True
                            break
                self.generic_visit(node)

        F().visit(tree)
        if not found and sym == "*":  # allow wildcard file fetch
            out.append((f"{rel}:1", src))

    return out


# -----------------------------------------------------------------------------
# Main tool factory
# -----------------------------------------------------------------------------

def make_arch_review_tool(
        project_path: str,
        llm: BaseChatModel,
        bm25_k: int = 12,
        dense_k: int = 12,
        ensemble_weights: Tuple[float, float] = (0.5, 0.5),
        map_char_budget: int = 24000,
        evidence_char_budget: int = 20000,
        max_evidence_items: int = 8,
        answer_language: str = 'ru',
        embeddings: Embeddings = None,
        use_gpu: bool = None
) -> StructuredTool:
    """Create a LangChain tool that performs architecture-focused Q&A over a Python repo.

    Args:
        project_path: path to the Python project (repo root)
        llm: a LangChain chat model used for compression/planning/answering
        bm25_k / dense_k: candidates from each retriever
        ensemble_weights: weights for [BM25, dense] in EnsembleRetriever
        map_char_budget: max chars of repo map passed to the planner/answer prompts
        evidence_char_budget: max chars of fetched bodies
        max_evidence_items: planner budget for how many code bodies to fetch
        answer_language: answer language
        embeddings: embeddings model (если None, создается автоматически)
        use_gpu: принудительно использовать GPU (None = автоопределение)

    Returns:
        StructuredTool ready to plug into an agent. Input schema: {"question": str}
    """
    
    # Создаем эмбеддинги по умолчанию если не переданы
    if embeddings is None:
        if use_gpu is None:
            use_gpu = GPU_AVAILABLE
        
        model_name = "BAAI/bge-code-v1" if use_gpu else "BAAI/bge-small-en-v1.5"
        embeddings = load_model(model_name, use_gpu=use_gpu)
    index = build_repo_index(
        project_path=project_path,
        bm25_k=bm25_k,
        dense_k=dense_k,
        ensemble_weights=ensemble_weights,
        llm=llm,
        embeddings=embeddings
    )

    def _run(question: str) -> str:
        # 1) Retrieve compact map snippets relevant to the question
        retrieved = index.retriever.invoke(question)
        map_text = _gather_map_snippets(retrieved, max_chars=map_char_budget)

        # 2) Ask for an evidence plan (which bodies to fetch)
        evidence_chain = EVIDENCE_PROMPT | llm | StrOutputParser()
        raw_plan = evidence_chain.invoke({
            "question": question,
            "context": map_text,
            "max_items": max_evidence_items,
        })
        # Tolerate malformed JSON slightly
        plan_json = raw_plan.strip()
        plan_json = plan_json[plan_json.find("[") : plan_json.rfind("]") + 1] if "[" in plan_json and "]" in plan_json else "[]"
        try:
            plan = json.loads(plan_json)
            if not isinstance(plan, list):
                plan = []
        except Exception:
            plan = []

        # 2.5) Normalize planner paths to repo-relative
        plan = _normalize_plan_paths(plan, index.root, index.files) if plan else []

        # 3) Fetch requested bodies
        evidence_pairs = _extract_bodies(index.root, plan[:max_evidence_items]) if plan else []
        evidence_text = "\n\n".join([f"### {lbl}\n" + code for (lbl, code) in evidence_pairs])
        evidence_text = _trim_to_chars(evidence_text, evidence_char_budget)

        # 4) Final answer using map + evidence
        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        final = answer_chain.invoke({
            "question": question,
            "map_text": map_text,
            "answer_language": answer_language,
            "evidence_text": evidence_text if evidence_text else "(no additional bodies requested)",
        })
        return final

    return StructuredTool.from_function(
        func=_run,
        name="architectural_code_review",
        description=(
            "Answer architecture & design questions about a Python repository. "
            "Uses a compact repo map (signatures/imports) + hybrid retrieval + targeted body fetch by evidence plan. "
            "Input: a natural-language question about the codebase."
        ),
    )

llm = YandexGPT(
    iam_token=os.getenv('YANDEX_GPT_API_KEY'),
    folder_id=os.getenv('YANDEX_GPT_FOLDER_ID'),
    model_name="yandexgpt")

def load_model(model_name, local_dir="./models", wrapper_cls = HuggingFaceEmbeddings, use_gpu=None):
    """
    Загружает модель эмбеддингов с поддержкой GPU.
    
    Args:
        model_name: Имя модели
        local_dir: Локальная директория для кеширования
        wrapper_cls: Класс обертки для модели
        use_gpu: Принудительно использовать GPU (None = автоопределение)
    """
    if use_gpu is None:
        use_gpu = GPU_AVAILABLE
    
    local_path = Path(local_dir) / model_name.replace("/", "_")
    
    # Конфигурация для GPU
    model_kwargs = {}
    encode_kwargs = {}
    
    if use_gpu and GPU_AVAILABLE:
        model_kwargs.update({
            'device': DEVICE,
            'trust_remote_code': True,
        })
        encode_kwargs.update({
            'batch_size': 32,  # Увеличиваем batch_size для GPU
        })
        print(f"🔥 Загружаем {model_name} на GPU")
    else:
        model_kwargs.update({
            'device': 'cpu',
        })
        encode_kwargs.update({
            'batch_size': 8,  # Меньший batch_size для CPU
        })
        print(f"🐌 Загружаем {model_name} на CPU")
    
    if local_path.exists():
        return wrapper_cls(
            model_name=str(local_path),
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    else:
        emb = wrapper_cls(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        try:
            # Сохраняем модель локально
            if hasattr(emb, 'client'):
                emb.client.save(str(local_path))
            elif hasattr(emb, '_client'):
                emb._client.save(str(local_path))
        except Exception as e:
            print(f"⚠️  Не удалось сохранить модель локально: {e}")
        return emb

from langchain_community.embeddings import HuggingFaceBgeEmbeddings

emb_llm_encoder = load_model(
    model_name="BAAI/llm-embedder",
    wrapper_cls=HuggingFaceBgeEmbeddings
)

# emb_bge_code создается автоматически в make_arch_review_tool

tool_llm_encoder = make_arch_review_tool(
    project_path=os.getenv('TEST_PROJ_PATH'),
    llm=llm,
    embeddings=emb_llm_encoder
)

# Создаем инструмент с автоматическим определением GPU/CPU
print("🔧 Создаем RAG инструмент...")
tool_bge_code = make_arch_review_tool(
    project_path=os.getenv('TEST_PROJ_PATH'),
    llm=llm,
    use_gpu=True  # Принудительно используем GPU если доступно
)

# # Тестируем улучшенную систему на примере поиска вызовов функций
# result = tool_bge_code.invoke("Какие функции вызывают метод _render_pages_cached?")
# print("=== Результат поиска вызовов _render_pages_cached ===")
# print(result)

# print("\n" + "="*50 + "\n")

# # Дополнительный тест для понимания архитектуры
# print('123')
# result2 = tool_bge_code.invoke("Как используется класс PrefixedDBModel в проекте?")
# print("=== Результат поиска API эндпоинтов ===")
# print(result2)
# print("\n\n")


# print('123')
# result2 = tool_llm_encoder.invoke("Как используется класс PrefixedDBModel в проекте?")
# print("=== Результат поиска API эндпоинтов ===")
# print(result2)