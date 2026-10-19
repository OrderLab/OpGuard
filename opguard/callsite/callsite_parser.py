# opguard/callsite/callsite\_parser.py

from __future__ import annotations
import ast
import textwrap
from typing import Optional, Tuple, List, Iterable

from opguard.common_vars import OPGUARD_MULTI_OPS_PER_LINE
from .model import CallsiteInfo
from .tag_extractor import TagExtractor
from .ast_utils import (
    call_priority_kind_name,
    is_triton_kernel_call,
    dotted_name_from_ast,
)

class CallsiteParser:
    def __init__(self):
        self.file_cache = {}
        self.ctx_cache = {}
        self.fn_cache = {}
        self.extractor = TagExtractor()

    # ------------- file + stmt utilities -------------
    def parse_file(self, file_path: str) -> Optional[ast.AST]:
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            tree = ast.parse(source)
            self.file_cache[file_path] = (tree, source)
            return tree
        except Exception:
            return None

    def _module_and_source(self, file_path: str):
        tup = self.file_cache.get(file_path)
        if tup is not None:
            return tup[0], tup[1]
        self.parse_file(file_path)
        return self.file_cache.get(file_path, (None, None))

    def _smallest_stmt_spanning(self, tree: ast.AST, line: int) -> Optional[ast.stmt]:
        best = None
        best_span = None
        for node in ast.walk(tree):
            if isinstance(node, ast.stmt) and hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                if node.lineno <= line <= node.end_lineno:
                    span = (node.end_lineno - node.lineno, node.lineno, node.end_lineno)
                    if best_span is None or span < best_span:
                        best_span = span
                        best = node
        return best

    def _extract_stmt_text(self, source_text: str, node: ast.stmt) -> str:
        lines = source_text.splitlines(keepends=True)
        snippet = "".join(lines[node.lineno - 1 : node.end_lineno])
        return textwrap.dedent(snippet)

    # ------------- public API -------------
    def context_for_line(self, file_path: str, line_number: int):
        key = (file_path, line_number)
        if key in self.ctx_cache:
            return self.ctx_cache[key]
        tree, _src = self._module_and_source(file_path)
        if not tree:
            self.ctx_cache[key] = (None, None)
            return (None, None)
        cls_stack = []
        fn_stack = []
        for node in ast.walk(tree):
            if not hasattr(node, 'lineno') or not hasattr(node, 'end_lineno'):
                continue
            if node.lineno <= line_number <= node.end_lineno:
                if isinstance(node, ast.ClassDef):
                    cls_stack.append((node.lineno, node.end_lineno, node.name))
                elif isinstance(node, ast.FunctionDef):
                    fn_stack.append((node.lineno, node.end_lineno, node.name))
        cls = max(cls_stack, key=lambda x: x[0])[2] if cls_stack else None
        fn  = max(fn_stack,  key=lambda x: x[0])[2] if fn_stack  else None
        self.ctx_cache[key] = (cls, fn)
        return (cls, fn)

    def _build_info(self, kind: str, name: str, node: ast.AST,
                    stmt_root: ast.AST, source_text: str) -> CallsiteInfo:
        ln   = getattr(node, 'lineno', None)
        eln  = getattr(node, 'end_lineno', ln)
        col  = getattr(node, 'col_offset', 0)
        ecol = getattr(node, 'end_col_offset', col)

        # operation type
        op_type = {
            "triton_kernel": "triton_kernel",
            "method": "method",
            "function": "function",
            "augassign": "augmented_operator",
            "compare": "comparison",
            "binop": "operator",
            "setitem": "advanced_indexing_write",
        }.get(kind, "unknown")

        # build CallsiteInfo
        info = CallsiteInfo(function=name, operation_type=op_type, tags=[])

        # positions
        info.lineno = ln or 0
        info.end_lineno = eln or ln or 0
        info.col_offset = col
        info.end_col_offset = ecol

        # good snippet from ORIGINAL source
        try:
            snippet = ast.get_source_segment(source_text, node) or ""
        except Exception:
            snippet = ""
        info.snippet = snippet.strip()[:160]

        # tags (use stmt_root for context where needed)
        if kind == "triton_kernel":
            info.tags = self.extractor.for_triton_kernel(node)
        elif kind == "method":
            info.tags = self.extractor.for_method_call(stmt_root, node)
        elif kind == "function":
            info.tags = self.extractor.for_function_call(stmt_root, node)
        elif kind == "augassign":
            info.tags = self.extractor.for_augassign(node)
        elif kind == "compare":
            info.tags = self.extractor.for_compare(node)
        elif kind == "binop":
            info.tags = self.extractor.for_binop(node)
        elif kind == "setitem":
            info.tags = self.extractor.for_subscript_write(node, None)

        return info
    
    def _get_source_segment(self, source_text: str, node: ast.AST) -> str:
        # Python 3.8+ has end_lineno/end_col_offset; ast.get_source_segment handles tabs etc.
        try:
            seg = ast.get_source_segment(source_text, node)
            if seg:
                return seg.strip()
        except Exception:
            pass
        # Fallback: manual slice
        try:
            ln  = getattr(node, "lineno", None)
            eln = getattr(node, "end_lineno", None)
            col = getattr(node, "col_offset", 0)
            ecol= getattr(node, "end_col_offset", 0)
            if ln is None or eln is None:
                return ""
            lines = source_text.splitlines(keepends=True)
            if ln == eln:
                line = lines[ln - 1]
                return line[col:ecol].strip() or line.strip()
            snippet = "".join(lines[ln - 1:eln])
            # If we have valid columns, trim first/last line
            if col or ecol:
                first = lines[ln - 1][col:]
                last  = lines[eln - 1][:ecol] if ecol else lines[eln - 1]
                mid = lines[ln:eln - 1]
                snippet = "".join([first, *mid, last])
            return snippet.strip()
        except Exception:
            return ""

    def _iter_instrumentable_nodes(self, root: ast.AST) -> Iterable[Tuple[str, str, ast.AST]]:
        """
        Yield (kind, name, node) for ALL instrumentable ops contained in 'root'.
        IMPORTANT: we walk ORIGINAL NODES (no reparse), so positions map to file.
        """
        for n in ast.walk(root):
            if isinstance(n, ast.Call):
                prio, kind, name = call_priority_kind_name(n)  # your existing logic
                if kind not in ("triton_kernel", "method", "function"):
                    kind = "function"
                yield kind, name, n
            elif isinstance(n, ast.AugAssign):
                yield "augassign", "augassign", n
            elif isinstance(n, ast.Compare) and getattr(n, "ops", None):
                yield "compare", "compare", n
            elif isinstance(n, ast.BinOp):
                yield "binop", "binop", n
            elif isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Subscript):
                        yield "setitem", "setitem", t
            elif isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Subscript):
                yield "setitem", "setitem", n.target

    def analyze_line_multi(self, file_path: str, line_number: int):
        """Return ALL instrumentable ops on the line, using ORIGINAL nodes."""
        results = []
        tree, source_text = self._module_and_source(file_path)
        if not tree:
            return results

        # In FILE coordinates
        stmt_node = self._smallest_stmt_spanning(tree, line_number)
        if stmt_node is None:
            return results

        for kind, name, node in self._iter_instrumentable_nodes(stmt_node):
            # ✅ FIX: pass stmt_node so tag extractor can see full context
            info = self._build_info(kind, name, node, stmt_node, source_text)
            results.append(info)

        # Deterministic ordering for reproducibility (priority, then position)
        def rank_key(ci):
            kind_rank = {
                "triton_kernel": 0, "method": 1, "function": 2,
                "augmented_operator": 3, "comparison": 4,
                "operator": 5, "advanced_indexing_write": 6, "unknown": 7
            }.get(ci.operation_type, 9)
            # be robust if any fields are missing
            ln  = getattr(ci, "lineno", 0)
            eln = getattr(ci, "end_lineno", ln)
            co  = getattr(ci, "col_offset", 0)
            eco = getattr(ci, "end_col_offset", co)
            span_lines = max(0, eln - ln)
            span_cols  = max(0, eco - co)
            return (kind_rank, -span_lines, -span_cols, ln, co)

        results.sort(key=rank_key)
        return results

    def analyze_line(self, file_path: str, line_number: int) -> Optional[CallsiteInfo]:
        """Back-compat: return the top-ranked item only."""
        items = self.analyze_line_multi(file_path, line_number)
        if items:
            if OPGUARD_MULTI_OPS_PER_LINE:
                return items
            else:
                return [items[0]]
        return None
