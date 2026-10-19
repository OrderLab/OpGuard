# opguard/instrumentor/op_wrapper/ast_index.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Iterable
import ast

from .matchers import Tag, TagCategory

_MEANINGLESS = {
    "self", "cls", "true", "false", "none", "_"
}

def _norm(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = str(name).strip().lower()
    if not s or s.isdigit() or s in _MEANINGLESS:
        return None
    return s

def _names_in_expr(node: ast.AST) -> List[str]:
    """Collect meaningful identifier tokens from an expression."""
    out: List[str] = []
    def add(n: Optional[str]):
        n = _norm(n)
        if n and n not in out:
            out.append(n)

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name):
            add(n.id)
        def visit_Attribute(self, n: ast.Attribute):
            # include attribute token and descend into value
            add(n.attr)
            self.visit(n.value)
        def visit_Subscript(self, n: ast.Subscript):
            self.visit(n.value)
            # slice may be tuple, name, etc.
            try:
                self.visit(n.slice)  # py3.9+: expr
            except Exception:
                pass
        # containers
        def visit_Tuple(self, n: ast.Tuple):
            for e in n.elts: self.visit(e)
        def visit_List(self, n: ast.List):
            for e in n.elts: self.visit(e)
        def visit_Dict(self, n: ast.Dict):
            for e in n.keys + n.values:
                if e is not None:
                    self.visit(e)
        def generic_visit(self, n: ast.AST):
            super().generic_visit(n)

    try:
        V().visit(node)
    except Exception:
        pass
    # filter meaningless again
    return [t for t in out if t not in _MEANINGLESS]

def _lhs_names_from_target(tgt: ast.AST) -> List[str]:
    names: List[str] = []
    def add(n: Optional[str]):
        n = _norm(n)
        if n and n not in names:
            names.append(n)

    class V(ast.NodeVisitor):
        def visit_Name(self, n: ast.Name):
            add(n.id)
        def visit_Attribute(self, n: ast.Attribute):
            # attribute assignment like obj.x = ...
            add(n.attr)
            self.visit(n.value)
        def visit_Tuple(self, n: ast.Tuple):
            for e in n.elts: self.visit(e)
        def visit_List(self, n: ast.List):
            for e in n.elts: self.visit(e)

    V().visit(tgt)
    return names

def _lhs_names(assign: ast.AST) -> List[str]:
    if isinstance(assign, ast.Assign):
        out: List[str] = []
        for t in assign.targets:
            out.extend(_lhs_names_from_target(t))
        return out
    if isinstance(assign, ast.AugAssign):
        return _lhs_names_from_target(assign.target)
    if isinstance(assign, ast.AnnAssign):
        return _lhs_names_from_target(assign.target) if assign.target is not None else []
    return []

def _span(n: ast.AST) -> Tuple[int, int, int, int]:
    # (start_line, start_col, end_line, end_col)
    sl = getattr(n, "lineno", -1)
    sc = getattr(n, "col_offset", -1)
    el = getattr(n, "end_lineno", sl)
    ec = getattr(n, "end_col_offset", sc)
    return sl, sc, el, ec

class ASTCallIndex(ast.NodeVisitor):
    """
    Index all ast.Call nodes with their tags and precise spans.
    Provides lookup by (line, col) and containment.
    """
    def __init__(self, source: str):
        self._source = source
        self._mod = ast.parse(source)
        # records: list of dict with keys: node, span, tags
        self._recs: List[Dict] = []
        # fast index by exact (sl, sc)
        self._by_pos: Dict[Tuple[int, int], Dict] = {}
        # stack of current assign-like LHS tags
        self._lhs_stack: List[List[str]] = []

        self.visit(self._mod)

    # --------- visitor ---------
    def visit_Assign(self, node: ast.Assign):
        self._lhs_stack.append(_lhs_names(node))
        self.generic_visit(node)
        self._lhs_stack.pop()

    def visit_AugAssign(self, node: ast.AugAssign):
        self._lhs_stack.append(_lhs_names(node))
        self.generic_visit(node)
        self._lhs_stack.pop()

    def visit_AnnAssign(self, node: ast.AnnAssign):
        self._lhs_stack.append(_lhs_names(node))
        self.generic_visit(node)
        self._lhs_stack.pop()

    def visit_Call(self, node: ast.Call):
        rec = {"node": node, "span": _span(node), "tags": []}  # tags: List[Tag]

        # receiver_subexpr
        if isinstance(node.func, ast.Attribute):
            for nm in _names_in_expr(node.func.value):
                rec["tags"].append(Tag(nm, TagCategory.RECEIVER_SUBEXPR))
        elif isinstance(node.func, ast.Subscript):
            # Triton style: obj[grid](...)
            for nm in _names_in_expr(node.func.value):
                rec["tags"].append(Tag(nm, TagCategory.RECEIVER_SUBEXPR))
            try:
                for nm in _names_in_expr(node.func.slice):
                    rec["tags"].append(Tag(nm, TagCategory.GRID))
            except Exception:
                pass

        # args / kwargs
        for a in node.args:
            for nm in _names_in_expr(a):
                rec["tags"].append(Tag(nm, TagCategory.ARG))
        for kw in node.keywords or []:
            if kw.value is not None:
                for nm in _names_in_expr(kw.value):
                    rec["tags"].append(Tag(nm, TagCategory.KWARG))

        # LHS (from current assign context; includes comprehensions under Assign)
        if self._lhs_stack:
            for nm in self._lhs_stack[-1]:
                rec["tags"].append(Tag(nm, TagCategory.LHS))

        # dedup (name, category)
        seen = set()
        dedup_tags: List[Tag] = []
        for t in rec["tags"]:
            k = (t.name, t.category)
            if k not in seen:
                seen.add(k)
                dedup_tags.append(t)
        rec["tags"] = dedup_tags

        self._recs.append(rec)
        sl, sc, _, _ = rec["span"]
        self._by_pos.setdefault((sl, sc), rec)
        # continue traversal
        self.generic_visit(node)

    # --------- public API ---------
    def tags_at(self, line: int, col0: int) -> List[Tag]:
        """
        Retrieve tags for the call that contains (line, col0).
        Prefer exact (start_line, start_col), otherwise choose the innermost
        call span that contains the position. Returns [] if none.
        """
        # exact
        rec = self._by_pos.get((line, col0))
        if rec:
            return rec["tags"]

        # containment
        best: Optional[Dict] = None
        for r in self._recs:
            sl, sc, el, ec = r["span"]
            if (line < sl) or (line > el):
                continue
            if line == sl and col0 < sc:
                continue
            if line == el and col0 >= ec:
                continue
            # choose the smallest span that contains the point
            if best is None:
                best = r
            else:
                bsl, bsc, bel, bec = best["span"]
                if (el - sl) < (bel - bsl) or ((el == sl) and (ec - sc) < (bec - bsc)):
                    best = r
        return best["tags"] if best else []
