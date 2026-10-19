
# opguard/callsite/tag\_extractor.py

from __future__ import annotations
import ast
import os
from typing import List, Optional, Set
from .model import Tag, TagCategory

class TagExtractor:
    DEFAULT_MEANINGLESS = {
        # single letters, params, boilerplate
        'i','j','k','x','y','z','n','m','p','q','r','s','t','u','v','w',
        'a','b','c','d','e','f','g','h','l','o',
        'self','cls','args','kwargs','_',
        'true','false','none','if','for','while','def','class','import','from',
        'return','print','assert','raise','try','except','finally','with','as',
        'in','is','and','or','not',
        # framework roots
        'torch','triton','np','numpy','tf','jax','cupy','cuda','math',
    }

    DEFAULT_VERBS = {
        'view','reshape','contiguous','detach','cpu','cuda','numpy','item','tolist',
        'sum','mean','min','max','argmax','argmin','clip','astype','type','to',
        'flatten','squeeze','unsqueeze','permute','transpose','expand','repeat',
        'round','floor','ceil',
    }

    def __init__(self,
                 meaningless: Optional[Set[str]] = None,
                 verbs: Optional[Set[str]] = None,
                 max_tags: Optional[int] = None,
                 match_mode: Optional[str] = None):
        self.meaningless = set(n.lower() for n in (meaningless or self.DEFAULT_MEANINGLESS))
        self.verbs = set(n.lower() for n in (verbs or self.DEFAULT_VERBS))
        self.max_tags = int(os.getenv("OPGUARD_TAG_MAX", str(max_tags or 20)))
        self.match_mode = (os.getenv("OPGUARD_TAGS_MATCH_MODE", match_mode or "any_subexpr_plus_lhs")
                           .strip().lower())
        self.dedup_mode = os.getenv("OPGUARD_TAG_DEDUP_MODE", "name").strip().lower()  # "name" | "pair"

        # Add category priority (LHS first)
        from .model import TagCategory
        self.category_priority = [
            TagCategory.LHS,
            TagCategory.RECEIVER_SUBEXPR, TagCategory.RECEIVER,
            TagCategory.ARG, TagCategory.KWARG, TagCategory.KW_VALUE,
            TagCategory.INDEX, TagCategory.CONTAINER,
            TagCategory.OPERAND_LEFT, TagCategory.OPERAND_RIGHT,
            TagCategory.COMPARE_LEFT, TagCategory.COMPARE_RIGHT,
            TagCategory.VALUE, TagCategory.GRID,
        ]

    # ---------- core utils ----------
    def _norm(self, s: str) -> Optional[str]:
        s = (s or "").strip().lower()
        if not s or s.isdigit() or s in self.meaningless:
            return None
        return s

    def _dedup(self, items: List[Tag]) -> List[Tag]:
        prio = {cat: i for i, cat in enumerate(self.category_priority)}
        # stable-sort by priority; preserve relative order within a category
        items_sorted = sorted(range(len(items)), key=lambda i: prio.get(items[i].category, 999))
        out, seen = [], set()
        for idx in items_sorted:
            t = items[idx]
            key = t.name if self.dedup_mode == "name" else (t.name, t.category)
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
            if len(out) >= self.max_tags:
                break
        return out

    def _names_in_expr(self, node: ast.AST, category: TagCategory) -> List[Tag]:
        out: List[Tag] = []
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                nm = self._norm(sub.id)
                if nm:
                    out.append(Tag(nm, category))
            elif isinstance(sub, ast.Attribute):
                # include attribute tokens like self_attention, o_norm, weight, data, model_config, ...
                nm = self._norm(sub.attr)
                if nm and nm not in self.verbs:
                    out.append(Tag(nm, category))
        return out

    def _lhs_target_names(self, target: ast.AST) -> List[Tag]:
        if isinstance(target, ast.Name):
            nm = self._norm(target.id)
            return [Tag(nm, TagCategory.LHS)] if nm else []
        if isinstance(target, ast.Tuple):
            out: List[Tag] = []
            for e in target.elts:
                out.extend(self._lhs_target_names(e))
            return out
        if isinstance(target, ast.Subscript):
            return self._container_names(target.value)
        if isinstance(target, ast.Attribute):
            nm = self._norm(target.attr)
            if nm and nm not in self.verbs:
                return [Tag(nm, TagCategory.LHS)]
            return self._container_names(target.value)
        return []

    def _container_names(self, expr: ast.AST) -> List[Tag]:
        while isinstance(expr, (ast.Call, ast.Subscript, ast.Attribute)):
            if isinstance(expr, ast.Attribute):
                nm = self._norm(expr.attr)
                if nm and nm not in self.verbs:
                    return [Tag(nm, TagCategory.CONTAINER)]
                expr = expr.value
            elif isinstance(expr, ast.Subscript):
                expr = expr.value
            elif isinstance(expr, ast.Call):
                expr = expr.func
        if isinstance(expr, ast.Name):
            nm = self._norm(expr.id)
            return [Tag(nm, TagCategory.CONTAINER)] if nm else []
        try:
            import ast as _ast
            s = _ast.unparse(expr)
            nm = self._norm(s)
            return [Tag(nm, TagCategory.CONTAINER)] if nm else []
        except Exception:
            return []

    def _index_var_names(self, idx: ast.AST) -> List[Tag]:
        out: List[Tag] = []
        for sub in ast.walk(idx):
            if isinstance(sub, ast.Name):
                nm = self._norm(sub.id)
                if nm:
                    out.append(Tag(nm, TagCategory.INDEX))
        return out

    # ---------- per-kind extractors ----------
    def for_method_call(self, stmt_tree: ast.AST, call: ast.Call) -> List[Tag]:
        tags: List[Tag] = []
        if isinstance(call.func, ast.Attribute):
            base = call.func.value
            if self.match_mode in ("any_subexpr", "any_subexpr_plus_lhs"):
                tags.extend(self._names_in_expr(base, TagCategory.RECEIVER_SUBEXPR))
            else:
                while isinstance(base, (ast.Call, ast.Attribute, ast.Subscript)):
                    base = base.func if isinstance(base, ast.Call) else base.value
                if isinstance(base, ast.Name):
                    nm = self._norm(base.id)
                    if nm:
                        tags.append(Tag(nm, TagCategory.RECEIVER))
        # args
        for a in call.args:
            tags.extend(self._names_in_expr(a, TagCategory.ARG))
        for kw in call.keywords or []:
            if kw.value is not None:
                tags.extend(self._names_in_expr(kw.value, TagCategory.KWARG))
        # lhs
        if self.match_mode == "any_subexpr_plus_lhs":
            for n in ast.walk(stmt_tree):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        tags.extend(self._lhs_target_names(t))
                elif isinstance(n, ast.AugAssign):
                    tags.extend(self._lhs_target_names(n.target))
        return self._dedup(tags)

    def for_function_call(self, stmt_tree: ast.AST, call: ast.Call) -> List[Tag]:
        tags: List[Tag] = []
        for a in call.args:
            tags.extend(self._names_in_expr(a, TagCategory.ARG))
        for kw in call.keywords or []:
            if kw.value is not None:
                tags.extend(self._names_in_expr(kw.value, TagCategory.KWARG))
        if self.match_mode != "object_only":
            for n in ast.walk(stmt_tree):
                if isinstance(n, ast.Assign):
                    for t in n.targets:
                        tags.extend(self._lhs_target_names(t))
                elif isinstance(n, ast.AugAssign):
                    tags.extend(self._lhs_target_names(n.target))
        return self._dedup(tags)

    def for_triton_kernel(self, call: ast.Call) -> List[Tag]:
        tags: List[Tag] = []
        if isinstance(call.func, ast.Subscript):
            tags.extend(self._index_var_names(call.func.slice))  # grid
            # mark grid tags category explicitly
            tags = [Tag(t.name, TagCategory.GRID if t.category == TagCategory.INDEX else t.category) for t in tags]
        for kw in call.keywords or []:
            if kw.value is not None:
                tags.extend(self._names_in_expr(kw.value, TagCategory.KW_VALUE))
        return self._dedup(tags)

    def for_augassign(self, node: ast.AugAssign) -> List[Tag]:
        return self._dedup(self._lhs_target_names(node.target) + self._names_in_expr(node.value, TagCategory.VALUE))

    def for_binop(self, node: ast.BinOp) -> List[Tag]:
        return self._dedup(self._names_in_expr(node.left, TagCategory.OPERAND_LEFT) +
                           self._names_in_expr(node.right, TagCategory.OPERAND_RIGHT))

    def for_compare(self, node: ast.Compare) -> List[Tag]:
        left = self._names_in_expr(node.left, TagCategory.COMPARE_LEFT)
        right: List[Tag] = []
        if node.comparators:
            right = self._names_in_expr(node.comparators[0], TagCategory.COMPARE_RIGHT)
        return self._dedup(left + right)

    def for_subscript_write(self, sub: ast.Subscript, value: Optional[ast.AST] = None) -> List[Tag]:
        base = self._container_names(sub.value)
        idxs = self._index_var_names(sub.slice)
        extra: List[Tag] = self._names_in_expr(value, TagCategory.VALUE) if value is not None else []
        return self._dedup(base + idxs + extra)
