# opguard/instrumentor/op_wrapper/instrumentor.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import libcst as cst
from libcst import matchers as m
from libcst.metadata import PositionProvider, ParentNodeProvider, MetadataWrapper

from .cst_utils import (
    OP_CLASS_TO_SYMBOL, SYMBOL_TO_NAME,
    expr_to_dotted_name, key_from_subscript, extract_name,
    make_safe_identifier, canonical_expr_text, get_make_wrapper_params,
)
from .matchers import Tag, collect_call_tags, score_match, passes_threshold, collect_lhs_tags, TagCategory, names_in_expr
from .misslog import MissLogger
from .ast_index import ASTCallIndex
from opguard.instrument_skip import should_skip_torch_nn_functional_call

try:
    from opguard.common_vars import OPGUARD_LINENO_MODE
except Exception:
    OPGUARD_LINENO_MODE = os.environ.get("OPGUARD_LINENO_MODE", "begin")

_JIT_SCRIPT_DECORATORS = {
    "torch.jit.script",
    "torch.jit.script_if_tracing",
    "jit.script",
    "jit.script_if_tracing",
}


class RobustCSTFunctionInstrumentor(cst.CSTTransformer):
    """
    Phase-A transformer: selects call sites to wrap using *tag-based fuzzy matching*.
    - No line-number based filtering or logging.
    - Keeps "largest-call suppression" via claimed spans.
    - Collects prologue intents; Phase-B handles import/assign insertion.
    """
    METADATA_DEPENDENCIES = (PositionProvider, ParentNodeProvider)

    def __init__(self, targets: List[Dict[str, Any]], source_code: str, wrapper: MetadataWrapper):
        self.targets = targets or []
        self.source_lines = (source_code or "").splitlines()
        self.wrapper = wrapper
        self._ast_index = ASTCallIndex(source_code)

        self.current_class: Optional[str] = None
        self.method_depth = 0
        self.defer_prologue = True
        self.claimed_spans: List[Tuple[Tuple[int,int], Tuple[int,int]]] = []

        # Outputs collected during traversal
        self.wrapped_imports: List[Tuple[str, Optional[str], str, bool, bool]] = []
        self.need_flush_import = False
        self.need_operator_import = False
        self.wrapped_log: List[Dict[str, Any]] = []
        self._ast = ASTCallIndex(source_code)

        # Miss tracking (lineno-agnostic)
        self.miss = MissLogger()

        # Lineno is not used for matching; we still optionally curry it into wrappers
        self.lineno_mode = str(OPGUARD_LINENO_MODE).strip().lower()
        if self.lineno_mode not in ("begin", "end"):
            self.lineno_mode = "begin"

    # ---------- candidate selection & logging helpers ----------
    def _choose_best_by_tags(
        self,
        cands: List[Dict[str, Any]],
        observed: List[Tag],
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Pick the highest-scoring target by tag similarity, then enforce threshold.
        Returns (best_target_or_None, debug_meta).
        """
        best_t: Optional[Dict[str, Any]] = None
        best_ratio: float = -1.0
        best_meta: Optional[Dict[str, Any]] = None
        for cand in cands:
            ratio, meta = score_match(cand, observed)
            if ratio > best_ratio:
                best_ratio = ratio
                best_t = cand
                best_meta = meta
        if best_t is None:
            return None, None
        ok, thresh = passes_threshold(best_t, observed)
        if os.getenv("OPGUARD_DEBUG"):
            try:
                print(f"[DBG] choose_best_by_tags cands={cands} observed={observed} ratio={best_ratio} ok={ok} thresh={thresh}")
            except Exception:
                pass
        if not ok:
            return None, {"ratio": best_ratio, **(best_meta or {}), **(thresh or {})}
        return best_t, {"ratio": best_ratio, **(best_meta or {})}

    
    # ---------- observed-tag helper (AST index → CST → snippet AST) ----------
    # ---------- parent walking helpers ----------
    def _decorator_dotted_name(self, dec: cst.Decorator) -> str:
        node = dec.decorator
        if isinstance(node, cst.Call):
            node = node.func
        return expr_to_dotted_name(node)

    def _inside_torch_jit_script(self, node: cst.CSTNode) -> bool:
        fn = self._parent_of_type(node, (cst.FunctionDef,))
        if fn is None:
            return False
        return any(self._decorator_dotted_name(d) in _JIT_SCRIPT_DECORATORS for d in fn.decorators)

    def _should_skip_wrap(self, node: cst.CSTNode, callee: Optional[cst.BaseExpression] = None) -> bool:
        if self._inside_torch_jit_script(node):
            return True
        if callee is not None and should_skip_torch_nn_functional_call(expr_to_dotted_name(callee)):
            return True
        return False

    def _parent_of_type(self, node: cst.CSTNode, kinds: Tuple[type, ...]) -> Optional[cst.CSTNode]:
        """Walk ParentNodeProvider chain to the nearest node of given kinds."""
        cur: Optional[cst.CSTNode] = node
        while True:
            try:
                cur = self.get_metadata(ParentNodeProvider, cur)
            except Exception:
                return None
            if cur is None:
                return None
            if isinstance(cur, kinds):
                return cur

    class _ContainsVisitor(cst.CSTVisitor):
        def __init__(self, target: cst.CSTNode):
            self.target = target
            self.found = False
        def on_visit(self, node: cst.CSTNode) -> bool:
            # identity check is the safest for the current wrapper pass
            if node is self.target:
                self.found = True
                return False
            # keep descending until found
            return not self.found

    def _paren_generators_if_needed(self, args: List[cst.Arg]) -> List[cst.Arg]:
        """If a call has ≥2 args, wrap any GeneratorExp arg in parentheses."""
        if len(args) < 2:
            return args
        fixed: List[cst.Arg] = []
        for a in args:
            v = a.value
            if isinstance(v, cst.GeneratorExp):
                v = v.with_changes(lpar=[cst.LeftParen()], rpar=[cst.RightParen()])
                a = a.with_changes(value=v)
            fixed.append(a)
        return fixed

    def _assign_whose_value_contains(self, haystack: cst.CSTNode, call_node: cst.Call) -> Optional[cst.CSTNode]:
        """
        Search under `haystack` for an Assign/AugAssign/AnnAssign whose RHS/value subtree
        contains `call_node` by identity. Return that assign-like node if found.
        """
        # Fast path: exact ancestor assign
        anc = self._parent_of_type(call_node, (cst.Assign, cst.AugAssign, cst.AnnAssign))
        if anc is not None:
            return anc

        # Slow path: look within the closest SimpleStatementLine for siblings on same line
        ssl = self._parent_of_type(call_node, (cst.SimpleStatementLine,))
        if ssl is None:
            return None
        # Visit each small stmt and see whose value contains the call
        for small in ssl.body:
            try:
                if isinstance(small, cst.Assign):
                    v = self._ContainsVisitor(call_node)
                    small.value.visit(v)
                    if v.found:
                        return small
                elif isinstance(small, cst.AugAssign):
                    v = self._ContainsVisitor(call_node)
                    small.value.visit(v)
                    if v.found:
                        return small
                elif isinstance(small, cst.AnnAssign) and small.value is not None:
                    v = self._ContainsVisitor(call_node)
                    small.value.visit(v)
                    if v.found:
                        return small
            except Exception:
                # if traversal fails for a child, keep scanning siblings
                pass
        return None

    def _lhs_tags_for_call(self, call_node: cst.Call) -> List[Tag]:
        """
        Robustly locate the *correct* LHS for this call:
          1) Prefer the assign-like node whose RHS subtree actually contains this call.
          2) Fall back to the nearest ancestor assign if needed.
        Extract names from its LHS target(s).
        """
        assign = self._assign_whose_value_contains(call_node, call_node)
        if assign is None:
            return []

        lhs_tags: List[Tag] = []
        from .matchers import names_in_expr, TagCategory

        if isinstance(assign, cst.Assign):
            for t in assign.targets:
                for nm in names_in_expr(t.target):
                    lhs_tags.append(Tag(nm, TagCategory.LHS))
            return lhs_tags

        if isinstance(assign, cst.AugAssign):
            for nm in names_in_expr(assign.target):
                lhs_tags.append(Tag(nm, TagCategory.LHS))
            return lhs_tags

        if isinstance(assign, cst.AnnAssign) and assign.target is not None:
            for nm in names_in_expr(assign.target):
                lhs_tags.append(Tag(nm, TagCategory.LHS))
            return lhs_tags

        return lhs_tags
    
    def _cst_func_token(self, call_node: cst.Call) -> Optional[str]:
        f = call_node.func
        if isinstance(f, cst.Attribute):
            return f.attr.value
        if isinstance(f, cst.Name):
            return f.value
        return None

    # ---------- observed-tag helper (AST-first, exact same as callsite extraction) ----------
    def _observed_tags(self, call_node: cst.Call) -> List[Tag]:
        """Map CST call position to AST, return AST-indexed tags (receiver/arg/kwarg/LHS/GRID)."""
        pos = self.get_metadata(PositionProvider, call_node)
        line = pos.start.line
        col0 = pos.start.column  # LibCST columns are 0-based; Python AST col_offset is also 0-based
        tags = self._ast.tags_at(line, col0) or []
        if os.getenv("OPGUARD_DEBUG"):
            try:
                print("[DBG] _observed_tags(AST) ->", [(t.name, t.category) for t in tags])
            except Exception:
                pass
        return tags
    
    def _log_threshold_miss(
        self,
        cands: List[Dict[str, Any]],
        observed: List[Tag],
        reason: str = "tag_similarity_below_threshold",
    ) -> None:
        obs_names = [tg.name for tg in observed]
        for t in cands:
            self.miss.miss_once(reason, t, {"observed": obs_names})

    # ---------- small span helpers ----------
    def _pos_tuple(self, node_pos) -> Tuple[Tuple[int,int], Tuple[int,int]]:
        return (node_pos.start.line, node_pos.start.column), (node_pos.end.line, node_pos.end.column)

    def _span_contains(self, outer, inner) -> bool:
        (osl, osc), (oel, oec) = outer
        (isl, isc), (iel, iec) = inner
        return (isl, isc) >= (osl, osc) and (iel, iec) <= (oel, oec)

    def _inside_any_claimed_span(self, node_pos) -> bool:
        inner = self._pos_tuple(node_pos)
        return any(self._span_contains(sp, inner) for sp in self.claimed_spans)

    def _claim_span(self, node_pos):
        self.claimed_spans.append(self._pos_tuple(node_pos))

    def _inject_lineno_begin(self, original_node, call_expr: cst.Call) -> cst.Call:
        if self.lineno_mode != "begin":
            return call_expr
        pos = self.get_metadata(PositionProvider, original_node)
        lineno = pos.start.line
        return call_expr.with_changes(
            func=cst.Call(func=call_expr.func, args=[cst.Arg(value=cst.Integer(str(lineno)))])
        )

    # ---------- context tracking ----------
    def visit_ClassDef(self, node: cst.ClassDef):
        self.current_class = node.name.value

    def leave_ClassDef(self, original, updated):
        self.current_class = None
        return updated

    def visit_FunctionDef(self, node: cst.FunctionDef):
        self.method_depth += 1

    def _is_flush_stmt(self, stmt: cst.CSTNode) -> bool:
        return m.matches(
            stmt,
            m.SimpleStatementLine(
                body=[m.Expr(value=m.Call(func=m.Name("flush_trace_buffer")))]
            )
        )

    def leave_FunctionDef(self, original: cst.FunctionDef, updated: cst.FunctionDef):
        self.method_depth -= 1
        want_flush = original.name.value in ("forward", "backward")
        if not want_flush:
            return updated
        self.need_flush_import = True
        flush_stmt = cst.SimpleStatementLine([cst.Expr(cst.Call(func=cst.Name("flush_trace_buffer")))])
        new_body = []
        last_was_flush = False
        for stmt in updated.body.body:
            if self._is_flush_stmt(stmt):
                if not last_was_flush:
                    new_body.append(stmt); last_was_flush = True
                continue
            if m.matches(stmt, m.SimpleStatementLine(body=[m.Return()])):
                if not last_was_flush:
                    new_body.append(flush_stmt)
                new_body.append(stmt); last_was_flush = False
                continue
            new_body.append(stmt); last_was_flush = False
        return updated.with_changes(body=updated.body.with_changes(body=new_body))

    # ---------- Call sites ----------
    def leave_Call(self, original: cst.Call, updated: cst.Call) -> cst.Call:
        if self._should_skip_wrap(original, updated.func):
            return updated
        pos = self.get_metadata(PositionProvider, original)
        observed = self._observed_tags(original)

        # Suppress nested calls if a larger one already claimed the span
        if self._inside_any_claimed_span(pos):
            # This is informational; not line-based
            func_label = None
            if isinstance(updated.func, cst.Attribute):
                func_label = updated.func.attr.value
            elif isinstance(updated.func, cst.Name):
                func_label = updated.func.value
            for t in self.targets:
                if t.get("function") == func_label:
                    self.miss.miss_once("suppressed_by_larger_call_span", t, {})
            return updated

        # A) Triton kernel: obj[grid](...)
        if isinstance(updated.func, cst.Subscript):
            try:
                grid_expr = key_from_subscript(updated.func)
            except Exception:
                grid_expr = None

            dotted = expr_to_dotted_name(updated.func.value)
            base = dotted.split(".")[-1] if dotted else "kernel"

            # Candidates by type and name; may include function=None wildcard
            cands = [t for t in self.targets if t.get("operation_type") == "triton_kernel"
                     and t.get("function") in (base, dotted, None)]

            
            best_t, _dbg = self._choose_best_by_tags(cands, observed)

            if best_t is not None and grid_expr is not None:
                wrapped_name = f"opguard_wrapped_triton__{make_safe_identifier(base)}"
                tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(best_t)
                self.wrapped_imports.append((wrapped_name, dotted, tag_prefix, tensor_dump, global_sync))

                new_args = [cst.Arg(value=updated.func.value)]
                new_args.extend(updated.args)
                new_args.append(cst.Arg(keyword=cst.Name("__opg_triton_grid__"), value=grid_expr, equal=cst.AssignEqual()))
                new_call = updated.with_changes(func=cst.Name(wrapped_name), args=new_args)
                self._claim_span(pos)
                self.miss.mark_hit(best_t)
                return self._inject_lineno_begin(original, new_call)

            # Miss logging (either tags below threshold or missing grid)
            self._log_threshold_miss(cands, observed, reason="triton_tags_or_grid_below_threshold")

            return updated

        # B) Callable callee (factory)(...)
        if isinstance(updated.func, cst.Call):
            disp = canonical_expr_text(original.func, self.wrapper, self.source_lines)
            cands = [t for t in self.targets if t.get("function") == disp and t.get("operation_type") in (None, "function", "method")]
            best_t, _dbg = self._choose_best_by_tags(cands, observed)
            if best_t is None:
                self._log_threshold_miss(cands, observed, reason="tag_similarity_below_threshold")
                return updated

            cls = (best_t.get("class") or "").lower()
            tags = best_t.get("tag_names", [])
            wrapped_name = f"opguard_wrapped_{cls}_{(tags[0].lower() if tags else 'call')}_call"
            tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(best_t)
            self.wrapped_imports.append((wrapped_name, disp, tag_prefix, tensor_dump, global_sync))

            new_args = [cst.Arg(value=updated.func)]
            new_args.extend(updated.args)
            new_args = self._paren_generators_if_needed(new_args)
            new_call = updated.with_changes(func=cst.Name(wrapped_name), args=new_args)

            self._claim_span(pos)
            self.wrapped_log.append({
                "function": disp,
                "tag": tags,
                "wrapped_name": wrapped_name,
                "original_module": best_t.get("original_module"),
                "is_method": False,
                "operation_type": best_t.get("operation_type"),
                "file_path": best_t.get("file_path"),
                "context": best_t.get("context"),
            })
            self.miss.mark_hit(best_t)
            return self._inject_lineno_begin(original, new_call)

        # C) Regular functions / methods
        if m.matches(updated.func, m.Name()):
            func_name = updated.func.value; is_method = False
        elif m.matches(updated.func, m.Attribute()):
            func_name = updated.func.attr.value; is_method = True
        else:
            return updated

        cands = [t for t in self.targets if t.get("function") == func_name]
        if not cands:
            return updated

        best_t, _dbg = self._choose_best_by_tags(cands, observed)
        if best_t is None:
            self._log_threshold_miss(cands, observed, reason="tag_similarity_below_threshold")
            return updated

        cls = best_t.get("class", "").lower()
        tags = best_t.get("tag_names", [])
        first_tag = (tags[0].lower() if tags else "")
        wrapped_name = f"opguard_wrapped_{cls}_{first_tag + '_' if first_tag else ''}{func_name}"
        tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(best_t)
        self.wrapped_imports.append((wrapped_name, func_name, tag_prefix, tensor_dump, global_sync))

        new_args = [cst.Arg(value=updated.func)]
        new_args.extend(updated.args)
        new_args = self._paren_generators_if_needed(new_args)
        new_call = updated.with_changes(func=cst.Name(wrapped_name), args=new_args)

        self._claim_span(pos)
        self.wrapped_log.append({
            "function": func_name, "tag": tags, "wrapped_name": wrapped_name,
            "original_module": best_t.get("original_module"),
            "is_method": is_method, "operation_type": best_t.get("operation_type"),
            "file_path": best_t.get("file_path"),
            "context": best_t.get("context"),
        })
        self.miss.mark_hit(best_t)
        return self._inject_lineno_begin(original, new_call)

    # ---------- Binary operations (lineno-agnostic) ----------
    def leave_BinaryOperation(self, original: cst.BinaryOperation, updated: cst.BinaryOperation) -> cst.BaseExpression:
        if self._should_skip_wrap(original):
            return updated
        sym = None
        for cls, s in OP_CLASS_TO_SYMBOL.items():
            if isinstance(updated.operator, cls): sym = s; break
        if sym is None:
            return updated
        # Tokenize both operands robustly (handles a.b, a[b], etc.)
        left_tokens = {n.lower() for n in names_in_expr(updated.left)}
        right_tokens = {n.lower() for n in names_in_expr(updated.right)}

        # Build observed tag list for tag-based matcher (keeps logic consistent)
        observed = [Tag(n, "operand_left") for n in left_tokens] + \
                   [Tag(n, "operand_right") for n in right_tokens]

        # Collect operator candidates for this symbol (Path A: encoded; Path B: binop/tag_info)
        raw = [t for t in self.targets if t.get("operation_type") == "operator"]
        cands: List[Dict[str, Any]] = []
        for t in raw:
            func_str = str(t.get("function") or "")
            used_encoded = False
            matched_sym = False
            # Path A: "<lhs>_<op>_<rhs>"
            for candidate_sym in SYMBOL_TO_NAME.keys():
                sep = f"_{candidate_sym}_"
                if sep in func_str:
                    used_encoded = True
                    parts = func_str.split(sep)
                    if len(parts) == 2 and parts[0] and parts[1] and candidate_sym == sym:
                        matched_sym = True
                        # synthesize tag_info so tag matcher can work
                        t2 = dict(t)
                        if not t2.get("tag_info"):
                            t2["tag_info"] = [
                                {"name": parts[0], "category": "operand_left"},
                                {"name": parts[1], "category": "operand_right"},
                            ]
                        cands.append(t2)
                    break
            if used_encoded:
                continue
            # Path B: generic binop/operator with explicit tag_info
            if func_str in ("", "binop", "operator", "bin_op"):
                matched_sym = True  # same-op match is implicit — we use the observed 'sym'
                cands.append(t)

        if not cands:
            for t in raw:
                self.miss.miss_once(
                    "operator_no_candidates_for_binop",
                    t,
                    {"lhs": sorted(left_tokens), "rhs": sorted(right_tokens), "op": sym},
                )
            return updated

        best_t, _dbg = self._choose_best_by_tags(cands, observed)
        if best_t is None:
            self._log_threshold_miss(cands, observed, reason="operator_tag_similarity_below_threshold")
            return updated

        safe_op = SYMBOL_TO_NAME[sym]
        lname = next(iter(left_tokens), "lhs")
        rname = next(iter(right_tokens), "rhs")
        wrapped_name = make_safe_identifier(f"opguard_wrapped_{lname}_{safe_op}_{rname}")
        tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(best_t)
        self.wrapped_imports.append((wrapped_name, None, tag_prefix, tensor_dump, global_sync))
        self.need_operator_import = True

        call_expr = cst.Call(
            func=cst.Name(wrapped_name),
            args=[
                cst.Arg(value=cst.Attribute(value=cst.Name("operator"), attr=cst.Name(safe_op))),
                cst.Arg(value=updated.left),
                cst.Arg(value=updated.right),
            ]
        )
        self.miss.mark_hit(best_t)
        return self._inject_lineno_begin(original, call_expr)

    # ---------- Advanced indexing (reads) (lineno-agnostic) ----------
    def leave_Subscript(self, original: cst.Subscript, updated: cst.Subscript) -> cst.BaseExpression:
        base_name = (extract_name(updated.value) or "").lower()
        cands = [t for t in self.targets if t.get("operation_type") == "advanced_indexing"]
        # prefer candidate whose first tag matches container/base
        for t in cands:
            tag = ((t.get("tag_names") or [""])[0] or "").lower()
            if tag and tag in base_name:
                wrapped_name = f"opguard_wrapped_{(t.get('class') or '').lower()}_{tag}_getitem"
                key_expr = key_from_subscript(updated)
                tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(t)
                self.wrapped_imports.append((wrapped_name, None, tag_prefix, tensor_dump, global_sync))
                self.need_operator_import = True
                call_expr = cst.Call(
                    func=cst.Name(wrapped_name),
                    args=[
                        cst.Arg(value=cst.Attribute(value=cst.Name("operator"), attr=cst.Name("getitem"))),
                        cst.Arg(value=updated.value),
                        cst.Arg(value=key_expr),
                    ],
                )
                self.miss.mark_hit(t)
                return self._inject_lineno_begin(original, call_expr)
        # no logging to avoid noise (reads are frequent)
        return updated

    # ---------- Advanced indexing (writes) (lineno-agnostic) ----------
    def leave_Assign(self, original: cst.Assign, updated: cst.Assign) -> cst.BaseSmallStatement:
        if len(original.targets) != 1 or not isinstance(original.targets[0].target, cst.Subscript):
            return updated

        sub = original.targets[0].target
        base_name = (extract_name(sub.value) or "").lower()
        cands = [t for t in self.targets if t.get("operation_type") == "advanced_indexing_write"]
        for t in cands:
            tag = ((t.get("tag_names") or [""])[0] or "").lower()
            if tag and tag in base_name:
                wrapped_name = f"opguard_wrapped_{(t.get('class') or '').lower()}_{tag}_setitem"
                tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(t)
                self.wrapped_imports.append((wrapped_name, None, tag_prefix, tensor_dump, global_sync))
                self.need_operator_import = True
                key_expr = key_from_subscript(sub)
                inner_call = cst.Call(
                    func=cst.Name(wrapped_name),
                    args=[
                        cst.Arg(value=cst.Attribute(value=cst.Name("operator"), attr=cst.Name("setitem"))),
                        cst.Arg(value=sub.value),
                        cst.Arg(value=key_expr),
                        cst.Arg(value=updated.value),
                    ],
                )
                self.miss.mark_hit(t)
                return cst.Expr(self._inject_lineno_begin(original, inner_call))
        return updated

    # ---------- Augmented assignment (e.g. x += y) ----------
    def leave_AugAssign(self, original: cst.AugAssign, updated: cst.AugAssign) -> cst.BaseSmallStatement:
        """
        Wrap augmented operators using operator.i* functions:
          lhs += value  ==>  lhs = wrapped(operator.iadd, lhs, value)
        Matching is tag-based:
          - LHS tag:    category "lhs"
          - VALUE tags: category "value" across any names found in RHS expression
        Candidates: operation_type in {"augmented_operator","augassign","augmented_op","augmented"}
                    (also tolerate "operator" with function in {"augassign","binop",""})
        """
        # Map LibCST AugAssign operator nodes to operator.* names
        AUG_OP = {
            cst.AddAssign: "iadd",
            cst.SubtractAssign: "isub",
            cst.MultiplyAssign: "imul",
            cst.MatrixMultiplyAssign: "imatmul",
            cst.DivideAssign: "itruediv",
            cst.FloorDivideAssign: "ifloordiv",
            cst.ModuloAssign: "imod",
            cst.PowerAssign: "ipow",
            cst.LeftShiftAssign: "ilshift",
            cst.RightShiftAssign: "irshift",
            cst.BitAndAssign: "iand",
            cst.BitOrAssign: "ior",
            cst.BitXorAssign: "ixor",
        }
        # Figure out which inplace operator we have
        op_name = None
        for cls, name in AUG_OP.items():
            if isinstance(updated.operator, cls):
                op_name = name
                break
        if op_name is None:
            return updated

        # Build observed tag set: LHS + names in RHS (as "value")
        lhs_name = (extract_name(updated.target) or "").lower()
        observed: List[Tag] = []
        if lhs_name:
            observed.append(Tag(lhs_name, TagCategory.LHS))
        for nm in names_in_expr(updated.value):
            observed.append(Tag(nm, TagCategory.VALUE))

        # Candidate configs for augmented ops
        cands = []
        for t in self.targets:
            ot = (t.get("operation_type") or "").lower()
            fn = (t.get("function") or "").lower()
            if ot in {"augmented_operator", "augassign", "augmented_op", "augmented"}:
                cands.append(t)
            elif ot == "operator" and fn in {"augassign", "binop", ""}:
                cands.append(t)

        if not cands:
            return updated

        best_t, _dbg = self._choose_best_by_tags(cands, observed)
        if best_t is None:
            self._log_threshold_miss(cands, observed, reason="augassign_tag_similarity_below_threshold")
            return updated

        # Name & imports
        lhs_slug = make_safe_identifier(lhs_name or "lhs")
        wrapped_name = f"opguard_wrapped_{lhs_slug}_{op_name}_aug"
        tag_prefix, tensor_dump, global_sync = get_make_wrapper_params(best_t)
        self.wrapped_imports.append((wrapped_name, None, tag_prefix, tensor_dump, global_sync))
        self.need_operator_import = True

        # Build: lhs = wrapped(operator.i*, lhs, value)
        call_expr = cst.Call(
            func=cst.Name(wrapped_name),
            args=[
                cst.Arg(value=cst.Attribute(value=cst.Name("operator"), attr=cst.Name(op_name))),
                cst.Arg(value=updated.target),
                cst.Arg(value=updated.value),
            ],
        )
        call_expr = self._inject_lineno_begin(original, call_expr)
        assign_stmt = cst.Assign(targets=[cst.AssignTarget(target=updated.target)], value=call_expr)

        # bookkeeping
        self.miss.mark_hit(best_t)
        return assign_stmt

    # ---------- Module prologue (Phase-B handled elsewhere) ----------
    def leave_Module(self, original: cst.Module, updated: cst.Module) -> cst.Module:
        return updated