
# opguard/callsite/ast\_utils.py

from __future__ import annotations
import ast
import os
from typing import Optional, Tuple, Set

TRITON_META_HINTS = {
    "num_warps", "num_ctas", "num_stages", "maxnreg",
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "I64_COORDS",
}

def _load_deprioritized_funcs_from_env() -> Set[str]:
    """
    Read a comma-separated list of function names to deprioritize from env.
    e.g. OPGUARD_DEPRIORITIZE_FUNCS="range,len,zip"
    """
    raw = os.getenv("OPGUARD_DEPRIORITIZE_FUNCS", "").strip()
    if not raw:
        return set()
    out: Set[str] = set()
    for tok in raw.split(","):
        nm = tok.strip().lower()
        if nm:
            out.add(nm)
    return out

# Default: only 'range' unless overridden via env
DEPRIORITIZED_FUNCTIONS: Set[str] = {"range", "len", "zip"} | _load_deprioritized_funcs_from_env()

def dotted_name_from_ast(expr: ast.AST) -> str:
    if isinstance(expr, ast.Name):
        return expr.id
    parts = []
    cur = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    try:
        return ast.unparse(expr)
    except Exception:
        return "kernel"

def contains_str_constant(node: ast.AST) -> bool:
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return True
    return False

def is_all_caps_identifier(name: str) -> bool:
    return bool(name) and all(ch.isupper() or ch.isdigit() or ch == '_' for ch in name)

# Heuristic to identify Triton kernel call: kernel[grid](...)
# with weak negatives for registries and string-key subscripts

def is_triton_kernel_call(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Subscript):
        return False
    base = call.func.value
    grid = call.func.slice
    base_name = None
    if isinstance(base, ast.Name):
        base_name = base.id
    elif isinstance(base, ast.Attribute):
        x = base
        while isinstance(x, ast.Attribute):
            last = x.attr
            x = x.value
        base_name = last
    # Negative: registry/factory patterns
    looks_like_registry = (
        base_name is not None and (
            is_all_caps_identifier(base_name) or
            base_name.endswith(("REGISTRY","_REGISTRY","_CLASSES","_MAP","_DICT","_FACTORY","_IMPLS"))
        )
    )
    if looks_like_registry:
        return False
    if contains_str_constant(grid):
        return False
    # Positive signals
    has_triton_kw = any((kw.arg in TRITON_META_HINTS) for kw in (call.keywords or []) if kw.arg)
    base_is_kernelish = isinstance(base_name, str) and base_name.endswith("_kernel")
    ptr_kw_count = sum(1 for kw in (call.keywords or []) if kw.arg and kw.arg.endswith("_ptr"))
    return has_triton_kw or base_is_kernelish or (ptr_kw_count >= 2)


def call_priority_kind_name(call: ast.Call) -> Tuple[int, str, str]:
    """
    Rank calls for selection:
      0 → triton_kernel, 1 → function, 2 → method
    Returns (priority, kind, name)
    """
    func_expr = call.func
    if is_triton_kernel_call(call):
        kernel_dotted = dotted_name_from_ast(call.func.value)
        kernel_base   = kernel_dotted.split(".")[-1] if kernel_dotted else "kernel"
        return (0, "triton_kernel", kernel_base)
    if isinstance(func_expr, ast.Attribute):
        return (2, "method", func_expr.attr)
    if isinstance(func_expr, ast.Name):
        fname = func_expr.id
        # If a specific free-function name is in the deprioritized set,
        # push it behind methods by using a larger priority number.
        if fname and fname.lower() in DEPRIORITIZED_FUNCTIONS:
            return (3, "function", fname)
        return (1, "function", fname)
    if isinstance(func_expr, ast.Subscript):
        try:
            nm = ast.unparse(func_expr)
        except Exception:
            nm = "call"
        return (1, "function", nm)
    try:
        nm = ast.unparse(func_expr)
    except Exception:
        nm = "call"
    return (1, "function", nm)

