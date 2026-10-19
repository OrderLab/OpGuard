# opguard/instrumentor/op_wrapper/cst_utils.py
from __future__ import annotations
from typing import Optional, Tuple, List
import ast, re, keyword, os
import libcst as cst
from libcst import matchers as m
from libcst.metadata import PositionProvider

# ----------------- operator maps -----------------
from libcst import (
    Add, Subtract, Multiply, Divide, Power, Modulo, FloorDivide,
    LeftShift, RightShift, MatrixMultiply,
    BitAnd, BitOr, BitXor,
    GreaterThan, LessThan, GreaterThanEqual, LessThanEqual,
    Equal, NotEqual,
)

OP_CLASS_TO_SYMBOL = {
    Add: "+", Subtract: "-", Multiply: "*", Divide: "/", Power: "**",
    Modulo: "%", FloorDivide: "//", LeftShift: "<<", RightShift: ">>",
    MatrixMultiply: "@", BitAnd: "&", BitOr: "|", BitXor: "^",
    GreaterThan: ">", LessThan: "<", GreaterThanEqual: ">=", LessThanEqual: "<=",
    Equal: "==", NotEqual: "!=",
}

SYMBOL_TO_NAME = {
    "+": "add", "-": "sub", "*": "mul", "@": "matmul", "/": "truediv", 
    "**": "pow", "%": "mod", "//": "floordiv", ">": "gt", "<": "lt",
    ">=": "ge", "<=": "le", "==": "eq", "!=": "ne",
    "&": "and_", "|": "or_", "^": "xor",
}

TRITON_META_HINTS = {
    "num_warps", "num_ctas", "num_stages", "maxnreg",
    "BLOCK_M", "BLOCK_N", "BLOCK_K", "GROUP_SIZE_M", "I64_COORDS",
}

# ----------------- name / dotted utils -----------------
def extract_name(expr: cst.BaseExpression) -> Optional[str]:
    if m.matches(expr, m.Name()):
        return expr.value
    if m.matches(expr, m.Attribute()):
        return extract_name(expr.value)
    if m.matches(expr, m.Subscript()):
        return extract_name(expr.value)
    if m.matches(expr, m.Call()):
        return extract_name(expr.func)
    if m.matches(expr, m.BinaryOperation()):
        return extract_name(expr.left)
    if m.matches(expr, m.UnaryOperation()):
        return extract_name(expr.expression)
    return None

def expr_to_dotted_name(expr: cst.BaseExpression) -> str:
    if isinstance(expr, cst.Name):
        return expr.value
    if isinstance(expr, cst.Attribute):
        parts: List[str] = []
        cur = expr
        while isinstance(cur, cst.Attribute):
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
        parts.reverse()
        return ".".join(parts)
    return "kernel"

def make_safe_identifier(s: str) -> str:
    safe = re.sub(r'\W', '_', s)
    safe = re.sub(r'_+', '_', safe).strip('_') or "wrapped"
    if safe[0].isdigit():
        safe = "_" + safe
    if keyword.iskeyword(safe):
        safe += "_id"
    return safe

# ----------------- slicing / subscript utils -----------------
def _slice_to_expr(bs: cst.BaseSlice) -> cst.BaseExpression:
    if isinstance(bs, cst.Index):
        return bs.value
    if isinstance(bs, cst.Slice):
        def _or_none(x): return x if x is not None else cst.Name("None")
        return cst.Call(func=cst.Name("slice"),
                        args=[cst.Arg(_or_none(bs.lower)),
                              cst.Arg(_or_none(bs.upper)),
                              cst.Arg(_or_none(bs.step))])
    raise NotImplementedError(f"Unsupported slice node: {type(bs)}")

def key_from_subscript(sub: cst.Subscript) -> cst.BaseExpression:
    parts = [_slice_to_expr(el.slice) for el in sub.slice]
    if len(parts) == 1:
        return parts[0]
    return cst.Tuple(elements=[cst.Element(value=p) for p in parts])

# ----------------- source text helpers -----------------
def node_source_text(node: cst.CSTNode, wrapper: "cst.MetadataWrapper", source_lines: List[str]) -> str:
    # Try PositionProvider first; fallback to node.code for updated nodes
    try:
        pos = wrapper.resolve(PositionProvider)[node]
    except Exception:
        try:
            return node.code
        except Exception:
            return ""
    if not source_lines:
        try:
            return node.code
        except Exception:
            return ""
    if pos.start.line == pos.end.line:
        line = source_lines[pos.start.line - 1]
        return line[pos.start.column:pos.end.column]
    parts = [source_lines[pos.start.line - 1][pos.start.column:]]
    parts.extend(source_lines[pos.start.line:pos.end.line - 1])
    parts.append(source_lines[pos.end.line - 1][:pos.end.column])
    return "".join(parts)

def canonical_expr_text(node: cst.CSTNode, wrapper: "cst.MetadataWrapper", source_lines: List[str]) -> str:
    raw = node_source_text(node, wrapper, source_lines).strip()
    try:
        expr = ast.parse(raw, mode="eval")
        return ast.unparse(expr.body)
    except Exception:
        return " ".join(raw.split()).replace('"', "'")

# ----------------- lineno helpers -----------------
def lineno_for_mode(pos: PositionProvider, mode: str) -> int:
    return pos.end.line if mode == "end" else pos.start.line

# ----------------- config helpers -----------------
def get_make_wrapper_params(config: dict) -> tuple:
    # Kept compatible with your prior code
    tag_prefix = f"{config.get('file_path', 'unknown_file')}"
    dump_tensors = config.get("dump_tensors", False)
    global_sync = config.get("sync_streams", False)
    return tag_prefix, dump_tensors, global_sync
