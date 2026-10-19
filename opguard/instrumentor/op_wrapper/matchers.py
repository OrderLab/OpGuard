# opguard/instrumentor/op_wrapper/matcher.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
import libcst as cst
from libcst import matchers as m

# Keep categories aligned with callsite side
class TagCategory(str):
    RECEIVER = "receiver"
    RECEIVER_SUBEXPR = "receiver_subexpr"
    ARG = "arg"
    KWARG = "kwarg"
    LHS = "lhs"
    INDEX = "index"
    GRID = "grid"
    VALUE = "value"
    OPERAND_LEFT = "operand_left"
    OPERAND_RIGHT = "operand_right"
    COMPARE_LEFT = "compare_left"
    COMPARE_RIGHT = "compare_right"
    CONTAINER = "container"
    KW_VALUE = "kw_value"

@dataclass(frozen=True)
class Tag:
    name: str
    category: str

# ---------- extraction helpers (LibCST) ----------

def _norm(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    s = name.strip().lower()
    if not s or s.isdigit() or s in {"self", "cls", "true", "false", "none", "_"}:
        return None
    return s

class _NameAttrCollector(cst.CSTVisitor):
    def __init__(self) -> None:
        self.names: List[str] = []
    def visit_Name(self, node: cst.Name) -> None:
        nm = _norm(node.value)
        if nm:
            self.names.append(nm)
    def visit_Attribute(self, node: cst.Attribute) -> None:
        # add attribute token (rightmost identifier)
        nm = _norm(node.attr.value)
        if nm:
            self.names.append(nm)


def names_in_expr(expr: cst.CSTNode) -> List[str]:
    v = _NameAttrCollector()
    try:
        expr.visit(v)
    except Exception:
        pass
    # preserve order but unique
    out: List[str] = []
    seen = set()
    for n in v.names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def collect_call_tags(call: cst.Call, *, include_receiver_subexpr: bool = True) -> List[Tag]:
    tags: List[Tag] = []
    # receiver side
    if isinstance(call.func, cst.Attribute) and include_receiver_subexpr:
        tags.extend(Tag(n, TagCategory.RECEIVER_SUBEXPR) for n in names_in_expr(call.func.value))
    # function/attribute tokens themselves are intentionally omitted (we match by function name already)

    # args & kwargs
    for a in call.args:
        if a.keyword is None and a.star is None:
            tags.extend(Tag(n, TagCategory.ARG) for n in names_in_expr(a.value))
        elif a.keyword is not None:
            tags.extend(Tag(n, TagCategory.KWARG) for n in names_in_expr(a.value))
        # we ignore *args/**kwargs value names—they'd be inside a.value already
    
    collected_tags = _dedup_tags(tags)
    if os.getenv("OPGUARD_DEBUG"):
        try:
            print(f"[DBG] collect_call_tags collected={collected_tags}")
        except Exception:
            pass
    return collected_tags


def collect_lhs_tags(enclosing_stmt: cst.CSTNode) -> List[Tag]:
    tags: List[Tag] = []
    # Assign or AugAssign
    if m.matches(enclosing_stmt, m.SimpleStatementLine(body=[m.Assign()])):
        assign = enclosing_stmt.body[0]
        for t in assign.targets:
            tags.extend(Tag(n, TagCategory.LHS) for n in names_in_expr(t.target))
    elif m.matches(enclosing_stmt, m.SimpleStatementLine(body=[m.AugAssign()])):
        aug = enclosing_stmt.body[0]
        tags.extend(Tag(n, TagCategory.LHS) for n in names_in_expr(aug.target))
    return _dedup_tags(tags)


def _dedup_tags(items: Iterable[Tag]) -> List[Tag]:
    seen = set()
    out: List[Tag] = []
    for t in items:
        key = (t.name, t.category)
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out

# ---------- matching/scoring ----------

_DEFAULT_WEIGHTS = {
    TagCategory.LHS: 3.0,
    TagCategory.RECEIVER: 2.0,
    TagCategory.RECEIVER_SUBEXPR: 2.0,
    TagCategory.ARG: 1.0,
    TagCategory.KWARG: 1.0,
    TagCategory.KW_VALUE: 1.0,
    TagCategory.INDEX: 1.0,
    TagCategory.GRID: 1.0,
    TagCategory.CONTAINER: 1.0,
    TagCategory.OPERAND_LEFT: 1.0,
    TagCategory.OPERAND_RIGHT: 1.0,
    TagCategory.COMPARE_LEFT: 1.0,
    TagCategory.COMPARE_RIGHT: 1.0,
    TagCategory.VALUE: 0.5,
}

# parse env JSON like '{"lhs":3,"arg":1}' using our canonical keys
def _load_weights() -> Dict[str, float]:
    raw = os.getenv("OPGUARD_TAG_CATEGORY_WEIGHTS", "").strip()
    if not raw:
        return dict(_DEFAULT_WEIGHTS)
    try:
        obj = json.loads(raw)
        weights = dict(_DEFAULT_WEIGHTS)
        for k, v in obj.items():
            kk = str(k).lower()
            # remap possible short keys
            for cat in _DEFAULT_WEIGHTS.keys():
                if cat.lower() == kk:
                    weights[cat] = float(v)
                    break
        return weights
    except Exception:
        return dict(_DEFAULT_WEIGHTS)

_WEIGHTS = _load_weights()
_RATIO = float(os.getenv("OPGUARD_TAG_MATCH_RATIO", "0.75"))
_MIN = int(os.getenv("OPGUARD_TAG_MIN_MATCH", "2"))
_CATEGORY_MISMATCH_FACTOR = float(os.getenv("OPGUARD_TAG_CAT_MISMATCH_FACTOR", "0.5"))
# Categories that should be treated as equivalent (no penalty)
_CAT_EQUIV_DEFAULT = {
    "arg": {"kwarg", "kw_value"},
    "kwarg": {"arg", "kw_value"},
    "kw_value": {"kwarg", "arg"},
}
_CAT_EQUIV = _CAT_EQUIV_DEFAULT.copy()

def _cat_equiv(a: str, b: str) -> bool:
    if a == b:
        return True
    return b in _CAT_EQUIV.get(a, set())

def to_target_tags(target: Dict[str, Any]) -> List[Tag]:
    # prefer rich tag_info
    if target.get("tag_info"):
        return [Tag(t.get("name", ""), t.get("category", TagCategory.ARG))
                for t in target["tag_info"] if t.get("name")]
    # fallback to tag_names → category unknown, let them match any but with reduced weight
    if target.get("tag_names"):
        return [Tag(name, "*") for name in target["tag_names"] if name]
    return []


def score_match(target: Dict[str, Any], observed: List[Tag]) -> Tuple[float, Dict[str, Any]]:
    tgt = to_target_tags(target)
    if not tgt:
        return 0.0, {"matched": [], "needed": 0, "observed": [t.__dict__ for t in observed]}

    # build index by name for observed, with best category per name (max weight)
    obs_by_name: Dict[str, Tag] = {}
    for t in observed:
        cur = obs_by_name.get(t.name)
        if cur is None or _WEIGHTS.get(t.category, 0.0) > _WEIGHTS.get(cur.category, 0.0):
            obs_by_name[t.name] = t

    matched: List[Tuple[Tag, float]] = []
    total_weight = 0.0
    for t in tgt:
        w = _WEIGHTS.get(t.category, 1.0) if t.category != "*" else 1.0
        total_weight += w
        obs = obs_by_name.get(t.name)
        if obs is None:
            continue
        if t.category == "*" or _cat_equiv(t.category, obs.category):
            matched.append((t, w))
        else:
            matched.append((t, w * _CATEGORY_MISMATCH_FACTOR))

    got_weight = sum(w for _t, w in matched)
    ratio = got_weight / max(total_weight, 1e-9)
    return ratio, {
        "matched": [{"name": t.name, "category": t.category, "weight": w} for t, w in matched],
        "needed": total_weight,
        "observed": [t.__dict__ for t in observed],
        "ratio": ratio,
    }


def passes_threshold(target: Dict[str, Any], observed: List[Tag]) -> Tuple[bool, Dict[str, Any]]:
    tgt = to_target_tags(target)
    ratio, meta = score_match(target, observed)
    # also check count-based threshold
    unique_target_names = {t.name for t in tgt}
    unique_observed_names = {t.name for t in observed}
    count_match = len(unique_target_names & unique_observed_names)
    # Require at least MIN matches but never more than the number of target tags.
    # This lets single-tag targets (like lhs-only) pass when they match.
    required = min(max(_MIN, 1), len(unique_target_names) if unique_target_names else 1)
    ok = (count_match >= required) and (ratio >= _RATIO)
    meta.update({"count_match": count_match, "target_count": len(unique_target_names),
                 "required": required, "threshold_ratio": _RATIO, "threshold_min": _MIN})
    return ok, meta

