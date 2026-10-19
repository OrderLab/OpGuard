# opguard/callsite/model.py

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict, Any

class TagCategory(str, Enum):
    RECEIVER = "receiver"                  # object of a method call: obj.method(...)
    RECEIVER_SUBEXPR = "receiver_subexpr"  # any names inside the receiver expression
    ARG = "arg"                            # positional argument
    KWARG = "kwarg"                        # keyword argument value
    LHS = "lhs"                            # assignment target on same statement
    OPERAND_LEFT = "operand_left"          # left operand of an operator
    OPERAND_RIGHT = "operand_right"        # right operand of an operator
    COMPARE_LEFT = "compare_left"          # comparison left side
    COMPARE_RIGHT = "compare_right"        # comparison right side (first comparator)
    CONTAINER = "container"                # subscript base in writes: a[...]
    INDEX = "index"                        # names used in subscript indices
    VALUE = "value"                        # value part in writes / augassign
    GRID = "grid"                          # triton grid indices in kernel[grid](...)
    KW_VALUE = "kw_value"                  # value of any keyword in kernel/function

@dataclass(frozen=True)
class Tag:
    name: str
    category: TagCategory

@dataclass
class CallsiteInfo:
    function: str
    operation_type: str  # one of: 'method','function','triton_kernel','operator','augmented_operator','comparison','advanced_indexing_write','mapping_write'
    tags: List[Tag]

    def tag_names(self, max_n: int = 5) -> List[str]:
        seen = set()
        out: List[str] = []
        for t in self.tags:
            if t.name not in seen:
                seen.add(t.name)
                out.append(t.name)
            if len(out) >= max_n:
                break
        return out

    def tag_info(self, max_n: int = 5) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for t in self.tags:
            key = (t.name, t.category)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": t.name, "category": t.category.value})
            if len(out) >= max_n:
                break
        return out
