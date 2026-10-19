from __future__ import annotations
import json, os, re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from fnmatch import fnmatchcase

from .config import (
    SCALE_NS_PER_TICK, SLICE_STRIDE_TICKS, BAR_WIDTH_TICKS,
    DISPLAY_TIME_UNIT, _IO_STAT_KEYS, IGNORED_TOP, STACK_EXCLUDE_FUNC_GLOBS,
)
from . import config

GPU_RANK_RE = re.compile(r"gpu_(\d+)")

def _get(d: dict, *ks, default=None):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def parse_rank_from_rel(rel: str) -> int:
    m = GPU_RANK_RE.search(rel)
    return int(m.group(1)) if m else -1

def rel_tid(rank: int) -> str:
    return f"gpu_{rank}"

def short_name(fn: Optional[str]) -> str:
    if not fn: return "unknown"
    toks = fn.strip("<> ").split(".")
    return f"{toks[-2]}.{toks[-1]}" if len(toks) >= 2 else toks[-1]

def stable_name_with_counter(node: dict) -> str:
    base = short_name(node.get("function"))
    ctr = node.get("counter")
    return f"{base}#{ctr}" if ctr is not None else base

def strip_top(trace: dict) -> dict:
    return {k: v for k, v in trace.items()
            if k not in IGNORED_TOP and not str(k).startswith("__")}

def _fmt_ms(ts_ticks: Optional[int]) -> Optional[str]:
    if ts_ticks is None: return None
    return f"{(ts_ticks * SCALE_NS_PER_TICK)/1e6:.3f} ms"

# Debug helpers
def _dbg_open(name: str):
    if not config.DEBUG_ENABLED or config.DEBUG_DIR is None:
        return None
    Path(config.DEBUG_DIR).mkdir(parents=True, exist_ok=True)
    return (Path(config.DEBUG_DIR) / name).open("a", encoding="utf-8")

def _dbg_write(fp, rec: dict):
    if fp:
        fp.write(json.dumps(rec, ensure_ascii=False) + "\n")

# IO helpers
def _is_tensor_dict(x: Any) -> bool:
    # Treat entries with a stats dict as tensor-like, even if shape is missing (e.g., scalars)
    return isinstance(x, dict) and ("stats" in x)

def _flatten_tensors(obj: Any) -> List[dict]:
    out: List[dict] = []
    def rec(v):
        if _is_tensor_dict(v): out.append(v)
        elif isinstance(v, list):
            for t in v: rec(t)
        elif isinstance(v, dict):
            for t in v.values(): rec(t)
    rec(obj)
    return out

def io_summary_with_xor_values(t: dict) -> Dict[str, Any]:
    def entries(obj: Any) -> List[Dict[str, Any]]:
        L: List[Dict[str, Any]] = []
        for td in _flatten_tensors(obj):
            st = td.get("stats", {})
            L.append({
                "shape": td.get("shape"),
                "dtype": st.get("dtype"),
                "device": st.get("device"),
                "xor_signature": st.get("xor_signature"),
                "xor_sig_mode": st.get("xor_sig_mode"),
                "xor_sig_layout": st.get("xor_sig_layout"),
                "numel": st.get("numel"),
            })
        return L
    return {
        "inputs": entries(_get(t, "inputs", "args")) + entries(_get(t, "inputs", "kwargs")),
        "outputs": entries(t.get("outputs")),
    }

def signature_from_trace(t: dict) -> str:
    fn = t.get("function") or "fn"
    parts = []
    args = _get(t, "inputs", "args") or []
    for idx, v in enumerate(args):
        if _is_tensor_dict(v):
            shp = v.get("shape"); dt = v.get("stats", {}).get("dtype")
            parts.append(f"a{idx}[{tuple(shp) if isinstance(shp, list) else shp}:{dt}]")
        elif v is None:
            parts.append("None")
        else:
            parts.append(str(v)[:32])
    return f"{fn}({', '.join(parts)})"

def status_tid(rank: int) -> str:
    return f"{rel_tid(rank)}:status"

def _emit_thread_names(events, pid, tid, label):
    events.append({"ph": "M", "pid": pid, "tid": tid, "name": "thread_name",
                   "args": {"name": label}})
