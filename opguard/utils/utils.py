import json
import os
import signal
import threading
import operator
from typing import Any, List
from pathlib import Path
import orjson
import numpy as np
import torch
import hashlib


try:
    import torch
except Exception:
    torch = None

_COMPARATORS = {
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,   # 兼容单等号
}

def _infer_rank():
    from opguard.distributed import get_global_rank
    return get_global_rank()

def get_id():
    rank = _infer_rank()
    pid = os.getpid()
    try:
        tid = threading.get_native_id()  # py3.8+
    except Exception:
        tid = threading.get_ident()
    return f"{rank}-{pid}-{tid}"

def register_signal_handler(signal_num, new_handler):
    original_handler = signal.getsignal(signal_num)
    
    def wrapper(signum, frame):
        if callable(original_handler):
            original_handler(signum, frame)
        
        new_handler(signum, frame)
    
    signal.signal(signal_num, wrapper)

    return wrapper


def parse_val(v: str) -> Any:
    s = v.strip()
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        # 支持数字/对象/数组
        return json.loads(s)
    except Exception:
        return s
    
def passes_filter(trace: dict, filter_kv: str) -> bool:
    if not filter_kv:
        return True
    filters = filter_kv if isinstance(filter_kv, list) else filter_kv.split(",")
    for cond in filters:
        cond = cond.strip()
        found = False
        for op_str in sorted(_COMPARATORS, key=len, reverse=True):
            if op_str in cond:
                key_path, raw_val = cond.split(op_str, 1)
                expected = parse_val(raw_val)
                comp = _COMPARATORS[op_str]
                # 下钻
                keys = key_path.strip().split(".")
                val: Any = trace
                for k in keys:
                    if not isinstance(val, dict) or k not in val:
                        return False
                    val = val[k]
                try:
                    if not comp(val, expected):
                        return False
                except Exception:
                    return False
                found = True
                break
        if not found:
            print(f"⚠️ Invalid filter condition '{cond}'")
            return False
    return True


def _load_json(path: Path, filter: str = None) -> List[dict]:
    out: List[dict] = []
    with path.open("rb") as f:
        data = orjson.loads(f.read())
        if not isinstance(data, list):
            data = [data]
    
    for obj in data:
        if passes_filter(obj, filter):
            out.append(obj)
    
    return out

def _load_jsonl(path: Path, filter: str = None) -> List[dict]:
    out: List[dict] = []
    with path.open("rb") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line: continue
            try:
                obj = orjson.loads(line)
                if passes_filter(obj, filter):
                    out.append(obj)
            except Exception as e:
                print(f"⚠️ Corrupted line {i} in {path.name}: {e}")
    return out

def _load_pt(path: Path, filter: str = None) -> List[dict]:
    out: List[dict] = []
    if torch is None:
        print(f"⚠️ torch not available; skip {path}")
        return []
    try:
        data = torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"⚠️ Failed to load PT {path}: {e}")
        return []
    
    if not isinstance(data, list):
        data = [data]
    
    for obj in data:
        if passes_filter(obj, filter):
            out.append(obj)
    
    return out

def _suppress_unsupported_warning(path: Path, suf: str) -> bool:
    if suf not in {".py", ".yaml", ".yml"}:
        return False
    parts = {p.lower() for p in path.parts}
    return "source_files" in parts or "instrumentation" in parts


def load_traces_any(path: Path, filter: str = None) -> List[dict]:
    suf = path.suffix.lower()
    if suf == ".jsonl": return _load_jsonl(path, filter)
    if suf == ".json":  return _load_json(path, filter)
    if suf == ".pt":    return _load_pt(path, filter)
    if _suppress_unsupported_warning(path, suf):
        return []
    print(f"⚠️ Unsupported file type: {suf} ({path})"); return []


def placeholder_sig_from_meta(t: torch.Tensor) -> torch.Tensor:
    """Deterministic uint32[1] based on metadata (no data access)."""
    h = np.uint32(2166136261)  # FNV-1a offset
    FNV_PRIME = 16777619
    MASK32 = 0xFFFFFFFF

    def mix_py(v):
        nonlocal h
        v32 = int(v) & MASK32
        h ^= v32
        h = (h * FNV_PRIME) & MASK32

    shape = tuple(t.size()) if t.ndim is not None else ()
    stride = tuple(t.stride()) if hasattr(t, "stride") else ()
    mix_py(len(shape))
    for d in shape:  mix_py(d)
    mix_py(len(stride))
    for s in stride: mix_py(s)

    elem_size = int(t.element_size()) if hasattr(t, "element_size") else 0
    mix_py(elem_size)
    # dtype hash can vary across runs; mask to 32-bit for stability
    try:
        dtype_hash = int(hashlib.md5(str(t.dtype).encode()).hexdigest()[:8], 16)
        mix_py(dtype_hash)
    except Exception:
        mix_py(0)

    mix_py(int(bool(getattr(t, "requires_grad", False))))
    h_u32 = np.uint32(h)
    return torch.tensor([int(h_u32)], dtype=torch.uint32, device=torch.device("cpu"))
