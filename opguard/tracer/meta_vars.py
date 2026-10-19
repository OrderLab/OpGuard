# opguard/tracer/meta_vars.py
import sys
import threading
from opguard.utils.utils import get_id, _infer_rank
from opguard.distributed import get_world_size as _safe_world_size

META_VARS = None          # list[dict]
ID = None
LOCK = threading.Lock()

def _ensure_len(n: int):
    """Grow META_VARS to at least n entries."""
    global META_VARS
    if META_VARS is None:
        META_VARS = [{} for _ in range(n)]
    elif len(META_VARS) < n:
        META_VARS.extend({} for _ in range(n - len(META_VARS)))

def get_meta_vars() -> list[dict]:
    """Initialize to a safe size without requiring dist init; grow later as needed."""
    global META_VARS, ID
    target = _safe_world_size()
    if META_VARS is None or len(META_VARS) < target:
        with LOCK:
            target2 = _safe_world_size()  # recompute inside lock
            _ensure_len(target2)
            if ID is None:
                ID = get_id()
                print(
                    f"[OpGuard]: meta_vars {ID} init (size={len(META_VARS)})",
                    file=sys.stderr,
                    flush=True,
                )
    return META_VARS


def update_meta(prefix=None, rank: int | None = None, **kwargs):
    meta_vars = get_meta_vars()
    if rank is None:
        rank = _infer_rank()
    # If rank is larger than current capacity (e.g., we started at 1 then later learned world size), grow.
    if rank >= len(meta_vars):
        with LOCK:
            _ensure_len(rank + 1)
    meta = meta_vars[rank]
    if prefix:
        meta.update({f"{prefix}/{k}": v for k, v in kwargs.items()})
    else:
        meta.update(kwargs)

def get_meta(rank: int):
    meta_vars = get_meta_vars()
    if rank >= len(meta_vars):
        with LOCK:
            _ensure_len(rank + 1)
    return meta_vars[rank]
