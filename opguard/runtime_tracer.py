# opguard/runtime_tracer.py
import os
import sys
import time
import threading
import inspect
from datetime import datetime
from functools import wraps
from contextlib import contextmanager
import importlib
import contextvars
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from collections import deque

from opguard.distributed import get_global_rank
from opguard.common_vars import get_runtime_callstack
from opguard.utils.pkg_roots import filtered_stack_excluding_opguard
import logging

# ──────────────────────────────────────────────────────────────────────
# Trace-buffer lazy import + pending queues (avoid circular imports)
# ──────────────────────────────────────────────────────────────────────
_TB_PENDING_TRACES = deque()
_TB_PENDING_FULL   = deque()
_TB_LOCK = threading.Lock()
_TB_IMPORTING = False
_TB_DEBUG = os.getenv("OPGUARD_TB_DEBUG", "0") not in ("0", "", "false", "False", "no", "No")
_TB_INFO_PRINTED = False  # Flag to ensure we only print trace location info once


class WrapperHandle:
    def __init__(self, inner, trace):
        self._inner = inner
        self._trace = trace
        self._trace["is_async"] = True
        self._dumped = False
        self._lock = threading.Lock()

    def _dump_once(self):
        if self._dumped:
            return
        with self._lock:
            if not self._dumped:
                save_tensor_trace(**self._trace)
                self._dumped = True

    def __getattr__(self, name):
        val = getattr(self._inner, name)
        if name == "wait" and callable(val):
            def wait_wrapper(*args, **kwargs):
                r = val(*args, **kwargs)
                self._dump_once()
                return r
            return wait_wrapper
        if name == "is_completed" and callable(val):
            def is_completed_wrapper(*args, **kwargs):
                r = val(*args, **kwargs)
                if r:
                    self._dump_once()
                return r
            return is_completed_wrapper
        return val

def _dbg(msg: str):
    if _TB_DEBUG:
        try:
            print(f"[OpGuard][tracebuffer] {msg}", file=sys.stderr, flush=True)
        except Exception:
            pass

def _tracebuffer_ready():
    m = sys.modules.get("opguard.tracer.trace_buffer")
    if not m:
        return None
    need = ("ADD_TRACE", "ADD_FULL_TENSORS", "FLUSH", "FORCE_DUMP_ALL")
    return m if all(hasattr(m, a) for a in need) else None


def _maybe_init_trace_buffer():
    """Try to import the real sink exactly once when it's safe; return module if ready."""
    m = _tracebuffer_ready()
    if m:
        return m
    # Don't try to import while we're actively tracing (reduces cycle risk)
    if _in_tracer():
        return None
    # Only try after torch is present (your flow loads this early)
    if sys.modules.get("torch") is None:
        return None
    global _TB_IMPORTING
    if _TB_IMPORTING:
        return None
    try:
        _TB_IMPORTING = True
        importlib.import_module("opguard.tracer.trace_buffer")
        m = _tracebuffer_ready()
        if m:
            _dbg("trace_buffer imported; flushing pending traces")
            # Print trace location info once when trace buffer is first initialized
            global _TB_INFO_PRINTED
            if not _TB_INFO_PRINTED:
                try:
                    exec_output_dir = os.getenv("OPGUARD_EXEC_OUTPUT_DIR")
                    trace_output_base = os.getenv("OPGUARD_TRACE_OUTPUT_BASE")
                    if exec_output_dir:
                        exec_dir = exec_output_dir
                        source_files_dir = os.path.join(exec_dir, "source_files")
                    elif trace_output_base:
                        exec_dir = os.path.join(trace_output_base, "exec_trace")
                        source_files_dir = os.path.join(exec_dir, "source_files")
                    else:
                        from opguard._running import get_trace_output_dir
                        try:
                            base_dir = get_trace_output_dir("trace")
                            exec_dir = os.path.join(base_dir, "exec_trace")
                            source_files_dir = os.path.join(exec_dir, "source_files")
                        except Exception:
                            exec_dir = "unknown"
                            source_files_dir = "unknown"
                    print(f"[INFO] Execution traces: {exec_dir}, Source files: {source_files_dir}", file=sys.stderr)
                    _TB_INFO_PRINTED = True
                except Exception:
                    pass  # Don't fail if we can't determine paths
            # Flush anything we queued before the sink was ready
            with _TB_LOCK:
                while _TB_PENDING_TRACES:
                    tag, trace = _TB_PENDING_TRACES.popleft()
                    try:
                        m.ADD_TRACE(tag, trace)
                    except Exception:
                        pass
                while _TB_PENDING_FULL:
                    full = _TB_PENDING_FULL.popleft()
                    try:
                        m.ADD_FULL_TENSORS(full)
                    except Exception:
                        pass
        return m
    except Exception as e:
        _dbg(f"import failed (will retry later): {e}")
        return None
    finally:
        _TB_IMPORTING = False

def _emit_or_queue_trace(tag, trace, full_entry_or_none):
    """
    If the sink is ready (or can be safely imported now), write immediately.
    Otherwise, queue the trace and we'll flush later.
    """
    m = _tracebuffer_ready() or _maybe_init_trace_buffer()
    if m is not None:
        try:
            m.ADD_TRACE(tag, trace)
        except Exception:
            # Extremely defensive: if the sink just failed mid-run, enqueue instead
            with _TB_LOCK:
                _TB_PENDING_TRACES.append((tag, trace))
        if full_entry_or_none is not None:
            try:
                m.ADD_FULL_TENSORS(full_entry_or_none)
            except Exception:
                with _TB_LOCK:
                    _TB_PENDING_FULL.append(full_entry_or_none)
    else:
        # Sink not ready yet — queue it
        with _TB_LOCK:
            _TB_PENDING_TRACES.append((tag, trace))
            if full_entry_or_none is not None:
                _TB_PENDING_FULL.append(full_entry_or_none)

def flush_trace_buffer(force_dump=False):
    m = _tracebuffer_ready() or _maybe_init_trace_buffer()
    if m is None:
        _dbg("flush requested but sink not ready; pending traces will flush when ready")
        return
    with _TB_LOCK:
        while _TB_PENDING_TRACES:
            tag, trace = _TB_PENDING_TRACES.popleft()
            m.ADD_TRACE(tag, trace)
        while _TB_PENDING_FULL:
            full = _TB_PENDING_FULL.popleft()
            m.ADD_FULL_TENSORS(full)
    if force_dump:
        m.FLUSH(); m.FORCE_DUMP_ALL()
        print("[INFO] Force dumped all traces to disk", file=sys.stderr, flush=True)
    else:
        m.FLUSH()

# ──────────────────────────────────────────────────────────────────────
# Reentrancy guard
# ──────────────────────────────────────────────────────────────────────
_IN_TRACER = contextvars.ContextVar("opguard_in_tracer", default=False)

@contextmanager
def _reentry_guard():
    """
    Enter once sets the flag; nested uses become no-ops.
    This makes `with _reentry_guard():` a proper context manager.
    """
    if _IN_TRACER.get():
        # Already inside tracer; just yield without toggling.
        yield
        return
    token = _IN_TRACER.set(True)
    try:
        yield
    finally:
        _IN_TRACER.reset(token)

def _in_tracer():
    try:
        return _IN_TRACER.get()
    except Exception as e:
        print(f"Error while checking in_tracer: {e}", file=sys.stderr, flush=True)
        return False

# ──────────────────────────────────────────────────────────────────────
# Lazy module accessors (avoid importing torch/trace_buffer at import time)
# ──────────────────────────────────────────────────────────────────────
def _lazy_mod(modname: str):
    """Import only on demand; reuse from sys.modules if present."""
    m = sys.modules.get(modname)
    if m is None:
        m = importlib.import_module(modname)
    return m

class _NoopMV:
    __slots__ = ()
    def get_meta(self, *_a, **_k): return {}

def _mv():
    """meta_vars lazy accessor with safe fallback (no eager imports)."""
    m = sys.modules.get("opguard.tracer.meta_vars")
    if m is not None and hasattr(m, "get_meta"): return m
    try:
        return _lazy_mod("opguard.tracer.meta_vars")
    except Exception:
        return _NoopMV()

def _maybe_torch():
    return sys.modules.get("torch")

# ──────────────────────────────────────────────────────────────────────
# Tensor serialization (summary-only; no eager torch)
# ──────────────────────────────────────────────────────────────────────
def _maybe_torch_is_tensor(x):
    """Recognize tensors without importing torch."""
    tm = _maybe_torch()
    if tm is not None:
        try:
            return tm.is_tensor(x)
        except Exception:
            pass
    # Duck-typing fallback: torch tensors expose these attributes and come from torch.* modules
    t = type(x)
    mod = getattr(t, "__module__", "")
    return (("torch" in mod) and hasattr(x, "dtype") and hasattr(x, "device") and hasattr(x, "shape"))

def _summarize_tensor_safe(tensor):
    """
    Return a cheap metadata summary; if torch is loaded and we're not reentering,
    try the real summarizer. Under _reentry_guard() we always stay metadata-only.
    """
    if _in_tracer():
        # Do not reenter with any heavy ops; return metadata we can get without torch functions.
        return {
            "type": "Tensor",
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(getattr(tensor, "dtype", None), "name", getattr(tensor, "dtype", None))),
            "device": str(getattr(tensor, "device", None)),
            "note": "reentry-safe summary",
        }
    tm = _maybe_torch()
    if tm is None:
        # Torch not fully imported yet; avoid importing it here.
        return {
            "type": "Tensor",
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(getattr(tensor, "dtype", None), "name", getattr(tensor, "dtype", None))),
            "device": str(getattr(tensor, "device", None)),
            "note": "torch-not-loaded summary",
        }
    # Now torch is loaded; we can try the real summarizer lazily
    try:
        from opguard.tracer.tensor_dump import summarize_tensor  # lazy import
        with _reentry_guard():
            return summarize_tensor(tensor)
    except Exception as e:
        print(f"[OpGuard]: summarize_tensor failed: {e}", file=sys.stderr, flush=True)
        # As a last resort, return a cheap metadata summary
        return {
            "type": "Tensor",
            "shape": tuple(getattr(tensor, "shape", ())),
            "dtype": str(getattr(getattr(tensor, "dtype", None), "name", getattr(tensor, "dtype", None))),
            "device": str(getattr(tensor, "device", None)),
            "note": "fallback summary",
        }

def is_serializable(obj):
    import numbers
    return isinstance(obj, (numbers.Number, str, list, dict, tuple, type(None)))

def serialize_func_name(func):
    if callable(func):
        try:
            name = getattr(func, "__name__", "<anonymous>")
            module = getattr(func, "__module__", "<unknown>")
            return f"<{module}.{name}>"
        except Exception:
            return "<function>"
    return "<non-function>"

def serialize_tensor_io(obj, dump_mode="summary"):
    assert dump_mode in ["summary"], f"Invalid dump_mode: {dump_mode}"
    if _maybe_torch_is_tensor(obj):
        return _summarize_tensor_safe(obj)
    if isinstance(obj, (list, tuple)):
        return [serialize_tensor_io(x, dump_mode=dump_mode) for x in obj]
    if isinstance(obj, dict):
        return {k: serialize_tensor_io(v, dump_mode=dump_mode) for k, v in obj.items()}
    if callable(obj):
        return serialize_func_name(obj)
    if is_serializable(obj):
        return obj
    return f"<nonserializable: {type(obj).__name__}>"

# ──────────────────────────────────────────────────────────────────────
# Trace building/sink
# ──────────────────────────────────────────────────────────────────────
_counter_lock = threading.Lock()
_counter = {"counter": 0}

def _reserve_counter() -> int:
    with _counter_lock:
        _counter["counter"] += 1
        return _counter["counter"]


# Stack-trace cache: one entry per (thread_id, tag_prefix, lineno).
# This is intentionally unbounded per user request.
_STACK_TRACE_CACHE = {}


def _get_cached_stack_trace(thread_id: int, tag_prefix: str, lineno):
    key = (thread_id, tag_prefix, lineno)
    cached = _STACK_TRACE_CACHE.get(key)
    if cached is not None:
        return cached

    stack = filtered_stack_excluding_opguard()
    # Lock-free insert: if another thread already stored this key, reuse its value.
    return _STACK_TRACE_CACHE.setdefault(key, stack)

def save_tensor_trace(
    tag,
    self,
    args,
    kwargs,
    outputs,
    is_async=False,
    metadata=None,
    tag_prefix="trace",
    dump_mode="full",
    launch_time=None,
    *,
    counter_override: int | None = None,
    thread_id_override: int | None = None,
    stack_trace_override: str | None = None,
):
    gpu_rank = get_global_rank()
    tm = _maybe_torch()

    # Use reserved counter/thread captured at call time when provided
    if counter_override is None:
        counter = _reserve_counter()
    else:
        counter = counter_override

    thread_id = thread_id_override if thread_id_override is not None else threading.get_ident()

    fname = tag_prefix
    fn_name = serialize_func_name(args[0]) if args else "<unknown>"

    line_num = (metadata or {}).get("lineno", None)
    # Use provided stack trace if available (captured at call site before thread dispatch),
    # otherwise capture it here (may only show worker thread stack for async operations)
    if stack_trace_override is not None:
        stack_trace_str = stack_trace_override if get_runtime_callstack() else None
    else:
        stack_trace_str = filtered_stack_excluding_opguard() if get_runtime_callstack() else None

    summary_in = {
        "self": serialize_tensor_io(self, dump_mode="summary"),
        "args": serialize_tensor_io(args, dump_mode="summary"),
        "kwargs": serialize_tensor_io(kwargs, dump_mode="summary"),
    }
    summary_out = serialize_tensor_io(outputs, dump_mode="summary")

    trace = {
        "counter": counter,
        "pid": os.getpid(),
        "thread_id": thread_id,
        "timestamp": launch_time,
        "function": fn_name,
        "inputs": summary_in,
        "outputs": summary_out,
        "is_async": is_async,
        "meta": metadata,
        "line_num": line_num,
        "dump_mode": dump_mode,
        "fname": fname,
        "tag_prefix": tag_prefix,
        "gpu_rank": gpu_rank,
    }
    if stack_trace_str:
        trace["stack_trace"] = stack_trace_str

    # Write now if the sink is ready, else queue until it is.
    full_entry = None
    if dump_mode in ("full", "both") and tm is not None:
        full_entry = {
            "fname": fname,
            "inputs": {"args": args, "kwargs": kwargs},
            "outputs": outputs,
            "tag_prefix": tag_prefix,
            "gpu_rank": gpu_rank,
            "function": fn_name,
            "lineno": line_num,
            "meta": {**(metadata or {}), **_mv().get_meta(gpu_rank)},
        }

    _emit_or_queue_trace(tag, trace, full_entry)



def is_worklike(obj) -> bool:
    try:
        if obj is None:
            return False
        if hasattr(obj, "is_completed") and callable(getattr(obj, "is_completed")):
            return True
        if hasattr(obj, "wait") and callable(getattr(obj, "wait")):
            return True
        if hasattr(obj, "get_future"):
            gf = getattr(obj, "get_future")
            if callable(gf):
                return True
    except Exception:
        print(f"[OpGuard] Failed to check if obj is worklike: {obj}", file=sys.stderr, flush=True)
    return False


def is_namedtuple_instance(x) -> bool:
    return isinstance(x, tuple) and hasattr(x, "_fields")

def wrap_and_trace(obj, trace):
    if is_worklike(obj):
        return WrapperHandle(obj, trace), True

    if isinstance(obj, dict):
        changed = False
        new_dict = None
        for k, v in obj.items():
            nv, ch = wrap_and_trace(v, trace)
            if ch:
                if new_dict is None:
                    new_dict = dict(obj)
                new_dict[k] = nv
                changed = True
        return (new_dict, True) if changed else (obj, False)

    if isinstance(obj, list):
        changed = False
        new_list = None
        for i, x in enumerate(obj):
            nx, ch = wrap_and_trace(x, trace)
            if ch:
                if new_list is None:
                    new_list = list(obj)
                new_list[i] = nx
                changed = True
        return (new_list, True) if changed else (obj, False)

    if is_namedtuple_instance(obj):
        field_names = obj._fields
        updates = {}
        changed = False
        for name in field_names:
            val = getattr(obj, name)
            nval, ch = wrap_and_trace(val, trace)
            if ch:
                updates[name] = nval
                changed = True
        if changed:
            return obj._replace(**updates), True
        return obj, False

    if isinstance(obj, tuple):
        changed = False
        tmp = None
        for i, x in enumerate(obj):
            nx, ch = wrap_and_trace(x, trace)
            if ch:
                if tmp is None:
                    tmp = list(obj)
                tmp[i] = nx
                changed = True
        return (tuple(tmp), True) if changed else (obj, False)

    return obj, False

def _get_tensor_from_bound_method(obj):
    if hasattr(obj, "__self__"):
        if _maybe_torch_is_tensor(obj.__self__):
            return obj.__self__
    return None


def _get_meta_info(lineno=None):
    gpu_rank = get_global_rank()

    return {"lineno": lineno, **_mv().get_meta(gpu_rank)}

def scoped_tensor_tracer(tag_prefix="trace", lineno=None, dump_tensors=False, global_sync=False):
    """
    Decorator to wrap any function with tensor I/O tracing.
    A unique tag is attached to each dump via `tag_prefix`.
    """
    call_id = [0]

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if _in_tracer():
                return fn(*args, **kwargs)

            # Capture call-time invariants we must preserve even if we emit later
            call_id[0] += 1
            launch_time = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            counter_now = _reserve_counter()  # reserve counter now
            thread_id_now = threading.get_ident()
            
            # Capture stack trace at call site (before function execution, which may spawn threads)
            # This ensures we capture the original call stack even if the operation runs in a worker thread
            if get_runtime_callstack():
                stack_trace_at_call = _get_cached_stack_trace(
                    thread_id=thread_id_now,
                    tag_prefix=tag_prefix,
                    lineno=lineno,
                )
            else:
                stack_trace_at_call = None

            with _reentry_guard():
                result = fn(*args, **kwargs)
            
            trace = {
                "tag": f"{tag_prefix}_{call_id[0]}",
                "self": _get_tensor_from_bound_method(args[0]),
                "args": args,
                "kwargs": kwargs,
                "outputs": result,
                "is_async": False,
                "metadata": _get_meta_info(lineno),
                "tag_prefix": tag_prefix,
                "dump_mode": "summary" if not dump_tensors else "full",
                "launch_time": launch_time,
                "counter_override": counter_now,
                "thread_id_override": thread_id_now,
                "stack_trace_override": stack_trace_at_call,
            }
            result, is_sync = wrap_and_trace(result, trace)
            if global_sync:
                tm = _maybe_torch()
                if tm is not None:
                    try: tm.cuda.synchronize()
                    except Exception: pass
            if not is_sync:
                save_tensor_trace(**trace)
            
            return result
        return wrapper
    return decorator
