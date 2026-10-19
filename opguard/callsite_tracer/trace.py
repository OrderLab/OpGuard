# --- Unified CallsiteTracer: CUDA-API + CUPTI + Python ------------------------
import os
from opguard.local_sync.paths import atomic_output
import json
import traceback
import functools
from collections import defaultdict
import threading
import sys
import ctypes
from pathlib import Path

import torch
from torch._ops import OpOverloadPacket
from torch.utils._python_dispatch import TorchDispatchMode

# ──────────────────────────────────────────────────────────────────────────────
# Optional backends
try:
    import opguard.callsite_tracer.opguard_cupti as _opguard_cupti
    _CUPTI_AVAILABLE = True
except Exception:
    _opguard_cupti = None
    _CUPTI_AVAILABLE = False

try:
    import triton.runtime.jit
    _TRITON_AVAILABLE = True
except Exception:
    _TRITON_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Env knobs (keep it lean)
def _truthy(v, default=True):
    if v is None:
        return default
    return v not in ("0", "false", "False", "", "no", "No", "off", "OFF")

_ENABLE_CUDAAPI = _truthy(os.getenv("OPGUARD_CALLSITE_CUDAAPI"), True)
_ENABLE_CUPTI   = _truthy(os.getenv("OPGUARD_CALLSITE_CUPTI"),   True)
_ENABLE_PYTHON  = _truthy(os.getenv("OPGUARD_CALLSITE_PYTHON"),  True)

_PY_STRATEGY    = os.getenv("OPGUARD_PYTHON_STRATEGY", "ops+dispatch").lower()  # "dispatch"|"ops+dispatch"
_VALID_PY_STRATS = {"dispatch", "ops+dispatch"}
if _PY_STRATEGY not in _VALID_PY_STRATS:
    _PY_STRATEGY = "ops+dispatch"


def _env_nonneg_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return default


# Memory caps for the in-process unique-stack aggregator.
# 0 = unlimited. Non-zero defaults avoid unbounded GCPU growth on long jobs.
# Instrumentation mainly needs unique sites, not every distinct full stack.
_CS_MAX_FRAMES = _env_nonneg_int("OPGUARD_CS_MAX_FRAMES", 32)
_CS_MAX_STACKS_PER_SITE = _env_nonneg_int("OPGUARD_CS_MAX_STACKS_PER_SITE", 8)
_CS_MAX_SITES = _env_nonneg_int("OPGUARD_CS_MAX_SITES", 20000)
_CS_MAX_TOTAL_STACKS = _env_nonneg_int("OPGUARD_CS_MAX_TOTAL_STACKS", 50000)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
def _rank_or_zero():
    from opguard.distributed import get_global_rank
    return get_global_rank()

def _with_prefix(base_path: str, prefix: str) -> str:
    base_dir = os.path.dirname(base_path) or "."
    base_name = os.path.basename(base_path) or "callsite_tracer.json"
    name, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".json"
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{prefix}_{name}{ext}")

def _ser_streams(sset): return [[d, s] for (d, s) in sorted(sset)]
def _ser_counts(dd):    return {k: int(v) for k, v in dd.items()}

def _load_handle(path):
    try:
        return ctypes.PyDLL(path)   # prefer PyDLL so GIL is held during drain
    except OSError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Stats structure (shared across backends)
def _entry():
    return {
        "count": 0,
        "callstacks": set(),  # set[tuple[ (file,line,func), ... ]]
        "ops": set(),
        "streams": set(),                      # set[(device,stream)]
        "stream_counts": defaultdict(int),     # "dev:stream" -> int
        "by_op": defaultdict(lambda: {
            "count": 0,
            "streams": set(),
            "stream_counts": defaultdict(int),
        }),
    }

# ──────────────────────────────────────────────────────────────────────────────
# Optional Native CUDA-API bridge (ctypes)
class _NativeCudaApi:
    def __init__(self):
        self.available = False
        self._lib_cdll = None      # for start/flush/stop (no Python C-API use)
        self._lib_pydll = None     # for drain (must hold the GIL)

        self._start_mem = None
        self._drain     = None
        self._flush     = None
        self._stop      = None
        self._set_ready = None     # optional gate

        try:
            RTLD_NOW      = getattr(ctypes, "RTLD_NOW",      2)
            RTLD_GLOBAL   = getattr(ctypes, "RTLD_GLOBAL",   0x100)
            RTLD_NOLOAD   = getattr(ctypes, "RTLD_NOLOAD",   0x4)
            RTLD_NODELETE = getattr(ctypes, "RTLD_NODELETE", 0x1000)

            # ---------- 1) Prefer already-loaded handles ----------
            # CDLL(None) and PyDLL(None) refer to the main program handle
            cdll_none = ctypes.CDLL(None)
            pydll_none = ctypes.PyDLL(None)

            have_all = all(
                hasattr(cdll_none, nm) for nm in
                ("pyhook_start_mem", "pyhook_flush", "pyhook_stop")
            ) and hasattr(pydll_none, "pyhook_drain")

            if have_all:
                self._lib_cdll  = cdll_none
                self._lib_pydll = pydll_none
            else:
                # ---------- 2) Try path from env or packaged fallback ----------
                path = os.getenv("OPGUARD_NATIVE_HOOK_PATH")
                if not path:
                    try:
                        import opguard
                        path = str(
                            Path(opguard.__file__).resolve().parent /
                            "callsite_tracer" / "native" / "libpylaunch_hook.so"
                        )
                    except Exception:
                        path = None

                if path and os.path.exists(path):
                    # Try NOLOAD first to reuse an LD_PRELOADed instance (shares globals/queue)
                    try:
                        self._lib_cdll = ctypes.CDLL(path, mode=RTLD_NOLOAD | RTLD_GLOBAL)
                    except OSError:
                        # load it if not resident
                        self._lib_cdll = ctypes.CDLL(path, mode=RTLD_NOW | RTLD_GLOBAL | RTLD_NODELETE)
                    # Always open a PyDLL handle for drain so the GIL is held
                    try:
                        self._lib_pydll = ctypes.PyDLL(path)   # separate handle is fine
                    except OSError:
                        # If PyDLL(path) fails but symbols are in the main handle, fall back to PyDLL(None)
                        self._lib_pydll = pydll_none

            # ---------- 3) Bind symbols from the chosen handles ----------
            lib_c = self._lib_cdll
            lib_p = self._lib_pydll

            if lib_c and lib_p:
                # functions that don't touch Python C-API → CDLL
                self._start_mem = getattr(lib_c, "pyhook_start_mem", None)
                self._flush     = getattr(lib_c, "pyhook_flush", None)
                self._stop      = getattr(lib_c, "pyhook_stop", None)

                # drain builds Python objects → PyDLL (GIL held)
                self._drain     = getattr(lib_p, "pyhook_drain", None)

                # optional ready gate (either handle is fine)
                self._set_ready = getattr(lib_c, "pyhook_set_ready", None) or \
                                getattr(lib_p, "pyhook_set_ready", None)

                if self._start_mem and self._drain and self._flush and self._stop:
                    # signatures
                    self._start_mem.argtypes = [ctypes.c_int]
                    self._start_mem.restype  = None

                    self._drain.argtypes     = [ctypes.c_int]
                    self._drain.restype      = ctypes.py_object  # IMPORTANT

                    self._flush.argtypes     = []
                    self._flush.restype      = None

                    self._stop.argtypes      = []
                    self._stop.restype       = None

                    if self._set_ready:
                        self._set_ready.argtypes = [ctypes.c_int]
                        self._set_ready.restype  = None

                    self.available = True

        except Exception:
            print("[CallsiteTracer] _NativeCudaApi failed to initialize", file=sys.stderr, exc_info=True)
            self.available = False


    # API
    def start(self, queue_limit: int | None = None):
        if not self.available: return
        self._start_mem(int(queue_limit or 200000))

    def drain(self, max_items: int = 5000):
        if not self.available: return []
        try:
            return self._drain(int(max_items)) or []
        except Exception:
            return []

    def flush(self):
        if not self.available: return
        try: self._flush()
        except Exception: pass

    def stop(self):
        if not self.available: return
        try: self._stop()
        except Exception: pass


# ──────────────────────────────────────────────────────────────────────────────
# The coordinator
class CallsiteTracer(TorchDispatchMode):
    """
    Runs any combination of:
      - CUDA-API/native tracer (enqueue via native hook -> Python drain thread)
      - CUPTI tracer           (opguard_cupti.start_mem + drain thread)
      - Python tracer          (TorchDispatch/Triton hooks)

    Each backend aggregates into its own map and dumps to its own file:
      cudaapi_<name>.json, cupti_<name>.json, python_<name>.json
    """
    def __init__(self, dump_every=10000, dump_path=None,
                 *,
                 enable_cudaapi: bool | None = None,
                 enable_cupti:   bool | None = None,
                 enable_python:  bool | None = None,
                 python_strategy: str | None = None):
        # knobs
        self.dump_every = int(dump_every or 0)
        self._next_dump_total = self.dump_every if self.dump_every else None
        self._lock = threading.Lock()

        # enable flags (default to envs)
        self.enable_cudaapi = _ENABLE_CUDAAPI if enable_cudaapi is None else bool(enable_cudaapi)
        self.enable_cupti   = _ENABLE_CUPTI   if enable_cupti   is None else bool(enable_cupti)
        self.enable_python  = _ENABLE_PYTHON  if enable_python  is None else bool(enable_python)
        # CPU position/index calculations can affect subsequent GPU operators.
        self.include_cpu = os.getenv("OPGUARD_CALLSITE_INCLUDE_CPU", "0").lower() in ("1", "true", "yes")
        self.python_strategy = (python_strategy or _PY_STRATEGY)
        self._python_patch_ops = (self.python_strategy == "ops+dispatch")

        # paths
        if dump_path is None:
            from opguard._running import get_output_subdir
            dump_path = os.path.join(get_output_subdir("callsite_trace"), "callsite_tracer.json")
        self._base_dump_path = dump_path
        self._paths = {}  # src -> file path

        # stats per backend
        self.stats = {
            "cudaapi": defaultdict(_entry),
            "cupti":   defaultdict(_entry),
            "python":  defaultdict(_entry),
        }
        self._totals = {"cudaapi": 0, "cupti": 0, "python": 0}
        self._total_all = 0
        self._total_unique_stacks = 0
        self._limit_hits = set()

        # Unique-stack / site caps (0 = unlimited). Env defaults are applied above.
        self.max_frames = _CS_MAX_FRAMES
        self.max_stacks_per_site = _CS_MAX_STACKS_PER_SITE
        self.max_sites = _CS_MAX_SITES
        self.max_total_stacks = _CS_MAX_TOTAL_STACKS

        # python mode internals
        self._in_ops = False
        self._in_dispatch = False
        self._in_triton = False
        self._orig_ops_call = OpOverloadPacket.__call__
        if _TRITON_AVAILABLE:
            self._orig_triton_run = triton.runtime.jit.JITFunction.run
        self._mode_pushed = False

        # backend handles
        self._cupti_active = False
        self._pump_cupti_stop = None
        self._pump_cupti_thread = None

        self._cudaapi = _NativeCudaApi()
        self._cudaapi_active = False
        self._pump_cudaapi_stop = None
        self._pump_cudaapi_thread = None

        self._start_count = 0

    # ── Context mgr ───────────────────────────────────────────────────────────
    def __enter__(self):
        self._begin()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._end(exc_type, exc_val, exc_tb)
        return False

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def _begin(self):
        # rank-aware final paths with prefixes
        rank = _rank_or_zero()
        base_dir = os.path.dirname(self._base_dump_path) or "."
        base_name = os.path.basename(self._base_dump_path)
        name, ext = os.path.splitext(base_name)
        if "rank" not in name:
            name = f"{name}_rank{rank}"
        base_path = os.path.join(base_dir, f"{name}{ext or '.json'}")
        self._paths["cudaapi"] = _with_prefix(base_path, "cudaapi")
        self._paths["cupti"]   = _with_prefix(base_path, "cupti")
        self._paths["python"]  = _with_prefix(base_path, "python")
        print(
            f"[CallsiteTracer] memory caps  frames={self.max_frames or 'unlimited'} "
            f"stacks/site={self.max_stacks_per_site or 'unlimited'} "
            f"sites={self.max_sites or 'unlimited'} "
            f"total_stacks={self.max_total_stacks or 'unlimited'}  "
            f"(0 via OPGUARD_CS_MAX_* = unlimited)",
            file=sys.stderr,
        )

        # Start CUDA-API/native
        if self.enable_cudaapi and not self._cudaapi.available:
            print("[CallsiteTracer] Native CUDA-API hook unavailable; CUDA-API capture is disabled. Run `opguard doctor`.", file=sys.stderr)
        if self.enable_cudaapi and self._cudaapi.available and not self._cudaapi_active:
            try:
                self._cudaapi.start(queue_limit=int(os.getenv("OPGUARD_NATIVE_QUEUE_LIMIT", "200000")))
                self._cudaapi_active = True
                self._pump_cudaapi_stop = threading.Event()
                self._pump_cudaapi_thread = threading.Thread(target=self._cudaapi_pump, daemon=True)
                self._pump_cudaapi_thread.start()
                print(f"[CallsiteTracer] CUDA-API backend ON (mem queue)", file=sys.stderr)
            except Exception as e:
                print(f"[CallsiteTracer] CUDA-API backend failed to start: {e}", file=sys.stderr)

        # Start CUPTI
        if self.enable_cupti and not _CUPTI_AVAILABLE:
            print("[CallsiteTracer] CUPTI extension unavailable; CUPTI capture is disabled. Run `opguard doctor`.", file=sys.stderr)
        if self.enable_cupti and _CUPTI_AVAILABLE and not self._cupti_active:
            try:
                _opguard_cupti.start_mem()
                self._cupti_active = True
                self._pump_cupti_stop = threading.Event()
                self._pump_cupti_thread = threading.Thread(target=self._cupti_pump, daemon=True)
                self._pump_cupti_thread.start()
                print(f"[CallsiteTracer] CUPTI backend ON (mem queue)", file=sys.stderr)
            except Exception as e:
                print(f"[CallsiteTracer] CUPTI start failed: {e}", file=sys.stderr)
                self._cupti_active = False

        # Start Python (TorchDispatch/Triton)
        if self.enable_python:
            if self._python_patch_ops:
                self._patch_ops()
            if _TRITON_AVAILABLE:
                self._patch_triton()
            super().__enter__()               # push TorchDispatchMode
            self._mode_pushed = True
            print(f"[CallsiteTracer] Python backend ON ({self.python_strategy})", file=sys.stderr)

    def _end(self, exc_type=None, exc_val=None, exc_tb=None):
        # stop pumps first to stabilize stats
        # CUDA-API
        if self._cudaapi_active:
            try:
                if self._pump_cudaapi_stop: self._pump_cudaapi_stop.set()
                if self._pump_cudaapi_thread: self._pump_cudaapi_thread.join(timeout=2.0)
            except Exception: pass
            try: self._cudaapi.flush()
            except Exception: pass
            try:
                # final drain
                rem = self._cudaapi.drain(1_000_000)
                if rem: self._ingest_event_list(rem, src="cudaapi")
            except Exception as e:
                print(f"[CallsiteTracer] native final drain error: {e}", file=sys.stderr)
            try: self._cudaapi.stop()
            except Exception as e:
                print(f"[CallsiteTracer] native stop error: {e}", file=sys.stderr)
            self._cudaapi_active = False

        # CUPTI
        if self._cupti_active:
            try:
                if self._pump_cupti_stop: self._pump_cupti_stop.set()
                if self._pump_cupti_thread: self._pump_cupti_thread.join(timeout=2.0)
            except Exception: pass
            try: _opguard_cupti.flush()
            except Exception: pass
            try:
                remaining = _opguard_cupti.drain(1_000_000)
                if remaining: self._ingest_event_list(remaining, src="cupti")
            except Exception as e:
                print(f"[CallsiteTracer] CUPTI final drain error: {e}", file=sys.stderr)
            try: _opguard_cupti.stop()
            except Exception as e:
                print(f"[CallsiteTracer] CUPTI stop error: {e}", file=sys.stderr)
            self._cupti_active = False

        # Python backend unpatch/pop mode
        if self.enable_python:
            try:
                if self._python_patch_ops:
                    OpOverloadPacket.__call__ = self._orig_ops_call
                if _TRITON_AVAILABLE:
                    triton.runtime.jit.JITFunction.run = self._orig_triton_run
            except Exception:
                pass
            try:
                if self._mode_pushed:
                    super().__exit__(exc_type, exc_val, exc_tb)
            except Exception:
                pass

        # final dumps
        self._dump_all(final=True)

    # ── Python backend patching ────────────────────────────────────────────────
    def _patch_ops(self):
        tracer = self
        orig = self._orig_ops_call

        @functools.wraps(orig)
        def _wrapped_call(self_op, *args, **kwargs):
            if tracer._in_dispatch or tracer._in_triton:
                return orig(self_op, *args, **kwargs)
            tracer._in_ops = True
            try:
                result = orig(self_op, *args, **kwargs)
                tracer._record_callsite(traceback.extract_stack(limit=tracer.max_frames or None), func=str(self_op), src="python")
                return result
            finally:
                tracer._in_ops = False

        OpOverloadPacket.__call__ = _wrapped_call

    def _patch_triton(self):
        tracer = self
        orig = self._orig_triton_run

        @functools.wraps(orig)
        def _wrapped_triton_run(self_jit, *args, **kwargs):
            if tracer._in_ops or tracer._in_dispatch:
                return orig(self_jit, *args, **kwargs)
            if not getattr(self_jit, "_compiled", False):
                return orig(self_jit, *args, **kwargs)
            tracer._in_triton = True
            try:
                result = orig(self_jit, *args, **kwargs)
                kernel_name = getattr(self_jit.fn, "__name__", "<triton_kernel>")
                tracer._record_callsite(traceback.extract_stack(limit=tracer.max_frames or None), func=kernel_name, src="python")
                return result
            finally:
                tracer._in_triton = False

        triton.runtime.jit.JITFunction.run = _wrapped_triton_run

    # ── TorchDispatch: also tags streams for CUPTI if active ──────────────────
    def _first_cuda_device_from_args(self, args, kwargs):
        for x in (args or ()):
            if isinstance(x, torch.Tensor) and x.is_cuda:
                return x.device.index
        for v in (kwargs or {}).values():
            if isinstance(v, torch.Tensor) and v.is_cuda:
                return v.device.index
        return torch.cuda.current_device()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        # Tag streams for CUPTI if that backend is active
        if self._cupti_active and self._found_cuda(args, kwargs):
            stack  = traceback.extract_stack(limit=self.max_frames or None)
            py_stack = [[f.filename, f.lineno, f.name] for f in stack]
            try:
                dev = self._first_cuda_device_from_args(args, kwargs)
                s   = torch.cuda.current_stream(dev)
                stream_ptr = int(s.cuda_stream)
                tag_id = _opguard_cupti.tag_stream_push(dev, stream_ptr, py_stack, str(func))
            except Exception:
                tag_id = None
            try:
                out = func(*args, **kwargs)
            finally:
                if tag_id is not None:
                    try: _opguard_cupti.tag_stream_pop(dev, stream_ptr, tag_id)
                    except Exception: pass
        else:
            out = func(*args, **kwargs)

        # Always record a Python callsite if python backend is enabled
        if self.enable_python and (self._found_cuda(args, kwargs) or
                                   self._found_cuda((out,), {}) or
                                   (self.include_cpu and self._found_tensor((args, kwargs, out)))):
            self._record_callsite(traceback.extract_stack(limit=self.max_frames or None), func=str(func), src="python")
        return out

    def _found_cuda(self, args, kwargs):
        return self._found_tensor((args, kwargs), cuda_only=True)

    @staticmethod
    def _found_tensor(value, cuda_only=False):
        if isinstance(value, torch.Tensor):
            return value.is_cuda if cuda_only else True
        if isinstance(value, dict):
            return any(CallsiteTracer._found_tensor(v, cuda_only) for v in value.values())
        if isinstance(value, (tuple, list)):
            return any(CallsiteTracer._found_tensor(v, cuda_only) for v in value)
        return False

    # ── Pumps (CUPTI & CUDA-API) ──────────────────────────────────────────────
    def _cupti_pump(self):
        SLEEP = 1.0
        BATCH = 5000
        stop = self._pump_cupti_stop
        while stop and not stop.is_set():
            try:
                events = _opguard_cupti.drain(BATCH)
                if events:
                    self._ingest_event_list(events, src="cupti")
                    self._maybe_dump()
            except Exception as e:
                print(f"[CallsiteTracer] _cupti_pump error: {e}", file=sys.stderr)
            stop.wait(SLEEP)

    def _cudaapi_pump(self):
        SLEEP = 1.0
        BATCH = 5000
        stop = self._pump_cudaapi_stop
        while stop and not stop.is_set():
            try:
                events = self._cudaapi.drain(BATCH)
                if events:
                    self._ingest_event_list(events, src="cudaapi")
                    self._maybe_dump()
            except Exception as e:
                print(f"[CallsiteTracer] _cudaapi_pump error: {e}", file=sys.stderr)
            stop.wait(SLEEP)

    # ── Memory caps ───────────────────────────────────────────────────────────
    def _note_limit(self, kind: str):
        if kind in self._limit_hits:
            return
        self._limit_hits.add(kind)
        print(f"[CallsiteTracer] hit {kind} cap; further unique entries are dropped "
              f"(counts for existing sites still increment)", file=sys.stderr)

    def _trim_frames(self, frames):
        if not frames or not self.max_frames:
            return frames
        if len(frames) <= self.max_frames:
            return frames
        # Keep innermost frames (root→leaf order, same as extract_stack(limit=N))
        return frames[-self.max_frames:]

    def _get_site(self, sdict, key):
        e = sdict.get(key)
        if e is not None:
            return e
        if self.max_sites and len(sdict) >= self.max_sites:
            self._note_limit("sites")
            return None
        e = _entry()
        sdict[key] = e
        return e

    def _add_callstack(self, e, frames):
        frames = self._trim_frames(frames)
        key = tuple(tuple(x) for x in frames)
        stacks = e["callstacks"]
        if key in stacks:
            return
        if self.max_stacks_per_site and len(stacks) >= self.max_stacks_per_site:
            self._note_limit("stacks_per_site")
            return
        if self.max_total_stacks and self._total_unique_stacks >= self.max_total_stacks:
            self._note_limit("total_stacks")
            return
        stacks.add(key)
        self._total_unique_stacks += 1

    # ── Ingestion (shared for CUPTI & CUDA-API) ───────────────────────────────
    def _ingest_event_list(self, events, src: str):
        # events: list[dict] with keys: kernel, device, stream, start, end, stack
        FILTER_FUNCS = {"_wrapped_call", "_wrapped_triton_run", "__torch_dispatch__", "__call__", "_call_impl", "_wrapped_call_impl"}

        def lowest_frame(stack):
            for f in reversed(stack):
                file, line, func = f
                if func in FILTER_FUNCS:   # keep it lean; no path filtering here
                    continue
                return (file, line, func)
            return None

        with self._lock:
            sdict = self.stats[src]
            for ev in events:
                kernel = ev.get("kernel", "unknown")
                stack  = ev.get("stack") or []
                lf = lowest_frame(stack)
                if lf is None:
                    lf = tuple(stack[-1]) if stack else ("<unknown>", 0, "<unknown>")
                key = f"{lf[0]}:{lf[1]} in {lf[2]}"
                e = self._get_site(sdict, key)
                if e is None:
                    continue
                e["count"] += 1
                e["ops"].add(kernel)
                self._add_callstack(e, stack)

                dev = ev.get("device"); st = ev.get("stream")
                if dev is not None and st is not None:
                    dev = int(dev); st = int(st)
                    e["streams"].add((dev, st))
                    ds_key = f"{dev}:{st}"
                    e["stream_counts"][ds_key] = e["stream_counts"].get(ds_key, 0) + 1
                    opb = e["by_op"][kernel]
                    opb["count"] += 1
                    opb["streams"].add((dev, st))
                    opb["stream_counts"][ds_key] += 1

                self._totals[src] += 1
                self._total_all  += 1

    # ── Python record ─────────────────────────────────────────────────────────
    def _record_callsite(self, stack, func=None, src="python"):
        lowest = None
        for fr in reversed(stack):
            if "site-packages/torch" in fr.filename or "/torch/" in fr.filename:
                continue
            if fr.name in ("_wrapped_call", "_wrapped_triton_run", "__torch_dispatch__"):
                continue
            lowest = fr
            break
        if lowest is None:
            return

        key = f"{lowest.filename}:{lowest.lineno} in {lowest.name}"
        callstack = [(f.filename, f.lineno, f.name) for f in stack
                     if f.name not in ("_wrapped_call", "_wrapped_triton_run")]
        try:
            opname = str(func) if func is not None else "unknown"
        except Exception:
            opname = "unknown"

        with self._lock:
            sdict = self.stats[src]
            e = self._get_site(sdict, key)
            if e is None:
                return
            e["count"] += 1
            self._add_callstack(e, callstack)
            e["ops"].add(opname)
            self._totals[src] += 1
            self._total_all   += 1
        self._maybe_dump()


    # ── Dumping ───────────────────────────────────────────────────────────────
    def _maybe_dump(self):
        if not self.dump_every:
            return
        if self._next_dump_total is None:
            self._next_dump_total = self.dump_every
        while self._total_all >= self._next_dump_total:
            self._dump_all(final=False)
            self._next_dump_total += self.dump_every

    def _dump_one(self, src: str, to_path: str):
        # Take a *snapshot* of the stats under the lock so we don't race
        with self._lock:
            sdict = self.stats[src]
            if not sdict:
                return

            out = {}
            for key, v in sdict.items():
                node = {
                    "count": v["count"],
                    "callstacks": [
                        [list(frame) for frame in cs]
                        for cs in v["callstacks"]
                    ],
                    "ops": sorted(list(v["ops"])),
                    "streams": _ser_streams(v["streams"]),
                    "stream_counts": _ser_counts(v["stream_counts"]),
                    "by_op": {},
                }
                for kname, ob in v["by_op"].items():
                    node["by_op"][kname] = {
                        "count": ob["count"],
                        "streams": _ser_streams(ob["streams"]),
                        "stream_counts": _ser_counts(ob["stream_counts"]),
                    }
                out[key] = node

        # After we’ve built `out`, no need to hold the lock for IO.
        if not out:
            return

        os.makedirs(os.path.dirname(to_path) or ".", exist_ok=True)
        with atomic_output(to_path) as temporary:
            with open(temporary, "w") as f:
                json.dump(out, f, indent=2)
        print(f"[CallsiteTracer] [{src}] dumped {len(out)} entries to {to_path}", file=sys.stderr)


    def _dump_all(self, final: bool):
        # dump each active backend to its own prefixed file
        if self.enable_cudaapi and self._paths.get("cudaapi"):
            self._dump_one("cudaapi", self._paths["cudaapi"])
        if self.enable_cupti and self._paths.get("cupti"):
            self._dump_one("cupti", self._paths["cupti"])
        if self.enable_python and self._paths.get("python"):
            self._dump_one("python", self._paths["python"])

    # ── Public API ────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        return self._start_count > 0

    def start(self):
        """
        Begin tracing until `stop()` is called.
        Safe to call multiple times: uses a ref-count.
        """
        print("[CallsiteTracer] start", file=sys.stderr)
        if self._start_count == 0:
            self._begin()
        self._start_count += 1
        return self

    def stop(self, *, exc: tuple | None = None):
        """
        End tracing started by `start()`. If nested, only stops when ref-count hits zero.
        Optionally pass an exc triple like in __exit__.
        """
        print("[CallsiteTracer] stop", file=sys.stderr)
        if self._start_count == 0:
            return
        self._start_count -= 1
        if self._start_count == 0:
            exc = exc or (None, None, None)
            self._end(*exc)


# --- Decorator API -------------------------------------------------------------
def trace_cuda_calls(dump_path, dump_every=10000, *,
                     enable_cudaapi=None, enable_cupti=None, enable_python=None,
                     python_strategy=None):
    def deco(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            tracer = CallsiteTracer(
                dump_every=dump_every,
                dump_path=dump_path,
                enable_cudaapi=enable_cudaapi,
                enable_cupti=enable_cupti,
                enable_python=enable_python,
                python_strategy=python_strategy,
            )
            try:
                tracer.start()
                return fn(*args, **kwargs)
            finally:
                tracer.stop()
        return wrapped
    return deco



# Usage:
# def main():
#     tracer = CallsiteTracer(
#         dump_every=5000,
#         dump_path="cuda_op_callsites_rank{}.json".format(torch.distributed.get_rank())
#     )
#     with tracer:
#         pretrain(…)
