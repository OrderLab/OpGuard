# opguard/callsite_tracer/_autostart.py
from __future__ import annotations

import os, atexit, ctypes, sys, platform, threading
from pathlib import Path

# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------
def _truthy(v: str | None, default: bool = False) -> bool:
    if v is None:
        return default
    return v.lower() not in ("0", "false", "no", "")

def _rank_or_zero():
    from opguard.distributed import get_global_rank
    return get_global_rank()

# -----------------------------------------------------------------------------
# Native hook preload (optional but recommended: ensures one shared instance)
# -----------------------------------------------------------------------------
_NATIVE = None
_FORCE_DUMP = None

def _dlopen_native_hook_once():
    """
    Preload the native CUDA-API interposer .so if CUDAAPI backend is enabled.
    Registers pyhook_force_dump() at-exit when available.
    """
    global _NATIVE, _FORCE_DUMP

    # Only when CUDA-API backend is requested
    if not _truthy(os.getenv("OPGUARD_CALLSITE_CUDAAPI"), True):
        return

    # Optional global kill switch (kept for compatibility)
    if os.getenv("OPGUARD_NATIVE_HOOK", "1").lower() in ("0","false","no"):
        print("[OpGuard-Autostart] Native hook disabled by OPGUARD_NATIVE_HOOK=0", file=sys.stderr)
        return

    if _NATIVE is not None:
        print("[OpGuard-Autostart] Native hook already loaded", file=sys.stderr)
        return

    print("[OpGuard-Autostart] Loading native CUDA-API interposer", file=sys.stderr)

    RTLD_NOW      = getattr(ctypes, "RTLD_NOW",      getattr(os, "RTLD_NOW",      2))
    RTLD_GLOBAL   = getattr(ctypes, "RTLD_GLOBAL",   getattr(os, "RTLD_GLOBAL",   0x100))
    RTLD_NOLOAD   = getattr(ctypes, "RTLD_NOLOAD",   0x4)
    RTLD_NODELETE = getattr(ctypes, "RTLD_NODELETE", getattr(os, "RTLD_NODELETE", 0x1000))

    # Candidate locations
    cands: list[Path] = []
    env_path = os.getenv("OPGUARD_NATIVE_HOOK_PATH")
    if env_path:
        p = Path(env_path)
        if p.exists(): cands.append(p)

    try:
        import opguard
        pkg_so = (Path(opguard.__file__).resolve().parent /
                  "callsite_tracer" / "native" / "libpylaunch_hook.so")
        if pkg_so.exists():
            cands.append(pkg_so)
    except Exception:
        pass

    if not cands:
        # Don’t fail hard; CUDA-API backend will just disable itself in tracer if not present
        print("[OpGuard-Autostart] Native hook not installed; CUDA-API capture is unavailable. Run `opguard doctor`.", file=sys.stderr)
        return

    # Prefer NOLOAD reuse (e.g., if LD_PRELOADed)
    if platform.system() != "Windows":
        for c in cands:
            try:
                _NATIVE = ctypes.CDLL(str(c), mode=RTLD_NOLOAD | RTLD_GLOBAL)
                if _NATIVE:
                    print(f"[OpGuard-Autostart] Reusing already-loaded native hook: {c}", file=sys.stderr)
                    break
            except OSError as e:
                # Keep trying normal load
                pass

    # Load if not already resident
    if _NATIVE is None:
        mode = RTLD_NOW | RTLD_GLOBAL
        if os.getenv("OPGUARD_NATIVE_NODELETE", "1").lower() not in ("0","false","no"):
            mode |= RTLD_NODELETE
        for c in cands:
            try:
                _NATIVE = ctypes.CDLL(str(c), mode=mode) if platform.system() != "Windows" else ctypes.CDLL(str(c))
                print(f"[OpGuard-Autostart] Loaded native hook: {c}", file=sys.stderr)
                break
            except OSError as e:
                print(f"[OpGuard-Autostart] Failed to load native hook {c}: {e}", file=sys.stderr)

    if _NATIVE is None:
        return

    # Register optional force dump
    try:
        _FORCE_DUMP = _NATIVE.pyhook_force_dump
        _FORCE_DUMP.restype = None
        _FORCE_DUMP.argtypes = []
        atexit.register(_FORCE_DUMP)
    except Exception:
        _FORCE_DUMP = None


# -----------------------------------------------------------------------------
# Autostart coordinator (unified CallsiteTracer)
# -----------------------------------------------------------------------------
_started = False
_tracer  = None

def autostart():
    """
    Unified autostart:
      • Respects OPGUARD_AUTOSTART (default off) and OPGUARD_CALLSITE_TRACER (default on).
      • Uses OPGUARD_CALLSITE_{CUDAAPI,CUPTI,PYTHON} to select backends.
      • Starts a single CallsiteTracer that runs the chosen backends concurrently.
    """
    global _started, _tracer
    if _started:
        return

    # Gate
    if not _truthy(os.getenv("OPGUARD_AUTOSTART"), False):
        print("[OpGuard-Autostart] Disabled by OPGUARD_AUTOSTART=0", file=sys.stderr)
        return
    if not _truthy(os.getenv("OPGUARD_CALLSITE_TRACER"), True):
        return

    # Back-compat shims → new envs
    # If users set the old knobs, translate them once.
    if os.getenv("OPGUARD_CUPTI_ON") is not None and os.getenv("OPGUARD_CALLSITE_CUPTI") is None:
        os.environ["OPGUARD_CALLSITE_CUPTI"] = "1" if _truthy(os.getenv("OPGUARD_CUPTI_ON")) else "0"
    if os.getenv("OPGUARD_NATIVE") is not None and os.getenv("OPGUARD_CALLSITE_CUDAAPI") is None:
        os.environ["OPGUARD_CALLSITE_CUDAAPI"] = "1" if _truthy(os.getenv("OPGUARD_NATIVE")) else "0"

    # Preload native interposer if CUDAAPI is enabled
    _dlopen_native_hook_once()

    from opguard._running import get_output_subdir
    base_dir = get_output_subdir("callsite_trace")

    pid  = os.getpid()
    rank = _rank_or_zero()
    dump_path = os.path.join(base_dir, f"callsite_rank{rank}_pid{pid}.json")
    os.makedirs(os.path.dirname(dump_path) or ".", exist_ok=True)

    # Start unified tracer
    try:
        from opguard.callsite_tracer.trace import CallsiteTracer
        dump_every = int(os.getenv("OPGUARD_CS_DUMP_EVERY", os.getenv("OPGUARD_DUMP_EVERY", "10000")))
        py_strategy = os.getenv("OPGUARD_PYTHON_STRATEGY", "ops+dispatch")

        _tracer = CallsiteTracer(
            dump_every=dump_every,
            dump_path=dump_path,
            enable_cudaapi=_truthy(os.getenv("OPGUARD_CALLSITE_CUDAAPI"), True),
            enable_cupti=_truthy(os.getenv("OPGUARD_CALLSITE_CUPTI"), True),
            enable_python=_truthy(os.getenv("OPGUARD_CALLSITE_PYTHON"), True),
            python_strategy=py_strategy,
        )
        _tracer.start()
        atexit.register(_tracer.stop)
        _started = True

        print(f"[OpGuard-Autostart] Tracer ON  (cudaapi={_tracer._cudaapi_active}, "
              f"cupti={_tracer._cupti_active}, "
              f"python={_tracer.enable_python})  → {dump_path}",
              file=sys.stderr)

    except Exception as e:
        print(f"[OpGuard-Autostart] Failed to start unified tracer: {e}", file=sys.stderr)

# NOTE: Do NOT auto-run here; usercustomize should call autostart() once.
