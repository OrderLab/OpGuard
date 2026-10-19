# usercustomize.py
import os, sys, atexit, glob, json, time, hashlib, errno
from pathlib import Path


# ---------- tiny utils ----------
def _truthy(v, default=False):
    if v is None: return default
    return v not in ("0","false","False","","no","No")

def _log(msg):
    if _truthy(os.environ.get("OPGUARD_VERBOSE"), True):
        print(f"[OpGuard-usercustomize] {msg}", file=sys.stderr)

# --- Python-side aggregator: merge NDJSON into a single aggregate at exit ---

def _py_agg_truthy(v, default=True):
    if v is None: return default
    return v not in ("0","false","False","","no","No")

def _py_agg_rank():
    from opguard.distributed import get_global_rank
    return str(get_global_rank())

def _py_agg_base_dir():
    from opguard._running import get_trace_output_dir
    return Path(get_trace_output_dir())

def _py_agg_ndjson_globs(rank: str, pid: int):
    # We match both per-thread and per-process NDJSON patterns
    base = _py_agg_base_dir()
    pat1 = str(base / f"callsite_rank{rank}_pid{pid}_tid*.ndjson")
    pat2 = str(base / f"callsite_rank{rank}_pid{pid}.ndjson")
    return [pat1, pat2]

def _py_agg_target_path(rank: str, pid: int):
    # Sibling of NDJSON, but per-process
    base = _py_agg_base_dir()
    return base / f"callsite_rank{rank}_pid{pid}.aggregate.json"

def _py_agg_parse_line(line, _loads):
    try:
        obj = _loads(line)
        # expected keys: "site", "op", "frames" (list[[file, line, func], ...] in leaf->root order)
        site = obj.get("site") or "(no Python frames)"
        op   = obj.get("op") or ""
        frames = obj.get("frames") or []
        return site, op, frames
    except Exception:
        return None, None, None

def _register_python_ndjson_aggregator():
    if not _py_agg_truthy(os.environ.get("OPGUARD_PY_AGG"), True):
        return  # user disabled python aggregate

    try:
        try:
            import orjson as _orjson
            loads = _orjson.loads
            dumps = lambda o: _orjson.dumps(o, option=_orjson.OPT_INDENT_2)
        except Exception:
            loads = json.loads
            dumps = lambda o: json.dumps(o, indent=2).encode()

        pid  = os.getpid()
        rank = _py_agg_rank()
        out_path = _py_agg_target_path(rank, pid)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        def _do_aggregate():
            # Aggregate structure
            by_site = {}   # site -> {"count": int, "callstacks": [frames...], "stack_counts":[...], "ops": set(), "seen": {key: idx}}
            def _frames_key_exact(frames):
                # exact key with line numbers, leaf->root order matches native NDJSON writer
                # format: file:line:func\n...
                parts = []
                for f, ln, fn in frames:
                    parts.append(f"{f}:{ln}:{fn}\n")
                return "".join(parts)

            matched_files = set()
            for pat in _py_agg_ndjson_globs(rank, pid):
                for p in glob.glob(pat):
                    try:
                        st = os.stat(p)
                        if st.st_size == 0:
                            continue
                        matched_files.add(p)
                    except FileNotFoundError:
                        pass

            if not matched_files:
                # Nothing to merge—don’t write an empty file
                return

            # Merge line by line
            for p in sorted(matched_files):
                try:
                    with open(p, "rb") as fh:
                        for raw in fh:
                            # tolerate trailing whitespace
                            line = raw.strip()
                            if not line:
                                continue
                            site, op, frames = _py_agg_parse_line(line, loads)
                            if site is None:
                                continue
                            b = by_site.get(site)
                            if b is None:
                                b = {"count": 0, "callstacks": [], "stack_counts": [], "ops": set(), "seen": {}}
                                by_site[site] = b
                            b["count"] += 1
                            if op:
                                b["ops"].add(op)
                            k = _frames_key_exact(frames)
                            idx = b["seen"].get(k)
                            if idx is None:
                                idx = len(b["callstacks"])
                                b["seen"][k] = idx
                                b["callstacks"].append(frames)
                                b["stack_counts"].append(1)
                            else:
                                b["stack_counts"][idx] += 1
                except Exception as e:
                    print(f"[OpGuard-PyAgg] read failed: {p}: {e}", file=sys.stderr)

            # Emit aggregate JSON
            # Schema mirrors your native aggregate (minus internal 'seen')
            out_obj = {}
            for site, b in by_site.items():
                out_obj[site] = {
                    "count": b["count"],
                    "callstacks": b["callstacks"],
                    "ops": sorted(b["ops"]),
                    # optional debug info; uncomment if you want counts per stack in output:
                    # "stack_counts": b["stack_counts"],
                }

            tmp_path = out_path.with_suffix(".aggregate.json.tmp")
            try:
                with open(tmp_path, "wb") as f:
                    f.write(dumps(out_obj))
                os.replace(tmp_path, out_path)
                print(f"[OpGuard-PyAgg] wrote {out_path}", file=sys.stderr)
            except Exception as e:
                print(f"[OpGuard-PyAgg] write failed: {out_path}: {e}", file=sys.stderr)
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass

        # Register at exit (Python-level; independent of .so lifetime)
        atexit.register(_do_aggregate)
    except Exception as e:
        print(f"[OpGuard-PyAgg] aggregator init failed: {e}", file=sys.stderr)


# Numerical interventions are independent of tracing/autostart.
def _enable_torch_deterministic_mode():
    from opguard.runtime_policy import apply_requested_policy
    apply_requested_policy()

def _ensure_torch_deterministic():
    """
    Ensure torch deterministic mode is enabled.
    If torch is already imported, configure determinism immediately.
    Otherwise, install a meta-path finder that applies determinism right after torch loads.
    Simple approach: just enable determinism when torch is imported, no periodic enforcement needed.
    """
    if 'torch' in sys.modules:
        _enable_torch_deterministic_mode()
        return

    import importlib.abc, importlib.machinery

    class _TorchDeterministicFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "torch":
                return None
            spec = importlib.machinery.PathFinder.find_spec(fullname, path)
            if not spec or not spec.loader:
                return None
            real_loader = spec.loader

            class _Loader(importlib.abc.Loader):
                def create_module(self, spec_):
                    if hasattr(real_loader, "create_module"):
                        return real_loader.create_module(spec_)
                    return None
                def exec_module(self, module):
                    real_loader.exec_module(module)
                    # Enable determinism when torch is imported
                    _enable_torch_deterministic_mode()
                    # Keep finder in place to catch re-imports (though unlikely)

            spec.loader = _Loader()
            return spec

    # Check if finder already exists
    for finder in sys.meta_path:
        if isinstance(finder, _TorchDeterministicFinder):
            return  # Already installed
    
    _finder = _TorchDeterministicFinder()
    sys.meta_path.insert(0, _finder)

# ---------- small file helpers ----------
def _hash_file(p):
    try:
        with open(p, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except Exception:
        return None

def _try_lock(path, timeout_ms=0):
    import fcntl, time as _time
    start = _time.time()
    f = open(path, "a+")
    while True:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return f
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                f.close(); raise
            if timeout_ms <= 0:
                f.close(); return None
            if (_time.time() - start) * 1000 >= timeout_ms:
                f.close(); return None
            _time.sleep(0.05)

def _unlock_file(f):
    try:
        import fcntl
        fcntl.flock(f, fcntl.LOCK_UN)
    finally:
        try: f.close()
        except Exception: pass

# ---------- instrumentation (gated by TRACING_BUILD) ----------
def _instrument_once():
    # Only instrument if explicitly requested. This is orthogonal to determinism/tracing.
    if not _truthy(os.environ.get("OPGUARD_AUTO_INSTRUMENT"), False):
        # _log("OPGUARD_AUTO_INSTRUMENT not set, skipping instrumentation.")
        return
    _log("OPGUARD_AUTO_INSTRUMENT set, instrumenting.")
    from opguard._running import get_trace_output_dir
    base = Path(get_trace_output_dir())
    base.mkdir(parents=True, exist_ok=True)
    manifest = base / "instrument_manifest.json"
    lockfile = base / ".instrument.lock"

    cfg_path  = os.environ.get("OPGUARD_CALLSITE_CONFIG")
    cfg_hash  = _hash_file(cfg_path) if cfg_path else "nocfg"
    key = {"cfg_hash": cfg_hash, "py": sys.version.split()[0]}

    try:
        if manifest.exists():
            cur = json.loads(manifest.read_text())
            if cur.get("cfg_hash") == key["cfg_hash"] and cur.get("py") == key["py"] and cur.get("ok") is True:
                _log("manifest up-to-date; no-op")
                return
    except Exception:
        pass

    # Non-blocking lock (default 0ms). Configure via OPGUARD_INSTR_LOCK_TIMEOUT_MS if you want a small wait.
    try:
        timeout_ms = int(os.environ.get("OPGUARD_INSTR_LOCK_TIMEOUT_MS", "0"))
    except Exception:
        timeout_ms = 0

    lk = _try_lock(str(lockfile), timeout_ms=timeout_ms)
    if lk is None:
        _log("instrument lock busy; skip to avoid blocking site import")
        return

    try:
        # Re-check under lock
        try:
            if manifest.exists():
                cur = json.loads(manifest.read_text())
                if cur.get("cfg_hash") == key["cfg_hash"] and cur.get("py") == key["py"] and cur.get("ok") is True:
                    _log("manifest updated by another proc; no-op")
                    return
        except Exception:
            pass

        ok = False
        try:
            from opguard.instrument_all_files import run_instrumentation as instrument_all
            _log("starting instrumentation")
            ok = bool(instrument_all(cfg_path))
            _log(f"instrumentation result: ok={ok}")
        except Exception as e:
            _log(f"instrumenter failed: {e}")
            ok = False

        # Write/update manifest
        out = dict(key); out.update({"ok": bool(ok), "ts": time.time()})
        manifest.write_text(json.dumps(out))
    finally:
        _unlock_file(lk)

# ---------- tracer autostart (delegates to _autostart) ----------
def _maybe_start_tracer():
    # Only when requested:
    if not _truthy(os.environ.get("OPGUARD_AUTOSTART"), False):
        return
    if not _truthy(os.environ.get("OPGUARD_CALLSITE_TRACER"), False):
        return
    try:
        # Defer backend selection to _autostart
        from opguard.callsite_tracer import load_autostart_module
        _as = load_autostart_module()
        _as.autostart()
    except Exception as e:
        _log(f"Tracer autostart failed: {e}")

# ---------- entry ----------
try:
    # Tracing preserves the application's numerical settings by default.
    # Only this explicit experiment flag installs the deferred Torch policy.
    from opguard.runtime_policy import determinism_requested
    if determinism_requested():
        _ensure_torch_deterministic()

    # B) Optional instrumentation (independent gate)
    _instrument_once()

    # C) Optional tracer (needs AUTOSTART + CALLSITE_TRACER)
    _maybe_start_tracer()

except Exception as _e:
    _log(f"entry failed: {_e}")
