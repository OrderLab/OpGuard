"""Deterministic output selection without cross-process coordination."""
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import yaml

from opguard.distributed import GLOBAL_RANK_ENV, LOCAL_RANK_ENV, get_global_rank, get_world_size

_CFG_PATH = os.path.join(os.path.dirname(__file__), "_running_config.yaml")
_cfg_lock = threading.Lock()
# Cache the target, not its local shadow; local dumping may be enabled later.
_trace_output_cache = {}
TRACE_BASE = os.path.expanduser("~/.opguard/traces")


def _load_config():
    if not os.path.exists(_CFG_PATH):
        return {}
    with open(_CFG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _saved_output():
    try:
        with (Path.home() / ".opguard_state.yaml").open(encoding="utf-8") as f:
            state = yaml.safe_load(f) or {}
        path = state.get("output_dir")
        return path if isinstance(path, str) and os.path.isabs(path) else None
    except (OSError, ValueError, AttributeError, yaml.YAMLError):
        return None


def _run_id():
    # RUN_TOKEN remains a compatibility alias; no lease or lock files are used.
    value = os.environ.get("OPGUARD_RUN_ID") or os.environ.get("OPGUARD_RUN_TOKEN")
    if value and (value in (".", "..") or any(c in value for c in "/\\\0{}")):
        raise ValueError("OPGUARD_RUN_ID must be a single directory name")
    return value


def get_trace_target_dir(tag_prefix="trace"):
    """Resolve explicit output > run ID > CLI state > configured output.

    All ranks must receive the same explicit path or run ID. Date templates
    are for single-process use only. tag_prefix is retained for API compatibility;
    all trace types in one process share the same run directory.
    """
    root = os.path.abspath(os.path.expanduser(os.getenv("OPGUARD_TRACE_ROOT") or TRACE_BASE))
    explicit = os.getenv("OPGUARD_TRACE_OUTPUT_BASE")
    run_id = None if explicit else _run_id()
    saved = None if explicit or run_id else _saved_output()
    raw = explicit or (f"run_{run_id}" if run_id else saved)
    if not raw:
        raw = _load_config().get("trace", {}).get("output_base", "trace_{date_time}")
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("trace.output_base must be a non-empty path")
    has_date = any(token in raw for token in ("{date}", "{date_time}", "{datetime}"))
    if has_date and (get_world_size() > 1 or get_global_rank() != 0 or
                     any(name in os.environ for name in GLOBAL_RANK_ENV + LOCAL_RANK_ENV)):
        raise ValueError("Distributed tracing requires OPGUARD_TRACE_OUTPUT_BASE or OPGUARD_RUN_ID; date templates are process-local")

    key = (os.getpid(), root, raw)
    with _cfg_lock:
        target = _trace_output_cache.get(key)
        if target is None:
            if has_date:
                now = datetime.now()
                raw = raw.replace("{date}", now.strftime("%Y%m%d"))
                raw = raw.replace("{date_time}", now.strftime("%Y%m%d-%H%M%S-%f"))
                raw = raw.replace("{datetime}", now.strftime("%Y%m%d-%H%M%S-%f"))
                raw += "_" + uuid.uuid4().hex[:8]
            raw = os.path.expanduser(raw)
            target = os.path.abspath(raw if os.path.isabs(raw) else os.path.join(root, raw))
            _trace_output_cache[key] = target

    return target


def get_trace_output_dir(tag_prefix="trace"):
    """Resolve the target, then apply local staging at the writer boundary."""
    from opguard.local_dump_manager import translate_path
    output = translate_path(get_trace_target_dir(tag_prefix))
    # Errors propagate: falling back per node would silently split a distributed run.
    os.makedirs(output, exist_ok=True)
    os.environ["OPGUARD_TRACE_OUTDIR"] = output
    return output


def get_output_subdir(kind):
    """Stable layout, independent of which files or folders already exist."""
    overrides = {'exec_trace': 'OPGUARD_EXEC_OUTPUT_DIR', 'callsite_trace': 'OPGUARD_CALLSITE_OUTPUT_DIR'}
    override = os.getenv(overrides[kind])
    if override:
        from opguard.local_dump_manager import translate_path
        return translate_path(os.path.abspath(os.path.expanduser(override)))
    return os.path.join(get_trace_output_dir(), kind)
