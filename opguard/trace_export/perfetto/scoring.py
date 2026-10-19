from __future__ import annotations
import os
from typing import Any, Dict, List, Optional, Tuple

from .utils import _get, _flatten_tensors, strip_top
from .config import _IO_STAT_KEYS, DIFF_ATOMS_HARD, DIFF_ATOMS_MAX

# --------------- concise diff ---------------

def _pretty(v: Any) -> Any:
    if isinstance(v, (list, tuple)):
        return tuple(_pretty(x) for x in v)
    if isinstance(v, (int, float, bool, type(None))): return v
    if isinstance(v, str): return v if len(v) <= 128 else (v[:125] + "...")
    if isinstance(v, dict): return {str(k): _pretty(v[k]) for k in sorted(v.keys(), key=lambda x: str(x))}
    r = repr(v)
    return r if len(r) <= 128 else (r[:125] + "...")

def _flatten_for_diff(value: Any, base: str, out: Dict[str, Any]) -> None:
    from .utils import _is_tensor_dict
    from .config import _IO_STAT_KEYS
    if _is_tensor_dict(value):
        shp = value.get("shape")
        out[f"{base}::shape"] = _pretty(tuple(shp) if isinstance(shp, list) else shp)
        st = value.get("stats", {}) or {}
        for k in _IO_STAT_KEYS:
            if k in st:
                out[f"{base}::{k}"] = _pretty(st.get(k))
        return
    if isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            _flatten_for_diff(v, f"{base}[{i}]", out)
        return
    if isinstance(value, dict):
        for k in sorted(value.keys(), key=lambda x: str(x)):
            _flatten_for_diff(value[k], f"{base}.{k}", out)
        return
    out[base] = _pretty(value)

def _compare_io(a_val: Any, b_val: Any, base: str) -> List[Dict[str, Any]]:
    flat_a: Dict[str, Any] = {}; flat_b: Dict[str, Any] = {}
    _flatten_for_diff(a_val, base, flat_a)
    _flatten_for_diff(b_val, base, flat_b)
    diffs: List[Dict[str, Any]] = []
    for k in sorted(set(flat_a.keys()) | set(flat_b.keys())):
        va = flat_a.get(k, "<MISSING>"); vb = flat_b.get(k, "<MISSING>")
        if va != vb:
            diffs.append({"path": k, "run1": va, "run2": vb})
    return diffs

def _atoms_from_diffs(top_level_changes, args_diffs, kwargs_diffs, outputs_diffs, hard_cap=DIFF_ATOMS_HARD) -> List[str]:
    atoms: List[str] = []
    for k, va, vb in top_level_changes:
        atoms.append(f"{k}: {_pretty(va)} vs {_pretty(vb)}")
        if len(atoms) >= hard_cap: return atoms
    for d in args_diffs:
        atoms.append(f"{d['path']}: {_pretty(d['run1'])} vs {_pretty(d['run2'])}")
        if len(atoms) >= hard_cap: return atoms
    for d in kwargs_diffs:
        atoms.append(f"{d['path']}: {_pretty(d['run1'])} vs {_pretty(d['run2'])}")
        if len(atoms) >= hard_cap: return atoms
    for d in outputs_diffs:
        atoms.append(f"{d['path']}: {_pretty(d['run1'])} vs {_pretty(d['run2'])}")
        if len(atoms) >= hard_cap: return atoms
    return atoms

def _summary_from_atoms(atoms: List[str], limit: int = DIFF_ATOMS_MAX) -> str:
    if not atoms: return ""
    if len(atoms) <= limit: return "; ".join(atoms)
    head = "; ".join(atoms[:limit])
    return f"{head}  (+{len(atoms)-limit} more)"

def _match_pattern(pattern: str, function_name: str) -> bool:
    """
    Match a pattern against a function name.
    Supports:
    - Exact match: "forward_gpu_no_cpy"
    - Substring match: "no_cpy" (if pattern in function_name)
    - Wildcard match: "*_no_cpy" (converts to fnmatch pattern)
    
    Function names may have angle brackets around them (e.g., "<module.function>"),
    which are stripped before matching.
    """
    if not function_name:
        return False
    
    # Strip angle brackets if present (e.g., "<module.function>" -> "module.function")
    cleaned_name = function_name.strip("<>")
    
    # Exact match
    if pattern == function_name or pattern == cleaned_name:
        return True
    
    # Wildcard pattern (contains *)
    if "*" in pattern:
        import fnmatch
        # Try matching against both original and cleaned name
        return fnmatch.fnmatch(function_name, pattern) or fnmatch.fnmatch(cleaned_name, pattern)
    
    # Substring match (default behavior for backward compatibility)
    return pattern in function_name or pattern in cleaned_name

def _extract_file_path_from_stack(stack_trace: str) -> str:
    """
    Extract the first file path from a stack trace string.
    Stack trace format: "path:line in function\\n..."
    """
    if not stack_trace:
        return ""
    import re
    # Pattern to match stack trace lines: "path:line in function"
    stack_line_re = re.compile(r"^(.*?):(\d+)\s+in\s+(.*)$")
    for line in stack_trace.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        m = stack_line_re.match(line)
        if m:
            file_path = m.group(1)
            if file_path and file_path not in ("<unknown>", "?"):
                return file_path
    return ""

def _match_file_path_pattern(pattern: str, file_path: str) -> bool:
    """
    Match a pattern against a file path.
    Wildcards are NOT supported for file path matching - only exact directory segment matching.
    Supports:
    - Exact match: "/path/to/file.py"
    - Directory segment match: "ndtimeline" matches ".../ndtimeline/..." (complete directory name)
    
    Note: Wildcard patterns (containing *) are ignored for file paths - they only work for function names.
    """
    if not file_path:
        return False
    
    # Exact match
    if pattern == file_path:
        return True
    
    # Wildcard patterns are NOT supported for file paths - skip them
    if "*" in pattern:
        return False
    
    # For non-wildcard patterns, check if pattern appears as a complete directory segment
    # This ensures "ndtimeline" matches "/path/ndtimeline/file.py" but not "/path/ndtimeline_backup/file.py"
        path_parts = file_path.replace(os.sep, '/').split('/')
    return pattern in path_parts

def _build_ignore_keys(ignore_fields: Optional[Dict[str, List[str]]], function_name: str, file_path: str = "") -> set[str]:
    """
    Build ignore keys set for a given function name and/or file path based on ignore_fields patterns.
    Supports exact match, substring match, and wildcard patterns (e.g., "*_no_cpy") for function names.
    For file paths, only exact directory segment matching is supported (no wildcards).
    """
    ignore_keys = set()
    if not ignore_fields:
        return ignore_keys
    for pattern, keys in ignore_fields.items():
        # Match against function name (using function name matching) or file path (using file path matching)
        if _match_pattern(pattern, function_name) or (file_path and _match_file_path_pattern(pattern, file_path)):
            ignore_keys.update(keys)
    return ignore_keys

def _is_ignored_path(field_path: str, ignore_keys: set[str]) -> bool:
    """
    Check if a field path (e.g., 'outputs', 'kwargs.outputs', 'outputs[0].stats.xor_signature')
    should be ignored based on ignore_keys.
    """
    if not ignore_keys:
        return False
    return any(
        field_path == ignore or field_path.startswith(ignore + ".") or field_path.startswith(ignore + "[")
        for ignore in ignore_keys
    )

def _filter_io_diffs(diffs: List[Dict[str, Any]], base_path: str, ignore_keys: set[str]) -> List[Dict[str, Any]]:
    """
    Filter out diffs that match ignore_keys patterns.
    """
    if not ignore_keys:
        return diffs
    filtered = []
    for diff in diffs:
        path = diff.get("path", "")
        full_path = f"{base_path}.{path}" if base_path else path
        if not _is_ignored_path(full_path, ignore_keys):
            filtered.append(diff)
    return filtered

def compare_simple(a: dict, b: dict, ignore_output_only: bool = False, ignore_fields: Optional[Dict[str, List[str]]] = None):
    # Extract file path BEFORE stripping (stack_trace might be removed by strip_top)
    function_name = a.get("function", "") or b.get("function", "")
    # Extract file path from stack trace (primary) or fname field (fallback)
    stack_trace1 = a.get("stack_trace", "") or ""
    stack_trace2 = b.get("stack_trace", "") or ""
    file_path_from_stack = _extract_file_path_from_stack(stack_trace1) or _extract_file_path_from_stack(stack_trace2)
    # Fallback to fname if stack trace doesn't have file path
    fname1 = a.get("fname", "") or ""
    fname2 = b.get("fname", "") or ""
    file_path = file_path_from_stack or fname1 or fname2
    
    # Build ignore keys for this function and file path
    ignore_keys = _build_ignore_keys(ignore_fields, function_name, file_path)
    
    ac, bc = strip_top(a), strip_top(b)
    
    tl_changes: List[Tuple[str, Any, Any]] = []
    diff_fields: List[str] = []
    details: Dict[str, Any] = {}

    # Check top-level fields (excluding inputs/outputs/meta)
    for k in sorted(ac.keys()):
        if k in {"inputs", "outputs", "meta"}: continue
        if _is_ignored_path(k, ignore_keys): continue
        va, vb = ac.get(k), bc.get(k)
        if va != vb: diff_fields.append(k); tl_changes.append((k, va, vb))

    # Compare I/O fields
    a_args = _get(ac, "inputs", "args") or []
    b_args = _get(bc, "inputs", "args") or []
    a_kwargs = _get(ac, "inputs", "kwargs") or {}
    b_kwargs = _get(bc, "inputs", "kwargs") or {}
    a_outs = ac.get("outputs") or []; b_outs = bc.get("outputs") or []

    # Compare I/O and filter ignored paths
    args_diffs = _compare_io(a_args, b_args, "args")
    kwargs_diffs = _compare_io(a_kwargs, b_kwargs, "kwargs")
    outputs_diffs = _compare_io(a_outs, b_outs, "outputs")
    
    # Filter out ignored paths
    args_diffs = _filter_io_diffs(args_diffs, "inputs.args", ignore_keys)
    kwargs_diffs = _filter_io_diffs(kwargs_diffs, "inputs.kwargs", ignore_keys)
    outputs_diffs = _filter_io_diffs(outputs_diffs, "outputs", ignore_keys)
    
    # Check if entire sections should be ignored
    if not _is_ignored_path("args", ignore_keys) and args_diffs:
        diff_fields.append("args")
    if not _is_ignored_path("kwargs", ignore_keys) and kwargs_diffs:
        diff_fields.append("kwargs")
    if not _is_ignored_path("outputs", ignore_keys) and outputs_diffs:
        diff_fields.append("outputs")

    atoms = _atoms_from_diffs(tl_changes, args_diffs, kwargs_diffs, outputs_diffs)

    only_outputs_changed = (not tl_changes) and (not args_diffs) and (not kwargs_diffs) and bool(outputs_diffs)
    if ignore_output_only and only_outputs_changed:
        return [], {}, []

    return diff_fields, details, atoms

def detect_anomaly(t: dict) -> Tuple[bool, Dict[str, Any]]:
    notes = []; found = False
    if _get(t, "anomaly") or _get(t, "meta", "anomaly"):
        return True, {"notes": "trace flagged anomaly"}
    def check(obj):
        nonlocal found, notes
        for td in _flatten_tensors(obj):
            st = td.get("stats", {})
            if st.get("contiguous") is False: found = True; notes.append("non_contiguous")
            if st.get("has_nan"):             found = True; notes.append("has_nan")
            if st.get("has_inf"):             found = True; notes.append("has_inf")
    check(_get(t, "inputs")); check(t.get("outputs"))
    return found, ({"notes": ", ".join(sorted(set(notes)))} if notes else {})
