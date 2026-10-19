from __future__ import annotations
import os
import json
import re
import time
import shutil
import argparse
import tempfile
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, List, Tuple, Callable, Deque
from functools import partial
import operator
import difflib
from collections import deque
from opguard.utils.io_tokenization import _bag_for_io, bag_similarity
import yaml
import orjson
import torch
from opguard.utils.utils import load_traces_any
from opguard.common_vars import OPGUARD_SAVE_ORI_TRACE, OPGUARD_STACK_TAIL_LINES

COMBINED_DIFF_FILE_NAME = "combined_diff_log.jsonl"
FIRST_DIFF_FILE_NAME = "first_diff_log.jsonl"

# 每个子任务日志写入缓冲（字节）
LOG_BUFFER_SIZE = 1024 * 1024
# ===========================================

# 比较操作符映射
_COMPARATORS = {
    "!=": operator.ne,
    ">=": operator.ge,
    "<=": operator.le,
    ">": operator.gt,
    "<": operator.lt,
    "=": operator.eq,   # 兼容单等号
}

IGNORED_TOP = {"pid", "thread_id", "timestamp", "counter", "fname", "stack_trace", "source_files"}


def get_stack_tail(trace_str: str, tail_lines: int = 1) -> str:
    lines = (trace_str or "").strip().splitlines()
    return "\n".join(lines[-tail_lines:]) if lines else "<empty>"


def load_ignore_config(path: Optional[str]) -> dict[str, dict]:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg or {}


def strip_nonessential(trace: dict) -> dict:
    return {k: v for k, v in trace.items() if k not in IGNORED_TOP}


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
    import os
    path_parts = file_path.replace(os.sep, '/').split('/')
    return pattern in path_parts

def build_ignore_prefixes(ignore_fields: dict[str, list[str]], function_name: str) -> set[str]:
    """
    将 per-function ignore_fields 映射成路径前缀集合。
    Supports exact match, substring match, and wildcard patterns (e.g., "*_no_cpy").
    """
    prefixes: set[str] = set()
    if not ignore_fields:
        return prefixes
    for pattern, keys in ignore_fields.items():
        if _match_pattern(pattern, function_name):
            if keys:
                prefixes.update(keys)
    return prefixes


def is_ignored_path(field_path: str, ignore_prefixes: set[str]) -> bool:
    return any(field_path == p or field_path.startswith(p + ".") for p in ignore_prefixes)


def serialize_clean(trace: dict, ignore_prefixes: set[str]) -> bytes:
    """
    去除顶级非关键字段后，根据 ignore_prefixes 过滤路径，再稳定序列化。
    """
    def filter_obj(obj: Any, prefix: str = "") -> Any:
        if isinstance(obj, dict):
            res = {}
            for k, v in obj.items():
                path = f"{prefix}.{k}" if prefix else k
                if is_ignored_path(path, ignore_prefixes):
                    continue
                res[k] = filter_obj(v, path)
            return res
        elif isinstance(obj, list):
            return [filter_obj(v, f"{prefix}[]") for v in obj]
        else:
            return obj

    clean = strip_nonessential(trace)
    filtered = filter_obj(clean, "")
    return orjson.dumps(filtered, option=orjson.OPT_SORT_KEYS)

def is_ignored(field_path: str, ignore_keys: set[str]) -> bool:
    """
    Check if a full field path (e.g., 'kwargs.outputs') is in the ignore list.
    """
    return any(
        field_path == ignore or field_path.startswith(ignore + ".")
        for ignore in ignore_keys
    )

def _get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def trace_fingerprint(trace: dict) -> tuple:
    """
    Build a stable fingerprint for alignment.
    Avoid unstable fields (pid, timestamp, counter).
    Prefer semantic identity: function, output shape/numel/dtype, and a few meta hints.
    """
    func = trace.get("function", None)
    # outputs.{shape, stats.{numel,dtype}}
    shape = _get(trace, "outputs", "shape", default=None)
    numel = _get(trace, "outputs", "stats", "numel", default=None)
    dtype = _get(trace, "outputs", "stats", "dtype", default=None)
    # optional hints
    lineno = trace.get("line_num", None) or _get(trace, "meta", "lineno", default=None)
    layer = _get(trace, "meta", "layer_number", default=None)
    prop_dir = _get(trace, "meta", "propagation_direction", default=None)
    # NOTE: we intentionally do NOT include xor_signature (it changes across differing runs)
    return (func, tuple(shape) if isinstance(shape, list) else shape, numel, dtype, lineno, layer, prop_dir)

def log_insert_or_delete(
    writer: Callable[[dict], None],
    rel_path: str,
    trace: dict,
    entry_idx: int,
    side: str,  # "run1" (delete) or "run2" (insert)
    kind: str,  # "delete" or "insert"
):
    clean = strip_nonessential(trace)
    writer({
        "counter": trace.get("counter", 0),
        "timestamp": trace.get("timestamp", "20000101-000000-000000"),
        "file": str(rel_path),
        "entry": entry_idx,
        "stack_tail": get_stack_tail(clean.get("stack_trace", ""), tail_lines=OPGUARD_STACK_TAIL_LINES),
        "fname": trace.get("fname"),
        "diff_fields": ["event"],
        "function": trace.get("function", {}),
        "meta": clean.get("meta", {}),
        "diff_details": {
            "event": kind,
            "side": side,
            "trace": clean,   # include the lone trace snapshot for inspection
        },
    })

def score_match(a: dict, b: dict, ignored_paths: set[str]) -> int:
    s = 0
    # (existing: function, fname, lineno, dtype/shape/numel on outputs, tag_prefix, etc.)
    # ... keep your current top-level fields scoring here ...

    # --- IO-aware scoring ---
    argsA = _get(a, "inputs", "args");   argsB = _get(b, "inputs", "args")
    kwargsA = _get(a, "inputs", "kwargs"); kwargsB = _get(b, "inputs", "kwargs")
    outsA = a.get("outputs");            outsB = b.get("outputs")

    bag_argsA = _bag_for_io(argsA, ignored_paths, "inputs.args")
    bag_argsB = _bag_for_io(argsB, ignored_paths, "inputs.args")
    bag_kwargsA = _bag_for_io(kwargsA, ignored_paths, "inputs.kwargs")
    bag_kwargsB = _bag_for_io(kwargsB, ignored_paths, "inputs.kwargs")
    bag_outsA = _bag_for_io(outsA, ignored_paths, "outputs")
    bag_outsB = _bag_for_io(outsB, ignored_paths, "outputs")

    # Weights — tune to taste
    w_args   = 25
    w_kwargs = 25
    w_outs   = 35

    s += int( round( w_args   * bag_similarity(bag_argsA,   bag_argsB) ) )
    s += int( round( w_kwargs * bag_similarity(bag_kwargsA, bag_kwargsB) ) )
    s += int( round( w_outs   * bag_similarity(bag_outsA,   bag_outsB) ) )

    # Cap overall score
    return min(s, 100)

def align_with_lookahead(
    t1: list[dict],
    t2: list[dict],
    ignored_paths: set[str],
    lookahead: int = 12,
    gap_penalty: int = 8,
    match_threshold: int = 85,
    accept_threshold: int = 70,
    tie_eps: int = 5,
    diag_penalty: int = 4,   # NEW: penalize deviation from current diagonal (offset j-i)
):
    """
    Returns (pairs, inserts, deletes)
      pairs   : list of (i, j)
      inserts : list of j indices (present only in t2)
      deletes : list of i indices (present only in t1)
    """
    i, j = 0, 0
    n1, n2 = len(t1), len(t2)
    pairs: list[tuple[int, int]] = []
    inserts: list[int] = []
    deletes: list[int] = []

    prefer_insert = (n2 >= n1)  # global bias if lengths differ

    def in1(ii): return 0 <= ii < n1
    def in2(jj): return 0 <= jj < n2

    def safe_score(ii: int, jj: int) -> int:
        if in1(ii) and in2(jj):
            return score_match(t1[ii], t2[jj], ignored_paths)
        return -10**9

    while i < n1 and j < n2:
        s00 = safe_score(i, j)
        if s00 >= match_threshold:
            # Good local match: pair it and advance both
            pairs.append((i, j))
            i += 1; j += 1
            continue

        # --- try look-ahead on both sides
        max_dj = min(lookahead, max(0, n2 - j - 1))
        max_di = min(lookahead, max(0, n1 - i - 1))
        cur_offset = j - i  # diagonal offset we are on

        # best insert candidate (keep i, move j -> j_ins)
        best_ins = None  # (adj, raw, gap, j_ins, next)
        for dj in range(1, max_dj + 1):
            j_ins = j + dj
            raw = safe_score(i, j_ins)
            # continuity: after align, next would be (i+1, j_ins+1)
            next_sc = safe_score(i + 1, j_ins + 1)
            # diagonal penalty: prefer staying near current offset
            adj = raw - gap_penalty * dj - diag_penalty * abs((j_ins - i) - cur_offset)
            cand = (adj, raw, dj, j_ins, next_sc)
            if (best_ins is None) or (cand > best_ins):
                best_ins = cand

        # best delete candidate (keep j, move i -> i_del)
        best_del = None  # (adj, raw, gap, i_del, next)
        for di in range(1, max_di + 1):
            i_del = i + di
            raw = safe_score(i_del, j)
            next_sc = safe_score(i_del + 1, j + 1)
            adj = raw - gap_penalty * di - diag_penalty * abs((j - i_del) - cur_offset)
            cand = (adj, raw, di, i_del, next_sc)
            if (best_del is None) or (cand > best_del):
                best_del = cand

        # Extract fields
        ins_adj, ins_raw, ins_gap, j_ins, ins_next = best_ins if best_ins else (-10**9, -10**9, 0, -1, -10**9)
        del_adj, del_raw, del_gap, i_del, del_next = best_del if best_del else (-10**9, -10**9, 0, -1, -10**9)

        # If neither candidate clears accept, DO NOT force a pair: take a single-step skip by bias
        if ins_adj < accept_threshold and del_adj < accept_threshold:
            if prefer_insert and j + 1 < n2:
                inserts.append(j)  # treat t2[j] as an insertion
                j += 1
                continue
            elif i + 1 < n1:
                deletes.append(i)
                i += 1
                continue
            else:
                # last resort: if one side exhausted, break
                break

        # Choose between acceptable candidates
        chosen = None
        if ins_adj > del_adj + tie_eps:
            chosen = ("insert", j_ins, ins_gap, ins_next)
        elif del_adj > ins_adj + tie_eps:
            chosen = ("delete", i_del, del_gap, del_next)
        else:
            # close call: use continuity next-score, then global bias
            if ins_next > del_next + tie_eps:
                chosen = ("insert", j_ins, ins_gap, ins_next)
            elif del_next > ins_next + tie_eps:
                chosen = ("delete", i_del, del_gap, del_next)
            else:
                chosen = ("insert", j_ins, ins_gap, ins_next) if prefer_insert else ("delete", i_del, del_gap, del_next)

        kind, idx, gap, _ = chosen
        if kind == "insert":
            # emit all intervening inserts, but DO NOT pair yet; re-evaluate (i, j) afterwards
            for k in range(j, idx):
                inserts.append(k)
            j = idx
            continue
        else:
            for k in range(i, idx):
                deletes.append(k)
            i = idx
            continue

    # tails
    while j < n2:
        inserts.append(j); j += 1
    while i < n1:
        deletes.append(i); i += 1

    return pairs, inserts, deletes

# -------- I/O diff helpers (path-aware deep compare) --------

def _path_join(base: str, seg: str) -> str:
    return f"{base}.{seg}" if base else seg

def _collect_leaf_diffs(a, b, base_path: str, ignore_keys: set[str], diffs: list[tuple[str, Any, Any]]):
    # Ignore entire subtree?
    if any(base_path == ig or base_path.startswith(ig + ".") for ig in ignore_keys if ig):
        return

    # Equal (fast path)
    if a == b:
        return

    # Dict-vs-dict: dive
    if isinstance(a, dict) and isinstance(b, dict):
        keys = set(a.keys()) | set(b.keys())
        for k in sorted(keys):
            pa = a.get(k, None); pb = b.get(k, None)
            _collect_leaf_diffs(pa, pb, _path_join(base_path, str(k)), ignore_keys, diffs)
        return

    # List/tuple-vs-list/tuple: align by index
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        n = max(len(a), len(b))
        for i in range(n):
            va = a[i] if i < len(a) else None
            vb = b[i] if i < len(b) else None
            _collect_leaf_diffs(va, vb, _path_join(base_path, f"[{i}]"), ignore_keys, diffs)
        return

    # Fallback: leaf differs
    diffs.append((base_path, a, b))

def _summarize_io_diffs(io_paths: list[tuple[str, Any, Any]]) -> dict:
    """
    Returns:
      { 
        "self": [ {field, run1, run2}, ... ],
        "args": { "<idx>": [ {field, run1, run2}, ... ], ... },
        "kwargs": { "<key>": [ {field, run1, run2}, ... ], ... },
        "outputs": [ {field, run1, run2}, ... ]
      }
    """
    summary = {"self": [], "args": {}, "kwargs": {}, "outputs": []}

    for path, v1, v2 in io_paths:
        # Normalize root prefixes to decide the bucket
        if path.startswith("inputs.self"):
            summary["self"].append({"field": path[len("inputs.self"):].lstrip("."), "run1": v1, "run2": v2})
            continue

        if path.startswith("inputs.args"):
            # Expect something like "inputs.args[3].stats.dtype"
            # Extract [idx] and the remainder as 'field'
            rest = path[len("inputs.args"):].lstrip(".")
            # rest starts with "[<idx>]" possibly followed by ".<field...>"
            m = re.match(r"\[(\d+)\](?:\.(.*))?$", rest)
            if not m:
                # Unusual path (e.g., args itself); bucket under index -1
                idx, field = "-1", rest
            else:
                idx, field = m.group(1), (m.group(2) or "")
            field = field or "<value>"
            summary["args"].setdefault(idx, []).append({"field": field, "run1": v1, "run2": v2})

        elif path.startswith("inputs.kwargs"):
            # "inputs.kwargs.weight.stats.dtype"
            rest = path[len("inputs.kwargs."):].lstrip(".")
            # first token is the kwarg key
            if "." in rest:
                key, field = rest.split(".", 1)
            else:
                key, field = rest, "<value>"
            summary["kwargs"].setdefault(key, []).append({"field": field, "run1": v1, "run2": v2})

        elif path.startswith("outputs"):
            # "outputs.shape" or "outputs.stats.xor_signature" etc.
            field = path[len("outputs."):].lstrip(".") or "<value>"
            summary["outputs"].append({"field": field, "run1": v1, "run2": v2})

    # Optional: sort lists for stable logs
    summary["self"] = sorted(summary["self"], key=lambda d: d["field"])
    for k in list(summary["args"].keys()):
        summary["args"][k] = sorted(summary["args"][k], key=lambda d: d["field"])
    for k in list(summary["kwargs"].keys()):
        summary["kwargs"][k] = sorted(summary["kwargs"][k], key=lambda d: d["field"])
    summary["outputs"] = sorted(summary["outputs"], key=lambda d: d["field"])
    return summary

def _filtered_view_kwargs(kwargs_dict: dict, ignore_keys: set[str]) -> dict:
    """Apply nested-ignore to kwargs while keeping structure (for equality fast-path)."""
    res = {}
    for k, v in kwargs_dict.items():
        kp = f"kwargs.{k}"
        if any(kp == ig or kp.startswith(ig + ".") for ig in ignore_keys if ig):
            continue
        res[k] = v
    return res


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

def compare_traces(trace1: dict, trace2: dict, 
                   path: str,  # this is your rel_path + entry index
                   ignore_output_only=True, 
                   ignore_fields: dict[str, list[str]] = None,
                   entry_idx: int = None,
                   rel_path: str = None,
                   log_writer: Optional[Callable[[dict], None]] = None) -> int:

    function_name = trace1.get("function", "") or trace2.get("function", "")
    # Extract file path from stack trace (primary) or fname field (fallback)
    stack_trace1 = trace1.get("stack_trace", "") or ""
    stack_trace2 = trace2.get("stack_trace", "") or ""
    file_path_from_stack = _extract_file_path_from_stack(stack_trace1) or _extract_file_path_from_stack(stack_trace2)
    # Fallback to fname if stack trace doesn't have file path
    fname1 = trace1.get("fname", "") or ""
    fname2 = trace2.get("fname", "") or ""
    file_path = file_path_from_stack or fname1 or fname2

    ignore_keys = set()
    if ignore_fields:
        for pattern, keys in ignore_fields.items():
            # Match against function name (using function name matching) or file path (using file path matching)
            if _match_pattern(pattern, function_name) or _match_file_path_pattern(pattern, file_path):
                ignore_keys.update(keys)

    trace1_clean = strip_nonessential(trace1)
    trace2_clean = strip_nonessential(trace2)

    # Split inputs and outputs explicitly
    inputs1 = trace1_clean.get("inputs", {})
    inputs2 = trace2_clean.get("inputs", {})
    outputs1 = trace1_clean.get("outputs", None)
    outputs2 = trace2_clean.get("outputs", None)

    args1 = inputs1.get("args", [])
    args2 = inputs2.get("args", [])
    self1 = inputs1.get("self", None)
    self2 = inputs2.get("self", None)
    kwargs1 = inputs1.get("kwargs", {})
    kwargs2 = inputs2.get("kwargs", {})

    diff_fields = []
    diff_details = {}

    # Compare top-level fields (excluding inputs/outputs)
    for key in trace1_clean.keys():
        if key in {"inputs", "outputs"}:
            continue
        if is_ignored(key, ignore_keys):
            continue
        if trace1_clean.get(key) != trace2_clean.get(key):
            diff_fields.append(key)
            diff_details[key] = {
                "run1": trace1_clean.get(key),
                "run2": trace2_clean.get(key),
            }

    # ---- I/O deep diffs (path-aware) ----
    io_paths: list[tuple[str, Any, Any]] = []

    # args (path root: inputs.args)
    if not is_ignored("self", ignore_keys):
        if self1 != self2:
            _collect_leaf_diffs(self1, self2, "inputs.self", ignore_keys, io_paths)

    # args (path root: inputs.args)
    if not is_ignored("args", ignore_keys):
        if args1 != args2:
            _collect_leaf_diffs(args1, args2, "inputs.args", ignore_keys, io_paths)

    # kwargs (path root: inputs.kwargs) – respect nested ignores
    if not is_ignored("kwargs", ignore_keys):
        # fast-path equality on filtered views
        if _filtered_view_kwargs(kwargs1, ignore_keys) != _filtered_view_kwargs(kwargs2, ignore_keys):
            _collect_leaf_diffs(kwargs1, kwargs2, "inputs.kwargs", ignore_keys, io_paths)

    # outputs (path root: outputs)
    if not is_ignored("outputs", ignore_keys):
        if outputs1 != outputs2:
            _collect_leaf_diffs(outputs1, outputs2, "outputs", ignore_keys, io_paths)

    # Translate I/O paths to summary buckets (args/kwargs/outputs with indices/keys)
    if io_paths:
        io_summary = _summarize_io_diffs(io_paths)
        # Mark fields at the coarse level so legacy readers still work
        # Decide whether to include "args", "kwargs", "outputs" in diff_fields
        if any(p[0].startswith("inputs.self") for p in io_paths) and "self" not in diff_fields:
            diff_fields.append("self")
        if any(p[0].startswith("inputs.args") for p in io_paths) and "args" not in diff_fields:
            diff_fields.append("args")
        if any(p[0].startswith("inputs.kwargs") for p in io_paths) and "kwargs" not in diff_fields:
            diff_fields.append("kwargs")
        if any(p[0].startswith("outputs") for p in io_paths) and "outputs" not in diff_fields:
            diff_fields.append("outputs")

        # Attach structured details
        diff_details.setdefault("io_diff_paths", [
            {"path": p, "run1": a, "run2": b} for (p, a, b) in io_paths
        ])
        diff_details["io_diff_summary"] = io_summary


    # Optional: global output-only flag
    if ignore_output_only and set(diff_fields) == {"outputs"}:
        return 0

    if diff_fields:
        stack_tail = get_stack_tail(trace1.get("stack_trace", ""), tail_lines=OPGUARD_STACK_TAIL_LINES)
        log_writer({
            "counter": trace2.get("counter", 0),
            "timestamp": trace2.get("timestamp", "20000101-000000-000000"),
            "file": str(rel_path or path),
            "entry": entry_idx,
            "stack_tail": stack_tail,
            "fname": trace2.get("fname"),
            "diff_fields": diff_fields,
            "function": function_name,
            "meta": trace2_clean.get("meta", {}),
            "diff_details": diff_details,
            "ori_trace1": trace1 if OPGUARD_SAVE_ORI_TRACE else None,
            "ori_trace2": trace2 if OPGUARD_SAVE_ORI_TRACE else None,
        })

        return len(diff_fields)

    return 0

@dataclass
class FilePair:
    p1: Path
    p2: Path
    rel: str


def list_pairs_and_uniques(base1: Path, base2: Path) -> Tuple[List[FilePair], List[Path], List[Path]]:
    base1 = base1.resolve()
    base2 = base2.resolve()

    pairs: List[FilePair] = []
    only1: List[Path] = []
    only2: List[Path] = []

    base2_map = {}
    n2 = len(str(base2)) + 1
    for q in base2.rglob("*"):
        if q.is_file():
            rel = str(q.resolve())[n2:]
            base2_map[rel] = q.resolve()

    n1 = len(str(base1)) + 1
    matched_rels = set()
    for p in base1.rglob("*"):
        if not p.is_file():
            continue
        rel = str(p.resolve())[n1:]
        q = base2_map.get(rel)
        if q:
            pairs.append(FilePair(p.resolve(), q, rel))
            matched_rels.add(rel)
        else:
            only1.append(p.resolve())

    for rel, q in base2_map.items():
        if rel not in matched_rels:
            only2.append(q)

    return pairs, only1, only2


def bucketize_pairs(
    pairs: List[FilePair],
    num_buckets: int
) -> List[List[FilePair]]:
    if num_buckets <= 1:
        return [pairs]
    buckets: List[List[FilePair]] = [[] for _ in range(num_buckets)]

    for i, fp in enumerate(pairs):
        buckets[i % num_buckets].append(fp)

    return buckets


class JsonlWriter:
    def __init__(self, path: Path, buffer_size: int = LOG_BUFFER_SIZE):
        self._path = path
        self._f = path.open("ab", buffering=buffer_size)

    def write(self, obj: dict):
        self._f.write(orjson.dumps(obj))
        self._f.write(b"\n")

    def close(self):
        try:
            self._f.flush()
        finally:
            self._f.close()

    @property
    def path(self) -> Path:
        return self._path

def is_async_trace(traces: list) -> bool:
    for trace in traces:
        if trace.get("is_async", False):
            return True
    return False

def process_bucket(
    bucket: List[FilePair],
    ignore_output_only: bool,
    ignore_files: Optional[list],
    ignore_fields: Optional[dict],
    filter_kv: Optional[str | list],
    log_path: Path,
    align: bool = False,
) -> Tuple[int, int, int, str]:
    """
    return (matched_files, mismatched_counts, total_diffs, log_file_name)
    """
    matched_files = 0
    mismatched_counts = 0
    total_diffs = 0
    writer = JsonlWriter(log_path)
    try:
        for pair in bucket:
            ignore_this = False
            for ignore_file in ignore_files:
                if ignore_file in pair.rel:
                    ignore_this = True
                    break
            if ignore_this:
                print(f"Ignore compare {pair.rel}")
                continue
            
            try:
                t1 = load_traces_any(pair.p1, filter_kv)
                t2 = load_traces_any(pair.p2, filter_kv)
            except Exception as e:
                print(f"Error loading {pair.rel}: {e}")
                continue
            
            if pair.p1.stem.endswith("_async") or pair.p2.stem.endswith("_async") or is_async_trace(t1):
                print(f"Sorting async traces in {pair.rel}")
                t1 = sorted(t1, key=lambda x: x["timestamp"])
                t2 = sorted(t2, key=lambda x: x["timestamp"])

            if not align:
                # existing fast path (zip by index)
                if len(t1) != len(t2):
                    mismatched_counts += 1
                    print(f"⚠️ Trace count mismatch in {pair.rel}: {len(t1)} vs {len(t2)}")

                local_diff = 0
                for i, (a, b) in enumerate(zip(t1, t2)):
                    local_diff += compare_traces(
                        a, b,
                        pair.rel,
                        ignore_output_only=ignore_output_only,
                        ignore_fields=ignore_fields,
                        rel_path=pair.rel,
                        entry_idx=i,
                        log_writer=writer.write,
                    )
                total_diffs += local_diff
            else:
                # alignment path
                if len(t1) != len(t2):
                    mismatched_counts += 1  # still useful stat
                ignored_paths = set()
                # fold in CLI/global ignores (you already compute ignore_keys in compare_traces; reuse or rebuild here)
                # Example: build from ignore_fields config matching all functions (pattern=""):
                if ignore_fields and "" in ignore_fields:
                    ignored_paths.update(ignore_fields[""])

                pairs, inserts, deletes = align_with_lookahead(
                    t1, t2,
                    ignored_paths=ignored_paths,
                    lookahead=12, gap_penalty=8,
                    match_threshold=85, accept_threshold=70,
                )

                local_diff = 0
                # log inserts
                for j_idx in inserts:
                    log_insert_or_delete(writer.write, pair.rel, t2[j_idx], j_idx, side="run2", kind="insert")
                    local_diff += 1

                # log deletes
                for i_idx in deletes:
                    log_insert_or_delete(writer.write, pair.rel, t1[i_idx], i_idx, side="run1", kind="delete")
                    local_diff += 1

                # compare aligned pairs
                for (i_idx, j_idx) in pairs:
                    local_diff += compare_traces(
                        t1[i_idx], t2[j_idx],
                        pair.rel,
                        ignore_output_only=ignore_output_only,
                        ignore_fields=ignore_fields,
                        filter_kv=filter_kv,
                        rel_path=pair.rel,
                        entry_idx=j_idx,
                        log_writer=writer.write,
                    )

                total_diffs += local_diff

            if local_diff == 0:
                matched_files+=1
    finally:
        writer.close()

    return matched_files, mismatched_counts, total_diffs, writer.path.name

def _dispatch_process_bucket(job, fn):
    bucket, log_path = job
    return fn(bucket=bucket, log_path=log_path)

def _aggregate_diff_summary(combined_path: Path, out_path: Path):
    counts = {}  # key -> int
    try:
        with combined_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                evt = json.loads(line)
                det = evt.get("diff_details", {})
                # Prefer structured summary if present
                summ = det.get("io_diff_summary", {})
                if summ:
                    for idx, items in (summ.get("args") or {}).items():
                        for it in items:
                            key = f"args[{idx}].{it['field']}"
                            counts[key] = counts.get(key, 0) + 1
                    for k, items in (summ.get("kwargs") or {}).items():
                        for it in items:
                            key = f"kwargs.{k}.{it['field']}"
                            counts[key] = counts.get(key, 0) + 1
                    for it in (summ.get("outputs") or []):
                        key = f"outputs.{it['field']}"
                        counts[key] = counts.get(key, 0) + 1
                else:
                    # Fallback: count io_diff_paths if present
                    for dp in det.get("io_diff_paths", []):
                        key = dp.get("path", "<unknown>")
                        counts[key] = counts.get(key, 0) + 1
        out_path.write_text(orjson.dumps(counts, option=orjson.OPT_INDENT_2).decode("utf-8"))
    except Exception as e:
        print(f"Failed to aggregate diff summary: {e}")


def compare_by_buckets(
    dir1: str, dir2: str, 
    ignore_output_only=False, 
    ignore_config=None, 
    filter_kv=None, 
    log_dir=None,
    worker=0,
    align=False,
) -> Tuple[int, int, int, Path]:
    
    base1 = Path(dir1).resolve()
    base2 = Path(dir2).resolve()
    out_dir_p = Path(log_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    # 1. list matched file pairs and only files
    pairs, only1, only2 = list_pairs_and_uniques(base1, base2)
    total_pairs = len(pairs)

    print(f"🔍 Found {total_pairs} matching file pairs.")
    print(f"⚠️ Found {len(only1)} files only in {dir1}.")
    for p in only1:
        print(f" - {p}")
    
    print(f"⚠️ Found {len(only2)} files only in {dir2}.")
    for p in only2:
        print(f" - {p}")

    combined = out_dir_p / COMBINED_DIFF_FILE_NAME
    if total_pairs == 0:
        combined.write_bytes(b"")
        print("No matching files. Nothing to compare.")
        return 0, 0, 0, combined

    if worker == 0:
        worker = os.cpu_count()

    # 2. bucketize pairs
    num_buckets = min(total_pairs, worker * 8)
    print(f"Using {num_buckets} buckets.")
    buckets = bucketize_pairs(pairs, num_buckets)

    t0 = time.time()
    if combined.exists():
        combined.unlink(missing_ok=True)

    results: List[Tuple[int, int, int, str]] = []

    # 3. dispatch jobs
    with tempfile.TemporaryDirectory() as tmpd:
        tmpdir = Path(tmpd)
        jobs: List[Tuple[List[FilePair], Path]] = []
        for i, b in enumerate(buckets):
            jobs.append((b, tmpdir / f"bucket_{i:05d}.jsonl"))

        fn = partial(
            process_bucket,
            ignore_output_only=ignore_output_only,
            ignore_files=ignore_config.get("inspector_ignore_files", []),
            ignore_fields=ignore_config.get("inspector_ignore_fields", {}),
            filter_kv=filter_kv,
            align=align,
        )

        # 4. compare trace in buckets in parallel
        with mp.Pool(processes=worker) as pool:
            for res in pool.imap_unordered(
                partial(_dispatch_process_bucket, fn=fn),  
                jobs,
            ):
                results.append(res)
                done = len(results)
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[bucket] {done}/{num_buckets} done ({rate:.2f} it/s)")

        # 5. merge logs
        with combined.open("wb") as fout:
            for _, _, _, log_name in sorted(results, key=lambda x: x[3]):
                with (tmpdir / log_name).open("rb") as fin:
                    shutil.copyfileobj(fin, fout)

    # 7. summarize results
    total_matched_files = sum(r[0] for r in results)
    total_mismatched_count_files = sum(r[1] for r in results)
    total_diff_fields = sum(r[2] for r in results)

    print("\n=====================================")
    print(f"✅ Files with matched trace: {total_matched_files}")
    print(f"⚠️ Files with mismatched trace count: {total_mismatched_count_files}")
    print(f"🔍 Fields with different values found: {total_diff_fields}")
    print(f"📄 Combined log saved to: {combined}")
    print(f"⏰ Elapsed: {time.time()-t0:.2f}s")
    print("=====================================")

    summary_path = out_dir_p / "diff_summary_counts.json"
    _aggregate_diff_summary(combined, summary_path)
    print(f"🧾 Diff summary counts saved to: {summary_path}")


    return total_matched_files, total_mismatched_count_files, total_diff_fields, combined

def get_fisrt_diff_by_rank(log_dir:str):
    first_diff_in_rank = {} # rank -> (timestamp, event)

    path = Path(log_dir) / COMBINED_DIFF_FILE_NAME
    if not path.exists():
        return
    
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            rank = -1
            file_name = event.get("file", "")
            m = re.search(r"gpu_(\d+)", file_name)
            if m:
                rank = int(m.group(1))

            timestamp = event.get("timestamp", None)
            if timestamp is None or rank == -1:
                print("Event without timestamp or wrong rank: ", event)
                continue

            if rank not in first_diff_in_rank or timestamp < first_diff_in_rank[rank][0]:
                first_diff_in_rank[rank] = (timestamp, event)
    
    if not first_diff_in_rank:
        print("No diff and skip to find the first diff!")
        return
    
    _, earliest_event = min(first_diff_in_rank.values(), key=lambda x: x[0])

    with open(os.path.join(log_dir, FIRST_DIFF_FILE_NAME), "w") as f:
        f.write("⚠️ First diff event: ")
        f.write(format_diff_event(earliest_event))
        f.write("\n⚠️ First diff event by rank: ")

        for rank in sorted(first_diff_in_rank.keys()):
            f.write(format_diff_event(first_diff_in_rank[rank][1]))

def format_diff_event(event):
    return """
• Rank: {rank}, Timestamp: {timestamp}, Func: {func}, Fname: {fname}
  Meta: {meta}
  Diff fields: {diff_fields}
  Diff params: {diff_detail}
    """.format(
        rank=event["file"].split("/")[0],
        timestamp=event["timestamp"],
        func=event["function"],
        fname=event["fname"],
        meta=event["meta"],
        diff_fields=event["diff_fields"],
        diff_detail=event["diff_details"],
    )

def main():
    parser = argparse.ArgumentParser(description="Compare two trace runs by bucketing file pairs for parallelism.")
    parser.add_argument("run1", help="Path to first trace output directory")
    parser.add_argument("run2", help="Path to second trace output directory")
    parser.add_argument("--ignore-output-only", action="store_true",
                        help="Ignore differences where only the 'outputs' field differs")

    default_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    parser.add_argument("--ignore-config", type=str, default=default_config_path,
                        help=f"YAML file with per-function ignore fields (default: {default_config_path})")
    
    parser.add_argument("--filter", type=str, default=None,
                        help="Key=val filter (e.g. meta.global_step=0)")

    parser.add_argument("--out-dir", type=str, required=True,
                        help="Directory to store the combined trace diff output")
    
    parser.add_argument("--worker", type=int, default=os.cpu_count(),
                        help="Number of workers to use for parallelism")

    parser.add_argument("--align", action="store_true",
                        help="Enable alignment (handles insertions/deletions) using difflib; slower.")

    parser.add_argument("--ignore-fields", type=str, nargs="*", default=[],
                        help="List of top-level fields (or nested paths) to ignore, e.g. --ignore-fields meta outputs.stats.dtype")

    args = parser.parse_args()

    # Load YAML config if provided
    ignore_config = load_ignore_config(args.ignore_config) if args.ignore_config else {}

    # Turn CLI list into the same dict format {pattern: [keys]} used downstream
    cli_ignore = args.ignore_fields or []
    if cli_ignore:
        # Empty string pattern "" means global
        ignore_config.setdefault("inspector_ignore_fields", {}).setdefault("", []).extend(cli_ignore)


    compare_by_buckets(
        dir1=args.run1,
        dir2=args.run2,
        ignore_output_only=args.ignore_output_only,
        ignore_config=ignore_config,
        filter_kv=args.filter,
        log_dir=args.out_dir,
        worker=args.worker,
        align=args.align,
    )

    get_fisrt_diff_by_rank(args.out_dir)

if __name__ == "__main__":
    mp.freeze_support()
    main()
