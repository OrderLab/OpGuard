#!/usr/bin/env python3
from __future__ import annotations
import argparse
import os
import re
import sys
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Tuple
import shutil
import multiprocessing as mp
from functools import partial
import tempfile
from concurrent.futures import ThreadPoolExecutor
import yaml as _yaml

# Optional fast JSON
try:
    import orjson as _orjson
    def _dumps(o: Any) -> bytes: return _orjson.dumps(o)
    def _loads(b: bytes | str) -> Any:
        if isinstance(b, str): b = b.encode('utf-8', 'surrogatepass')
        return _orjson.loads(b)
except Exception:
    _orjson = None
    def _dumps(o: Any) -> bytes: return json.dumps(o, separators=(",", ":")).encode("utf-8")
    def _loads(b: bytes | str) -> Any:
        return json.loads(b)

COMPARATORS = {
    "!=": lambda a,b: a != b,
    ">=": lambda a,b: (a is not None and b is not None and a >= b),
    "<=": lambda a,b: (a is not None and b is not None and a <= b),
    ">":  lambda a,b: (a is not None and b is not None and a >  b),
    "<":  lambda a,b: (a is not None and b is not None and a <  b),
    "=":  lambda a,b: a == b,   # alias for equality
    "==": lambda a,b: a == b,
}

_FILTER_RE = re.compile(r"""
    ^\s*
    (?P<path>[A-Za-z0-9_.\[\]]+)
    \s*(?P<op>!=|>=|<=|==|=|>|<)\s*
    (?P<val>.+?)
    \s*$
""", re.X)

# ---------- Path access helpers (supports dotted paths and [idx]) ----------

_SEG_RE = re.compile(r"""
    (?P<name>[A-Za-z0-9_]+)
    (?P<rest>(\[[0-9]+\])*)
""", re.X)

_IDX_RE = re.compile(r"\[([0-9]+)\]")

# --- deletion helpers for dotted paths with [idx] (supports negatives) ---

_IDX_ANY_RE = re.compile(r"\[(-?[0-9]+)\]")

def _step_into(container: Any, name: str) -> Any | None:
    """Step into dict key `name` on container; return None if missing."""
    if isinstance(container, dict) and name in container:
        return container[name]
    return None

def _iter_indices(rest: str) -> list[int]:
    return [int(m.group(1)) for m in _IDX_ANY_RE.finditer(rest or "")]

def _normalize_index(idx: int, n: int) -> int | None:
    if idx < 0:
        idx = n + idx
    return idx if 0 <= idx < n else None

_SEG_ANY_RE = re.compile(r"(?P<name>[A-Za-z0-9_]+)(?P<rest>(\[-?[0-9]+\])*)")

def _resolve_parent_and_key(obj: Any, path: str) -> tuple[Any, str | int] | None:
    """
    Walks down `path` and returns (parent_container, final_key_or_index).
    Supports deletes of:
      - dict fields: e.g., 'outputs[0].stats.xor_signature'
      - list elements: e.g., 'outputs[-1]'
    """
    cur_parent = None
    cur_key: str | int | None = None
    cur = obj

    segs = path.split(".")
    for i, seg in enumerate(segs):
        m = _SEG_ANY_RE.fullmatch(seg)
        if not m:
            return None
        name, rest = m.group("name"), m.group("rest") or ""

        # Step into dict[name]
        nxt = _step_into(cur, name)
        if nxt is None:
            return None

        # If there are indices, step through them
        idxs = _iter_indices(rest)
        for idx in idxs:
            if not isinstance(nxt, (list, tuple)):
                return None
            norm = _normalize_index(idx, len(nxt))
            if norm is None:
                return None
            nxt = nxt[norm]

        # If this is the last segment, we want to delete the *field inside nxt* OR the *element of nxt*
        if i == len(segs) - 1:
            if idxs:
                # last hop addressed a list element; so parent is the list before the last index
                # we need to re-walk to get that parent list ref + the last index normalized
                base = _step_into(cur, name)
                if base is None or not isinstance(base, list):
                    return None
                # resolve all but last index to get the list container
                target_list = base
                for idx in idxs[:-1]:
                    if not isinstance(target_list, list):
                        return None
                    norm = _normalize_index(idx, len(target_list))
                    if norm is None:
                        return None
                    target_list = target_list[norm]
                if not isinstance(target_list, list):
                    return None
                last_norm = _normalize_index(idxs[-1], len(target_list))
                if last_norm is None:
                    return None
                return (target_list, last_norm)
            else:
                # delete dict field `name` from dict `cur`
                return (cur, name)

        # advance: keep track of parent/key to allow deeper nesting
        cur_parent, cur_key = cur, name
        cur = nxt

    return None

def _del_by_path(obj: Any, path: str) -> bool:
    """
    Delete a dotted path (with optional indices). Returns True if deletion happened.
    Cases:
      - ... .foo.bar   -> delete key 'bar' from dict ...
      - ... .arr[3]    -> delete element 3 from 'arr' list
      - ... .arr[-1].stats.xor_signature -> delete nested key from dict inside that element
    """
    hit = _resolve_parent_and_key(obj, path)
    if not hit:
        return False

    parent, key = hit
    try:
        if isinstance(parent, dict) and isinstance(key, str):
            if key in parent:
                del parent[key]
                return True
        elif isinstance(parent, list) and isinstance(key, int):
            if 0 <= key < len(parent):
                # remove element entirely
                del parent[key]
                return True
    except Exception:
        pass
    return False

def _ensure_parent_and_delete_nested(obj: Any, path: str) -> bool:
    """
    If final segment includes dotted child (e.g., outputs[-1].stats.xor_signature),
    _del_by_path already handles that by returning parent container (dict or list).
    For safety, attempt multi-hop deletes iteratively:
    """
    # Progressive trim from full path to find deepest deletable parent
    parts = path.split(".")
    # Try direct delete first
    if _del_by_path(obj, path):
        return True
    # If final segment is deeper, progressively delete the final leaf
    # by descending to the leaf parent and deleting there.
    cur = obj
    for i in range(len(parts) - 1):
        head = ".".join(parts[:i+1])
        tail = ".".join(parts[i+1:])
        # resolve head to object
        val = _get_by_path(obj, head, default=None)
        if val is None:
            return False
        # try delete tail within val
        if _del_by_path(val, tail):
            return True
    return False

def _normalize_func_name(f: Any) -> list[str]:
    """
    Return candidate keys for lookup:
      - raw string
      - stripped <...>
      - last token after last dot
    """
    if not isinstance(f, str):
        return []
    raw = f
    stripped = f.strip("<>")
    base = stripped.split(".")[-1]
    # include also just raw in case configs use it
    cands = []
    for x in (raw, stripped, base):
        if x and x not in cands:
            cands.append(x)
    return cands

def _match_pattern(pattern: str, function_name: Any) -> bool:
    """
    Match a pattern against a function name.
    Supports:
      - Exact match
      - Substring match (backward compatible)
      - Wildcard match via fnmatch when pattern contains '*'
    """
    if not isinstance(function_name, str):
        return False
    cands = _normalize_func_name(function_name)
    if "*" in pattern:
        import fnmatch
        return any(fnmatch.fnmatch(c, pattern) for c in cands)
    # Exact or substring match
    return any(pattern == c or pattern in c for c in cands)

def _load_ignore_map(path: str) -> dict[str, list[str]]:
    if not path or not os.path.exists(path) or _yaml is None:
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = _yaml.safe_load(fh) or {}
        m = cfg.get("inspector_ignore_fields", {}) or {}
        # normalize values to list[str]
        out = {}
        for k, v in m.items():
            if isinstance(v, (list, tuple)):
                out[str(k)] = [str(x) for x in v]
            elif isinstance(v, str):
                out[str(k)] = [v]
        return out
    except Exception as e:
        sys.stderr.write(f"[warn] failed to read ignore-config {path}: {e}\n")
        return {}

def _collect_ignores_for_record(rec: dict, global_ignores: list[str], by_fn: dict[str, list[str]]) -> list[str]:
    paths = list(global_ignores)
    func = rec.get("function")
    if func:
        for pattern, keys in by_fn.items():
            if _match_pattern(pattern, func):
                paths.extend(keys)
    return paths

def _apply_ignores_inplace(rec: dict, paths: list[str]) -> None:
    # Fast path: unique only
    seen = set()
    for p in paths:
        p = p.strip()
        if not p or p in seen:
            continue
        seen.add(p)
        # Try deleting; support nested dict fields and list elements
        # Both whole-field deletes (e.g., 'outputs') and leaf deletes
        # (e.g., 'outputs[-1].stats.xor_signature') are handled.
        _ensure_parent_and_delete_nested(rec, p)


def _get_by_path(obj: Any, path: str, default: Any=None) -> Any:
    """
    Get a nested value by dotted path with optional list indices, e.g.:
      'meta.global_step', 'inputs.args[0].stats.numel'
    """
    cur = obj
    for seg in path.split("."):
        if seg == "":
            return default
        m = _SEG_RE.fullmatch(seg)
        if not m:
            return default
        name = m.group("name")
        rest = m.group("rest") or ""

        # dict step
        if not isinstance(cur, dict) or name not in cur:
            return default
        cur = cur[name]

        # list indices like [0][3]...
        if rest:
            if not isinstance(cur, (list, tuple)):
                return default
            for i_m in _IDX_RE.finditer(rest):
                idx = int(i_m.group(1))
                if idx < 0 or idx >= len(cur):
                    return default
                cur = cur[idx]
    return cur

def _try_parse_value(s: str) -> Any:
    """
    Parse a filter value. Tries JSON first (so numbers/bool/null/strings in quotes work),
    then falls back to raw string.
    Examples:
      1500 -> int
      3.14 -> float
      true/false/null -> proper JSON values
      "bf16" -> string
      bf16 -> string (fallback)
    """
    # Trim a single pair of surrounding quotes if user typed '\"text\"' by shell
    st = s.strip()
    try:
        return _loads(st)
    except Exception:
        return st

# ---------- Filter parsing ----------

class Filter:
    def __init__(self, path: str, op: str, value: Any):
        self.path = path
        self.op = op
        self.value = value
        if op not in COMPARATORS:
            raise ValueError(f"Unsupported operator: {op}")

    def match(self, trace: dict) -> bool:
        lhs = _get_by_path(trace, self.path, default=None)
        rhs = self.value
        # Best-effort numeric normalization when both look like numbers but one is a str
        if isinstance(lhs, (int,float)) and isinstance(rhs, str):
            try: rhs = float(rhs) if ('.' in rhs or 'e' in rhs.lower()) else int(rhs)
            except Exception: pass
        if isinstance(rhs, (int,float)) and isinstance(lhs, str):
            try: lhs = float(lhs) if ('.' in lhs or 'e' in lhs.lower()) else int(lhs)
            except Exception: pass
        return bool(COMPARATORS[self.op](lhs, rhs))

    def __repr__(self) -> str:
        v = self.value
        try:
            v_str = _dumps(v).decode("utf-8")
        except Exception:
            v_str = str(v)
        return f"{self.path}{self.op}{v_str}"

def parse_filter(expr: str) -> Filter:
    m = _FILTER_RE.match(expr)
    if not m:
        raise ValueError(f"Invalid filter expression: {expr!r}")
    path = m.group("path")
    op   = m.group("op")
    val  = _try_parse_value(m.group("val"))
    return Filter(path, op, val)

# ---------- I/O: streaming readers/writers that preserve format ----------

def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield _loads(line)
            except Exception as e:
                sys.stderr.write(f"[warn] JSONL parse error in {path}: {e}\n")

def read_json_array(path: Path) -> list[dict]:
    try:
        data = _loads(path.read_bytes())
        if isinstance(data, list):
            return data
        sys.stderr.write(f"[warn] JSON expected list root: {path}\n")
        return []
    except Exception as e:
        sys.stderr.write(f"[warn] JSON parse error in {path}: {e}\n")
        return []

def write_jsonl(path: Path, items: Iterable[dict]) -> int:
    n = 0
    with path.open("wb") as f:
        for obj in items:
            f.write(_dumps(obj)); f.write(b"\n"); n += 1
    return n

def write_json_array(path: Path, items: list[dict]) -> int:
    path.write_bytes(_dumps(items))
    return len(items)

# ---------- Core filtering ----------

TRACE_EXTS = {".jsonl", ".ndjson", ".json"}  # treat ndjson as jsonl

def is_trace_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in TRACE_EXTS

def filter_trace_file(in_path: Path, out_path: Path, filters: List[Filter],
                      ignore_global: list[str]=None, ignore_by_fn: dict[str,list[str]]=None) -> Tuple[int,int,str]:
    """
    Returns (kept, total, kind) where kind in {"jsonl","json"}
    """
    ignore_global = ignore_global or []
    ignore_by_fn = ignore_by_fn or []
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = in_path.suffix.lower()
    total = 0
    kept = 0
    if ext in {".jsonl", ".ndjson"}:
        def _matched():
            nonlocal total, kept
            for rec in iter_jsonl(in_path):
                total += 1
                paths = _collect_ignores_for_record(rec, ignore_global, ignore_by_fn)
                if paths:
                    _apply_ignores_inplace(rec, paths)
                if all(f.match(rec) for f in filters):
                    kept += 1
                    yield rec
        kind = "jsonl"
        write_jsonl(out_path, _matched())
    elif ext == ".json":
        data = read_json_array(in_path)
        total = len(data)
        filtered = []
        for rec in data:
            paths = _collect_ignores_for_record(rec, ignore_global, ignore_by_fn)
            if paths:
                _apply_ignores_inplace(rec, paths)
            if all(f.match(rec) for f in filters):
                filtered.append(rec)
        kept = len(filtered)
        kind = "json"
        write_json_array(out_path, filtered)
    else:
        return (0, 0, "skip")
    return (kept, total, kind)

# ---------- Directory traversal ----------

def walk_files(root: Path) -> Iterator[Path]:
    if root.is_file():
        yield root
    else:
        for p in root.rglob("*"):
            if p.is_file():
                yield p

def filter_trace_file_count_only(in_path: Path, filters: List[Filter],
                                 ignore_global: list[str]=None, ignore_by_fn: dict[str,list[str]]=None) -> Tuple[int,int,str]:
    ignore_global = ignore_global or []
    ignore_by_fn = ignore_by_fn or {}
    ext = in_path.suffix.lower()
    total = 0; kept = 0
    if ext in {".jsonl", ".ndjson"}:
        for rec in iter_jsonl(in_path):
            total += 1
            # apply ignores
            paths = _collect_ignores_for_record(rec, ignore_global, ignore_by_fn)
            if paths:
                _apply_ignores_inplace(rec, paths)
            if all(f.match(rec) for f in filters):
                kept += 1
        return kept, total, "jsonl"
    elif ext == ".json":
        data = read_json_array(in_path)
        total = len(data)
        k = 0
        for rec in data:
            paths = _collect_ignores_for_record(rec, ignore_global, ignore_by_fn)
            if paths:
                _apply_ignores_inplace(rec, paths)
            if all(f.match(rec) for f in filters):
                k += 1
        return k, total, "json"
    return 0, 0, "skip"


def _filter_jsonl_chunk(in_path: Path, start: int, end: int, filters: list[Filter], shard_path: Path,
                        ignore_global: list[str], ignore_by_fn: dict[str,list[str]]) -> int:
    kept = 0
    with in_path.open("rb") as f, shard_path.open("wb") as out:
        f.seek(start)
        # align to next newline if not at 0
        if start != 0:
            f.readline()
        pos = f.tell()
        while pos < end:
            line = f.readline()
            if not line:
                break
            pos = f.tell()
            s = line.strip()
            if not s:
                continue
            try:
                rec = _loads(s)
            except Exception:
                continue
            paths = _collect_ignores_for_record(rec, ignore_global, ignore_by_fn)
            if paths:
                _apply_ignores_inplace(rec, paths)
            if all(fl.match(rec) for fl in filters):
                out.write(_dumps(rec)); out.write(b"\n")
                kept += 1
    return kept


def _split_jsonl_ranges(path: Path, chunk_bytes: int) -> list[tuple[int,int]]:
    size = path.stat().st_size
    if size == 0:
        return []
    ranges = []
    start = 0
    while start < size:
        end = min(start + chunk_bytes, size)
        ranges.append((start, end))
        start = end
    return ranges

def filter_jsonl_parallel(in_path: Path, out_path: Path, filters: list[Filter],
                          workers: int, chunk_bytes: int, intra_workers: int,
                          ignore_global: list[str], ignore_by_fn: dict[str,list[str]]) -> tuple[int,int,str]:
    
    """Parallelize a single large JSONL via chunking."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Short-circuit small files to simpler streaming path
    if in_path.stat().st_size <= chunk_bytes or intra_workers <= 1:
        return filter_trace_file(in_path, out_path, filters, ignore_global, ignore_by_fn)

    ranges = _split_jsonl_ranges(in_path, chunk_bytes)
    if not ranges:
        # ensure empty file written only if write-empty requested by caller
        out_path.write_bytes(b"")
        return (0, 0, "jsonl")

    kept_total = 0
    total = 0
    with tempfile.TemporaryDirectory() as tdir:
        tdir = Path(tdir)
        jobs = []
        for i, (st, ed) in enumerate(ranges):
            shard = tdir / f"shard_{i:05d}.jsonl"
            jobs.append((st, ed, shard))

        def _submit_chunk(exec, in_path, st, ed, filters, shard):
            return exec.submit(_filter_jsonl_chunk, in_path, st, ed, filters, shard, ignore_global, ignore_by_fn)

        with ThreadPoolExecutor(max_workers=max(1, intra_workers)) as ex:
            futures = [
                _submit_chunk(ex, in_path, st, ed, filters, shard)
                for (st, ed, shard) in jobs
            ]
            kept_per = [f.result() for f in futures]

        # Concatenate shards in order
        with out_path.open("wb") as out:
            for (st, ed, shard), kept in zip(jobs, kept_per):
                if shard.exists() and shard.stat().st_size > 0:
                    with shard.open("rb") as f:
                        shutil.copyfileobj(f, out)
                kept_total += kept

    # Count total records (single pass count only; cheaper than parsing twice)
    # If you want exact total across chunks without reread, you can optionally
    # count during chunking by summing every line read; here we do a quick pass:
    for _ in iter_jsonl(in_path):
        total += 1

    return kept_total, total, "jsonl"

def _filter_one_file(src: Path, rel: str, out_root: Path, filters: list[Filter],
                     write_empty: bool, copy_others: bool,
                     jsonl_chunk_bytes: int, jsonl_chunk_workers: int,
                     ignore_global: list[str], ignore_by_fn: dict[str,list[str]]) -> tuple:
    dst = out_root / rel
    if is_trace_file(src):
        ext = src.suffix.lower()
        if ext in {".jsonl", ".ndjson"}:
            kept, tot, kind = filter_jsonl_parallel(
                src, dst, filters,
                workers=jsonl_chunk_workers,
                chunk_bytes=jsonl_chunk_bytes,
                intra_workers=jsonl_chunk_workers,
                ignore_global=ignore_global,
                ignore_by_fn=ignore_by_fn,
            )
        elif ext == ".json":
            kept, tot, kind = filter_trace_file(src, dst, filters, ignore_global, ignore_by_fn)
        else:
            return (rel, 0, 0, "skip", False, 0)

        removed = False
        if kept == 0 and not write_empty:
            try:
                if dst.exists():
                    if dst.suffix.lower() in {".jsonl", ".ndjson"} and dst.stat().st_size == 0:
                        dst.unlink(missing_ok=True); removed = True
                    elif dst.suffix.lower() == ".json":
                        try:
                            if _loads(dst.read_bytes()) == []:
                                dst.unlink(missing_ok=True); removed = True
                        except Exception:
                            pass
            except Exception:
                pass
        return (rel, kept, tot, kind, removed, 1)
    else:
        if copy_others:
            dst.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(src, dst)
                return (rel, 0, 0, "copy", False, 1)
            except Exception:
                return (rel, 0, 0, "copy-failed", False, 1)
        return (rel, 0, 0, "skip", False, 1)

def main():
    # ... keep existing argparse ...
    ap = argparse.ArgumentParser(
        description="Filter execution traces by key/value predicates and emit a new trace set preserving file formats."
    )
    ap.add_argument("input", help="Input trace file or directory")
    ap.add_argument("--out", required=True, help="Output directory for filtered traces")
    ap.add_argument("--filter", dest="filters", action="append", required=True,
                    help='Filter expression, e.g. --filter "meta.global_step=1500"')
    ap.add_argument("--write-empty", action="store_true")
    ap.add_argument("--copy-others", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    # NEW parallelism knobs
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Process pool size for file-level parallelism (default: CPU count)")
    ap.add_argument("--jsonl-chunk-bytes", type=int, default=64 * 1024 * 1024,
                    help="Chunk size (bytes) for intra-file parallelism on large JSONL (default: 64MB)")
    ap.add_argument("--jsonl-chunk-workers", type=int, default=0,
                    help="Workers for intra-file JSONL chunking (default: min(workers, 8))")

    ap.add_argument("--ignore-fields", type=str, nargs="*", default=[],
                    help="Fields to strip before filtering/writing. Dotted paths with indices allowed, e.g. "
                        "outputs stats.xor_signature outputs[-1].stats.xor_signature inputs.args[10]")
    default_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    ap.add_argument("--ignore-config", type=str, default=default_config_path,
                    help=f"YAML with per-function ignore fields under inspector_ignore_fields (default: {default_config_path})")



    args = ap.parse_args()

    ignore_global: list[str] = args.ignore_fields or []
    ignore_by_fn: dict[str, list[str]] = _load_ignore_map(args.ignore_config)


    in_root = Path(args.input).resolve()
    out_root = Path(args.out).resolve()
    if not in_root.exists():
        sys.stderr.write(f"Input does not exist: {in_root}\n")
        sys.exit(2)

    filters = [parse_filter(expr) for expr in args.filters]

    # Decide intra-file workers
    if args.jsonl_chunk_workers and args.jsonl_chunk_workers > 0:
        intra = args.jsonl_chunk_workers
    else:
        intra = max(1, min(args.workers, 8))

    if args.dry_run:
        # Keep your previous dry-run loop (single process) or just print hint:
        print("[dry-run] Parallel summary not computed; run without --dry-run for full stats.")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    # Walk inputs and build job list
    root_is_file = in_root.is_file()
    base_len = len(str(in_root.parent if root_is_file else in_root))
    files = list(walk_files(in_root))
    jobs = []
    for src in files:
        rel = str(src)[base_len+1:] if base_len > 0 else src.name
        jobs.append((src, rel))

    # Dispatch in a pool (file-level)
    with mp.Pool(processes=args.workers) as pool:
        func = partial(
            _filter_one_file,
            out_root=out_root,
            filters=filters,
            write_empty=args.write_empty,
            copy_others=args.copy_others,
            jsonl_chunk_bytes=args.jsonl_chunk_bytes,
            jsonl_chunk_workers=intra,
            ignore_global=ignore_global,
            ignore_by_fn=ignore_by_fn,
        )
        results = pool.starmap(func, jobs)

    # Summarize
    total_files = len(files)
    processed = sum(1 for _,_,_,k,_,_ in results if k in {"jsonl","json"})
    total_records = sum(tot for _,_,tot,_,_,_ in results)
    kept_records = sum(kept for _,kept,_,_,_,_ in results)
    skipped_empty = sum(1 for _,_,_,_,removed,_ in results if removed)
    copied_others = sum(1 for _,_,_,k,_,_ in results if k == "copy")

    print("======================================")
    print(f"Input:           {in_root}")
    print(f"Output:          {out_root}")
    print(f"Filters (AND):   {[repr(f) for f in filters]}")
    print("--------------------------------------")
    print(f"Files total:     {total_files}")
    print(f"Trace files:     {processed}")
    print(f"Records total:   {total_records}")
    print(f"Records kept:    {kept_records}")
    print(f"Files skipped(empty): {skipped_empty} (use --write-empty to keep)")
    if args.copy_others:
        print(f"Other files copied: {copied_others}")
    print(f"Workers:         files={args.workers}, jsonl-chunk={intra}, chunk_bytes={args.jsonl_chunk_bytes}")
    print("======================================")

if __name__ == "__main__":
    mp.freeze_support()
    main()
