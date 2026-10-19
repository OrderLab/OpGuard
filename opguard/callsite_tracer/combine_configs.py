#!/usr/bin/env python3
import sys
import yaml
import argparse
import json
import os
from typing import Any, Dict, List, Tuple, Optional

def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_instrumentations(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "instrumentations" in obj:
        return obj["instrumentations"]
    if isinstance(obj, list):
        return obj
    return []

def _ctx_path(e: Dict[str, Any]) -> Tuple[str, ...]:
    ctx = e.get("context") or []
    return tuple(f"{c.get('type','?')}:{c.get('name','?')}" for c in ctx if isinstance(c, dict))

def entry_key(e: Dict[str, Any], ignore_lineno: bool) -> Tuple:
    """
    Build a key for matching entries.
    If ignore_lineno=True, exclude line_num from the key (allows fuzzy line matching).
    """
    base = (
        e.get("file_path"),
        # optionally skip line_num here
        None if ignore_lineno else e.get("line_num"),
        e.get("function"),
        e.get("operation_type"),
        _ctx_path(e),
        tuple(sorted(e.get("tag_names", []))),
    )
    return base

def normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(e))  # deep copy

def merge_ops(a: List[str], b: List[str]) -> List[str]:
    return sorted(set(a or []) | set(b or []))

def merge_stream_counts(a: Dict[str, int], b: Dict[str, int]) -> Dict[str, int]:
    out = {}
    a = a or {}
    b = b or {}
    for k in set(a.keys()) | set(b.keys()):
        out[k] = int(a.get(k, 0)) + int(b.get(k, 0))
    return out

def _equal_except(e1: Dict[str, Any], e2: Dict[str, Any], ignore_keys: set) -> bool:
    """Check equality of dicts except for keys in ignore_keys (missing vs None treated equal)."""
    keys = set(e1.keys()) | set(e2.keys())
    for k in keys:
        if k in ignore_keys:
            continue
        if (e1.get(k) != e2.get(k)):
            return False
    return True

def combine_entries(e1: Dict[str, Any], e2: Dict[str, Any], allow_lineno_drift: bool) -> Optional[Dict[str, Any]]:
    """
    Return combined entry if matchable, else None.
    If allow_lineno_drift is True, differences in 'line_num' are ignored.
    """
    keys_to_ignore = {"ops", "stream_counts"}
    if allow_lineno_drift:
        keys_to_ignore.add("line_num")

    if not _equal_except(e1, e2, keys_to_ignore):
        return None

    combined = normalize_entry(e1)
    combined["ops"] = merge_ops(e1.get("ops", []), e2.get("ops", []))
    combined["stream_counts"] = merge_stream_counts(e1.get("stream_counts"), e2.get("stream_counts"))

    # Preserve both line numbers if they differ (useful for diagnostics)
    ln1 = e1.get("line_num")
    ln2 = e2.get("line_num")
    if allow_lineno_drift and ln1 != ln2:
        combined["_line_num_pair"] = {"trace1": ln1, "trace2": ln2}
        # keep e1's line_num as the canonical one; you can change this policy if desired
        combined["line_num"] = ln1

    return combined

def closest_by_lineno(candidates: List[Dict[str, Any]], target_lineno: Optional[int]) -> List[Dict[str, Any]]:
    """
    Return candidates sorted by absolute distance of 'line_num' to target.
    If target or candidate line_num is missing, treat distance as +inf (push to end).
    """
    def dist(e):
        ln = e.get("line_num")
        if target_lineno is None or ln is None:
            return float("inf")
        try:
            return abs(int(ln) - int(target_lineno))
        except Exception:
            return float("inf")
    return sorted(candidates, key=dist)

def main():
    ap = argparse.ArgumentParser(description="Combine two YAML traces by extracting common entries.")
    ap.add_argument("trace1", help="First YAML trace file")
    ap.add_argument("trace2", help="Second YAML trace file")
    ap.add_argument("-o", "--output", required=True, help="Output directory")

    # Toggle for line number leniency (default: allow drift)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--strict-lineno", dest="allow_lineno_drift", action="store_false",
                       help="Require exact line_num match")
    group.add_argument("--fuzzy-lineno", dest="allow_lineno_drift", action="store_true",
                       help="Allow different line_num while matching other fields (default)")
    ap.set_defaults(allow_lineno_drift=True)

    args = ap.parse_args()

    trace1 = ensure_instrumentations(load_yaml(args.trace1))
    trace2 = ensure_instrumentations(load_yaml(args.trace2))

    # Build index for trace2; include/exclude line_num in the key depending on toggle
    idx2 = {}
    for e in trace2:
        key = entry_key(e, ignore_lineno=args.allow_lineno_drift)
        idx2.setdefault(key, []).append(e)

    combined = []
    for e1 in trace1:
        key = entry_key(e1, ignore_lineno=args.allow_lineno_drift)
        candidates = idx2.get(key, [])
        if not candidates:
            continue

        # If fuzzy, prefer the candidate whose line_num is closest to e1's
        if args.allow_lineno_drift:
            candidates = closest_by_lineno(candidates, e1.get("line_num"))

        merged: Optional[Dict[str, Any]] = None
        for e2 in candidates:
            merged = combine_entries(e1, e2, allow_lineno_drift=args.allow_lineno_drift)
            if merged is not None:
                combined.append(merged)
                break  # use first valid match

    result = {"instrumentations": combined}

    os.makedirs(args.output, exist_ok=True)
    out_path = os.path.join(args.output, "combined.yaml")
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(result, f, sort_keys=True, allow_unicode=True)

    print("=== Stats ===")
    print(f"Trace1 entries: {len(trace1)}")
    print(f"Trace2 entries: {len(trace2)}")
    print(f"Combined entries: {len(combined)}")
    print(f"Allow lineno drift: {args.allow_lineno_drift}")
    print(f"Output written to: {out_path}")

if __name__ == "__main__":
    main()
