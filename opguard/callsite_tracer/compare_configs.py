#!/usr/bin/env python3
import sys
import yaml
import json
import argparse
from typing import Any, Dict, List, Tuple

def load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def ensure_instrumentations(obj: Any) -> List[Dict[str, Any]]:
    # If the file already has a top-level 'instrumentations' list, use it;
    # otherwise, if the whole file is a list, treat that as instrumentations.
    if isinstance(obj, dict) and "instrumentations" in obj:
        val = obj["instrumentations"]
        if isinstance(val, list):
            return val
    if isinstance(obj, list):
        return obj
    # Fallback: empty
    return []

def norm_list_of_dicts(lst: List[Dict[str, Any]], key_fields: Tuple[str, ...]) -> List[Dict[str, Any]]:
    # Sort a list of dicts deterministically by a tuple of key fields (best-effort)
    def key_fn(d: Dict[str, Any]):
        return tuple(str(d.get(k, "")) for k in key_fields)
    return sorted(lst, key=key_fn)

def normalize_entry(e: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of entry with order-insensitive fields normalized."""
    e = json.loads(json.dumps(e))  # deep copy via JSON
    # Sort obvious list fields
    for k in ("ops", "tag_names"):
        if isinstance(e.get(k), list):
            e[k] = sorted(e[k])
    # Sort 'context' (list of {type,name,...}) deterministically
    if isinstance(e.get("context"), list):
        e["context"] = norm_list_of_dicts(e["context"], ("type", "name"))
    # Sort 'stream_counts' dict keys deterministically
    if isinstance(e.get("stream_counts"), dict):
        sc = e["stream_counts"]
        e["stream_counts"] = {k: sc[k] for k in sorted(sc.keys(), key=str)}
    return e

def entry_key(e: Dict[str, Any]) -> Tuple:
    """Build a stable identity key for an instrumentation entry."""
    ctx = e.get("context") or []
    ctx_path = tuple(f"{c.get('type','?')}:{c.get('name','?')}" for c in ctx if isinstance(c, dict))
    return (
        e.get("file_path"),
        e.get("line_num"),
        e.get("function"),
        e.get("operation_type"),
        ctx_path,
    )

def dict_diff(a: Any, b: Any, path: str = "") -> List[str]:
    """Deep diff two normalized JSON-like objects; return list of human-friendly differences."""
    diffs = []
    if type(a) != type(b):
        diffs.append(f"{path}: TYPE {type(a).__name__} -> {type(b).__name__}")
        return diffs

    if isinstance(a, dict):
        a_keys = set(a.keys()); b_keys = set(b.keys())
        for k in sorted(a_keys - b_keys, key=str):
            diffs.append(f"{path}.{k}: REMOVED ({json.dumps(a[k], ensure_ascii=False)})")
        for k in sorted(b_keys - a_keys, key=str):
            diffs.append(f"{path}.{k}: ADDED ({json.dumps(b[k], ensure_ascii=False)})")
        for k in sorted(a_keys & b_keys, key=str):
            diffs.extend(dict_diff(a[k], b[k], f"{path}.{k}" if path else k))
        return diffs

    if isinstance(a, list):
        # Treat lists as order-insensitive sequences of JSON-serializable items.
        # Convert elements to canonical JSON strings for multiset comparison.
        a_c = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in a]
        b_c = [json.dumps(x, sort_keys=True, ensure_ascii=False) for x in b]
        a_counts = {}
        b_counts = {}
        for x in a_c: a_counts[x] = a_counts.get(x, 0) + 1
        for x in b_c: b_counts[x] = b_counts.get(x, 0) + 1
        all_items = sorted(set(a_counts) | set(b_counts))
        for it in all_items:
            ca, cb = a_counts.get(it, 0), b_counts.get(it, 0)
            if ca != cb:
                pretty = json.loads(it)
                diffs.append(f"{path}: COUNT {ca} -> {cb} for item {json.dumps(pretty, ensure_ascii=False)}")
        return diffs

    # Primitive
    if a != b:
        diffs.append(f"{path}: {json.dumps(a, ensure_ascii=False)} -> {json.dumps(b, ensure_ascii=False)}")
    return diffs

def index_entries(entries: List[Dict[str, Any]]) -> Dict[Tuple, Dict[str, Any]]:
    idx = {}
    for e in entries:
        ne = normalize_entry(e)
        key = entry_key(ne)
        # Support duplicates: if duplicate key appears, append a numeric suffix index
        if key in idx:
            # Create a disambiguated key with a running counter
            i = 2
            while (key, i) in idx:
                i += 1
            idx[(key, i)] = ne
        else:
            idx[key] = ne
    return idx

def main():
    ap = argparse.ArgumentParser(description="Order-insensitive YAML comparator for instrumentations.")
    ap.add_argument("old_yaml", help="Old/left YAML file")
    ap.add_argument("new_yaml", help="New/right YAML file")
    ap.add_argument("-q", "--quiet", action="store_true", help="Only print counts")
    args = ap.parse_args()

    old = ensure_instrumentations(load_yaml(args.old_yaml))
    new = ensure_instrumentations(load_yaml(args.new_yaml))

    left = index_entries(old)
    right = index_entries(new)

    left_keys = set(left.keys())
    right_keys = set(right.keys())

    removed = sorted(list(left_keys - right_keys), key=str)
    added = sorted(list(right_keys - left_keys), key=str)
    common = sorted(list(left_keys & right_keys), key=str)

    modified = []
    diffs_by_key = {}

    for k in common:
        a = left[k]; b = right[k]
        diffs = dict_diff(a, b)
        if diffs:
            modified.append(k)
            diffs_by_key[k] = diffs

    # Output
    if args.quiet:
        print(json.dumps({
            "summary": {
                "old_count": len(old),
                "new_count": len(new),
                "added": len(added),
                "removed": len(removed),
                "modified": len(modified)
            }
        }, indent=2, ensure_ascii=False))
        return

    def fmt_key(k):
        # k can be (identity) or ((identity), n)
        if isinstance(k, tuple) and len(k) == 2 and isinstance(k[0], tuple):
            ident, dup = k
        else:
            ident, dup = k, None
        (file_path, line_num, function, op_type, ctx_path) = ident
        base = f"{file_path}:{line_num} :: {function} [{op_type}]"
        if ctx_path:
            base += " :: " + " / ".join(ctx_path)
        if dup is not None:
            base += f"  (dup#{dup})"
        return base

    print("=== Summary ===")
    print(json.dumps({
        "old_count": len(old),
        "new_count": len(new),
        "added": len(added),
        "removed": len(removed),
        "modified": len(modified)
    }, indent=2, ensure_ascii=False))
    print()

    if added:
        print("=== Added entries ===")
        for k in added:
            print(f"- {fmt_key(k)}")
        print()

    if removed:
        print("=== Removed entries ===")
        for k in removed:
            print(f"- {fmt_key(k)}")
        print()

    if modified:
        print("=== Modified entries (field-level diffs) ===")
        for k in modified:
            print(f"- {fmt_key(k)}")
            for d in diffs_by_key[k]:
                print(f"    • {d}")
        print()

if __name__ == "__main__":
    main()
